"""Reference transactions, their inclusion rules and the two-person review (PLAN.md 4.5b).

The repository's transactions must pass every rule; each rule must refuse
what it is there to refuse; the figures the screens derive are checked by
hand; a transaction joins the library only on two approvals by
administrators other than its proposer; and the fees and amortisation the
approved deals give are their medians, which move a deal's IRR only when
applied as Settings.
"""
import copy
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from api.auth import AuthUser, require_user
from api.main import app
from api.schemas import ReferenceDealIn
from db import engine as db_engine
from db.models import AuditEvent, ReferenceDeal
from library import fees, references
from tests.test_deals import PROFILE

client = TestClient(app)
A, B, C = "user_admin_a", "user_admin_b", "user_admin_c"
SEEDS = references.repository_deals()
BY_KEY = {d["key"]: d for d in SEEDS}


def seed(key: str) -> dict:
    return copy.deepcopy(BY_KEY[key])


# ---------------------------------------------------------------------------
# The repository's transactions and the inclusion rules
# ---------------------------------------------------------------------------
def test_the_repository_proposes_ten_transactions_across_regions_and_outcomes():
    assert len(SEEDS) == 10 and len(BY_KEY) == 10
    tags = [references.tags(d) for d in SEEDS]
    assert {t["region"] for t in tags} == {"us", "europe", "emerging", "other_developed"}
    assert {t["outcome"] for t in tags} == {"success", "distress"}
    assert {t["era"] for t in tags} == {"before_2000", "2000_2007", "2008_2014", "2015_2019"}


@pytest.mark.parametrize("key", sorted(BY_KEY))
def test_every_repository_transaction_passes_the_rules_and_the_api_schema(key):
    deal = seed(key)
    assert references.problems(deal, today=date(2026, 10, 7)) == []
    ReferenceDealIn.model_validate(deal)
    # Every source is an SEC filing a reviewer can open
    assert all(s["url"].startswith("https://www.sec.gov/Archives/edgar/data/") for s in deal["sources"].values())


def test_the_figures_derived_from_hca_match_a_hand_calculation():
    d = references.derived(seed("hca-2006"))
    assert d["entry_multiple"] == round(33000 / 4327, 2) == 7.63
    assert d["leverage"] == round(19964 / 4327, 2) == 4.61
    assert d["tx_fee_pct"] == round((131 + 77) / 33000 * 100, 2) == 0.63
    assert d["fin_fee_pct"] == round(568 / 19964 * 100, 2) == 2.85
    assert d["def_senior_amort"] is None


def test_a_deal_outside_the_dollar_is_sized_by_the_filings_own_dollar_value():
    nxp = references.tags(seed("nxp-2006"))
    # EUR 8,208m paid; the IPO prospectus states it as $10,601m
    assert nxp == {"region": "europe", "size": "over_10bn", "sector": "information_technology",
                   "era": "2000_2007", "outcome": "success"}
    assert references.tags(seed("focus-media-2013"))["region"] == "emerging"
    assert references.tags(seed("masonite-2005"))["region"] == "other_developed"


def broken(change) -> list[str]:
    deal = seed("dollar-general-2007")
    change(deal)
    return [p["code"] for p in references.problems(deal, today=date(2026, 10, 7))]


def _set(path: str, value):
    def change(deal):
        *head, last = path.split(".")
        node = deal
        for k in head:
            node = node[int(k)] if k.isdigit() else node[k]
        node[last] = value
    return change


@pytest.mark.parametrize("change,code", [
    (_set("sponsors", []), "no_sponsor"),
    (lambda d: d["figures"].pop("ebitda"), "missing_figure"),
    (_set("currency", "EUR"), "missing_figure"),
    (_set("sources.dg_10k_2007.url", "https://www.reuters.com/a-story"), "not_a_filing"),
    (_set("sources.dg_10k_2007.url", "http://www.sec.gov/Archives/x.htm"), "not_a_filing"),
    (_set("figures.equity.source", "nowhere"), "missing_source"),
    (lambda d: d["sources"].update(extra={"filer": "x", "form": "8-K", "filed": "2007-01-01",
                                          "url": "https://www.sec.gov/x"}), "unused_source"),
    (_set("figures.debt.parts.0.value", 2400), "parts_dont_add_up"),
    (_set("figures.equity.value", 0), "not_positive"),
    (_set("figures.transaction_value.value", 30000), "multiple_out_of_range"),
    (_set("figures.debt.parts.0.value", 5000), "parts_dont_add_up"),
    (_set("figures.financing_fees.value", 500), "fee_out_of_range"),
    (_set("figures.senior_amort_pct.value", 150), "amortisation_out_of_range"),
    (_set("closed.date", "2027-01-01"), "closed_in_future"),
    (_set("outcome.year", 2006), "outcome_before_close"),
    (_set("outcome.event", "bankruptcy"), "event_doesnt_match_outcome"),
    (_set("figures.bogus", {"value": 1, "source": "dg_10k_2007", "where": "x"}), "unknown_figure"),
])
def test_each_inclusion_rule_refuses_what_it_is_for(change, code):
    assert code in broken(change)


def test_debt_above_the_value_paid_is_refused():
    def change(deal):
        deal["figures"]["debt"] = {"value": 7200, "source": "dg_10k_2007", "where": "Item 1"}
    assert "debt_exceeds_value" in broken(change)


def test_the_balance_rules_flag_a_lopsided_bucket_and_name_the_empty_ones_filled():
    us = [seed(k) for k in ("hca-2006", "toys-r-us-2005", "dollar-general-2007", "dominos-1998",
                            "gymboree-2010")]
    over = references.balance(us, seed("dun-bradstreet-2019"))["over"]
    assert {"dimension": "region", "bucket": "us", "share_pct": 100.0} in over
    fills = references.balance(us, seed("nxp-2006"))["fills"]
    assert {"dimension": "region", "bucket": "europe"} in fills
    assert {"dimension": "region", "bucket": "us"} not in fills
    # Too small a library to call anything lopsided
    assert references.balance(us[:2], seed("dun-bradstreet-2019"))["over"] == []


# ---------------------------------------------------------------------------
# Fees and amortisation from the approved transactions
# ---------------------------------------------------------------------------
def test_the_sourced_fees_are_the_medians_of_the_deals_that_give_them():
    got = fees.sourced(SEEDS)
    # Transaction fees, % of value: HCA .63, Domino's .64, DG .89, Gymboree 1.58, D&B 3.29, Toys 3.53
    assert got["tx_fee_pct"] == {"value": round((0.89 + 1.58) / 2, 2), "n": 6, "low": 0.63, "high": 3.53,
                                 "first_year": 1998, "last_year": 2019,
                                 "deals": ["dollar-general-2007", "dominos-1998", "dun-bradstreet-2019",
                                           "gymboree-2010", "hca-2006", "toys-r-us-2005"]}
    # Financing fees, % of debt: 1.94, 2.85, 2.87, 3.07 (Toys 135/4,400), 3.08, 5.08, 6.00
    assert (got["fin_fee_pct"]["value"], got["fin_fee_pct"]["n"]) == (round(135 / 4400 * 100, 2), 7)
    assert (got["fin_fee_pct"]["low"], got["fin_fee_pct"]["high"]) == (1.94, 6.0)
    # Senior amortisation: 1% a year on Dollar General's, Gymboree's and D&B's term loans
    assert got["def_senior_amort"] == {"value": 1.0, "n": 3, "low": 1.0, "high": 1.0, "first_year": 2007,
                                       "last_year": 2019,
                                       "deals": ["dollar-general-2007", "dun-bradstreet-2019", "gymboree-2010"]}


def test_a_figure_fewer_than_three_deals_give_is_not_offered():
    got = fees.sourced([seed("dollar-general-2007"), seed("gymboree-2010")])
    assert got == {"tx_fee_pct": None, "fin_fee_pct": None, "def_senior_amort": None}


# ---------------------------------------------------------------------------
# The two-person review, through the API and a real database
# ---------------------------------------------------------------------------
@pytest.fixture
def admins(monkeypatch):
    monkeypatch.setenv("FSE_ADMINS", f"{A},{B},{C}")


@pytest.fixture
def as_user():
    made = set()

    def sign_in(subject: str):
        app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)
        if subject not in made:
            assert client.post("/api/account", json=PROFILE).status_code == 200
            made.add(subject)
    yield sign_in


def queue() -> dict:
    resp = client.get("/api/library/review")
    assert resp.status_code == 200, resp.text
    return resp.json()


def proposal_id(key: str) -> str:
    return next(p["id"] for p in queue()["proposals"] if p["key"] == key)


def verdict(pid: str, v: str, reason=None):
    return client.post(f"/api/library/review/{pid}", json={"verdict": v, **({"reason": reason} if reason else {})})


def approve_all(as_user):
    as_user(A)
    ids = [p["id"] for p in queue()["proposals"]]
    for pid in ids:
        assert verdict(pid, "approve").status_code == 200
    as_user(B)
    for pid in ids:
        assert verdict(pid, "approve").json()["status"] == "approved"


def test_only_administrators_see_the_queue_or_propose(fresh_db, admins, as_user):  # noqa: ARG001
    as_user("user_someone")
    assert client.get("/api/library/review").status_code == 403
    assert client.post("/api/library/review", json=seed("hca-2006")).status_code == 403


def test_the_repository_queue_is_added_once_with_every_rule_passing(fresh_db, admins, as_user):  # noqa: ARG001
    as_user(A)
    first = queue()
    assert len(first["proposals"]) == 10 and first["library_size"] == 0
    assert all(p["origin"] == "repository" and p["problems"] == [] and p["approvals"] == 0 for p in first["proposals"])
    assert len(queue()["proposals"]) == 10
    with db_engine.connect() as conn:
        assert conn.execute(select(ReferenceDeal.id)).all().__len__() == 10


def test_a_transaction_joins_the_library_on_the_second_approval(fresh_db, admins, as_user):  # noqa: ARG001
    as_user(A)
    pid = proposal_id("hca-2006")
    one = verdict(pid, "approve").json()
    assert (one["status"], one["approvals"], one["my_verdict"]) == ("proposed", 1, "approve")
    assert verdict(pid, "approve").status_code == 409          # once each
    assert client.get("/api/library/references").json()["deals"] == []

    as_user(B)
    two = verdict(pid, "approve").json()
    assert (two["status"], two["approvals"]) == ("approved", 2)
    refs = client.get("/api/library/references").json()
    assert [d["key"] for d in refs["deals"]] == ["hca-2006"] and refs["awaiting_review"] == 9
    assert refs["deals"][0]["derived"]["entry_multiple"] == 7.63
    assert refs["deals"][0]["approved_at"]
    cov = client.get("/api/library/coverage").json()["collections"][0]
    assert (cov["id"], cov["count"], cov["awaiting_review"]) == ("reference_deals", 1, 9)
    assert {b["bucket"]: b["count"] for b in cov["dimensions"]["region"]}["us"] == 1
    assert {b["bucket"]: b["count"] for b in cov["dimensions"]["sector"]}["health_care"] == 1
    assert verdict(pid, "approve").status_code == 409          # decided
    with db_engine.connect() as conn:
        rows = conn.execute(select(AuditEvent.action, AuditEvent.detail)
                            .where(AuditEvent.action == "reference_reviewed").order_by(AuditEvent.id)).all()
    assert [tuple(r) for r in rows] == [("reference_reviewed", {"reference": pid, "verdict": "approve"})] * 2
    history = client.get("/api/account/history").json()["entries"]
    assert (history[0]["action"], history[0]["reference"], history[0]["verdict"]) == (
        "reference_reviewed", pid, "approve")


def test_a_proposer_never_reviews_their_own_and_two_others_decide(fresh_db, admins, as_user):  # noqa: ARG001
    as_user(A)
    deal = seed("hca-2006")
    deal["key"] = "hca-2006-restated"
    made = client.post("/api/library/review", json=deal)
    assert made.status_code == 201, made.text
    pid = made.json()["id"]
    assert made.json()["mine"] is True and made.json()["origin"] == "user"
    assert verdict(pid, "approve").status_code == 403
    assert client.post("/api/library/review", json=deal).status_code == 409   # the same content again
    as_user(B)
    assert verdict(pid, "approve").json()["status"] == "proposed"
    as_user(C)
    assert verdict(pid, "approve").json()["status"] == "approved"
    with db_engine.connect() as conn:
        assert conn.execute(select(AuditEvent.action).where(AuditEvent.action == "reference_proposed")).all()


def test_one_rejection_with_a_reason_decides_it(fresh_db, admins, as_user):  # noqa: ARG001
    as_user(A)
    pid = proposal_id("avago-2005")
    assert verdict(pid, "reject").status_code == 422           # a rejection says why
    rejected = verdict(pid, "reject", "figure_wrong").json()
    assert (rejected["status"], rejected["reasons"]) == ("rejected", ["figure_wrong"])
    as_user(B)
    assert verdict(pid, "approve").status_code == 409
    assert "avago-2005" not in [p["key"] for p in queue()["proposals"]]
    assert queue()["decided"][0]["key"] == "avago-2005"


def test_the_rules_block_approval_of_a_proposal_that_breaks_them(fresh_db, admins, as_user):  # noqa: ARG001
    as_user(A)
    deal = seed("gymboree-2010")
    deal["key"] = "gymboree-2010-press"
    deal["sources"]["gym_10k_2011"]["url"] = "https://www.example.com/press-release"
    made = client.post("/api/library/review", json=deal).json()
    assert {"code": "not_a_filing", "at": "transaction_value"} in made["problems"]
    as_user(B)
    refused = verdict(made["id"], "approve")
    assert refused.status_code == 409
    assert {"code": "not_a_filing", "at": "transaction_value"} in refused.json()["detail"]["problems"]


def test_an_approved_correction_supersedes_the_older_version(fresh_db, admins, as_user):  # noqa: ARG001
    as_user(A)
    pid = proposal_id("masonite-2005")
    verdict(pid, "approve")
    as_user(B)
    verdict(pid, "approve")
    corrected = seed("masonite-2005")
    corrected["figures"]["ebitda"]["value"] = 287.4
    made = client.post("/api/library/review", json=corrected).json()
    assert made["replaces_approved"] is True
    as_user(A)
    verdict(made["id"], "approve")
    as_user(C)
    verdict(made["id"], "approve")
    refs = client.get("/api/library/references").json()["deals"]
    assert [(d["key"], d["figures"]["ebitda"]["value"]) for d in refs] == [("masonite-2005", 287.4)]
    with db_engine.connect() as conn:
        statuses = conn.execute(select(ReferenceDeal.status).where(ReferenceDeal.key == "masonite-2005")
                                .order_by(ReferenceDeal.created_at)).scalars().all()
    assert statuses == ["superseded", "approved"]


def test_the_sourced_fees_follow_the_approved_deals_and_move_the_irr_only_when_applied(fresh_db, admins, as_user):  # noqa: ARG001,E501
    deal = {"ebitda": 100.0, "entry_mult": 10.0, "exit_mult": 10.0, "hold": 5}

    def irr(settings: dict) -> float:
        return client.post("/api/deal/run", json={"inputs": deal, "settings": settings}).json()["returns"]["irr"]

    plain = irr({})
    before = client.get("/api/library/fees").json()
    assert before["enabled"] and before["library_size"] == 0
    assert before["settings"] == {"tx_fee_pct": None, "fin_fee_pct": None, "def_senior_amort": None}
    approve_all(as_user)
    after = client.get("/api/library/fees").json()
    assert after["library_size"] == 10 and after["settings"] == fees.sourced(SEEDS)
    assert irr({}) == plain          # approving deals moves no result by itself
    applied = {k: v["value"] for k, v in after["settings"].items()}
    # Lower fees (1.24% of value against 2.3%) leave a smaller equity cheque, so a higher IRR
    assert irr(applied) > plain


def test_with_the_library_off_nothing_of_it_answers(fresh_db, admins, as_user, monkeypatch):  # noqa: ARG001
    as_user(A)
    monkeypatch.setenv("FSE_EXAMPLE_LIBRARY", "0")
    assert client.get("/api/library/references").json() == {"enabled": False, "deals": [], "awaiting_review": 0}
    assert client.get("/api/library/review").json()["enabled"] is False
    assert client.get("/api/library/fees").json()["enabled"] is False
    assert client.post("/api/library/review", json=seed("hca-2006")).status_code == 409


def test_without_a_database_the_library_has_no_transactions_and_review_says_why(admins, as_user, monkeypatch):  # noqa: ARG001
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app.dependency_overrides[require_user] = lambda: AuthUser(subject=A)
    assert client.get("/api/library/references").json() == {"enabled": True, "deals": [], "awaiting_review": 0}
    assert client.get("/api/library/review").status_code == 503


def test_the_browser_tests_recorded_answers_are_current():
    from tests import e2e_references
    assert e2e_references.OUT.read_text(encoding="utf-8") == e2e_references.render(), (
        "web/e2e/fixtures/references.json is stale: run python -m tests.e2e_references")
