"""Model validation (PLAN.md 4.6).

Done when: a report is generated with all splits (here from a real database
and the task the scheduler calls; staging.yml on staging); anonymisation is
tested (below: no deal, owner, name or figure in a report, small groups
suppressed, the complement rule, opting out); the deal summary shows the
full metric set (the deal answer's ``credit`` block and the simulation's
``p_loss`` here, web/e2e/validation.spec.ts on the screen).

Every statistic is checked by hand on small sets of cases, and every case
against the model's own answers (/api/deal/run, plan vs actual).
"""
import json
import math
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text, update

from api.auth import AuthUser, require_user
from api.main import app
from core.risk_sources import cumulative_default_pct, rating_for_coverage
from db import references as stored_references
from db.engine import connect, transaction
from db.models import AuditEvent, Deal
from jobs.scheduled import TASKS
from library import references
from validation import cases, report, run, tags
from validation.cases import Case

client = TestClient(app)

TODAY = date(2026, 10, 8)


def deal(key: str) -> dict:
    return next(d for d in references.repository_deals() if d["key"] == key)


# ---------------------------------------------------------------------------
# The deal summary's full metric set: coverage, default risk, probability of loss
# ---------------------------------------------------------------------------
def test_every_deal_answer_carries_its_coverage_and_default_risk_worked_by_hand():
    answer = client.post("/api/deal/run", json={"inputs": {}, "settings": {}}).json()
    om = answer["operating_model"]
    coverage = om["ebit"][0] / om["interest_expense"][0]
    rating, low, high = rating_for_coverage(coverage)
    credit = answer["credit"]
    assert credit["coverage"] == pytest.approx(coverage, rel=1e-12)
    assert (credit["rating"], credit["band_low"], credit["band_high"]) == (rating, low, high)
    assert credit["years"] == 5
    assert credit["default_pct"] == cumulative_default_pct(rating, 5)
    assert [s["id"] for s in credit["sources"]] == ["damodaran_ratings_2026", "sp_default_study_2024",
                                                   "deal_model"]


def test_the_credit_block_is_the_implied_rating_warnings_reading_whenever_that_is_raised():
    # Heavy debt: a speculative grade, so the warning is raised too
    answer = client.post("/api/deal/run", json={"inputs": {"debt_pct": 80, "base_rate": 9}}).json()
    warning = next(w for w in answer["risk_warnings"] if w["id"] == "implied_rating")
    credit = answer["credit"]
    assert credit["speculative"] is True
    for k in ("coverage", "band_low", "band_high", "default_pct"):
        assert warning["figures"][k] == pytest.approx(credit[k])
    assert warning["labels"] == {"rating": credit["rating"], "study_row": credit["study_row"]}


def test_a_deal_without_debt_has_no_coverage_and_says_so():
    credit = client.post("/api/deal/run", json={"inputs": {"debt_pct": 0}}).json()["credit"]
    assert credit["coverage"] is None and credit["default_pct"] is None and credit["rating"] is None
    assert credit["years"] == 5


def test_the_simulations_probability_of_loss_is_its_share_of_paths_below_1x():
    body = {"mc": {"n": 4000, "hold": 5}, "deal": {}, "settings": {}, "seed": 7}
    answer = client.post("/api/montecarlo/run", json=body)
    assert answer.status_code == 200, answer.text
    summary = answer.json()["summary"]
    from core.montecarlo import probability_of_loss
    assert probability_of_loss([0.5, 0.99, 1.0, 2.0]) == 0.5
    # Below 1x is exactly an IRR below zero on a single investment and exit
    assert 0.0 <= summary["p_loss"] <= 1.0
    assert summary["p_loss"] <= 1.0 - summary["p_above_hurdle"] + 1e-12


# ---------------------------------------------------------------------------
# Cases: a reference transaction against the app's own answer
# ---------------------------------------------------------------------------
def headline_inputs(d: dict) -> dict:
    v, e, debt = (references.value(d, k) for k in ("transaction_value", "ebitda", "debt"))
    return {"ebitda": e, "entry_mult": v / e, "debt_pct": debt / v * 100}


@pytest.mark.parametrize("key", ["hca-2006", "masonite-2005", "avago-2005", "focus-media-2013"])
def test_a_reference_transactions_prediction_is_what_the_deal_screen_answers_for_its_headline_figures(key):
    d = deal(key)
    screen = client.post("/api/deal/run", json={"inputs": headline_inputs(d)}).json()["credit"]
    predicted = cases.predicted_default(d)
    assert predicted["coverage"] == pytest.approx(screen["coverage"], rel=1e-12)
    assert predicted["default_pct"] == screen["default_pct"]


def test_distress_within_the_hold_happened_and_an_exit_or_later_distress_did_not():
    masonite = cases.library_case(deal("masonite-2005"), today=TODAY)   # closed 2005, distress 2009
    toys = cases.library_case(deal("toys-r-us-2005"), today=TODAY)      # distress 2017, after the hold
    hca = cases.library_case(deal("hca-2006"), today=TODAY)              # IPO 2011
    assert (masonite.happened, toys.happened, hca.happened) == (True, False, False)
    assert masonite.predicted == pytest.approx(cases.predicted_default(deal("masonite-2005"))["default_pct"] / 100)
    # Masonite: a Canadian building products maker, $2.9bn (filed in US dollars)
    assert masonite.groups == {"region": "other_developed", "sector": "industrials", "size": "1bn_10bn",
                               "era": "2000_2007"}


def test_a_deal_still_held_counts_only_once_its_hold_has_passed():
    held = {**deal("dun-bradstreet-2019"), "outcome": {"kind": "held", "event": "held", "year": 2019,
                                                        "source": "dnb_10k_2019", "where": "x"}}
    assert cases.library_case(held, today=date(2023, 6, 1)) is None
    assert cases.library_case(held, today=date(2024, 6, 1)).happened is False


def test_only_outcomes_newer_than_the_published_tables_are_out_of_time():
    # S&P's sample ends 2024; Damodaran's January 2026 table reads 2025
    assert cases.fit_until() == 2025
    for d in references.repository_deals():
        found = cases.library_case(d, today=TODAY)
        assert found is None or not found.out_of_time, d["key"]
    future = {**deal("masonite-2005"), "closed": {**deal("masonite-2005")["closed"], "date": "2024-03-01"},
              "outcome": {**deal("masonite-2005")["outcome"], "year": 2026}}
    assert cases.library_case(future, today=TODAY).out_of_time is True
    early = {**future, "outcome": {**future["outcome"], "year": 2025}}
    assert cases.library_case(early, today=TODAY).out_of_time is False


def test_a_users_deal_is_grouped_by_its_country_industry_size_and_year():
    groups = tags.deal_tags(country="DE", industry="machinery", ev=900.0, currency="EUR", year=2026,
                            usd_per=lambda ccy: 1.2 if ccy == "EUR" else None)
    assert groups == {"region": "europe", "sector": "industrials", "size": "1bn_10bn", "era": "2020_on"}
    unknown = tags.deal_tags(country="", industry="", ev=900.0, currency="JPY", year=2026,
                             usd_per=lambda ccy: None)
    assert unknown == {"region": "unknown", "sector": "unknown", "size": "unknown", "era": "2020_on"}


def test_every_industry_a_deal_can_start_from_has_a_sector():
    fixture = json.loads((tags.references.DATA.parents[1] / "web/e2e/fixtures/benchmarks.json")
                         .read_text(encoding="utf-8"))
    ids = {i["id"] for i in fixture["industries"]["industries"]} - {"all", "diversified"}
    assert ids == set(tags.INDUSTRY_SECTOR)
    assert set(tags.INDUSTRY_SECTOR.values()) <= set(references.SECTORS)


# ---------------------------------------------------------------------------
# Statistics, by hand
# ---------------------------------------------------------------------------
def prob(p, happened, origin="library", **groups):
    g = {"region": "us", "sector": "industrials", "size": "1bn_10bn", "era": "2000_2007", **groups}
    return Case(origin=origin, check="default", groups=g, out_of_time=True, predicted=p, happened=happened)


def ranged(percentile, error, origin="contributed", **groups):
    g = {"region": "us", "sector": "industrials", "size": "1bn_10bn", "era": "2020_on", **groups}
    return Case(origin=origin, check="irr_range", groups=g, out_of_time=True, percentile=percentile,
                irr_error_pp=error)


def test_probability_calibration_and_bias_worked_by_hand():
    found = [prob(0.1, False), prob(0.2, True), prob(0.3, False), prob(0.4, True), prob(0.5, False)]
    s = report.probability_stats(found)
    # expected 1.5, observed 2; variance .09+.16+.21+.24+.25 = .95
    assert (s["expected"], s["observed"]) == (1.5, 2)
    assert (s["predicted_pct"], s["observed_pct"], s["bias_pp"]) == (30.0, 40.0, 10.0)
    assert s["brier"] == pytest.approx((0.01 + 0.64 + 0.09 + 0.36 + 0.25) / 5, abs=1e-4)
    assert s["z"] == pytest.approx(0.5 / math.sqrt(0.95), abs=0.005)
    assert s["consistent"] is True
    assert report.probability_stats([prob(0.01, True)] * 5)["consistent"] is False


def test_range_calibration_worked_by_hand():
    # Percentiles 10, 30, 50, 70, 95: inside the central 50% (25-75): 3; 80% (10-90): 4; 90% (5-95): 5
    s = report.range_stats([ranged(p, e) for p, e in ((10, -4), (30, -1), (50, 0), (70, 2), (95, 8))])
    assert [lv["inside_pct"] for lv in s["levels"]] == [60.0, 80.0, 100.0]
    assert (s["bias_pp"], s["mean_percentile"]) == (1.0, 51.0)
    low, high = report.wilson(3, 5)
    assert s["levels"][0]["interval_pct"] == [round(low, 2), round(high, 2)]
    # Wilson's interval for 3 of 5 by its formula
    p, z, n = 0.6, 1.96, 5
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    assert low == pytest.approx((centre - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)) * 100)
    assert s["consistent"] is True
    always_outside = report.range_stats([ranged(99.5, 20)] * 10)
    assert always_outside["consistent"] is False and always_outside["levels"][2]["inside_pct"] == 0.0


def test_a_group_needs_min_cases_for_statistics_and_every_bucket_is_listed():
    out = report.sample("default", [prob(0.1, False)] * 4 + [prob(0.1, False, region="europe")])
    assert out["overall"]["enough"] is True and out["overall"]["stats"]["observed"] == 0
    regions = {b["bucket"]: b for b in out["splits"]["region"]}
    assert list(regions) == ["us", "europe", "emerging", "other_developed"]
    assert (regions["us"]["n"], regions["us"]["enough"], regions["us"]["stats"]) == (4, False, None)
    assert regions["emerging"]["n"] == 0
    assert set(out["splits"]) == {"region", "sector", "size", "era"}
    assert len(out["splits"]["sector"]) == len(references.SECTORS)


# ---------------------------------------------------------------------------
# Anonymity
# ---------------------------------------------------------------------------
def test_a_group_with_fewer_than_five_users_deals_shows_neither_its_figures_nor_their_count():
    found = [ranged(50, 0)] * 6 + [ranged(50, 0, region="europe")] * 3 + [ranged(50, 0, region="europe",
                                                                                origin="library")] * 2
    regions = {b["bucket"]: b for b in report.split("irr_range", found, "region")}
    assert regions["europe"] == {"bucket": "europe", "n": None, "library_n": 2, "contributed_n": None,
                                 "suppressed": True, "enough": False, "stats": None}


def test_the_shown_groups_never_leave_fewer_than_five_users_deals_to_work_out_by_subtraction():
    # Europe holds 3: the overall figures less the US's and emerging markets'
    # would be Europe's alone, so the smaller of those (emerging, 5) is hidden
    # too; the US stays. Library-only groups always stay.
    found = ([ranged(50, 0)] * 6 + [ranged(50, 0, region="europe")] * 3
             + [ranged(50, 0, region="emerging")] * 5
             + [ranged(50, 0, origin="library", region="other_developed")] * 5)
    regions = {b["bucket"]: b for b in report.split("irr_range", found, "region")}
    assert [b for b, g in regions.items() if g["suppressed"]] == ["europe", "emerging"]
    assert regions["us"]["contributed_n"] == 6 and regions["us"]["stats"] is not None
    assert regions["other_developed"]["n"] == 5


@pytest.mark.parametrize("seed", range(40))
def test_no_dimension_ever_isolates_fewer_than_five_users_deals(seed):
    rng = np.random.default_rng(seed)
    buckets = tags.BUCKETS["region"]
    found = [ranged(50, 0, origin=rng.choice(["library", "contributed"]),
                    region=str(rng.choice(buckets))) for _ in range(int(rng.integers(1, 30)))]
    groups = report.split("irr_range", found, "region")
    hidden = [g["bucket"] for g in groups if g["suppressed"]]
    hidden_contributed = sum(1 for c in found if c.origin == "contributed" and c.groups["region"] in hidden)
    total = sum(1 for c in found if c.origin == "contributed")
    if total >= report.MIN_CONTRIBUTED:      # else the overall figures are hidden too
        assert hidden_contributed == 0 or hidden_contributed >= report.MIN_CONTRIBUTED
    else:
        assert report.sample("irr_range", found)["overall"]["suppressed"] is (total > 0)
    for g in groups:
        assert g["suppressed"] or g["contributed_n"] in (0, None) or g["contributed_n"] >= 5


def test_a_report_with_too_few_users_deals_does_not_count_them():
    found = [ranged(50, 0)] * 3
    built = report.build(found, generated_at="t", engine_version="1.0.0", library_included=True, fit_until=2025)
    assert built["cases"]["contributed_deals"] is None or built["cases"]["contributed_deals"] >= 5


# ---------------------------------------------------------------------------
# End to end: a real database, the scheduled task, the endpoint
# ---------------------------------------------------------------------------
PROFILE = {"country": "DE", "preferred_currency": "EUR", "locale": "de-DE", "time_zone": "Europe/Berlin"}
PLAN = {"ebitda": 240.0, "entry_mult": 9.0, "exit_mult": 10.5, "hold": 6, "growth": 7.5,
        "gross_margin": 44.0, "opex": 17.0, "tax": 29.0, "da": 3.5, "debt_pct": 55.0,
        "senior_pct": 75.0, "base_rate": 5.25, "mezz_spread": 4.5, "capex": 3.0, "nwc": 0.5,
        "mincash": 15.0, "currency": "EUR", "unit": "millions", "country": "DE", "industry": "machinery"}
NAME = "Project Edelweiss Secret"


def sign_in(subject: str):
    app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)
    assert client.post("/api/account", json=PROFILE).status_code == 200


def actuals_for(plan: dict, ev_factor: float) -> dict:
    run_ = client.post("/api/deal/run", json={"inputs": plan}).json()
    om, r = run_["operating_model"], run_["returns"]
    years = [{"ebitda": e} for e in om["ebitda"][:plan["hold"]]]
    return {"currency": plan["currency"], "unit": plan["unit"], "years": years,
            "exit": {"exit_ev": r["exit_ev"] * ev_factor, "net_debt_at_exit": r["net_debt_at_exit"],
                     "sponsor_equity_entry": r["entry_equity"]}}


def contribute(subject: str, ev_factor: float, *, opt_in: bool = True, plan: dict = PLAN) -> str:
    sign_in(subject)
    created = client.post("/api/deals", json={"name": NAME, "inputs": plan})
    assert created.status_code == 201, created.text
    deal_id = created.json()["id"]
    assert client.put(f"/api/deals/{deal_id}/actuals", json=actuals_for(plan, ev_factor)).status_code == 200
    if opt_in:
        assert client.put(f"/api/deals/{deal_id}/validation", json={"opt_in": True}).json() == {"opt_in": True}
    return deal_id


def approve_repository():
    content = [(d["key"], references.content_hash(d), d) for d in references.repository_deals()]
    stored_references.sync_repository(content)
    with transaction() as conn:
        conn.execute(text("UPDATE reference_deals SET status = 'approved', decided_at = now()"))


def by_check(built: dict, check: str) -> dict:
    return next(c for c in built["checks"] if c["id"] == check)["samples"]


def test_the_scheduled_task_writes_a_report_with_every_check_and_split(fresh_db):  # noqa: ARG001
    approve_repository()
    ids = [contribute(f"user_{i}", 0.6 + 0.15 * i) for i in range(6)]
    contribute("user_shy", 2.0, opt_in=False)
    summary = TASKS["validation-report"].run()
    assert all(isinstance(v, (int, float, bool, str, type(None))) for v in summary.values())
    assert summary["library_cases"] == 10 and summary["contributions"] == 6
    assert summary["contributions_used"] == 6 and summary["checks"] == 3 and summary["splits"] == 4

    got = client.get("/api/validation/report").json()["report"]
    assert got["cases"] == {"library": 10, "contributed_deals": 6}
    assert [c["id"] for c in got["checks"]] == ["default", "irr_range", "loss"]
    for check in got["checks"]:
        for name in ("out_of_time", "in_sample"):
            assert set(check["samples"][name]["splits"]) == {"region", "sector", "size", "era"}
    # The library's ten are in-sample; each deal's plan predates its actuals
    assert by_check(got, "default")["in_sample"]["overall"]["n"] == 10
    ranges = by_check(got, "irr_range")["out_of_time"]["overall"]
    assert (ranges["n"], ranges["contributed_n"], ranges["library_n"]) == (6, 6, 0)
    europe = {b["bucket"]: b for b in by_check(got, "irr_range")["out_of_time"]["splits"]["region"]}["europe"]
    assert europe["n"] == 6
    machinery = {b["bucket"]: b for b in by_check(got, "loss")["out_of_time"]["splits"]["sector"]}["industrials"]
    assert machinery["n"] == 6

    # Anonymity: nothing names a deal, its owner, or carries a deal's figure
    flat = json.dumps(got)
    for secret in [*ids, NAME, "user_0", "user_shy", "Edelweiss"]:
        assert secret not in flat
    for field in ("ebitda", "entry_mult", "exit_ev", "currency", "owner", "subject", "deal_id", "name"):
        assert f'"{field}"' not in flat


def test_each_contribution_is_what_plan_vs_actual_answers_for_that_deal(fresh_db):  # noqa: ARG001
    deal_id = contribute("user_anna", 0.8)
    found = run.build()[0]
    irr = by_check(found, "irr_range")["out_of_time"]["overall"]
    assert irr["suppressed"] and irr["stats"] is None            # one user's deal: hidden
    # The case itself, checked against the plan-vs-actual endpoint for the same deal
    from db import validation as store
    (contribution,) = store.contributions()
    irr_case, loss_case = cases.contributed_cases(contribution, usd_per=lambda c, d: 1.1)
    stored = client.get(f"/api/deals/{deal_id}/actuals").json()["actuals"]
    answer = client.post("/api/backtesting/plan-vs-actual",
                         json={"plan": PLAN, "settings": {}, "actuals": stored, "n": cases.PATHS}).json()
    assert irr_case.out_of_time is True
    assert irr_case.percentile == pytest.approx(answer["actual"]["percentile"])
    assert irr_case.irr_error_pp == pytest.approx((answer["actual"]["irr"] - answer["plan"]["irr"]) * 100)
    assert loss_case.happened == (answer["actual"]["irr"] < 0)


def test_a_plan_saved_after_the_first_actuals_is_in_sample(fresh_db):  # noqa: ARG001
    contribute("user_anna", 0.8)
    from db import validation as store
    with transaction() as conn:
        conn.execute(update(Deal).values(actuals_first_saved_at=datetime(2020, 1, 1, tzinfo=timezone.utc)))
    (contribution,) = store.contributions()
    assert all(not c.out_of_time for c in cases.contributed_cases(contribution, usd_per=lambda c, d: 1.1))
    # Clearing the actuals keeps the first save: what was seen stays seen
    sign_in("user_anna")
    deal_id = client.get("/api/deals").json()["deals"][0]["id"]
    client.delete(f"/api/deals/{deal_id}/actuals")
    with connect() as conn:
        assert conn.execute(select(Deal.actuals_first_saved_at)).scalar() == datetime(2020, 1, 1, tzinfo=timezone.utc)


def test_first_actuals_are_dated_once_and_kept(fresh_db):  # noqa: ARG001
    deal_id = contribute("user_anna", 0.8, opt_in=False)
    with connect() as conn:
        first = conn.execute(select(Deal.actuals_first_saved_at)).scalar()
    assert first is not None
    client.put(f"/api/deals/{deal_id}/actuals", json=actuals_for(PLAN, 0.9))
    with connect() as conn:
        assert conn.execute(select(Deal.actuals_first_saved_at)).scalar() == first


def test_opting_out_takes_a_deal_out_of_the_next_report_and_each_change_is_in_the_history(fresh_db):  # noqa: ARG001
    ids = [contribute(f"user_{i}", 0.7 + 0.1 * i) for i in range(5)]
    assert run.build()[1]["contributions"] == 5
    sign_in("user_0")
    assert client.put(f"/api/deals/{ids[0]}/validation", json={"opt_in": True}).json() == {"opt_in": True}
    assert client.put(f"/api/deals/{ids[0]}/validation", json={"opt_in": False}).json() == {"opt_in": False}
    assert client.get(f"/api/deals/{ids[0]}/validation").json() == {"opt_in": False}
    report_, counts = run.build()
    assert counts["contributions"] == 4
    assert report_["cases"]["contributed_deals"] is None         # four: too few to count
    with connect() as conn:
        actions = conn.execute(select(AuditEvent.action).where(AuditEvent.deal_id == ids[0])
                               .where(AuditEvent.action.like("validation%"))
                               .order_by(AuditEvent.id)).scalars().all()
    # Opting in twice is one entry; opting out another
    assert actions == ["validation_opted_in", "validation_opted_out"]


def test_the_choice_belongs_to_the_deals_owner(fresh_db):  # noqa: ARG001
    deal_id = contribute("user_anna", 0.8, opt_in=False)
    sign_in("user_ben")
    assert client.get(f"/api/deals/{deal_id}/validation").status_code == 404
    assert client.put(f"/api/deals/{deal_id}/validation", json={"opt_in": True}).status_code == 404
    sign_in("user_anna")
    assert client.get(f"/api/deals/{deal_id}/validation").json() == {"opt_in": False}


def test_opting_in_is_not_an_edit_of_the_deal(fresh_db):  # noqa: ARG001
    deal_id = contribute("user_anna", 0.8, opt_in=False)
    before = client.get(f"/api/deals/{deal_id}").json()
    client.put(f"/api/deals/{deal_id}/validation", json={"opt_in": True})
    after = client.get(f"/api/deals/{deal_id}").json()
    assert (after["updated_at"], after["latest_version"]) == (before["updated_at"], before["latest_version"])


def test_with_the_library_off_the_report_leaves_it_out(fresh_db, monkeypatch):  # noqa: ARG001
    approve_repository()
    monkeypatch.setenv("FSE_EXAMPLE_LIBRARY", "0")
    built, counts = run.build()
    assert built["library_included"] is False and counts["library_cases"] == 0


def test_reports_are_kept_to_the_newest_thirty(fresh_db):  # noqa: ARG001
    from db import validation as store
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(report.KEEP_REPORTS + 3):
        store.save_report({"i": i}, keep=report.KEEP_REPORTS, now=start + timedelta(days=i))
    with connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM validation_reports")).scalar() == report.KEEP_REPORTS
    assert store.latest_report() == {"i": report.KEEP_REPORTS + 2}


def test_before_the_first_report_and_without_a_database_the_answer_is_null(fresh_db, monkeypatch):  # noqa: ARG001
    assert client.get("/api/validation/report").json() == {"report": None}
    monkeypatch.delenv("DATABASE_URL")
    assert client.get("/api/validation/report").json() == {"report": None}


def test_the_browser_tests_recorded_report_is_current():
    from tests import e2e_validation
    recorded = json.loads(e2e_validation.OUT.read_text(encoding="utf-8"))
    assert recorded == e2e_validation.answer(), "rerun: python -m tests.e2e_validation"
