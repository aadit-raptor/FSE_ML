"""The deal risk score against companies and deals like it (PLAN.md 5.2).

"Done when: it works with the library off; each region has its own
comparison group (test); thin regions show 'not enough data'."

The industry averages are Damodaran's real January 2026 workbooks, as
recorded in tests/fixtures/benchmarks and kept in
ml/evaluation/data/peer_tables.json (tests/test_model_cards.py checks the
file is current). Every expected number below was read off those tables.
"""
from __future__ import annotations

import statistics
from datetime import date
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from benchmarks.damodaran import Table
from library import references
from ml.anomaly_detector import MIN_INDUSTRIES, DealShape, assess, compare, shown_score, similar_deals
from ml.evaluation.deal_risk import peer_tables

TABLES = peer_tables()
client = TestClient(app)


def deal(country: str, industry: str = "machinery", leverage: float = 5.0, multiple: float = 11.0,
         margin: float = 15.0) -> DealShape:
    return DealShape(country, industry, leverage, multiple, margin)


def by_metric(found: dict) -> dict:
    return {c["metric"]: c for c in found["comparisons"]}


# ---------------------------------------------------------------------------
# Each region has its own comparison group
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("country, group", [
    ("US", "us"), ("JP", "japan"), ("CN", "china"), ("IN", "india"), ("DE", "europe"), ("FR", "europe"),
    ("CA", "aus_nz_canada"), ("AU", "aus_nz_canada"), ("BR", "emerging"), ("ZA", "emerging"),
])
def test_each_country_is_compared_with_the_companies_of_its_own_region(country, group):
    found = compare(deal(country), TABLES)
    assert found["status"] == "ok"
    for c in found["comparisons"]:
        assert c["group"] == group
        assert c["firms"] == TABLES[f"debt.{group}"].rows["machinery"]["firms"]
    assert found["sample"] == {"firms": TABLES[f"debt.{group}"].rows["machinery"]["firms"], "group": group}


def test_the_regions_comparison_groups_differ_and_so_does_the_answer():
    """Seven groups, seven industry figures: the same deal sits differently in each."""
    found = {c: by_metric(compare(deal(c), TABLES))["leverage"] for c in ("US", "JP", "CN", "IN", "DE", "CA", "BR")}
    assert len({f["group"] for f in found.values()}) == 7
    assert len({f["peer"] for f in found.values()}) == 7
    assert len({f["z"] for f in found.values()}) == 7
    # Read off the tables: 105 US machinery companies at 1.8x... against 987 in emerging markets
    assert found["US"]["firms"] == 105 and found["BR"]["firms"] == 987 and found["DE"]["firms"] == 210


# ---------------------------------------------------------------------------
# Thin regions show "not enough data"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("country, industry, group, firms", [
    ("CA", "shipbuilding_marine", "aus_nz_canada", 8),     # 8 Australian, NZ and Canadian companies
    ("JP", "aerospace_defense", "japan", 5),
    ("GB", "information_services", "europe", 6),
])
def test_a_thin_region_says_not_enough_data_and_never_borrows_the_global_group(country, industry, group, firms):
    found = assess(deal(country, industry), TABLES, library=None)
    assert found["status"] == "not_enough_data" and found["reason"] == "no_peers"
    assert found["sample"] is None and found["unusual"] is False
    assert found["score"]["shown"] is False and found["score"]["value"] is None
    for c in found["comparisons"]:
        assert c["status"] == "not_enough_data" and c["peer"] is None and c["group"] is None
        assert c["skipped"] == [{"area": group, "reason": "thin", "sample": firms}]
    # The global group has the industry (it is not thin): it is simply never used
    assert TABLES[f"debt.global"].rows[industry]["firms"] >= 20


def test_a_country_file_too_thin_hands_over_to_its_region_but_not_further():
    """China's own file has 15 air transport companies; emerging markets 79."""
    lev = by_metric(compare(deal("CN", "air_transport"), TABLES))["leverage"]
    assert lev["group"] == "emerging" and lev["firms"] == 79
    assert lev["skipped"] == [{"area": "china", "reason": "thin", "sample": 15}]


def test_a_group_with_too_few_industries_can_not_say_how_far_off_a_deal_is():
    rows = {k: r for k, r in list(TABLES["debt.europe"].rows.items())[:MIN_INDUSTRIES]}
    rows["machinery"] = TABLES["debt.europe"].rows["machinery"]
    tables = {**TABLES, "debt.europe": Table("debt.europe", date(2026, 1, 5), "u", rows)}
    lev = by_metric(compare(deal("DE"), tables))["leverage"]
    assert lev["status"] == "not_enough_data" and lev["firms"] == 210 and lev["z"] is None
    assert lev["skipped"][-1]["reason"] == "few_industries"
    assert compare(deal("DE"), tables)["score"] is None      # a scored figure is missing


def test_without_a_country_there_is_nothing_to_compare_with():
    found = assess(deal(""), TABLES, library=None)
    assert found["status"] == "not_enough_data" and found["reason"] == "no_country"
    assert found["comparisons"] == [] and found["score"]["value"] is None


def test_without_stored_averages_nothing_is_compared():
    found = assess(deal("DE"), {}, library=None)
    assert found["status"] == "not_enough_data" and found["reason"] == "no_peers"
    assert {s["reason"] for c in found["comparisons"] for s in c["skipped"]} == {"missing"}


# ---------------------------------------------------------------------------
# The figures, by hand
# ---------------------------------------------------------------------------
def test_a_german_machinery_deal_by_hand():
    """Developed Europe's 210 machinery companies carry 1.871635x debt / EBITDA,
    trade at 14.98x EBITDA and earn a 12.97% EBITDA margin. Across Europe's 65
    industries with 20 companies or more, debt / EBITDA has a median absolute
    deviation of 1.22149..., so the robust spread is 1.4826 x that = 1.81096."""
    europe = TABLES["debt.europe"]
    values = [r["debt_ebitda"] for k, r in europe.rows.items() if k != "all" and r["firms"] >= 20]
    assert len(values) == 65
    middle = statistics.median(values)
    assert statistics.median(abs(v - middle) for v in values) * 1.4826 == pytest.approx(1.81096, abs=5e-6)

    found = compare(deal("DE", leverage=5.0, multiple=11.0, margin=15.0), TABLES)
    lev, mult, margin = (by_metric(found)[m] for m in ("leverage", "entry_multiple", "ebitda_margin"))
    assert lev["peer"] == 1.8716 and lev["spread"] == 1.811
    assert lev["z"] == pytest.approx((5.0 - 1.871635) / 1.81096, abs=1e-4) == pytest.approx(1.7275, abs=1e-4)
    assert lev["position"] == "above" and lev["risk_z"] == lev["z"]
    assert mult["peer"] == 14.9805 and mult["z"] == pytest.approx(-1.0427, abs=1e-4)
    assert mult["position"] == "below"
    # The margin is compared in per cent; lower is riskier, so its risk z is minus z
    assert margin["peer"] == 12.9671 and margin["z"] == pytest.approx(0.3346, abs=1e-4)
    assert margin["risk_z"] == -margin["z"] and margin["position"] == "in_line"
    # The score: leverage's 1.73 spreads above; the cheaper price counts nothing
    assert found["score"] == lev["risk_z"]
    assert found["unusual"] is False


def test_the_score_counts_only_leverage_and_price_above_the_industrys():
    base = compare(deal("DE"), TABLES)
    dear = compare(deal("DE", multiple=20.0), TABLES)
    mult = by_metric(dear)["entry_multiple"]
    assert mult["risk_z"] > 0
    assert dear["score"] == pytest.approx(base["score"] + mult["risk_z"], abs=1e-4)
    # A margin far below the industry's is unusual but not scored (the card can't test it)
    thin_margin = compare(deal("DE", margin=-1.0), TABLES)
    assert by_metric(thin_margin)["ebitda_margin"]["risk_z"] >= 2
    assert thin_margin["unusual"] is True and thin_margin["score"] == base["score"]


def test_leverage_two_spreads_above_the_industrys_is_unusual():
    lev = by_metric(compare(deal("DE"), TABLES))["leverage"]
    edge = 1.871635 + 2 * lev["spread"]
    assert compare(deal("DE", leverage=edge + 0.01), TABLES)["unusual"] is True
    assert compare(deal("DE", leverage=edge - 0.01), TABLES)["unusual"] is False
    assert by_metric(compare(deal("DE", leverage=edge + 0.01), TABLES))["leverage"]["position"] == "well_above"


def test_a_deal_without_a_sector_is_compared_with_the_whole_market():
    found = assess(deal("DE", industry=""), TABLES, library=None)
    assert found["industry"] == "all" and found["industry_name"] == "Total Market (without financials)"
    assert found["sample"] == {"firms": TABLES["debt.europe"].rows["all"]["firms"], "group": "europe"}


# ---------------------------------------------------------------------------
# The score is shown only where its card says it beats the baseline
# ---------------------------------------------------------------------------
def test_the_score_is_shown_in_the_us_where_the_card_beats_leverage_alone():
    found = assess(deal("US"), TABLES, library=None)
    assert found["score"]["shown"] is True and found["score"]["verdict"] == "beats_baseline"
    assert found["score"]["value"] == compare(deal("US"), TABLES)["score"]
    assert found["score"]["cases"] == 5 and found["score"]["model"] > found["score"]["baseline"]


@pytest.mark.parametrize("country", ["DE", "CA", "BR", "JP"])
def test_elsewhere_the_comparison_shows_but_the_score_says_not_enough_data(country):
    found = assess(deal(country), TABLES, library=None)
    assert found["status"] == "ok" and all(c["status"] == "ok" for c in found["comparisons"])
    assert found["score"]["shown"] is False and found["score"]["value"] is None
    assert found["score"]["verdict"] == "not_enough_data"


def test_a_card_that_does_not_beat_the_baseline_hides_the_score():
    card = {"evaluation": {"headline_set": "s", "headline_metric": "auc", "sets": {"s": {"by_region": {
        "us": {"cases": 9, "verdict": "does_not_beat_baseline", "model": {"auc": 0.5}, "baseline": {"auc": 0.6}}}}}}}
    shown = shown_score(compare(deal("US"), TABLES), card)
    assert shown == {"shown": False, "value": None, "region": "us", "verdict": "does_not_beat_baseline",
                     "cases": 9, "model": 0.5, "baseline": 0.6}


# ---------------------------------------------------------------------------
# Deals like it: the library when on; everything else with it off
# ---------------------------------------------------------------------------
def test_deals_like_it_are_the_reference_transactions_of_the_same_region_and_sector():
    library = references.repository_deals()
    # A US speciality retailer worth 5,000 (US dollar millions): consumer discretionary, 1bn-10bn
    found = similar_deals(library, "US", "retail_special_lines", 5_000.0)
    assert found["region"] == "us" and found["sector"] == "consumer_discretionary" and found["size"] == "1bn_10bn"
    keys = [d["key"] for d in found["deals"]]
    assert set(keys) == {"toys-r-us-2005", "dollar-general-2007", "dominos-1998", "gymboree-2010"}
    same = [d["same_size"] for d in found["deals"]]
    assert same == sorted(same, reverse=True)                # the same size first
    for d in found["deals"]:
        assert d["same_size"] == (d["size"] == "1bn_10bn")
    toys = next(d for d in found["deals"] if d["key"] == "toys-r-us-2005")
    assert (toys["outcome"], toys["event"], toys["leverage"], toys["year"]) == ("distress", "bankruptcy", 6.69, 2005)
    # Health care (HCA) and other regions (NXP) are not like it
    assert similar_deals(library, "NL", "retail_special_lines", 5_000.0)["deals"] == []


def test_with_the_library_off_the_comparison_and_score_are_the_same_and_no_deal_is_listed():
    on = assess(deal("US", "retail_special_lines"), TABLES, library=references.repository_deals(), ev_usd_m=5_000.0)
    off = assess(deal("US", "retail_special_lines"), TABLES, library=None, ev_usd_m=5_000.0)
    assert off["deals"] == {"enabled": False, "region": None, "sector": None, "size": None, "deals": []}
    assert on["deals"]["enabled"] is True and len(on["deals"]["deals"]) == 4
    assert {k: v for k, v in on.items() if k != "deals"} == {k: v for k, v in off.items() if k != "deals"}


# ---------------------------------------------------------------------------
# The endpoint scores the deal's own figures
# ---------------------------------------------------------------------------
def post(inputs: dict, library=None, usd_per=1.0, **body):
    with mock.patch("api.routers.integrations._peer_data", lambda currency: (TABLES, library, usd_per)):
        resp = client.post("/api/ml/deal-risk", json={"inputs": inputs, **body})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_the_endpoint_compares_the_deals_leverage_price_and_margin():
    body = post({"country": "DE", "industry": "machinery", "currency": "EUR"}, senior_x=3.4, mezz_x=0.8)
    lev, mult, margin = (next(c for c in body["comparisons"] if c["metric"] == m)
                         for m in ("leverage", "entry_multiple", "ebitda_margin"))
    assert lev["deal"] == pytest.approx(4.2) and mult["deal"] == 10.0
    assert margin["deal"] == pytest.approx(body["inputs"]["ebitda_margin"])
    assert body["sample"] == {"firms": 210, "group": "europe"} and body["region"] == "europe"
    assert body["score"]["shown"] is False and body["model"]["engine_version"]


def test_more_leverage_moves_the_deals_place_among_its_peers():
    low = post({"country": "US", "industry": "machinery"}, senior_x=2.0, mezz_x=0.0)
    high = post({"country": "US", "industry": "machinery"}, senior_x=6.0, mezz_x=1.0)
    z = [next(c for c in b["comparisons"] if c["metric"] == "leverage")["z"] for b in (low, high)]
    assert z[1] > z[0]
    assert high["score"]["value"] > low["score"]["value"]
    assert high["unusual"] is True and low["unusual"] is False


def test_the_endpoint_works_with_the_library_off_and_lists_deals_with_it_on():
    inputs = {"country": "US", "industry": "retail_special_lines", "ebitda": 500.0, "entry_mult": 10.0}
    off = post(inputs, library=None)
    on = post(inputs, library=references.repository_deals())
    assert off["status"] == "ok" and off["deals"]["enabled"] is False and off["deals"]["deals"] == []
    assert on["deals"]["enabled"] is True and on["deals"]["size"] == "1bn_10bn"
    assert len(on["deals"]["deals"]) == 4


def test_a_deal_in_thousands_of_another_currency_is_sized_in_us_dollars():
    """500,000 thousand euros of EBITDA at 10x is 5,000 million euros: at 1.1
    US dollars a euro, 5,500 million US dollars."""
    body = post({"country": "US", "industry": "retail_special_lines", "currency": "EUR", "unit": "thousands",
                 "ebitda": 500_000.0, "entry_mult": 10.0}, library=[], usd_per=1.1)
    assert body["deals"]["size"] == "1bn_10bn"
    small = post({"country": "US", "industry": "retail_special_lines", "currency": "EUR", "unit": "thousands",
                  "ebitda": 5_000.0, "entry_mult": 10.0}, library=[], usd_per=1.1)
    assert small["deals"]["size"] == "under_100m"
    unknown = post({"country": "US", "industry": "retail_special_lines", "currency": "EUR"}, library=[], usd_per=None)
    assert unknown["deals"]["size"] is None


def test_a_server_without_a_database_says_not_enough_data():
    with mock.patch("api.routers.integrations.is_configured", lambda: False):
        resp = client.post("/api/ml/deal-risk", json={"inputs": {"country": "DE", "industry": "machinery"}})
    body = resp.json()
    assert resp.status_code == 200 and body["status"] == "not_enough_data" and body["reason"] == "no_peers"
    assert body["deals"]["enabled"] is False


def test_the_browser_tests_recorded_answers_are_current():
    from tests import e2e_deal_risk
    assert e2e_deal_risk.OUT.read_text(encoding="utf-8") == e2e_deal_risk.text(), (
        "web/e2e/fixtures/deal-risk.json is stale: run python -m tests.e2e_deal_risk")


@pytest.mark.parametrize("retired, current, group", [("UK", "GB", "europe"), ("DD", "DE", "europe")])
def test_a_retired_country_code_is_compared_as_its_current_country(retired, current, group):
    """The API accepts codes accounts saved before the country list dropped
    them; the starting figures read them as the current country, and so does this."""
    old, new = compare(deal(retired), TABLES), compare(deal(current), TABLES)
    assert old["sample"]["group"] == group
    assert old == new
