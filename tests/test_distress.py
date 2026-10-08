"""The distress predictor (PLAN.md 5.3, ml/distress_model.py).

"Done when: implied default rates match base rates for comparable bands
within the card's tolerance; higher leverage raises risk (test)."

Every table read here is transcribed from its source: S&P's Corporate
Methodology (January 2024) Tables 3 and 17, S&P's 2024 default study Tables
24 and 25 (library/base_rates.py) and Damodaran's coverage table
(core/risk_sources.py). Expected figures below are read off those tables.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.config import resolve_config
from core.debt import equivalent_tranches
from core.deal import DealInputs, in_millions, run_deal
from core.risk_sources import COVERAGE_BANDS, rating_for_coverage
from db.deals import clean_inputs
from library import base_rates
from ml import distress_model as dm
from ml.evaluation import distress as ev
from simulation.vectorized_simulation import _run_vectorized_core, run_vectorized_simulation_full
from tests.test_montecarlo_tranches import draws_at, params_for

client = TestClient(app)

# A card under which every region shows the probabilities
SHOWN = ev._ALWAYS_SHOWN


def view(deal: DealInputs, card=SHOWN) -> dict:
    d, cfg = in_millions(deal, resolve_config())
    return dm.deal_view(d, run_deal(d, cfg), card=card)


# ---------------------------------------------------------------------------
# The published tables, as transcribed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("leverage, profile", [
    (0.0, 1), (1.49, 1), (1.5, 2), (1.99, 2), (2.0, 3), (2.99, 3), (3.0, 4), (3.99, 4), (4.0, 5), (5.0, 5),
    (5.01, 6), (12.0, 6), (float("inf"), 6), (float("nan"), 6),
])
def test_table_17_turns_leverage_into_a_financial_risk_profile(leverage, profile):
    """Minimal under 1.5x, modest 1.5-2, intermediate 2-3, significant 3-4,
    aggressive 4-5, highly leveraged "greater than 5"."""
    assert int(dm.leverage_profile(leverage)) == profile


def test_table_3_is_read_at_the_weaker_of_two_anchors():
    assert dm.anchor(1, 1) == "aa+"      # "aaa/aa+"
    assert dm.anchor(1, 6) == "bb+"      # "bbb-/bb+"
    assert dm.anchor(4, 4) == "bb"
    assert dm.anchor(4, 6) == "b"
    assert dm.anchor(5, 6) == "b-"       # "b/b-"
    assert dm.anchor(6, 3) == "b+"       # "bb-/b+"
    # Every row and column printed, and a weaker business never reads better
    assert all(len(row) == 6 for row in dm.ANCHORS.values()) and len(dm.ANCHORS) == 6
    for profile in range(1, 7):
        grades = [dm.band_of(dm.anchor(b, profile)) for b in range(1, 7)]
        assert grades == sorted(grades)


def test_the_coverage_read_is_the_deal_summarys_rating_folded_to_letter_grades():
    for coverage in np.linspace(-1.0, 10.0, 1101):
        rating, _, _ = rating_for_coverage(float(coverage))
        assert dm.BANDS[int(dm.coverage_band(coverage, 1.0))] == dm.BANDS[dm.band_of(rating)]
    # Exactly on a bound reads the band above it, as rating_for_coverage does
    for low, rating in COVERAGE_BANDS[1:]:
        assert int(dm.coverage_band(low, 1.0)) == dm.band_of(rating)


def test_forward_default_rates_are_never_negative_and_rise_toward_the_weakest_band():
    for table in base_rates.CUMULATIVE:
        h = dm.hazards(table)
        assert (h >= 0).all()
        assert h[:, 0].tolist() == sorted(h[:, 0].tolist())   # year one: weaker band, more defaults


# ---------------------------------------------------------------------------
# Calibration: a deal in one band defaults as often as S&P says
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("table", ["us", "europe", "emerging", "global"])
def test_a_deal_held_in_one_band_defaults_as_often_as_the_table_says(table):
    for band in dm.BANDS:
        printed = base_rates.CUMULATIVE[table][band]
        _, cumulative = dm.probabilities([dm.BANDS.index(band)] * len(printed), table)
        assert cumulative * 100 == pytest.approx(printed, abs=1e-9)


def test_past_the_tables_last_year_its_last_rate_holds():
    """Europe prints ten years: B's tenth-year rate is (16.81 - 16.28) / (100 - 16.28)."""
    yearly, _ = dm.probabilities([dm.BANDS.index("B")] * 12, "europe")
    tenth = (16.81 - 16.28) / (100 - 16.28)
    _, cum = dm.probabilities([dm.BANDS.index("B")] * 12, "europe")
    survive10 = 1 - 16.81 / 100
    assert cum[10] == pytest.approx(1 - survive10 * (1 - tenth))
    assert cum[11] == pytest.approx(1 - survive10 * (1 - tenth) ** 2)
    assert yearly[10] == pytest.approx(survive10 * tenth)


def test_the_cards_calibration_is_within_its_tolerance():
    card = json.loads(dm.CARD.read_text(encoding="utf-8"))["evaluation"]
    calibration = card["sets"]["calibration"]
    assert calibration["overall"]["cases"] == sum(
        len(rows[b]) for t, rows in base_rates.CUMULATIVE.items() for b in dm.BANDS)
    for group in [calibration["overall"], *calibration["by_region"].values()]:
        assert group["model"]["max_abs_gap_pp"] <= card["calibration_tolerance_pp"]


def test_comparable_bands_from_real_figures_match_the_table():
    """A deal whose coverage and leverage keep it in one band, built from
    metrics rather than from the band itself: business risk 4, leverage
    4.5x (aggressive, 'bb-') and coverage 2.1x ('BB') read BB every year."""
    years = 7
    path = dm.bands(np.full(years, 2.1), np.full(years, 1.0), np.full(years, 1.0), np.full(years, 4.5), 4)
    assert [dm.BANDS[b] for b in path] == ["BB"] * years
    _, cumulative = dm.probabilities(path, "us")
    assert cumulative * 100 == pytest.approx(base_rates.CUMULATIVE["us"]["BB"][:years], abs=1e-9)


# ---------------------------------------------------------------------------
# Higher leverage raises risk
# ---------------------------------------------------------------------------
def test_higher_leverage_raises_the_chance_of_default():
    over_hold = [view(DealInputs(country="US", debt_pct=pct))["years"][-1]["cumulative"]
                 for pct in (30.0, 45.0, 60.0, 75.0)]
    assert over_hold == sorted(over_hold) and len(set(over_hold)) == 4
    # ... and never improves a year's band
    low, high = view(DealInputs(country="US", debt_pct=30.0)), view(DealInputs(country="US", debt_pct=75.0))
    for a, b in zip(low["years"], high["years"]):
        assert dm.BANDS.index(b["band"]) >= dm.BANDS.index(a["band"])


def test_a_weaker_business_raises_it_too():
    risk = [view(DealInputs(country="US", business_risk=b))["years"][-1]["cumulative"] for b in (1, 4, 6)]
    assert risk[0] < risk[1] < risk[2]


# ---------------------------------------------------------------------------
# The deal, by hand
# ---------------------------------------------------------------------------
def test_the_default_deal_in_the_uk_by_hand():
    """Debt 600 at close (60% of 10x 100), year-one EBITDA 105: 5.71x, highly
    leveraged, 'b' at fair business risk; coverage 1.92x is B+ -> B. Year
    two 5.19x is still B; year three 4.66x is aggressive ('bb-'), coverage
    2.32x BB+, so BB. Europe's table (Table 25): B 1.75, 4.72; BB 1.15, 1.90."""
    v = view(DealInputs(country="GB"))
    y = v["years"]
    assert (v["region"], v["table"], v["business_risk"]) == ("europe", "europe", 4)
    assert y[0]["leverage"] == pytest.approx(600 / 105)
    assert [r["leverage_profile"] for r in y[:3]] == [6, 6, 5]
    assert [r["band"] for r in y[:3]] == ["B", "B", "BB"]
    assert (y[0]["coverage_band"], y[2]["coverage_band"]) == ("B", "BB")
    h1 = 1.75 / 100
    h2 = (4.72 - 1.75) / (100 - 1.75)
    h3 = (1.90 - 1.15) / (100 - 1.15)
    assert y[0]["probability"] == pytest.approx(h1)
    assert y[1]["probability"] == pytest.approx((1 - h1) * h2)
    assert y[2]["cumulative"] == pytest.approx(1 - (1 - h1) * (1 - h2) * (1 - h3))


@pytest.mark.parametrize("country, region, table", [
    ("US", "us", "us"), ("DE", "europe", "europe"), ("BR", "emerging", "emerging"), ("IN", "emerging", "emerging"),
    ("JP", "other_developed", "global"), ("", None, "global"), ("UK", "europe", "europe"),
])
def test_each_region_reads_its_own_table(country, region, table):
    v = view(DealInputs(country=country))
    assert (v["region"], v["table"]) == (region, table)


def test_leases_count_as_the_deal_is_priced():
    """Post-IFRS 16: the liability is debt and the lease cost is added back,
    as the leverage warning reads them."""
    plain = view(DealInputs(country="US"))
    leased = view(DealInputs(country="US", accounting_standard="ifrs", lease_cost=10.0, lease_liability=80.0))
    assert leased["years"][0]["leverage"] != plain["years"][0]["leverage"]
    d, cfg = in_millions(DealInputs(country="US", accounting_standard="ifrs", lease_cost=10.0,
                                    lease_liability=80.0), resolve_config())
    r = run_deal(d, cfg)
    assert leased["years"][0]["leverage"] == pytest.approx(
        (r.debt_schedule.total_beginning_debt[0] + 80.0) / (r.operating_model.ebitda[0] + 10.0))


# ---------------------------------------------------------------------------
# The card decides what is shown
# ---------------------------------------------------------------------------
def test_today_the_card_shows_the_probabilities_nowhere():
    card = json.loads(dm.CARD.read_text(encoding="utf-8"))["evaluation"]
    verdicts = {r: g["verdict"] for r, g in card["sets"]["reference_deals"]["by_region"].items()}
    assert "beats_baseline" not in verdicts.values()
    for country in ("US", "GB", "BR", "JP", ""):
        v = view(DealInputs(country=country), card=None)
        assert not v["shown"]
        assert all(y["probability"] is None and y["cumulative"] is None for y in v["years"])
        assert all(y["band"] in dm.BANDS for y in v["years"])


def test_the_api_sends_bands_always_and_probabilities_only_where_the_card_beats(monkeypatch):
    answer = client.post("/api/deal/run", json={"inputs": {"country": "US"}}).json()["distress"]
    assert answer["shown"] is False and answer["card"]["verdict"] == "does_not_beat_baseline"
    assert [y["band"] for y in answer["years"]] == ["B", "B", "BB", "BB", "BB"]
    assert all(y["probability"] is None for y in answer["years"])
    beats = {"evaluation": {"headline_set": "s", "headline_metric": "auc", "sets": {"s": {"by_region": {
        "us": {"verdict": "beats_baseline", "cases": 6, "model": {"auc": 0.9}, "baseline": {"auc": 0.5}}}}}}}
    monkeypatch.setattr(dm, "_card", lambda: beats)
    answer = client.post("/api/deal/run", json={"inputs": {"country": "US"}}).json()["distress"]
    assert answer["shown"] is True
    assert answer["years"][0]["probability"] == pytest.approx(base_rates.CUMULATIVE["us"]["B"][0] / 100)
    gb = client.post("/api/deal/run", json={"inputs": {"country": "GB"}}).json()["distress"]
    assert gb["shown"] is False and gb["years"][0]["probability"] is None


def test_business_risk_reaches_the_answer_and_is_stored_only_when_changed():
    weak = client.post("/api/deal/run", json={"inputs": {"country": "US", "business_risk": 1}}).json()
    assert weak["distress"]["business_risk"] == 1
    assert weak["distress"]["years"][0]["leverage_band"] == "BB"     # 5.71x at excellent: 'bbb-/bb+'
    assert "business_risk" not in clean_inputs({})
    assert clean_inputs({"business_risk": 5})["business_risk"] == 5
    assert client.post("/api/deal/run", json={"inputs": {"business_risk": 7}}).status_code == 422


# ---------------------------------------------------------------------------
# The simulation
# ---------------------------------------------------------------------------
def test_a_path_at_the_mean_lands_on_the_deal_models_bands():
    """The tranche path is the deal model's schedule on arrays, so a path at
    the mean reads the same coverage, leverage and band every year."""
    deal = DealInputs(country="US", tranches=equivalent_tranches(DealInputs(), resolve_config()))
    params = dataclasses.replace(params_for(deal, n=1), n_interest_passes=50)
    paths = _run_vectorized_core(params, draws_at(deal, params), credit=True)["credit_paths"]
    expected = view(deal)
    for t, y in enumerate(expected["years"]):
        assert paths["debt"][0, t] / paths["ebitda"][0, t] == pytest.approx(y["leverage"], abs=1e-3)
        assert paths["ebit"][0, t] / paths["interest"][0, t] == pytest.approx(y["coverage"], abs=1e-3)
    got = dm.simulated_view(paths["ebit"], paths["ebitda"], paths["interest"], paths["debt"],
                            country="US", business=4, card=SHOWN)
    assert [max(y["band_shares"], key=y["band_shares"].get) for y in got["years"]] == \
        [y["band"] for y in expected["years"]]
    assert got["years"][-1]["cumulative"] == pytest.approx(expected["years"][-1]["cumulative"], abs=1e-9)


@pytest.mark.parametrize("tranches", [False, True])
def test_asking_for_the_credit_paths_changes_nothing_else(tranches):
    deal = DealInputs(tranches=equivalent_tranches(DealInputs(), resolve_config()) if tranches else ())
    params = params_for(deal, n=500)
    plain = run_vectorized_simulation_full(params, seed=7)
    with_credit = run_vectorized_simulation_full(params, seed=7, credit=True)
    assert plain.credit_paths is None
    assert with_credit.credit_paths["debt"].shape == (500, params.holding_period)
    assert plain.df.equals(with_credit.df)


def test_the_monte_carlo_answer_counts_paths_by_band_and_higher_leverage_moves_them():
    def run(debt_pct):
        body = {"deal": {"country": "US", "debt_pct": debt_pct}, "mc": {"n": 2000}, "seed": 3}
        return client.post("/api/montecarlo/run", json=body).json()["distress"]
    low, high = run(40.0), run(75.0)
    for y in low["years"] + high["years"]:
        assert sum(y["band_shares"].values()) == pytest.approx(1.0)
        assert y["probability"] is None
    def weak_share(years):    # share of paths at B or below, over the hold
        return sum(y["band_shares"]["B"] + y["band_shares"]["CCC/C"] for y in years)
    assert weak_share(high["years"]) > weak_share(low["years"])
    # The mean probability, as the card would show it, rises with leverage too
    def shown(debt_pct):
        d = DealInputs(country="US", debt_pct=debt_pct)
        sim = run_vectorized_simulation_full(params_for(d, n=2000), seed=3, credit=True)
        p = sim.credit_paths
        return dm.simulated_view(p["ebit"], p["ebitda"], p["interest"], p["debt"], country="US", business=4,
                                 card=SHOWN)["years"][-1]["cumulative"]
    assert shown(75.0) > shown(40.0)


def test_the_two_bucket_path_reads_debt_at_the_start_of_each_year():
    """Year one opens with the debt raised at close (60% of 10x 100 = 600);
    each later year opens with what the year before left, which is never
    more (the two-bucket path never redraws)."""
    deal = DealInputs(country="US")
    params = params_for(deal, n=1)
    out = _run_vectorized_core(params, draws_at(deal, params), credit=True)
    debt = out["credit_paths"]["debt"][0]
    assert debt[0] == pytest.approx(600.0)
    assert all(b < a for a, b in zip(debt, debt[1:]))
    # The year's interest is charged on that opening debt
    rate = params.interest_mean
    senior, mezz = 600.0 * params.senior_pct, 600.0 * (1 - params.senior_pct)
    assert out["credit_paths"]["interest"][0, 0] == pytest.approx(senior * rate + mezz * (rate + params.mezz_spread))


def test_the_monte_carlo_reads_the_deals_business_risk():
    def run(business):
        body = {"deal": {"country": "US", "business_risk": business}, "mc": {"n": 1000}, "seed": 3}
        return client.post("/api/montecarlo/run", json=body).json()["distress"]
    strong, weak = run(1), run(6)
    assert (strong["business_risk"], weak["business_risk"]) == (1, 6)
    assert weak["years"][0]["band_shares"]["B"] > strong["years"][0]["band_shares"]["B"]
