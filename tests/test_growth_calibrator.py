"""The growth calibrator by sector and region (PLAN.md 5.5).

"Done when: ranges match observed growth on held-out years per region
(card); using them changes the simulation."

The companies are the SEC's real filers, as kept in
ml/evaluation/data/firm_growth.json (python -m tests.ml_firm_growth).
"""
from __future__ import annotations

import json
import math
from unittest import mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.limits import RUN_PATHS
from api.main import app
from ml import growth_calibrator as gc
from ml.evaluation import growth as card_eval

client = TestClient(app)
DATA = gc.load()
CARD = json.loads(gc.CARD.read_text(encoding="utf-8"))
ANCHOR = {"field": "growth", "value": 4.0, "source": "imf", "dataset": "NGDP_RPCH+PCPIPCH", "area": "US",
          "level": "country", "sample": 1, "sample_kind": "economies", "as_of": "2026", "url": "u",
          "skipped": [], "detail": {}}


# ---------------------------------------------------------------------------
# Industry codes
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sic,sector", [
    (2834, "health_care"), (7372, "information_technology"), (3674, "information_technology"),
    (3711, "consumer_discretionary"), (4911, "utilities"), (5812, "consumer_discretionary"),
    (5411, "consumer_staples"), (1311, "energy"), (2911, "energy"), (4813, "communication_services"),
    (3721, "industrials"), (6512, "real_estate"), (3841, "health_care"), (2086, "consumer_staples"),
    (6022, None), (6311, None), (6798, None), (6770, None), (9995, None), (None, None),
])
def test_sic_codes_map_to_their_gics_sector_and_financials_are_left_out(sic, sector):
    assert gc.sector(sic) == sector


def test_every_industry_code_in_the_data_is_placed_or_deliberately_left_out():
    excluded = [(6000, 6499), (6700, 6799), (9100, 9999)]
    for firm in DATA["companies"]:
        sic = firm["sic"]
        if sic is None or any(lo <= sic <= hi for lo, hi in excluded) or sic < 100:
            continue
        assert gc.sector(sic) is not None, sic


def test_sectors_are_the_ones_deals_are_tagged_with():
    from validation.tags import INDUSTRY_SECTOR
    assert {s for _, _, s in gc.SIC_SECTORS if s} == set(INDUSTRY_SECTOR.values())


# ---------------------------------------------------------------------------
# Cases by hand
# ---------------------------------------------------------------------------
def tiny(companies: list[dict], nominal: dict | None = None, fx: dict | None = None) -> dict:
    nominal = nominal or {"US": {y: 4.0 for y in range(2000, 2026)}}
    fx = fx or {"USD": {y: 1.0 for y in range(2008, 2026)}}
    return {"written_on": "2026-10-08", "years": [2008, 2025], "sources": {}, "usd_per_unit": fx,
            "nominal_growth": nominal, "companies": companies}


def firm(i: int, series: dict, country: str = "US", sic: int = 2834) -> dict:
    return {"id": i, "country": country, "sic": sic, "series": series}


def test_a_span_by_hand():
    """100 to 133.1 over three years is 10% a year; the economy grew 4% a
    year, so the company grew ln(1.1 / 1.04) faster."""
    d = tiny([firm(1, {"us-gaap:Revenues|USD": {2010: 100.0, 2013: 133.1}})])
    [s] = gc.spans(d, horizons=(3,))
    assert (s.start, s.end, s.region, s.sector, s.country) == (2010, 2013, "us", "health_care", "US")
    assert s.growth == pytest.approx(0.1)
    assert s.economy == pytest.approx(0.04)
    assert s.excess == pytest.approx(math.log(1.1 / 1.04))


def test_spans_take_the_first_revenue_concept_with_both_years_and_one_currency():
    d = tiny([firm(1, {
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax|USD": {2010: 100.0, 2013: 200.0},
        "us-gaap:Revenues|USD": {2010: 100.0, 2013: 121.0, 2012: 1.0},
    })])
    [s] = gc.spans(d, horizons=(3,))
    assert s.growth == pytest.approx(1.21 ** (1 / 3) - 1)
    # A concept with only one of the years is not mixed with another
    d = tiny([firm(1, {"us-gaap:Revenues|USD": {2010: 100.0}, "us-gaap:SalesRevenueNet|USD": {2013: 500.0}})])
    assert gc.spans(d, horizons=(3,)) == []


def test_the_size_floor_reads_us_dollars_at_the_start_years_rate():
    fx = {"USD": {y: 1.0 for y in range(2008, 2026)}, "EUR": {2010: 1.25, 2011: 1.0}}
    d = tiny([firm(1, {"us-gaap:Revenues|USD": {2010: 49.0, 2013: 60.0}}),
              firm(2, {"ifrs-full:Revenue|EUR": {2010: 45.0, 2013: 60.0}}, country="DE"),
              firm(3, {"ifrs-full:Revenue|EUR": {2011: 45.0, 2014: 60.0}}, country="DE"),
              firm(4, {"ifrs-full:Revenue|GBP": {2010: 900.0, 2013: 950.0}}, country="GB")],
             nominal={"US": {y: 4.0 for y in range(2000, 2026)}, "DE": {y: 2.0 for y in range(2000, 2026)}}, fx=fx)
    found = gc.spans(d, horizons=(3,))
    # 49M is under the floor; 45M euros at 1.25 is 56M; at 1.0 it is 45M; no rate for pounds
    assert [s.company for s in found] == [2]
    assert found[0].size_usd_m == pytest.approx(56.25) and found[0].region == "europe"


def test_financials_and_companies_without_a_code_are_left_out():
    series = {"us-gaap:Revenues|USD": {2010: 100.0, 2013: 120.0}}
    d = tiny([firm(1, series, sic=6022), firm(2, series, sic=None), firm(3, series, sic=7372)])
    assert [s.company for s in gc.spans(d, horizons=(3,))] == [3]


def test_a_country_the_app_does_not_cover_takes_its_regions_median_economy():
    nominal = {"DE": {2011: 2.0, 2012: 2.0}, "FR": {2011: 4.0, 2012: 4.0}, "IT": {2011: 9.0, 2012: 9.0}}
    with mock.patch.object(gc, "COUNTRIES", ("DE", "FR", "IT", "US")):
        assert gc.economy_growth(nominal, "GR") == {2011: 4.0, 2012: 4.0}
        assert gc.economy_growth(nominal, "DE") == {2011: 2.0, 2012: 2.0}
    assert gc.compound({2011: 10.0, 2012: 21.0 / 1.1 - 10.0}, 2011, 2012) == pytest.approx(0.1, abs=0.06)
    assert gc.compound({2011: 10.0}, 2011, 2012) is None


# ---------------------------------------------------------------------------
# Ranges by hand
# ---------------------------------------------------------------------------
def spans_with(excesses: list[float], sector: str = "health_care", region: str = "us", h: int = 5,
               first_id: int = 0) -> list[gc.Span]:
    return [gc.Span(first_id + i, "US", region, sector, 2010, 2010 + h, math.exp(x) * 1.03 - 1, 0.03, 100.0)
            for i, x in enumerate(excesses)]


def test_a_group_needs_enough_companies_and_a_thin_sector_takes_its_region():
    health = spans_with([0.01 * i for i in range(gc.MIN_COMPANIES - 1)])
    tech = spans_with([0.02] * 5, sector="information_technology", first_id=1000)
    model = gc.fit(health + tech)
    assert ("us", "health_care", 5) not in model
    group, sector = gc.lookup(model, "us", "health_care", 5)
    assert sector is None and group["companies"] == gc.MIN_COMPANIES - 1 + 5
    assert gc.lookup(model, "us", "health_care", 4) == (None, None)


def test_the_range_is_the_groups_percentiles_around_the_economy_and_its_draw_has_them_as_its_80_percent():
    xs = [0.01 * i - 0.2 for i in range(41)]           # -0.2 ... 0.2
    group = gc.fit_group(spans_with(xs))
    assert group["quantiles"] == pytest.approx([-0.16, 0.0, 0.16])
    b = gc.band(0.04, group)
    assert b["low"] == pytest.approx(1.04 * math.exp(-0.16) - 1)
    assert b["high"] == pytest.approx(1.04 * math.exp(0.16) - 1)
    assert b["median"] == pytest.approx(0.04)
    assert b["mean"] == pytest.approx(round((b["low"] + b["high"]) / 2 * 100, 2))
    # The simulation's normal draw: its 10th and 90th percentiles are the range, to rounding
    assert b["draw_low"] == pytest.approx(b["low"], abs=1e-4)
    assert b["draw_high"] == pytest.approx(b["high"], abs=1e-4)


# ---------------------------------------------------------------------------
# The data and the ranges served
# ---------------------------------------------------------------------------
def test_the_data_holds_listed_companies_in_every_region():
    found = gc.spans(DATA)
    by_region = {r: len({s.company for s in found if s.region == r}) for r in gc.SP_REGIONS}
    assert by_region["us"] > 2000
    assert all(n >= gc.MIN_COMPANIES for n in by_region.values()), by_region
    assert all(s.size_usd_m >= gc.FLOOR_USD_M for s in found)


def test_the_browser_tests_replay_what_the_endpoint_answers():
    from tests import e2e_growth
    assert e2e_growth.OUT.read_text(encoding="utf-8") == e2e_growth.text(), \
        "web/e2e/fixtures/growth.json is stale: run python -m tests.e2e_growth"


def test_the_served_ranges_are_what_the_data_gives():
    assert json.loads(gc.RANGES.read_text(encoding="utf-8")) == json.loads(json.dumps(gc.ranges(DATA))), \
        "ml/growth_ranges.json is stale: run python -m tests.ml_firm_growth"


# ---------------------------------------------------------------------------
# The card: held-out years, per region
# ---------------------------------------------------------------------------
def test_the_card_tests_strictly_newer_spans():
    """Each start year's spans are predicted by a fit on spans that had
    ended by then. ``scored`` fits once per start year that has spans, in
    order, so the fits line up with those years."""
    ends = []
    real_fit = card_eval.fit

    def spy(cases):
        cases = list(cases)
        ends.append(max(c.end for c in cases))
        return real_fit(cases)

    small = {**DATA, "companies": DATA["companies"][:400]}
    with mock.patch.object(card_eval, "fit", spy):
        rows = card_eval.scored(small)
    starts = sorted({s.start for s in gc.spans(small) if s.start >= card_eval.FIRST_START})
    assert rows and len(ends) == len(starts)
    assert all(end <= start for end, start in zip(ends, starts)), list(zip(ends, starts))


def test_ranges_hold_observed_growth_on_held_out_years_in_every_region():
    growth = CARD["evaluation"]["sets"]["growth"]
    for region, r in growth["by_region"].items():
        assert r["verdict"] in ("beats_baseline", "does_not_beat_baseline", "not_enough_data")
        if r["verdict"] == "not_enough_data":
            continue
        # Held-out companies fall inside the central 80% about as often as it says
        assert 0.65 <= r["model"]["inside"] <= 0.9, (region, r["model"]["inside"])
        # The economy-wide range in force holds far fewer of them
        assert r["baseline"]["inside"] < r["model"]["inside"], region
    assert growth["by_region"]["us"]["verdict"] == "beats_baseline"


# ---------------------------------------------------------------------------
# A deal's answer
# ---------------------------------------------------------------------------
def test_a_us_software_deal_gets_its_sectors_range_around_the_us_economy():
    a = gc.calibrate("US", "software_system_application", 5, ANCHOR)
    assert a["status"] == "ok" and a["shown"] and a["region"] == "us"
    assert a["sector"] == "information_technology" and a["group_sector"] == "information_technology"
    group = gc.served()["ranges"]["us"]["information_technology"]["5"]
    b = gc.band(0.04, group)
    assert a["range"] == {"low": round(b["low"] * 100, 2), "median": round(b["median"] * 100, 2),
                          "high": round(b["high"] * 100, 2)}
    assert a["settings"] == {"mc_growth_mean": b["mean"], "mc_growth_std": b["std"]}
    assert a["observed"]["companies"] == group["companies"]
    # A company's growth spreads far wider than an economy's
    assert a["settings"]["mc_growth_std"] > 5


def test_sectors_differ():
    soft = gc.calibrate("US", "software_system_application", 5, ANCHOR)["settings"]
    util = gc.calibrate("US", "utility_general", 5, ANCHOR)["settings"]
    assert util["mc_growth_std"] < soft["mc_growth_std"]


def test_nothing_is_shown_where_the_card_does_not_say_it_beats_the_baseline():
    with mock.patch.object(gc, "card_result", lambda region: {"region": region, "verdict": "does_not_beat_baseline",
                                                               "cases": 9, "model": 1.0, "baseline": 0.5}):
        a = gc.calibrate("US", "software_system_application", 5, ANCHOR)
    assert a["status"] == "ok" and not a["shown"] and a["hidden"] == "does_not_beat_baseline"
    assert a["range"] is None and a["settings"] is None


def test_untested_holds_no_country_and_no_economy_say_why():
    assert gc.calibrate("US", "software_system_application", 9, ANCHOR)["hidden"] == "untested_horizon"
    assert gc.calibrate("", "software_system_application", 5, ANCHOR)["reason"] == "no_country"
    a = gc.calibrate("US", "software_system_application", 5, None)
    assert a["reason"] == "no_growth" and a["settings"] is None and a["observed"] is not None


# ---------------------------------------------------------------------------
# The API, and the simulation it moves
# ---------------------------------------------------------------------------
def deal(**kw) -> dict:
    return {"inputs": {"country": "US", "industry": "software_system_application", "hold": 5, **kw}}


def test_the_endpoint_answers_the_range_from_the_stored_economy():
    assert "/api/ml/growth" in RUN_PATHS
    with mock.patch("api.routers.integrations.is_configured", lambda: True), \
            mock.patch("db.economy.all_series", lambda: ({}, None)), \
            mock.patch("benchmarks.starting.growth_figure", lambda series, country, today: ANCHOR):
        r = client.post("/api/ml/growth", json=deal())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["shown"] and body["anchor"]["value"] == 4.0
    assert body["settings"] == gc.calibrate("US", "software_system_application", 5, ANCHOR)["settings"]
    assert body["model"]["engine_version"]


def test_without_a_database_the_endpoint_says_there_is_no_economy():
    with mock.patch("api.routers.integrations.is_configured", lambda: False):
        body = client.post("/api/ml/growth", json=deal()).json()
    assert body["reason"] == "no_growth" and body["settings"] is None


def test_using_the_range_changes_the_simulation_and_draws_it():
    s = gc.calibrate("US", "software_system_application", 5, ANCHOR)["settings"]
    base = {"mc": {"n": 20000}, "seed": 11}
    used = {"mc": {"n": 20000, "growth_mean": s["mc_growth_mean"], "growth_std": s["mc_growth_std"]}, "seed": 11}
    a = client.post("/api/montecarlo/run", json=base).json()
    b = client.post("/api/montecarlo/run", json=used).json()
    assert b["params"]["growth_mean"] == pytest.approx(s["mc_growth_mean"] / 100)
    assert b["params"]["growth_std"] == pytest.approx(s["mc_growth_std"] / 100)
    assert b["summary"]["mean_irr"] != a["summary"]["mean_irr"]
    assert b["summary"]["p5_irr"] < a["summary"]["p5_irr"]      # a wider growth draw, a lower tail

    from core.config import resolve_config
    from core.deal import DealInputs
    from core.montecarlo import MCInputs, build_sim_params
    from simulation.vectorized_simulation import run_vectorized_simulation_full
    sim = run_vectorized_simulation_full(build_sim_params(
        MCInputs(n=50000, growth_mean=s["mc_growth_mean"], growth_std=s["mc_growth_std"]), DealInputs(),
        resolve_config()), seed=5)
    rng = gc.calibrate("US", "software_system_application", 5, ANCHOR)["range"]
    assert np.percentile(sim.df["Growth"], 10) * 100 == pytest.approx(rng["low"], abs=0.3)
    assert np.percentile(sim.df["Growth"], 90) * 100 == pytest.approx(rng["high"], abs=0.3)
