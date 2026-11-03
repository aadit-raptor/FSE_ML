"""The lender view (PLAN.md 7.5): covenants tested each year, their breach
odds from the simulation, and expected loss from regional base rates.

What these tests hold it to:

* a deal built to breach in year 2 is flagged in year 2, and not in the
  years around it;
* each test's figure is the convention's definition worked out again here
  from the deal answer's own tables (net or gross debt, leases frozen or as
  reported, the three fixed-charge ratios, liquidity with a revolver's
  undrawn commitment), and the sponsor's loan is never a lender's debt;
* a simulated path at the deal's own assumptions gives the deal model's
  figures for every test in every convention, so the breach odds are the
  simulation's paths read by the same definitions;
* tracking the paths moves no simulated figure;
* expected loss is S&P's regional rate x Global Credit Data's LGD x exposure.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from core import covenants as cv
from core.config import resolve_config
from core.deal import DealInputs, in_millions, run_deal
from core.debt import TrancheSpec, spec_from_kind
from core.lender import expected_loss
from core.montecarlo import LENDER_CELLS, MCInputs, build_sim_params, lender_paths
from db.deals import clean_inputs
from library import base_rates
from simulation.vectorized_simulation import _run_vectorized_core, run_vectorized_simulation_full
from tests.test_montecarlo_tranches import STRUCTURES, draws_at

client = TestClient(app)


def cfg(**overrides):
    return resolve_config(overrides)


def run(inputs: dict, settings: dict | None = None) -> dict:
    r = client.post("/api/deal/run", json={"inputs": inputs, **({"settings": settings} if settings else {})})
    assert r.status_code == 200, r.text
    return r.json()


def lev(facility="Senior Term Loan", levels=(6.0,), scope="total", metric="leverage"):
    return {"facility": facility, "metric": metric, "levels": list(levels), "scope": scope}


# ---------------------------------------------------------------------------
# Done when: a deal built to breach in year 2 is flagged in year 2
# ---------------------------------------------------------------------------
def test_a_deal_built_to_breach_in_year_two_is_flagged_in_year_two():
    """The default deal's net leverage falls each year as it repays
    (5.45x, 4.89x, 4.34x ...). A maximum of 6.0x stepping down to 4.5x in
    year 2 and back up to 6.0x is breached in year 2 only."""
    answer = run({"covenants": [lev(levels=(6.0, 4.5, 6.0))]})
    test = answer["lender"]["covenants"][0]
    assert [y["breach"] for y in test["years"]] == [False, True, False, False, False]
    assert test["first_breach"] == 2 and answer["lender"]["first_breach"] == 2
    year2 = test["years"][1]
    assert year2["value"] > 4.5 and year2["headroom"] == pytest.approx((4.5 - year2["value"]) / 4.5)
    assert year2["headroom"] < 0


def test_a_cover_breaches_below_its_minimum_and_a_level_path_repeats_its_last_year():
    answer = run({"covenants": [lev(metric="interest_cover", levels=(1.0, 2.6))]})
    years = answer["lender"]["covenants"][0]["years"]
    values = [y["value"] for y in years]
    # Cover rises as debt is repaid: under 2.6 early, over it later; the
    # second level stands for every year after the second
    assert [y["level"] for y in years] == [1.0, 2.6, 2.6, 2.6, 2.6]
    assert [y["breach"] for y in years] == [False] + [v < 2.6 for v in values[1:]]
    assert any(y["breach"] for y in years) and not all(y["breach"] for y in years[1:])


# ---------------------------------------------------------------------------
# Each convention's definition, worked out again from the answer's tables
# ---------------------------------------------------------------------------
def _tables(answer):
    om, cf, ds = answer["operating_model"], answer["cash_flow"], answer["debt_schedule"]
    return om, cf, ds, answer["tranches"]


@pytest.mark.parametrize("convention", ["us", "lma", "india"])
def test_every_test_is_the_conventions_definition_of_the_answers_own_figures(convention):
    inputs = {"mincash": 20.0, "covenant_convention": convention, "covenants": [
        lev(), lev(metric="interest_cover", levels=(1.0,)), lev(metric="fixed_charge_cover", levels=(1.0,)),
        lev(metric="liquidity", levels=(5.0,))]}
    answer = run(inputs)
    om, cf, ds, tr = _tables(answer)
    senior, mezz = tr["Senior Term Loan"], tr["Mezzanine"]
    tests = {c["metric"]: c["years"] for c in answer["lender"]["covenants"]}
    assert answer["lender"]["convention"] == convention and answer["lender"]["chosen"]
    for t in range(5):
        ebitda, cash = om["ebitda"][t], ds["cash_balance"][t]
        debt = senior[t]["ending_balance"] + mezz[t]["ending_balance"]
        expect_lev = (debt - cash if convention != "india" else debt) / ebitda
        assert tests["leverage"][t]["value"] == pytest.approx(expect_lev, rel=1e-9)
        interest = ds["total_cash_interest"][t] if convention != "india" else om["interest_expense"][t]
        assert tests["interest_cover"][t]["value"] == pytest.approx(ebitda / interest, rel=1e-9)
        # Regular amortisation only: the senior loan's 5% a year; the
        # mezzanine's bullet is refinanced, not paid from the year's cash
        amort = min(senior[t]["mandatory_repayment"], senior[0]["beginning_balance"] * 0.05)
        service = ds["total_cash_interest"][t] + amort
        flow = {"us": ebitda - cf["capex"][t] - om["taxes"][t],
                "lma": ebitda - cf["capex"][t] - om["taxes"][t] - cf["delta_nwc"][t],
                "india": om["net_income"][t] + om["da"][t] + om["interest_expense"][t]}[convention]
        assert tests["fixed_charge_cover"][t]["value"] == pytest.approx(flow / service, rel=1e-9)
        assert tests["liquidity"][t]["liquidity"] == pytest.approx(cash)
        assert tests["liquidity"][t]["breach"] is False


def test_the_convention_follows_the_country_unless_chosen():
    assert cv.convention_for("", "US") == "us"
    assert cv.convention_for("", "IN") == "india"
    assert cv.convention_for("", "DE") == "lma" and cv.convention_for("", "JP") == "lma"
    assert cv.convention_for("", "") == "lma"
    assert cv.convention_for("india", "US") == "india"
    answer = run({"country": "IN", "covenants": [lev()]})["lender"]
    assert (answer["convention"], answer["chosen"]) == ("india", False)


def test_leases_are_frozen_out_under_the_lma_and_counted_as_reported_in_india():
    """A post-IFRS 16 deal: the LMA's agreements freeze the accounts before
    IFRS 16 (no liability, no add-back); India's count them as the deal does."""
    base = {"accounting_standard": "ifrs", "lease_cost": 10.0, "lease_liability": 60.0, "covenants": [lev()]}
    lma = run({**base, "covenant_convention": "lma"})
    india = run({**base, "covenant_convention": "india"})
    om, _, ds, tr = _tables(lma)
    debt = sum(rows[0]["ending_balance"] for rows in tr.values())
    assert lma["lender"]["covenants"][0]["years"][0]["value"] == pytest.approx(
        (debt - ds["cash_balance"][0]) / om["ebitda"][0])
    assert india["lender"]["covenants"][0]["years"][0]["value"] == pytest.approx(
        (debt + 60.0) / (om["ebitda"][0] + 10.0))


SENIOR = spec_from_kind("institutional_term_loan", "TLB", amount=400.0, reference_level=4.0, margin=4.0)
NOTES = spec_from_kind("senior_notes", "Notes", amount=150.0, fixed_rate=9.0)
SPONSOR = spec_from_kind("shareholder_loan", "Sponsor loan", amount=80.0, fixed_rate=10.0)
RCF = spec_from_kind("revolver", "RCF", amount=100.0, reference_level=4.0, margin=3.0)


def _specs(*specs):
    return [dataclasses.asdict(s) | {"reference_path": [], "amort_schedule": []} for s in specs]


def _api_tranche(spec: TrancheSpec) -> dict:
    d = dataclasses.asdict(spec)
    d.pop("currency")
    return d | {"reference_path": list(spec.reference_path), "amort_schedule": list(spec.amort_schedule)}


def test_the_sponsors_loan_is_never_debt_and_senior_leverage_counts_the_secured_only():
    tranches = [_api_tranche(s) for s in (RCF, SENIOR, NOTES, SPONSOR)]
    answer = run({"tranches": tranches, "covenant_convention": "india", "covenants": [
        lev("TLB"), lev("TLB", scope="senior"), lev("RCF", metric="liquidity", levels=(1.0,))]})
    tr, tests = answer["tranches"], answer["lender"]["covenants"]
    om = answer["operating_model"]
    end = {name: rows[0]["ending_balance"] for name, rows in tr.items()}
    assert end["Sponsor loan"] > 0
    assert tests[0]["years"][0]["value"] == pytest.approx((end["RCF"] + end["TLB"] + end["Notes"]) / om["ebitda"][0])
    assert tests[1]["years"][0]["value"] == pytest.approx((end["RCF"] + end["TLB"]) / om["ebitda"][0])
    # Liquidity: cash plus what the revolver still offers at the year end
    cash = answer["debt_schedule"]["cash_balance"][0]
    assert tests[2]["years"][0]["liquidity"] == pytest.approx(cash + 100.0 - end["RCF"])
    # The revolver tests while it is committed, drawn or not
    assert all(y["tested"] for y in tests[2]["years"])


def test_a_covenant_on_a_facility_already_repaid_no_longer_tests():
    notes = spec_from_kind("senior_notes", "Short notes", amount=50.0, fixed_rate=8.0, maturity_years=2)
    answer = run({"tranches": [_api_tranche(SENIOR), _api_tranche(notes)],
                  "covenants": [lev("Short notes", levels=(0.1,))]})
    years = answer["lender"]["covenants"][0]["years"]
    assert [y["tested"] for y in years] == [True, True, False, False, False]
    assert [y["breach"] for y in years] == [True, True, False, False, False]
    assert all(y["headroom"] is None for y in years[2:])


def test_a_liquidity_level_is_money_in_the_deals_unit():
    """The same deal in thousands: the level is typed in thousands and the
    figure comes back in thousands, a thousand times the millions answer."""
    m = run({"mincash": 20.0, "covenants": [lev(metric="liquidity", levels=(25.0,))]})
    k = run({"ebitda": 100_000.0, "mincash": 20_000.0, "unit": "thousands",
             "covenants": [lev(metric="liquidity", levels=(25_000.0,))]})
    ym, yk = m["lender"]["covenants"][0]["years"][0], k["lender"]["covenants"][0]["years"][0]
    assert yk["liquidity"] == pytest.approx(ym["liquidity"] * 1000)
    assert yk["minimum_liquidity"] == pytest.approx(25_000.0)
    assert ym["breach"] and yk["breach"]
    assert k["lender"]["exposure"] == pytest.approx(m["lender"]["exposure"] * 1000)


def test_covenants_are_stored_only_when_set_and_move_no_figure():
    assert "covenants" not in clean_inputs({}) and "covenant_convention" not in clean_inputs({})
    stored = clean_inputs({"covenants": [lev()], "covenant_convention": "us"})
    assert stored["covenants"][0]["metric"] == "leverage" and stored["covenant_convention"] == "us"
    plain, tested = run({}), run({"covenants": [lev(levels=(1.0,))]})
    assert tested["returns"] == plain["returns"] and tested["debt_schedule"] == plain["debt_schedule"]
    assert plain["lender"]["covenants"] == [] and plain["lender"]["first_breach"] is None
    # A new covenant's level starts from each test's year-one figure, covenants or none
    first = tested["lender"]["covenants"][0]["years"][0]
    assert plain["lender"]["year_one"]["leverage"] == pytest.approx(first["value"])


def test_a_covenant_needs_a_level_and_a_known_metric():
    assert client.post("/api/deal/run", json={"inputs": {"covenants": [lev(levels=())]}}).status_code == 422
    bad = {**lev(), "metric": "net_worth"}
    assert client.post("/api/deal/run", json={"inputs": {"covenants": [bad]}}).status_code == 422


# ---------------------------------------------------------------------------
# The simulation reads its own paths by the same definitions
# ---------------------------------------------------------------------------
ALL_TESTS = (cv.Covenant("X", "leverage", (5.0,)), cv.Covenant("X", "leverage", (5.0,), "senior"),
             cv.Covenant("X", "interest_cover", (2.0,)), cv.Covenant("X", "fixed_charge_cover", (1.0,)),
             cv.Covenant("X", "liquidity", (1.0,)))


@pytest.mark.parametrize("convention", ["us", "lma", "india"])
@pytest.mark.parametrize("name", sorted(STRUCTURES))
def test_a_path_at_the_mean_gives_the_deal_models_covenant_figures(name, convention):
    """One path with nothing uncertain in it is the deal model, so every
    test's figure on it equals the deal model's, to the engine's rounding."""
    deal = DealInputs(tranches=STRUCTURES[name])
    params = dataclasses.replace(build_sim_params(MCInputs(n=1), deal, cfg()), n_interest_passes=50)
    out = _run_vectorized_core(params, draws_at(deal, params), lender=1)
    result = run_deal(deal, cfg())
    for c in ALL_TESTS:
        c = dataclasses.replace(c, facility=params.tranches[0].name)
        sim = cv.evaluate([c], convention, cv.simulation_facilities(deal, params),
                          cv.simulated_figures(out["lender_paths"], params))[0]
        model = cv.evaluate([c], convention, cv.deal_facilities(deal, result), cv.deal_figures(deal, result))[0]
        finite = np.isfinite(model.value)
        np.testing.assert_allclose(sim.value[0][finite], model.value[finite], rtol=2e-3, atol=0.05,
                                   err_msg=f"{c.metric} {c.scope}")
        np.testing.assert_array_equal(sim.tested[0], model.tested)


def test_the_two_bucket_path_at_the_mean_gives_the_deal_models_figures():
    deal = DealInputs()
    params = dataclasses.replace(build_sim_params(MCInputs(n=1), deal, cfg()), n_interest_passes=50)
    out = _run_vectorized_core(params, draws_at(deal, params), lender=1)
    result = run_deal(deal, cfg())
    for c in ALL_TESTS[:4]:
        c = dataclasses.replace(c, facility="Senior Term Loan")
        sim = cv.evaluate([c], "us", cv.simulation_facilities(deal, params),
                          cv.simulated_figures(out["lender_paths"], params))[0]
        model = cv.evaluate([c], "us", cv.deal_facilities(deal, result), cv.deal_figures(deal, result))[0]
        np.testing.assert_allclose(sim.value[0], model.value, rtol=2e-3, err_msg=c.metric)


@pytest.mark.parametrize("deal", [DealInputs(), DealInputs(tranches=STRUCTURES["loan and PIK notes"])],
                         ids=["two-bucket", "tranches"])
def test_tracking_the_paths_moves_no_simulated_figure(deal):
    params = build_sim_params(MCInputs(n=5000), deal, cfg())
    a = run_vectorized_simulation_full(params, seed=7, credit=True)
    b = run_vectorized_simulation_full(params, seed=7, credit=True, lender=3000)
    assert a.df.equals(b.df)
    assert b.lender_paths["closing"].shape[0] == 3000 and a.lender_paths is None


def mc(inputs: dict, **mc_inputs) -> dict:
    r = client.post("/api/montecarlo/run", json={"deal": inputs, "mc": {"n": 4000, **mc_inputs}, "seed": 42})
    assert r.status_code == 200, r.text
    return r.json()


def test_breach_odds_come_from_the_simulation():
    """With the narrowest spreads the rail allows (only the simulation's own
    one-year EBITDA shock is left) nearly every path breaches where the deal
    model does, in year 2 of the stepped covenant, and none in the other
    years; with the usual spreads the odds lie between, and widen as the
    spreads do."""
    inputs = {"covenants": [lev(levels=(6.0, 4.5, 6.0))]}
    still = mc(inputs, growth_std=0.1, exit_std=0.1, rate_std=0.1, gm_std=0.1)["covenants"]
    odds = [y["breach"] for y in still["covenants"][0]["years"]]
    assert odds[1] > 0.9 and max(odds[0], *odds[2:]) < 0.01
    usual = mc(inputs)["covenants"]
    year2 = usual["covenants"][0]["years"][1]["breach"]
    assert 0.05 < year2 < 0.99
    wide = mc(inputs, gm_std=8.0)["covenants"]["covenants"][0]["years"]
    # Wider margins spread year 2's leverage, so its odds move toward even
    assert abs(wide[1]["breach"] - 0.5) < abs(year2 - 0.5)
    assert usual["paths"] == 4000 and usual["convention"] == "lma"
    # breached_by never falls, and any covenant is at least each one
    by = [y["breached_by"] for y in usual["any"]]
    assert by == sorted(by) and usual["any"][1]["breach"] >= year2


def test_no_covenants_no_breach_odds():
    assert mc({})["covenants"] is None


def test_the_tracked_paths_stay_within_their_budget():
    big = DealInputs(hold=15, covenants=(cv.Covenant("A", "leverage", (5.0,)),), tranches=tuple(
        spec_from_kind("amortising_term_loan", f"T{i}", amount=10.0) for i in range(12)))
    params = build_sim_params(MCInputs(n=100_000, hold=15), big, cfg())
    k = lender_paths(big, params)
    assert k * 12 * 15 <= LENDER_CELLS and k >= 2000
    assert lender_paths(DealInputs(), params) == 0
    assert lender_paths(dataclasses.replace(big, tranches=()),
                        build_sim_params(MCInputs(n=3000), DealInputs(), cfg())) == 3000


# ---------------------------------------------------------------------------
# Expected loss from regional base rates
# ---------------------------------------------------------------------------
def test_expected_loss_is_the_regional_rate_times_lgd_times_exposure():
    tranches = [_api_tranche(s) for s in (RCF, SENIOR, NOTES, SPONSOR)]
    lender = run({"country": "DE", "tranches": tranches})["lender"]
    assert (lender["region"], lender["table"]) == ("europe", "europe")
    band = lender["band"]
    pd = base_rates.CUMULATIVE["europe"][band][4]
    assert lender["default_pct"] == pytest.approx(pd)
    rows = {f["name"]: f for f in lender["facilities"]}
    assert rows["RCF"]["exposure"] == pytest.approx(100.0)          # the whole commitment
    assert rows["TLB"]["lgd_pct"] == 22.0 and rows["Notes"]["lgd_pct"] == 26.0
    assert rows["Notes"]["expected_loss"] == pytest.approx(pd / 100 * 0.26 * 150.0)
    assert rows["Notes"]["expected_recovery"] == pytest.approx(150.0 * 0.74)
    assert rows["Sponsor loan"]["lender"] is False and rows["Sponsor loan"]["expected_loss"] is None
    assert lender["exposure"] == pytest.approx(650.0)
    assert lender["expected_loss"] == pytest.approx(sum(
        r["expected_loss"] for r in rows.values() if r["lender"]))


def test_a_country_without_a_regional_table_reads_the_global_one():
    lender = run({"country": "JP"})["lender"]
    assert (lender["region"], lender["table"]) == ("other_developed", "global")
    us = run({"country": "US"})["lender"]
    assert us["default_pct"] == base_rates.CUMULATIVE["us"][us["band"]][4]
    assert us["default_pct"] != base_rates.CUMULATIVE["global"][us["band"]][4]


def test_the_percentage_deals_mezzanine_is_subordinated():
    rows = {f["name"]: f for f in run({})["lender"]["facilities"]}
    assert rows["Senior Term Loan"]["lgd_row"] == "secured"
    assert rows["Mezzanine"]["lgd_row"] == "unsecured_subordinated"


def test_expected_loss_in_core_matches_the_answer():
    deal, c = in_millions(DealInputs(country="US"), cfg())
    view = expected_loss(deal, run_deal(deal, c))
    assert view["expected_loss"] == pytest.approx(run({"country": "US"})["lender"]["expected_loss"])
