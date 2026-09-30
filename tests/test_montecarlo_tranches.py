"""The simulation finances a deal the way its tranches say (PLAN.md 2.4b).

``simulation/vectorized_simulation.py`` keeps its two-bucket path for deals
sized by percentages (pinned by ``tests/test_montecarlo_baseline.py``) and
gains a tranche path beside it. What these tests hold it to:

* writing today's structure out as tranches simulates **exactly** what the
  percentages simulate, path by path, in every scenario;
* at the mean, with nothing uncertain, a path lands on the deal model's own
  answer for the same structure -- PIK, a revolver's draw and fee, a partial
  sweep share and a floating loan with a floor included;
* the rate draw moves **floating facilities only**, floored after the shock
  and before the margin, and a scenario's rate stress reaches them the same
  way;
* the heatmap runs on the deal's structure, not the percentages;
* the endpoints and background jobs take such a deal, and refuse one that
  cannot be financed without logging its figures.
"""
import dataclasses
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.config import resolve_config
from core.deal import DealInputs, run_deal
from core.debt import TrancheSpec, equivalent_tranches, simulation_tranches, spec_from_kind
from core.montecarlo import (
    MCInputs, apply_scenario, build_sim_params, driver_sensitivity, growth_exit_heatmap,
    run_scenarios,
)
from simulation.tranches import SimulationTranche, tranche_rates
from simulation.vectorized_simulation import (
    _draw_correlated_inputs, _run_vectorized_core, run_vectorized_simulation_full,
)
from tests.golden_compare import ABS, REL

client = TestClient(app)
SEED = 42
N = 20_000


def cfg(**overrides):
    return resolve_config(overrides)


def params_for(deal, n=N, **settings):
    return build_sim_params(MCInputs(n=n), deal, cfg(**settings))


def explicit(deal=None):
    """The default deal with its percentages written out as tranches."""
    deal = deal or DealInputs()
    return dataclasses.replace(deal, tranches=equivalent_tranches(deal, cfg()))


# ---------------------------------------------------------------------------
# Today's structure, written out, simulates exactly as before
# ---------------------------------------------------------------------------
def test_the_equivalent_structure_simulates_exactly_like_the_percentages():
    """The first click on "use explicit tranches" must not move the
    simulation either: every path's IRR, MOIC and net debt, not just the
    averages."""
    a = run_vectorized_simulation_full(params_for(DealInputs()), seed=SEED).df
    b = run_vectorized_simulation_full(params_for(explicit()), seed=SEED).df
    for col in ("IRR", "MOIC", "Net Debt Exit", "Exit Equity"):
        np.testing.assert_allclose(b[col].values, a[col].values, rtol=REL, atol=ABS, err_msg=col)


def test_the_equivalent_structure_matches_in_every_scenario():
    """A scenario's rate multiplier moves the senior and mezzanine rates in
    the percentages path; written out, both are floating, so it must move
    them by exactly as much."""
    a = run_scenarios(params_for(DealInputs(), n=5000), cfg(), seed=SEED)
    b = run_scenarios(params_for(explicit(), n=5000), cfg(), seed=SEED)
    for sc in a:
        np.testing.assert_allclose(b[sc].irr, a[sc].irr, rtol=REL, atol=ABS, err_msg=sc)


# ---------------------------------------------------------------------------
# At the mean, a path is the deal model
# ---------------------------------------------------------------------------
def draws_at(deal, params, n=1):
    """Draws pinned to the deal's own assumptions, with no shock and the rate
    exactly at its mean -- one path through the simulation with nothing
    uncertain in it."""
    full = lambda x: np.full(n, float(x))   # noqa: E731
    return {
        "growth": full(deal.growth / 100), "exit_multiple": full(deal.exit_mult),
        "interest": full(params.interest_mean), "gross_margin": full(deal.gross_margin / 100),
        "ebitda_shock": full(0.0),
    }


SONIA = TrancheSpec(
    name="GBP term loan", kind="amortising_term_loan", amount=400.0, floating=True,
    reference_rate="SONIA", reference_path=[3.0, 2.0, 1.0, 4.0, 5.0], floor=2.5, margin=4.0,
    amort_pct=10.0, sweep=True, maturity_years=7)
PIK = spec_from_kind("pik_notes", "PIK notes", amount=120.0, fixed_rate=12.0)
UNITRANCHE_HALF_SWEEP = spec_from_kind(
    "unitranche", "Unitranche", amount=500.0, reference_level=5.0, margin=5.5, floor=0.5,
    sweep_share=50.0)
# A bullet far bigger than a year's cash, due in year 3, and a revolver
# committed to cover it: the revolver draws, and pays its fee on what is left
BULLET = spec_from_kind("senior_notes", "Notes due year 3", amount=300.0, fixed_rate=8.0,
                        maturity_years=3)
REVOLVER = spec_from_kind("revolver", "RCF", amount=150.0, reference_level=4.0, margin=3.0)

STRUCTURES = {
    "floating loan with a floor": [SONIA],
    "loan and PIK notes": [SONIA, PIK],
    "unitranche taking half the sweep": [UNITRANCHE_HALF_SWEEP],
    "revolver funding a bullet": [BULLET, REVOLVER, SONIA],
    # Never drawn: five years of commitment fee, each one less cash to sweep
    "undrawn revolver beside a unitranche": [
        dataclasses.replace(UNITRANCHE_HALF_SWEEP, sweep_share=100.0), REVOLVER],
}


@pytest.mark.parametrize("name", sorted(STRUCTURES))
def test_a_path_at_the_mean_lands_on_the_deal_models_answer(name):
    """The tranche path is the deal model's debt schedule, vectorised. The
    only differences are the engine's rounding to hundredths of a million,
    so the tolerance is a few of those, far below any real mechanic's
    effect (a PIK coupon left out of the cash flow moves net debt by
    tens)."""
    deal = DealInputs(tranches=STRUCTURES[name])
    params = dataclasses.replace(params_for(deal, n=1), n_interest_passes=50)
    out = _run_vectorized_core(params, draws_at(deal, params))
    r = run_deal(deal, cfg())
    assert out["Net Debt Exit"][0] == pytest.approx(r.returns.net_debt_at_exit, abs=0.05)
    assert out["IRR"][0] == pytest.approx(r.returns.irr, abs=1e-4)


def test_the_revolver_case_really_draws():
    """The case above is only a test of the revolver if the deal model draws
    on it: the bullet in year 3 is more than that year's cash."""
    r = run_deal(DealInputs(tranches=STRUCTURES["revolver funding a bullet"]), cfg())
    rows = r.debt_schedule.schedule["RCF"]
    assert rows[2].redrawn > 0 and rows[0].commitment_fee > 0
    # ... and the undrawn case pays its fee every year and never draws
    r = run_deal(DealInputs(tranches=STRUCTURES["undrawn revolver beside a unitranche"]), cfg())
    rows = r.debt_schedule.schedule["RCF"]
    assert all(row.commitment_fee == 0.75 and row.redrawn == 0 for row in rows)


# ---------------------------------------------------------------------------
# The rate draw moves floating facilities only
# ---------------------------------------------------------------------------
def test_a_floating_rate_is_floored_after_the_shock_and_before_the_margin():
    """Reference 1.00%, floor 0.50%, margin 4.00%:

        shock  -3.00  ->  max(1.00 - 3.00, 0.50) + 4.00 = 4.50
        shock   0.00  ->  max(1.00,        0.50) + 4.00 = 5.00
        shock  +2.00  ->  max(3.00,        0.50) + 4.00 = 7.00

    Flooring the all-in rate instead would give max(-2.00 + 4.00, 0.50) =
    2.00 for the first, and so would flooring before the shock:
    max(1.00, 0.50) - 3.00 + 4.00."""
    t = SimulationTranche(name="loan", amount=100.0, rates=(0.01,), floating=True,
                          floor=0.005, margin=0.04)
    got = tranche_rates(t, np.array([-0.03, 0.0, 0.02]), n_years=2)
    np.testing.assert_allclose(got, [[0.045, 0.045], [0.05, 0.05], [0.07, 0.07]], rtol=1e-12)


def test_a_fixed_rate_ignores_the_shock_and_a_path_repeats_its_last_year():
    fixed = SimulationTranche(name="notes", amount=100.0, rates=(0.08,))
    np.testing.assert_array_equal(tranche_rates(fixed, np.array([-0.05, 0.05]), 3), np.full((2, 3), 0.08))
    path = SimulationTranche(name="loan", amount=100.0, rates=(0.02, 0.03), floating=True)
    np.testing.assert_allclose(tranche_rates(path, np.array([0.0]), 4), [[0.02, 0.03, 0.03, 0.03]])


def simulate_with(deal, interest_offset=0.0, corr=None, n=5000):
    params = params_for(deal, n=n)
    if corr is not None:
        params = dataclasses.replace(params, corr_matrix=corr)
    draws = _draw_correlated_inputs(params, seed=SEED)
    draws = {**draws, "interest": draws["interest"] + interest_offset}
    return _run_vectorized_core(params, draws)


FIXED_ONLY = [spec_from_kind("senior_notes", "Notes", amount=350.0, fixed_rate=7.0),
              spec_from_kind("pik_notes", "PIK", amount=150.0, fixed_rate=11.0)]
FLOATING = [spec_from_kind("institutional_term_loan", "TLB", amount=350.0, reference_level=4.0,
                           margin=3.5),
            spec_from_kind("pik_notes", "PIK", amount=150.0, fixed_rate=11.0)]


def test_a_structure_with_nothing_floating_does_not_feel_the_rate_draw():
    """Only the rate differs between the two runs, so every other draw is the
    same path; with every facility fixed, no path's answer may move."""
    base = simulate_with(DealInputs(tranches=FIXED_ONLY))
    moved = simulate_with(DealInputs(tranches=FIXED_ONLY), interest_offset=0.02)
    np.testing.assert_array_equal(moved["IRR"], base["IRR"])


def test_a_floating_facility_feels_the_rate_draw():
    base = simulate_with(DealInputs(tranches=FLOATING))
    moved = simulate_with(DealInputs(tranches=FLOATING), interest_offset=0.02)
    assert (moved["IRR"] < base["IRR"]).mean() > 0.99
    assert (moved["Net Debt Exit"] > base["Net Debt Exit"]).mean() > 0.99


def test_with_nothing_floating_the_interest_driver_only_shows_its_correlations():
    """With independent drivers, a structure of fixed facilities gives the
    rate draw no rank correlation with IRR at all: surprising on the screen,
    which is why the Monte Carlo rail says so. A floating one gives a small
    but clear negative one (350 of floating debt, a rate spread of 1.5
    points: about -0.04 against sampling noise of 0.007 on 20,000 paths)."""
    import pandas as pd
    independent = np.eye(5)

    def rho(tranches):
        out = simulate_with(DealInputs(tranches=tranches), corr=independent, n=N)
        return dict(driver_sensitivity(pd.DataFrame(out)))["Interest"]

    assert abs(rho(FIXED_ONLY)) < 0.015
    assert rho(FLOATING) < -0.03


def test_a_scenarios_rate_stress_reaches_floating_facilities_only():
    """Stagflation multiplies the rate mean. For a floating facility that is
    a rise in its reference rate; a fixed one does not care."""
    def irr(tranches, rate_mult):
        settings = cfg(stag_rate_mult=rate_mult)
        params = apply_scenario("stagflation", params_for(DealInputs(tranches=tranches), n=5000), settings)
        return run_vectorized_simulation_full(params, seed=SEED).irr

    np.testing.assert_array_equal(irr(FIXED_ONLY, 1.4), irr(FIXED_ONLY, 1.0))
    assert irr(FLOATING, 1.4).mean() < irr(FLOATING, 1.0).mean() - 0.005


# ---------------------------------------------------------------------------
# PIK and the sweep share
# ---------------------------------------------------------------------------
def test_a_pik_coupon_accrues_in_the_simulation_and_is_not_paid_in_cash():
    """The same notes paying cash and paying in kind: in kind, the coupon is
    added to what is owed at exit, so net debt is higher on every path."""
    cash_pay = [SONIA, dataclasses.replace(PIK, pik_share=0.0)]
    in_kind = [SONIA, PIK]
    a = simulate_with(DealInputs(tranches=cash_pay))
    b = simulate_with(DealInputs(tranches=in_kind))
    assert (b["Net Debt Exit"] > a["Net Debt Exit"]).all()


def test_a_partial_sweep_share_leaves_debt_outstanding():
    full = dataclasses.replace(UNITRANCHE_HALF_SWEEP, sweep_share=100.0)
    a = simulate_with(DealInputs(tranches=[full]))
    b = simulate_with(DealInputs(tranches=[UNITRANCHE_HALF_SWEEP]))
    # Half the cash stays on the balance sheet, so gross debt is higher, and
    # the cash kept is netted off: net debt is higher only by the interest
    # the kept cash failed to save
    assert (b["Net Debt Exit"] >= a["Net Debt Exit"] - 1e-9).all()
    assert (b["Net Debt Exit"] > a["Net Debt Exit"]).mean() > 0.99


# ---------------------------------------------------------------------------
# The heatmap
# ---------------------------------------------------------------------------
def heatmap(deal):
    mc = MCInputs(n=1000)
    return growth_exit_heatmap(build_sim_params(mc, deal, cfg()), mc, deal)[2]


def test_the_heatmap_runs_on_the_deals_own_structure():
    """Each cell is a deal-model run. Without the structure it silently
    reverted to the percentages -- the shape of finding 3."""
    percentages = heatmap(DealInputs())
    np.testing.assert_allclose(heatmap(explicit()), percentages, rtol=REL, atol=ABS)

    structure = [*equivalent_tranches(DealInputs(), cfg()), PIK]
    mc = MCInputs(n=1000)
    g_vals, em_vals, with_pik = growth_exit_heatmap(
        build_sim_params(mc, DealInputs(tranches=structure), cfg()), mc, DealInputs(tranches=structure))
    assert (with_pik != percentages).all()
    # A cell is the deal model's answer for that growth and exit multiple,
    # on the deal's own facilities
    for i, j in [(0, 0), (3, 4), (6, 7)]:
        deal = DealInputs(tranches=structure, growth=float(g_vals[j]) * 100, exit_mult=float(em_vals[i]))
        assert with_pik[i, j] == pytest.approx(run_deal(deal, cfg()).returns.irr, rel=1e-9)


def test_the_heatmap_feels_a_scenarios_rate_stress_on_floating_facilities():
    mc = MCInputs(n=1000)
    deal = DealInputs(tranches=FLOATING)
    base = build_sim_params(mc, deal, cfg())
    stressed = apply_scenario("stagflation", base, cfg(stag_rate_mult=1.4, stag_exit_mult=1.0,
                                                       stag_margin_mult=1.0, stag_growth_adj=0.0))
    calm = apply_scenario("stagflation", base, cfg(stag_rate_mult=1.0, stag_exit_mult=1.0,
                                                   stag_margin_mult=1.0, stag_growth_adj=0.0))
    assert (growth_exit_heatmap(stressed, mc, deal)[2] < growth_exit_heatmap(calm, mc, deal)[2]).all()


# ---------------------------------------------------------------------------
# What core/debt.py hands the simulation
# ---------------------------------------------------------------------------
def test_simulation_tranches_carry_the_deals_conventions():
    sim = simulation_tranches([SONIA, PIK, REVOLVER], hold=5)
    loan, notes, rcf = sim
    assert loan.floating and loan.rates == (0.03, 0.02, 0.01, 0.04, 0.05)
    assert (loan.floor, loan.margin) == (0.025, 0.04)
    assert loan.amort_type == "amortizing" and loan.amort_pct == 0.1 and loan.sweep
    assert not notes.floating and notes.rates == (0.12,) and notes.pik_share == 1.0
    assert rcf.amount == 0.0 and rcf.commitment == 150.0 and rcf.allow_redraw
    assert rcf.commitment_fee_pct == 0.005


def test_a_simulation_is_refused_a_structure_that_raises_more_than_the_deal_costs():
    from core.debt import UnfinanceableStructure
    too_much = [spec_from_kind("unitranche", "U", amount=5000.0, reference_level=5.0)]
    with pytest.raises(UnfinanceableStructure):
        params_for(DealInputs(tranches=too_much))


# ---------------------------------------------------------------------------
# Through the API and as a background job
# ---------------------------------------------------------------------------
TRANCHE_DEAL = {"tranches": [
    {"name": "TLB", "kind": "institutional_term_loan", "amount": 400.0, "floating": True,
     "reference_level": 4.0, "margin": 3.5, "amort_pct": 1.0, "sweep": True},
    {"name": "PIK", "kind": "pik_notes", "amount": 100.0, "fixed_rate": 11.0, "pik_share": 100.0}]}


def test_the_endpoints_simulate_a_deal_that_lists_tranches():
    body = {"mc": {"n": 3000}, "seed": 7, "deal": TRANCHE_DEAL}
    run = client.post("/api/montecarlo/run", json=body)
    assert run.status_code == 200, run.text
    deal = DealInputs(**TRANCHE_DEAL)
    sim = run_vectorized_simulation_full(build_sim_params(MCInputs(n=3000), deal, cfg()), seed=7)
    assert run.json()["summary"]["mean_irr"] == pytest.approx(float(sim.irr.mean()), rel=REL)
    # The echoed parameters stay the scalar ones: the tranches are the deal's own
    assert "tranches" not in run.json()["params"]
    scenarios = client.post("/api/montecarlo/scenarios", json=body)
    assert scenarios.status_code == 200, scenarios.text


def test_a_tranche_deal_in_thousands_simulates_exactly_as_in_millions():
    def deal(unit, k):
        return {**TRANCHE_DEAL, "ebitda": 100.0 * k, "unit": unit, "tranches": [
            {**t, "amount": t["amount"] * k} for t in TRANCHE_DEAL["tranches"]]}
    a = client.post("/api/montecarlo/run", json={"mc": {"n": 2000, "ebitda": 100.0}, "seed": 3,
                                                 "deal": deal("millions", 1)}).json()
    b = client.post("/api/montecarlo/run", json={"mc": {"n": 2000, "ebitda": 100_000.0}, "seed": 3,
                                                 "deal": deal("thousands", 1000)}).json()
    assert b["summary"] == a["summary"] and b["heatmap"] == a["heatmap"]


UNFINANCEABLE = {"tranches": [{"name": "U", "kind": "unitranche", "amount": 5000.0}]}


def test_the_endpoints_refuse_an_unfinanceable_structure_without_logging_it(caplog):
    with caplog.at_level(logging.DEBUG):
        for path in ("/api/montecarlo/run", "/api/montecarlo/scenarios"):
            resp = client.post(path, json={"mc": {"n": 1000}, "deal": UNFINANCEABLE})
            assert resp.status_code == 422, path
            assert "5,000.0" in resp.json()["detail"]
    assert "5,000.0" not in caplog.text


def test_a_background_job_refuses_it_the_same_way(job_queue, caplog):
    from jobs.runner import Runner
    body = {"kind": "montecarlo.run", "input": {"mc": {"n": 1000}, "deal": UNFINANCEABLE}}
    job = client.post("/api/jobs", json=body)
    assert job.status_code == 202, job.text
    with caplog.at_level(logging.DEBUG):
        assert Runner(lambda: job_queue).run_next() == "failed"
    done = client.get(f"/api/jobs/{job.json()['id']}").json()
    assert done["error_status"] == 422 and "5,000.0" in done["error"]
    assert "5,000.0" not in caplog.text and "job_failed" not in caplog.text
