"""Driver explanations (PLAN.md 5.6).

The Monte Carlo answer says why the deal's worst and best simulated outcomes
are where they are: the simulation's IRR with every driver at its mean, plus
each driver's contribution, equals the tail's mean IRR. What these tests hold
it to:

* the contributions **add up** to the simulated figure, to rounding error,
  for a deal sized by percentages, for every tranche structure, with tax
  rules and with leases -- and that figure is the simulation's own, read
  straight off its paths;
* a driver that cannot move the deal contributes exactly nothing, even where
  its rank correlation with IRR is large;
* the Shapley arithmetic is right on games solved by hand;
* explaining a run holds fewer paths than the run did, and a large run's
  tails are read at a bounded number of evenly spaced paths.
"""
import dataclasses

import numpy as np
import pytest
from fastapi.testclient import TestClient

import analytics.driver_attribution as attribution
from analytics.driver_attribution import (
    DRIVERS, MAX_TAIL_PATHS, TAIL_SHARE, coalition_irr, evenly_spaced, explain_tails, mean_draws,
    shapley_values, tail_paths,
)
from api.main import app
from core.config import resolve_config
from core.deal import DealInputs
from core.debt import TrancheSpec
from core.montecarlo import MCInputs, analysis_sample, build_sim_params, driver_sensitivity
from simulation.vectorized_simulation import _run_vectorized_core, run_vectorized_simulation_full
from tests.test_montecarlo_tranches import STRUCTURES

client = TestClient(app)
SEED = 42
N = 20_000
EXACT = 1e-12

FIXED_ONLY = [TrancheSpec(name="Fixed notes", kind="senior_notes", amount=450.0, fixed_rate=8.0, maturity_years=8)]
DEALS = {
    "percentages": DealInputs(),
    **{name: DealInputs(tranches=tranches) for name, tranches in STRUCTURES.items()},
    "tax rules": DealInputs(tax_interest_limit="ebitda_share", tax_interest_limit_pct=30.0,
                            tax_loss_carryforward=True, tax_loss_limit_pct=60.0),
    "leases, IFRS": DealInputs(accounting_standard="ifrs", lease_cost=8.0, lease_liability=40.0),
}


def simulate(deal, n=N, mc=None, **settings):
    params = build_sim_params(mc or MCInputs(n=n), deal, resolve_config(settings))
    return run_vectorized_simulation_full(params, seed=SEED)


# ---------------------------------------------------------------------------
# The contributions add up to the simulation's own figure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(DEALS))
def test_base_plus_contributions_is_the_tails_simulated_irr(name):
    sim = simulate(DEALS[name])
    out = explain_tails(sim)
    irr = np.sort(sim.irr)
    k = int(round(N * TAIL_SHARE))
    simulated = {"downside": irr[:k].mean(), "upside": irr[-k:].mean()}
    assert [c["case"] for c in out["cases"]] == ["downside", "upside"]
    for case in out["cases"]:
        total = out["base_irr"] + sum(c["irr"] for c in case["contributions"])
        assert total == pytest.approx(case["irr"], abs=EXACT)
        # ... and the figure explained is read off the simulation's paths, not refitted
        assert case["irr"] == pytest.approx(simulated[case["case"]], abs=EXACT)
        assert case["paths"] == case["tail_paths"] == k
        assert [c["driver"] for c in case["contributions"]] == [column for _, column in DRIVERS]
    assert simulated["downside"] < out["base_irr"] < simulated["upside"]


@pytest.mark.parametrize("name", ["percentages", "revolver funding a bullet"])
def test_the_base_is_the_simulation_with_every_driver_at_its_mean(name):
    sim = simulate(DEALS[name])
    params = dataclasses.replace(sim.params, n=1)
    p = sim.params
    at_mean = {"growth": np.array([p.growth_mean]), "exit_multiple": np.array([p.exit_mean]),
               "interest": np.array([p.interest_mean]), "gross_margin": np.array([p.gross_margin_mean]),
               "ebitda_shock": np.array([0.0])}   # no shock: the draw's own mean
    assert mean_draws(p) == {key: float(value[0]) for key, value in at_mean.items()}
    assert explain_tails(sim)["base_irr"] == pytest.approx(_run_vectorized_core(params, at_mean)["IRR"][0], abs=EXACT)


def test_the_default_deals_tails_are_growth_and_the_exit_multiple():
    """The comparison the task asked for, in numbers. Rank correlation makes
    the rate, the gross margin and the one-year shock look like real drivers
    (each about 0.4-0.5 with IRR); they are correlated with growth and the
    exit multiple, and themselves move the worst paths by well under a
    point."""
    sim = simulate(DealInputs(), n=50_000)
    rho = dict(driver_sensitivity(analysis_sample(sim)))
    down = {c["driver"]: c["irr"] for c in explain_tails(sim)["cases"][0]["contributions"]}
    for driver in ("Interest", "Gross Margin", "EBITDA Shock"):
        assert abs(rho[driver]) > 0.35
        assert abs(down[driver]) < 0.01
    assert down["Growth"] < -0.05 and down["Exit Multiple"] < -0.05


# ---------------------------------------------------------------------------
# A driver that moves nothing contributes nothing
# ---------------------------------------------------------------------------
def test_the_rate_contributes_exactly_nothing_to_a_deal_of_fixed_debt():
    sim = simulate(DealInputs(tranches=FIXED_ONLY))
    rho = dict(driver_sensitivity(analysis_sample(sim)))
    assert abs(rho["Interest"]) > 0.3   # what the rank correlation shows for it
    for case in explain_tails(sim)["cases"]:
        by_driver = {c["driver"]: c["irr"] for c in case["contributions"]}
        assert by_driver["Interest"] == 0.0
        assert by_driver["Growth"] != 0.0


def test_a_driver_with_no_spread_contributes_nothing():
    mc = MCInputs(n=N, growth_std=0.0)
    for case in explain_tails(simulate(DealInputs(), mc=mc))["cases"]:
        by_driver = {c["driver"]: c["irr"] for c in case["contributions"]}
        assert by_driver["Growth"] == pytest.approx(0.0, abs=EXACT)
        assert abs(by_driver["Exit Multiple"]) > 0.01


# ---------------------------------------------------------------------------
# The arithmetic, on games solved by hand
# ---------------------------------------------------------------------------
def worth(function, players):
    import itertools
    return {mask: function(*mask) for mask in itertools.product((0, 1), repeat=players)}


def test_shapley_values_of_a_game_solved_by_hand():
    """v = a + 2b + 4ab, with a third player who does nothing.

    The interaction 4ab is shared equally: a = 1 + 2 = 3, b = 2 + 2 = 4, c = 0;
    together 7 = v(1,1,1) - v(0,0,0)."""
    values = worth(lambda a, b, c: a + 2 * b + 4 * a * b, 3)
    assert shapley_values(values) == pytest.approx([3.0, 4.0, 0.0], abs=EXACT)


def test_shapley_values_of_the_glove_game():
    """One left glove (a) and two right gloves (b, c); a pair is worth 1.
    The textbook answer: 2/3, 1/6, 1/6."""
    values = worth(lambda a, b, c: float(a and (b or c)), 3)
    assert shapley_values(values) == pytest.approx([2 / 3, 1 / 6, 1 / 6], abs=EXACT)


def test_tails_are_never_larger_than_their_share():
    # 40 of 100 paths wiped out: the worst 5% is five of them, not all forty
    irr = np.concatenate([np.full(40, -1.0), np.linspace(0.0, 0.3, 60)])
    tails = tail_paths(irr, 0.05)
    assert len(tails["downside"]) == 5 and len(tails["upside"]) == 5
    assert set(irr[tails["downside"]]) == {-1.0}
    assert irr[tails["upside"]].min() == pytest.approx(np.sort(irr)[-5])
    one = tail_paths(np.array([0.2]))
    assert list(one["downside"]) == [0] and list(one["upside"]) == [0]


# ---------------------------------------------------------------------------
# Light enough for the free server
# ---------------------------------------------------------------------------
def test_explaining_a_run_holds_fewer_paths_than_the_run(monkeypatch):
    sim = simulate(DEALS["revolver funding a bullet"], n=10_000)
    expected = explain_tails(sim)
    sizes = []
    core = attribution._run_vectorized_core

    def recording(params, draws, *args, **kwargs):
        sizes.append(params.n)
        assert all(len(values) == params.n for values in draws.values())
        return core(params, draws, *args, **kwargs)

    monkeypatch.setattr(attribution, "_run_vectorized_core", recording)
    assert explain_tails(sim) == expected
    assert max(sizes) <= 5_000 and len(sizes) == 8   # 500 paths a tail, 156 a slice


def test_a_large_runs_tails_are_read_at_evenly_spaced_paths():
    """100,000 paths: each tail holds 5,000, of which 2,500 are explained,
    from the very worst to the 5,000th worst. Their mean is what the
    contributions add up to, and it sits within a tenth of a point of the
    whole tail's."""
    sim = simulate(DealInputs(), n=100_000)
    irr = np.sort(sim.irr)
    out = explain_tails(sim)
    for case, tail in zip(out["cases"], (irr[:5_000], irr[-5_000:])):
        assert (case["paths"], case["tail_paths"]) == (MAX_TAIL_PATHS, 5_000)
        picked = tail[np.linspace(0, 4_999, MAX_TAIL_PATHS).round().astype(int)]
        assert case["irr"] == pytest.approx(picked.mean(), abs=EXACT)
        assert case["irr"] == pytest.approx(tail.mean(), abs=0.001)
        assert out["base_irr"] + sum(c["irr"] for c in case["contributions"]) == pytest.approx(case["irr"], abs=EXACT)
    rows = np.arange(10)
    assert list(evenly_spaced(rows, 4)) == [0, 3, 6, 9] and evenly_spaced(rows, 10) is rows


def test_slicing_does_not_change_the_answer():
    sim = simulate(DealInputs(), n=2_000)
    paths = {key: sim.df[column].values[:100] for key, column in DRIVERS}
    whole = coalition_irr(dataclasses.replace(sim.params, n=1_000_000), paths)
    sliced = coalition_irr(dataclasses.replace(sim.params, n=64), paths)   # one path a slice
    assert sliced.keys() == whole.keys()
    for mask, value in whole.items():
        assert sliced[mask] == pytest.approx(value, abs=EXACT)


def test_explaining_changes_nothing_in_the_run():
    sim = simulate(DEALS["loan and PIK notes"])
    before, n = sim.df.copy(), sim.params.n
    explain_tails(sim)
    assert sim.params.n == n and sim.df.equals(before)


# ---------------------------------------------------------------------------
# The endpoint and the background job
# ---------------------------------------------------------------------------
def run(**body):
    response = client.post("/api/montecarlo/run", json={"mc": {"n": N}, "seed": SEED, **body})
    assert response.status_code == 200, response.text
    return response.json()


def test_the_endpoint_answers_explanations_that_add_up():
    body = run()
    out = body["explanations"]
    assert out == run()["explanations"]
    assert out["share"] == TAIL_SHARE
    summary = body["summary"]
    down, up = out["cases"]
    for case in (down, up):
        total = out["base_irr"] + sum(c["irr"] for c in case["contributions"])
        assert total == pytest.approx(case["irr"], abs=EXACT)
        assert case["paths"] == case["tail_paths"] == N * TAIL_SHARE
    # the tails lie beyond the summary's percentiles
    assert down["irr"] < summary["p5_irr"] < out["base_irr"] < summary["p95_irr"] < up["irr"]


def test_the_explanation_is_the_same_in_any_money_unit():
    """An IRR has no unit: a deal in thousands explains as it does in millions."""
    thousands = run(mc={"n": N, "ebitda": 100_000.0}, deal={"unit": "thousands", "ebitda": 100_000.0})
    assert thousands["explanations"] == run()["explanations"]


def test_a_scenario_moves_the_base_it_explains_from():
    base, recession = run()["explanations"], run(scenario="recession")["explanations"]
    assert recession["base_irr"] < base["base_irr"] - 0.02
    for case in recession["cases"]:
        total = recession["base_irr"] + sum(c["irr"] for c in case["contributions"])
        assert total == pytest.approx(case["irr"], abs=EXACT)


def test_a_background_job_answers_the_same_explanations(job_queue):
    from jobs.runner import Runner

    request = {"mc": {"n": N}, "seed": SEED}
    submitted = client.post("/api/jobs", json={"kind": "montecarlo.run", "input": request})
    assert submitted.status_code == 202, submitted.text
    assert Runner(lambda: job_queue).run_next() == "succeeded"
    job = client.get(f"/api/jobs/{submitted.json()['id']}").json()
    assert job["result"]["explanations"] == run()["explanations"]
