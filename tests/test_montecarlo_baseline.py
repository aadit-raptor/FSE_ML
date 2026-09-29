"""The Monte Carlo simulation's output, pinned (PLAN.md 2.4b).

Until this file nothing pinned what the simulation *answers*: the golden
snapshot records only the parameters it is handed (``sim_params``), and
``tests/test_api.py`` checks that a seed reproduces itself, which any change
to the maths would still pass. 2.4b adds a tranche path beside the two-bucket
one in ``simulation/vectorized_simulation.py``; these numbers are what proves
the two-bucket path -- every deal sized by percentages -- did not move.

The figures were recorded from the code before 2.4b touched it. They are
compared at the golden snapshot's tolerance (``tests/golden_compare.py``):
bit-for-bit on one machine, but the correlated draws go through a BLAS matrix
product whose last bits can differ between Windows and Linux CI. A real
change moves them by far more: pricing the senior loan at ``interest * 1.001``
moves the mean IRR in the fifth significant figure.

Never re-record these to make a test pass. A deliberate change to the
simulation's maths needs the user's approval (CLAUDE.md "Keep model logic as
is") and then updates this file explicitly, saying why.
"""
import numpy as np
import pytest

from core.config import resolve_config
from core.deal import DealInputs
from core.montecarlo import MCInputs, apply_scenario, build_sim_params
from simulation.vectorized_simulation import SimulationParams, run_vectorized_simulation_full
from tests.golden_compare import ABS, REL

SEED = 42
N = 20_000


def headline(sim) -> dict:
    irr = sim.irr
    return {
        "mean_irr": float(irr.mean()),
        "p5_irr": float(np.percentile(irr, 5)),
        "p95_irr": float(np.percentile(irr, 95)),
        "wipeout_rate": float(sim.wipeout_rate),
        "mean_moic": float(sim.moic.mean()),
        "mean_net_debt_at_exit": float(sim.df["Net Debt Exit"].mean()),
    }


def api_params():
    """What the Monte Carlo endpoint builds for the default deal and settings."""
    return build_sim_params(MCInputs(n=N), DealInputs(), resolve_config({}))


# Recorded 2026-09-30 from main at d995dc9, before 2.4b.
PINNED = {
    # SimulationParams' own defaults
    "engine_defaults": {
        "mean_irr": 0.20131168714213338, "p5_irr": 0.04575744106541201,
        "p95_irr": 0.33962064002930675, "wipeout_rate": 5e-05,
        "mean_moic": 2.641987968757912, "mean_net_debt_at_exit": 247.12307430786146,
    },
    # The default deal and Settings, as the endpoint builds them (fees, the
    # settings' correlation matrix and interest passes)
    "default_deal": {
        "mean_irr": 0.17938058733492665, "p5_irr": 0.026666133572437137,
        "p95_irr": 0.31516457731491737, "wipeout_rate": 5e-05,
        "mean_moic": 2.4094737517172025, "mean_net_debt_at_exit": 247.12307430786146,
    },
    # A scenario preset: shifted means, a higher rate
    "stagflation": {
        "mean_irr": 0.0654212275596628, "p5_irr": -0.11735855839205445,
        "p95_irr": 0.22298998547229587, "wipeout_rate": 0.002,
        "mean_moic": 1.5126306528718196, "mean_net_debt_at_exit": 298.4508093340895,
    },
    # Minimum cash, every fee and more leverage: the branches the defaults skip
    "stressed": {
        "mean_irr": 0.24749569144072656, "p5_irr": 0.0732014456998653,
        "p95_irr": 0.4031976447064944, "wipeout_rate": 0.00015,
        "mean_moic": 3.220847993648453, "mean_net_debt_at_exit": 345.72175330527756,
    },
}


def params_for(case):
    if case == "engine_defaults":
        return SimulationParams(n=N)
    if case == "default_deal":
        return api_params()
    if case == "stagflation":
        return apply_scenario("stagflation", api_params(), resolve_config({}))
    return SimulationParams(
        n=N, minimum_cash_pct=0.02, transaction_fees_pct=0.02, financing_fees_pct=0.03,
        other_uses=5.0, debt_pct=0.75)


@pytest.mark.parametrize("case", sorted(PINNED))
def test_the_two_bucket_simulation_answers_what_it_answered_before_tranches(case):
    got = headline(run_vectorized_simulation_full(params_for(case), seed=SEED))
    for key, expected in PINNED[case].items():
        assert got[key] == pytest.approx(expected, rel=REL, abs=ABS), f"{case}.{key}"
