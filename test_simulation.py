"""Smoke test for the vectorized Monte Carlo engine and risk analytics.

Run directly (`python test_simulation.py`) or under pytest.

Note: this previously imported simulation.monte_carlo.run_simulation, which is
a superseded scalar-loop implementation that no longer matches the run_lbo()
signature. The app runs simulation.vectorized_simulation, so the test now
covers that path instead.
"""

from simulation.vectorized_simulation import (
    SimulationParams,
    run_vectorized_simulation_full,
)
from analytics.risk_metrics import calculate_risk_metrics

N = 10_000


def build_simulation():
    params = SimulationParams(n=N)
    sim = run_vectorized_simulation_full(params, seed=42)   # seeded: reproducible
    metrics = calculate_risk_metrics(sim.df)
    return sim, metrics


def test_vectorized_simulation():
    sim, metrics = build_simulation()

    assert len(sim.df) == N, f"expected {N:,} paths, got {len(sim.df)}"
    for col in ("IRR", "MOIC", "Growth", "Exit Multiple", "Interest"):
        assert col in sim.df.columns, f"missing column: {col}"
    assert sim.n_valid + sim.n_wiped == N, "valid + wiped must equal n"
    assert sim.df["IRR"].notna().all(), "IRR contains NaN"

    assert metrics["5% Downside IRR"] <= metrics["Median IRR"] <= metrics["95% Upside IRR"], \
        "IRR percentiles are not monotonic"
    assert 0.0 <= metrics["Probability IRR > Target"] <= 1.0, "probability out of [0, 1]"


if __name__ == "__main__":
    test_vectorized_simulation()
    sim, metrics = build_simulation()

    print(f"\nPaths: {len(sim.df):,}  |  wiped out: {sim.n_wiped:,} "
          f"({sim.wipeout_rate * 100:.2f}%)\n")
    print("RISK ANALYTICS\n")
    for k, v in metrics.items():
        print(f"  {k:<28} {round(v, 4)}")
    print("\ntest_simulation: PASS")
