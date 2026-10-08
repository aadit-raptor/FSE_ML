"""Tests for the optional ML layer (ml/).

Skipped automatically when the ML dependencies (requirements-ml.txt) are not
installed, so the core test run does not depend on them.
"""

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")


def test_surrogate_tracks_the_simulation():
    # Checks what the Monte Carlo page relies on: the median, the 5th
    # percentile wherever the page shows it (predicted wipeout < 2%), and
    # that the network learned the fee-inclusive engine.
    pytest.importorskip("torch")
    import numpy as np
    from ml.surrogate.predict import SurrogatePredictor
    from ml.surrogate.generate_data import TRAINING_FIXED
    from simulation.vectorized_simulation import (
        SimulationParams, run_vectorized_simulation_full)

    surrogate = SurrogatePredictor.get_instance()
    if surrogate is None:
        pytest.skip("surrogate model not trained")

    def simulate(x, **overrides):
        p = SimulationParams(n=20_000, **{**TRAINING_FIXED, **overrides}, **x,
                             n_interest_passes=2)
        return run_vectorized_simulation_full(p, seed=7).irr

    rng = np.random.default_rng(11)
    p50_err, p5_err, closer_to_fee_free = [], [], 0
    for i in range(60):
        x = dict(growth_mean=rng.uniform(0.02, 0.16), growth_std=rng.uniform(0.01, 0.09),
                 exit_mean=rng.uniform(6, 18), exit_std=rng.uniform(0.5, 3.5),
                 interest_mean=rng.uniform(0.03, 0.12),
                 gross_margin_mean=rng.uniform(0.25, 0.75),
                 gross_margin_std=rng.uniform(0.015, 0.07), da_pct=rng.uniform(0.03, 0.08),
                 capex_pct=rng.uniform(0.02, 0.09), nwc_pct=rng.uniform(0.008, 0.045),
                 debt_pct=rng.uniform(0.35, 0.85))
        pred, irr = surrogate.predict(**x), simulate(x)
        p50_err.append(abs(pred.irr_p50 - np.median(irr)))
        if pred.p_wipeout < 0.02:
            p5_err.append(abs(pred.irr_p5 - np.percentile(irr, 5)))
        if i < 10:
            fee_free = np.median(simulate(x, transaction_fees_pct=0,
                                          financing_fees_pct=0))
            closer_to_fee_free += (abs(pred.irr_p50 - fee_free)
                                   < abs(pred.irr_p50 - np.median(irr)))

    assert np.median(p50_err) < 0.005, f"median-IRR error {np.median(p50_err):.4f}"
    assert max(p5_err) < 0.03, f"displayed 5th-percentile error {max(p5_err):.4f}"
    assert closer_to_fee_free == 0, "surrogate tracks the fee-free engine"


if __name__ == "__main__":
    test_surrogate_tracks_the_simulation()
    print("test_ml: PASS")
