"""Tests for the optional ML layer (ml/).

Skipped automatically when the ML dependencies (requirements-ml.txt) are not
installed, so the core test run does not depend on them.
"""

import json
import os

import pytest

pytest.importorskip("sklearn")
pytest.importorskip("joblib")

import ml.anomaly_detector as ad  # noqa: E402


def test_interest_coverage_is_ebitda_over_interest():
    # Leverage is debt/EBITDA, so coverage = 100 / (leverage * rate%).
    # 12.5x at 7.8%: interest is 97.5% of EBITDA -> 1.03x, a covenant breach.
    freescale = ad.check_deal(14.7, 12.5, 4.0, 9.8, 7.8)
    assert any("1.03x" in w for w in freescale.warnings), freescale.warnings
    # 4.2x at 7.26% -> 3.28x: well clear of the 1.5x covenant.
    default = ad.check_deal(10.0, 4.2, 5.0, 26.0, 7.26)
    assert not any("coverage" in w for w in default.warnings), default.warnings


def test_anomaly_flag_separates_historical_outcomes():
    # The flag must carry information: it should catch the historical
    # failures while passing nearly all of the successes.
    with open(os.path.join(ad.BASE, "anomaly_deals.json")) as f:
        deals = json.load(f)
    flagged = {True: [], False: []}
    for d in deals:
        r = ad.check_deal(d["entry_mult"], d["leverage"], d["growth"],
                          d["margin"], d["interest"])
        flagged[d["success"]].append(r.is_anomalous)
    assert all(flagged[False]), "every historical failure should be flagged"
    assert sum(flagged[True]) / len(flagged[True]) <= 0.10, (
        "no more than 10% of historical successes should be flagged")


def test_anomaly_docstring_states_the_real_sample():
    # It once claimed "~100 deals"; the trained file and the docstring must agree
    sample = ad.historical_sample()
    assert sample["deals"] == len(ad.HISTORICAL_DEALS) == 30
    assert "~100" not in ad.__doc__
    assert f"{sample['deals']} hand-entered" in ad.__doc__
    assert f"{sample['first_year']}-{sample['last_year']}" in ad.__doc__


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
    test_interest_coverage_is_ebitda_over_interest()
    test_anomaly_flag_separates_historical_outcomes()
    test_surrogate_tracks_the_simulation()
    print("test_ml: PASS")
