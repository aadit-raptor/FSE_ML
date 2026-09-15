"""Deliberate fixes to open model findings (see CLAUDE.md, "Model findings").

Each test pins the corrected behaviour with an independent check, so the
Streamlit golden snapshot stays the record of the old behaviour and every
departure from it is explained here.
"""
import threading

import numpy as np

from simulation.vectorized_simulation import SimulationParams, run_vectorized_simulation_full


# ---------------------------------------------------------------------------
# Finding 9: a fixed seed must be reproducible under concurrent simulations
# ---------------------------------------------------------------------------
def _irr(seed, n=20000):
    return run_vectorized_simulation_full(SimulationParams(n=n), seed=seed).df["IRR"].values


def test_seeded_simulations_are_reproducible_when_run_concurrently():
    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    expected = {s: _irr(s) for s in seeds}
    results, errors = {}, []
    barrier = threading.Barrier(len(seeds) * 3)

    def worker(seed, k):
        try:
            barrier.wait()
            results[(seed, k)] = _irr(seed)
        except Exception as e:  # pragma: no cover - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(s, k)) for s in seeds for k in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for (seed, _), irr in results.items():
        np.testing.assert_array_equal(irr, expected[seed])


def test_local_generator_keeps_the_legacy_seeded_stream():
    # Golden Monte Carlo numbers depend on np.random.seed(seed) followed by
    # standard_normal; the local RandomState must reproduce that stream.
    np.random.seed(42)
    legacy = np.random.standard_normal((5, 1000))
    local = np.random.RandomState(42).standard_normal((5, 1000))
    np.testing.assert_array_equal(legacy, local)
