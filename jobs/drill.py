"""The job drill (PLAN.md 1.9 "done when"): many simulations at once, on staging.

``POST /api/scheduled/drill`` queues ``count`` identical seeded Monte Carlo
jobs as the owner ``system:drill``; ``ops/job_drill.py`` (run by staging.yml
after each deploy) watches them while it polls ``/api/health``, and checks
that every job succeeded, that the health check kept answering, and that each
result's summary equals ``EXPECTED_SUMMARY`` -- the synchronous endpoint's
answer for the same seed, pinned here and re-checked against the live code by
``tests/test_jobs.py``. So "seeded results are unchanged" is checked on the
server that ran them, not only on a laptop.

Refused in production: the drill is load on purpose.
"""
from __future__ import annotations

DRILL_OWNER = "system:drill"
MAX_DRILL_JOBS = 10
# The app's default deal and simulation size, with a fixed seed
DRILL_REQUEST = {"mc": {"n": 50000}, "seed": 20260918}
# What POST /api/montecarlo/run answers for DRILL_REQUEST (summary only).
# Compared at 1e-9 relative, like the golden snapshot: Linux numpy can differ
# in the last bits.
EXPECTED_SUMMARY: dict[str, float] = {
    "mean_irr": 0.17943005567295175,
    "median_irr": 0.18450687170702174,
    "p5_irr": 0.0269408373574059,
    "p95_irr": 0.31468050755760724,
    "p_above_hurdle": 0.42886,
    "wipeout_rate": 0.0,
    "hurdle": 0.2,
}
RELATIVE_TOLERANCE = 1e-9


def summary_mismatches(summary: dict) -> list[str]:
    """Keys of ``summary`` that differ from ``EXPECTED_SUMMARY``; empty when all match."""
    bad = []
    for key, want in EXPECTED_SUMMARY.items():
        got = summary.get(key)
        if not isinstance(got, (int, float)) or abs(got - want) > RELATIVE_TOLERANCE * max(abs(want), 1e-12):
            bad.append(key)
    return bad
