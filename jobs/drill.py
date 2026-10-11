"""The job drill (PLAN.md 1.9 "done when"): many simulations at once, on staging.

``POST /api/scheduled/drill`` queues ``count`` identical seeded Monte Carlo
jobs as the owner ``system:drill``; ``ops/job_drill.py`` (run by staging.yml
after each deploy) watches them while it polls ``/api/health``, and checks
that every job succeeded, that the health check kept answering, and that each
result's summary equals ``EXPECTED_SUMMARY`` -- the synchronous endpoint's
answer for the same seed, pinned here and re-checked against the live code by
``tests/test_jobs.py``. So "seeded results are unchanged" is checked on the
server that ran them, not only on a laptop.

Then ``POST /api/scheduled/drill/live`` (PLAN.md 5.8 "done when") runs a
session of the live sliders in the same process, which the ten simulations
have just stretched, and answers how long the estimates took and how much
memory the API used at its peak against what the service has
(``live_drill``; ops/scheduled.py ``live_problems`` judges the answer).

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


# ---------------------------------------------------------------------------
# The live sliders on the server (PLAN.md 5.8 "done when")
# ---------------------------------------------------------------------------
# Render's free web service; a container that says less (cgroup) is believed
SERVICE_MEMORY_MB = 512.0
LIVE_MOVES = 40
# What POST /api/montecarlo/live answers for the default deal with no
# sliders (core.live.LIVE_SEED on LIVE_PATHS paths), re-checked against the
# live code by tests/test_jobs.py
EXPECTED_LIVE: dict[str, float] = {
    "irr_p5": 0.026343774265902644,
    "irr_p50": 0.18570548595366276,
    "irr_p95": 0.31428766513083767,
    "p_wipeout": 0.000125,
}
# Twelve facilities over fifteen years: the largest deal the API takes
_LOAN = {"kind": "institutional_term_loan", "amount": 30.0, "floating": True, "reference_level": 4.0,
         "margin": 3.5, "amort_pct": 1.0, "sweep": True, "maturity_years": 7}
_NOTES = {"kind": "senior_notes", "amount": 30.0, "floating": False, "fixed_rate": 9.0, "maturity_years": 8}
LARGEST_DEAL = {"hold": 15, "tax_interest_limit": "ebitda_share", "tax_loss_carryforward": True,
                "tranches": [{**_LOAN, "name": f"Loan {i}"} for i in range(6)]
                + [{**_NOTES, "name": f"Notes {i}"} for i in range(6)]}


def _megabytes(path: str) -> float | None:
    """A cgroup memory file's figure in MB; None when absent or unlimited."""
    try:
        with open(path, encoding="ascii") as f:
            text = f.read().strip()
    except OSError:
        return None
    return int(text) / 1e6 if text.isdigit() else None


def process_memory() -> dict[str, float | None]:
    """The API process's resident memory now and at its peak, and the
    container's limit (MB). Linux reads /proc and the cgroup; elsewhere the
    peak comes from getrusage and the rest is unknown."""
    now = peak = None
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            fields = dict(line.split(":", 1) for line in f if ":" in line)
        now = int(fields["VmRSS"].split()[0]) / 1000
        peak = int(fields["VmHWM"].split()[0]) / 1000
    except (OSError, KeyError, ValueError):
        import resource
        import sys
        used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak = used / 1e6 if sys.platform == "darwin" else used / 1000      # bytes on macOS, KB elsewhere
    limit = _megabytes("/sys/fs/cgroup/memory.max") or _megabytes("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    return {"rss_mb": now, "peak_mb": peak,
            "limit_mb": min(limit, SERVICE_MEMORY_MB) if limit else SERVICE_MEMORY_MB}


def live_drill(moves: int = LIVE_MOVES, memory=None) -> dict:
    """A session of the live sliders in this process: the default deal as it
    stands, ``moves`` slider positions, and the largest deal the API takes.
    Answers what it measured, flat, for the workflow to judge
    (ops/scheduled.py ``live_problems``)."""
    import time
    from dataclasses import replace

    from core.config import resolve_config
    from core.deal import DealInputs
    from core.live import live_estimate, sliders_of
    from core.montecarlo import MCInputs

    memory = memory or process_memory
    before = memory()
    mc, deal, cfg = MCInputs(), DealInputs(), resolve_config({})
    own = live_estimate(mc, deal, cfg)
    start, timings, medians = sliders_of(mc, deal), [], set()
    for i in range(moves):
        moved = replace(start, growth_mean=start.growth_mean + 0.1 * (i + 1), debt_pct=start.debt_pct - 0.5 * i)
        t0 = time.perf_counter()
        medians.add(live_estimate(mc, deal, cfg, moved)["prediction"]["irr_p50"])
        timings.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    largest = live_estimate(MCInputs(hold=15), DealInputs(**LARGEST_DEAL), cfg)
    largest_ms = (time.perf_counter() - t0) * 1000
    after = memory()
    added = (after["rss_mb"] - before["rss_mb"]) if after["rss_mb"] is not None and before["rss_mb"] is not None else None
    return {
        "live_paths": own["paths"], "live_moves": moves, "live_distinct": len(medians),
        "live_mismatched": ",".join(k for k, want in EXPECTED_LIVE.items()
                                    if abs(own["prediction"][k] - want) > RELATIVE_TOLERANCE * max(abs(want), 1e-12)),
        "live_median_irr": own["prediction"]["irr_p50"],
        "live_typical_ms": round(sorted(timings)[len(timings) // 2], 2), "live_slowest_ms": round(max(timings), 2),
        "live_largest_paths": largest["paths"], "live_largest_ms": round(largest_ms, 2),
        "live_added_mb": None if added is None else round(added, 1),
        "memory_peak_mb": None if after["peak_mb"] is None else round(after["peak_mb"], 1),
        "memory_limit_mb": round(after["limit_mb"], 1),
    }
