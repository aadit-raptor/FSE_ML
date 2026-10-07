"""The model validation report: calibration and bias per group (PLAN.md 4.6).

For every check and each sample (``out_of_time``, the headline, and
``in_sample``), the cases overall and split by each dimension in
``validation.tags.DIMENSIONS``, every bucket listed, empty ones too.

**Calibration.** A probability check (``default``, ``loss``) compares the
predicted probability with what happened: the mean predicted and the share
that happened (per cent), the expected and observed counts, the Brier score,
and ``z`` = (observed - expected) / sqrt(sum p(1 - p)); the group is
``consistent`` when |z| <= 1.96. The IRR range check counts how often the
actual IRR fell inside the plan's central 50%, 80% and 90% ranges of
simulated paths, with a 95% Wilson interval for each share; ``consistent``
when every claimed level lies inside its interval.

**Bias.** Probability checks: observed minus predicted, percentage points.
IRR: the mean of actual minus planned IRR, percentage points, and the mean
percentile at which the actual IRR fell (50 if unbiased).

**Statistics need** ``MIN_CASES`` cases in a group.

**Anonymity.** Users' deals appear only as part of a group: the report holds
no deal, owner, name or money figure. A group holding 1 to
``MIN_CONTRIBUTED - 1`` users' deals shows neither its statistics nor its
count of them. Within a dimension, if the suppressed groups together hold
fewer than ``MIN_CONTRIBUTED`` users' deals, the smallest groups holding
any are suppressed too until they hold that many, so subtracting the shown
groups from the overall figures never isolates fewer deals than that (the
overall figures themselves are suppressed below that many). Reference transactions are
public filings and always counted.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

from validation import tags
from validation.cases import Case

CHECKS = ("default", "irr_range", "loss")
PROBABILITY_CHECKS = ("default", "loss")
SAMPLES = ("out_of_time", "in_sample")
LEVELS = (50, 80, 90)
MIN_CASES = 5
MIN_CONTRIBUTED = 5
Z_95 = 1.96
# Reports kept: a month of nightly runs
KEEP_REPORTS = 30


def _r(x: Optional[float], places: int = 2) -> Optional[float]:
    return None if x is None or not math.isfinite(x) else round(float(x), places)


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """The Wilson score interval for a share, per cent."""
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half) * 100, min(1.0, centre + half) * 100


def probability_stats(cases: Sequence[Case]) -> dict:
    """Calibration and bias of predicted probabilities against outcomes."""
    p = [c.predicted for c in cases]
    o = [1.0 if c.happened else 0.0 for c in cases]
    n = len(cases)
    expected, observed = sum(p), sum(o)
    variance = sum(x * (1 - x) for x in p)
    z = (observed - expected) / math.sqrt(variance) if variance > 0 else None
    return {
        "predicted_pct": _r(expected / n * 100), "observed_pct": _r(observed / n * 100),
        "expected": _r(expected), "observed": int(observed),
        "bias_pp": _r((observed - expected) / n * 100),
        "brier": _r(sum((a - b) ** 2 for a, b in zip(p, o)) / n, 4),
        "z": _r(z), "consistent": None if z is None else abs(z) <= Z_95,
    }


def inside(percentile: float, level: int) -> bool:
    tail = (100 - level) / 2
    return tail <= percentile <= 100 - tail


def range_stats(cases: Sequence[Case]) -> dict:
    """How often the actual IRR fell inside the plan's central ranges."""
    n = len(cases)
    levels = []
    for level in LEVELS:
        hits = sum(1 for c in cases if inside(c.percentile, level))
        low, high = wilson(hits, n)
        levels.append({"claimed_pct": level, "inside_pct": _r(hits / n * 100),
                       "interval_pct": [_r(low), _r(high)], "consistent": low <= level <= high})
    return {
        "levels": levels,
        "bias_pp": _r(sum(c.irr_error_pp for c in cases) / n),
        "mean_percentile": _r(sum(c.percentile for c in cases) / n),
        "consistent": all(lv["consistent"] for lv in levels),
    }


def _stats(check: str, cases: Sequence[Case]) -> dict:
    return probability_stats(cases) if check in PROBABILITY_CHECKS else range_stats(cases)


def _cell(check: str, cases: Sequence[Case], suppressed: bool) -> dict:
    library_n = sum(1 for c in cases if c.origin == "library")
    contributed_n = len(cases) - library_n
    if suppressed:
        return {"n": None, "library_n": library_n, "contributed_n": None, "suppressed": True,
                "enough": False, "stats": None}
    enough = len(cases) >= MIN_CASES
    return {"n": len(cases), "library_n": library_n, "contributed_n": contributed_n, "suppressed": False,
            "enough": enough, "stats": _stats(check, cases) if enough else None}


def _contributed(cases: Iterable[Case]) -> int:
    return sum(1 for c in cases if c.origin == "contributed")


def _too_few(cases: Sequence[Case]) -> bool:
    return 0 < _contributed(cases) < MIN_CONTRIBUTED


def split(check: str, cases: Sequence[Case], dimension: str) -> list[dict]:
    """One dimension's groups, every bucket listed (``unknown`` only when used)."""
    buckets = list(tags.BUCKETS[dimension])
    if any(c.groups[dimension] == tags.UNKNOWN for c in cases):
        buckets.append(tags.UNKNOWN)
    grouped = {b: [c for c in cases if c.groups[dimension] == b] for b in buckets}
    hidden = {b for b, members in grouped.items() if _too_few(members)}
    # The complement rule: what the shown groups leave of the overall figures
    # must not isolate fewer than MIN_CONTRIBUTED users' deals, so the
    # smallest shown groups holding any are hidden too until it doesn't
    shown = sorted((b for b, members in grouped.items() if b not in hidden and _contributed(members)),
                   key=lambda b: (_contributed(grouped[b]), buckets.index(b)))
    while shown and 0 < sum(_contributed(grouped[b]) for b in hidden) < MIN_CONTRIBUTED:
        hidden.add(shown.pop(0))
    return [{"bucket": b, **_cell(check, grouped[b], b in hidden)} for b in buckets]


def sample(check: str, cases: Sequence[Case]) -> dict:
    return {
        "overall": _cell(check, cases, _too_few(cases)),
        "splits": {dim: split(check, cases, dim) for dim in tags.DIMENSIONS},
    }


def build(cases: Sequence[Case], *, generated_at: str, engine_version: str, library_included: bool,
          fit_until: int) -> dict:
    """The report for these cases. Nothing in it names a deal or its owner."""
    contributed = _contributed(cases)
    checks = []
    for check in CHECKS:
        mine = [c for c in cases if c.check == check]
        checks.append({"id": check, "samples": {
            "out_of_time": sample(check, [c for c in mine if c.out_of_time]),
            "in_sample": sample(check, [c for c in mine if not c.out_of_time]),
        }})
    return {
        "generated_at": generated_at,
        "engine_version": engine_version,
        "library_included": library_included,
        "fit_until": {"default": fit_until},
        "rules": {"min_cases": MIN_CASES, "min_contributed": MIN_CONTRIBUTED, "levels": list(LEVELS),
                  "z": Z_95},
        "cases": {
            "library": sum(1 for c in cases if c.origin == "library"),
            # One user's deal makes two cases (IRR range and loss)
            "contributed_deals": None if 0 < contributed // 2 < MIN_CONTRIBUTED else contributed // 2,
        },
        "dimensions": list(tags.DIMENSIONS),
        "checks": checks,
    }
