"""One evaluation harness for every model (PLAN.md 5.1).

Phase 5's rule: a model is tested on newer data than it learned from, split
by region, and appears only where it beats a simple baseline. This module is
that rule as code, the same for every model:

- **Time-based splits** (``walk_forward``): train on every case before a
  cutoff year, test on the cases from that year to the next cutoff, and pool
  the predictions, so no case is predicted by a model that saw it or anything
  after it.
- **Per-region results** (``evaluate_set``): the same statistics overall and
  for each of S&P's regions (``library.base_rates.SP_REGIONS``, as the
  validation report and Library -> Coverage use).
- **Baseline comparison** (``summarize``): each statistic for the model and
  for a simple baseline on the same cases; the verdict is ``beats_baseline``
  only when the model is strictly better on the headline statistic.

A group with fewer than ``MIN_CASES`` cases, or whose headline statistic
can't be computed (an AUC needs both outcomes), is ``not_enough_data``: the
app shows nothing there rather than an untested estimate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from library.base_rates import SP_REGIONS

REGIONS = SP_REGIONS
MIN_CASES = 5
PLACES = 4

BEATS = "beats_baseline"
DOES_NOT_BEAT = "does_not_beat_baseline"
NOT_ENOUGH = "not_enough_data"
VERDICTS = (BEATS, DOES_NOT_BEAT, NOT_ENOUGH)


@dataclass(frozen=True)
class Case:
    """One thing a model is asked about, with what really happened.

    ``year`` is when the case became known (None for data with no dates);
    ``truth`` is the outcome or the exact answer the model estimates.
    """
    id: str
    region: str
    year: Optional[int]
    features: Mapping[str, Any]
    truth: Any


@dataclass(frozen=True)
class Metric:
    """A statistic over cases and their predictions.

    ``better`` is ``higher`` or ``lower``; ``tolerance`` is how far it may
    move the wrong way before CI calls the model worse (ops/model_gate.py).
    """
    id: str
    better: str
    compute: Callable[[Sequence[Case], Sequence[Any]], Optional[float]]
    tolerance: float
    description: str


def walk_forward(cases: Sequence[Case], cutoffs: Sequence[int], fit: Callable[[Sequence[Case]], Any],
                 predict: Callable[[Any, Case], Any]) -> list[tuple[Case, Any]]:
    """Out-of-time predictions with an expanding training window.

    For each cutoff, ``fit`` sees the cases before it and ``predict`` answers
    the cases from it up to the next cutoff (the last fold has no end). A
    fold with no training or no test case is skipped; cases before the first
    cutoff, and undated ones, are never tested.
    """
    if list(cutoffs) != sorted(set(cutoffs)):
        raise ValueError("cutoffs must be distinct and increasing")
    dated = [c for c in cases if c.year is not None]
    out: list[tuple[Case, Any]] = []
    for i, cutoff in enumerate(cutoffs):
        end = cutoffs[i + 1] if i + 1 < len(cutoffs) else math.inf
        train = [c for c in dated if c.year < cutoff]
        test = [c for c in dated if cutoff <= c.year < end]
        if not train or not test:
            continue
        model = fit(train)
        out.extend((c, predict(model, c)) for c in test)
    return out


def auc(scores: Sequence[float], positives: Sequence[bool]) -> Optional[float]:
    """The chance a random positive scores above a random negative (ties
    count half): the Mann-Whitney form of the ROC area. None without both."""
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def mean_abs(values: Sequence[float]) -> Optional[float]:
    """The mean of the absolute values; None for no values."""
    return sum(abs(v) for v in values) / len(values) if values else None


def share(flags: Sequence[bool]) -> Optional[float]:
    """The share of true flags; None for no flags."""
    return sum(1 for f in flags if f) / len(flags) if flags else None


def _round(x: Optional[float]) -> Optional[float]:
    return None if x is None or not math.isfinite(x) else round(float(x), PLACES)


def better(metric: Metric, a: float, b: float) -> bool:
    """Whether ``a`` is strictly better than ``b`` on ``metric``."""
    return a > b if metric.better == "higher" else a < b


def summarize(cases: Sequence[Case], model: Sequence[Any], baseline: Sequence[Any],
              metrics: Sequence[Metric], headline: str, min_cases: int = MIN_CASES) -> dict:
    """Each statistic for the model and the baseline on the same cases, and
    the verdict on the headline statistic."""
    n = len(cases)
    if n < min_cases:
        return {"cases": n, "verdict": NOT_ENOUGH}
    m = {k.id: _round(k.compute(cases, model)) for k in metrics}
    b = {k.id: _round(k.compute(cases, baseline)) for k in metrics}
    head = next(k for k in metrics if k.id == headline)
    if m[headline] is None or b[headline] is None:
        verdict = NOT_ENOUGH
    else:
        verdict = BEATS if better(head, m[headline], b[headline]) else DOES_NOT_BEAT
    return {"cases": n, "model": m, "baseline": b, "verdict": verdict}


def evaluate_set(rows: Sequence[tuple[Case, Any, Any]], metrics: Sequence[Metric], headline: str,
                 regions: Sequence[str] = REGIONS, min_cases: int = MIN_CASES) -> dict:
    """``summarize`` over every case and over each region's cases.

    ``rows`` are ``(case, model prediction, baseline prediction)``. Every
    region is listed, an empty one as ``not_enough_data``.
    """
    def group(subset):
        return summarize([r[0] for r in subset], [r[1] for r in subset], [r[2] for r in subset],
                         metrics, headline, min_cases)
    return {"overall": group(rows),
            "by_region": {region: group([r for r in rows if r[0].region == region]) for region in regions}}


def metric_specs(metrics: Sequence[Metric]) -> list[dict]:
    """What a card records about each statistic."""
    return [{"id": k.id, "better": k.better, "tolerance": k.tolerance, "description": k.description}
            for k in metrics]
