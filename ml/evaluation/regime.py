"""The economic regime's evaluation (ml/macro_regime.py; PLAN.md 5.1, 5.7).

Question: when the app names a region's regime for a year from the IMF's
projection for it, is that the regime the year's final figures give?

- **Cases**: one per region and year from ``FIRST_TEST``, for each of the
  IMF's two reports a year (the World Economic Outlook historical forecasts
  database, ml/evaluation/data/regime_history.json, written by ``python -m
  tests.ml_regime_history``). The **spring** set is the April report's
  projection for its own year, the earliest figure the app ever calls a
  regime from; the **autumn** set is the October report's, which the app
  uses from October and, for the year just ended, until the next April.
- **Out of time**: a year's cut-offs come from the region's years before it
  (``macro_regime.call``), and its figure is the one the IMF published
  that spring or autumn, not what is known today.
- **Truth**: the same rule on the year's figures as the IMF gives them
  today, on the same cut-offs, so the only difference between the call and
  the truth is the projection.
- **Baseline**: "the same as last year" -- last year's regime from the same
  report's estimate of last year, on last year's own cut-offs. What a
  person would say without a projection.
- **Headline**: the share of years named right.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Sequence

from ml import macro_regime as mr
from ml.evaluation.harness import Case, Metric, evaluate_set, metric_specs, share

MODEL_ID = "regime"
DATA = Path(__file__).resolve().parent / "data" / "regime_history.json"
# The first year tested: the IMF's euro area figures start in 1992, and
# every region has its cut-offs' history well before this
FIRST_TEST = 2000
# The card's sets and the letter the IMF's file gives each report
SETS = {"spring": "S", "autumn": "F"}


def load(path: Path = DATA) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def series(data: Mapping, region: str) -> dict[int, tuple]:
    return {int(y): tuple(v) for y, v in data["actual"][region].items()}


def scored(data: Mapping, season: str) -> list[tuple[Case, str, str]]:
    """``(case, regime called, last year's regime)`` for every region and
    year the report, the final figures and the cut-offs all exist for."""
    rows = []
    for region in mr.REGIONS:
        actual = series(data, region)
        reports = data["reports"].get(region, {})
        for year in range(FIRST_TEST, max(actual) + 1):
            report = reports.get(f"{season}{year}") or {}
            now, before = report.get(str(year)), report.get(str(year - 1))
            truth = mr.call(actual, year)
            if truth is None or now is None or before is None:
                continue
            called = mr.call(actual, year, tuple(now))
            last = mr.call(actual, year - 1, tuple(before))
            if called is None or last is None:
                continue
            rows.append((Case(id=f"{region}.{year}", region=region, year=year,
                              features={"growth": now[0], "inflation": now[1]}, truth=truth["regime"]),
                         called["regime"], last["regime"]))
    return rows


def accuracy(cases: Sequence[Case], preds: Sequence[str]) -> Optional[float]:
    return share([c.truth == p for c, p in zip(cases, preds)])


def turn_accuracy(cases: Sequence[Case], preds: Sequence[str]) -> Optional[float]:
    """Accuracy on the years that were not ``base``: the ones a regime is for."""
    return share([c.truth == p for c, p in zip(cases, preds) if c.truth != "base"])


METRICS = (
    Metric("accuracy", "higher", accuracy, 0.02,
           "Share of years whose regime was named as the year's final figures give it"),
    Metric("turn_accuracy", "higher", turn_accuracy, 0.05,
           "The same share over the years that turned out a recession, stagflation or bull year"),
)
HEADLINE = "accuracy"


def evaluate(base: Optional[str] = None) -> dict:   # noqa: ARG001 -- nothing trained to read
    data = load()
    rows = {name: scored(data, season) for name, season in SETS.items()}
    years = sorted({r[0].year for r in rows[mr.GATE_SET]})
    described = {
        "spring": "The April report's projection for its own year",
        "autumn": "The October report's projection for its own year",
    }
    return {
        "headline_set": mr.GATE_SET,
        "headline_metric": HEADLINE,
        "metrics": metric_specs(METRICS),
        "regions_note": "the seven the regime is given for: United States, euro area, United Kingdom, Japan, "
                        "China, India and the world",
        "split": {"kind": "walk_forward", "dated_by": "the year named; its cut-offs come from the region's "
                  "earlier years and its figure from the IMF's report of that year", "cutoffs": years},
        "baseline": {"id": "last_year", "description":
                     "The same regime as last year, from the same report's estimate of last year"},
        "data": {"written_on": data["written_on"], "first_year": mr.FIRST_YEAR, "trail": mr.TRAIL,
                 "share": mr.SHARE, "min_years": mr.MIN_YEARS},
        "sets": {name: {"description": f"{described[name]} ({len(found)} region-years, "
                        f"{found[0][0].year if found else '-'}-{max((r[0].year for r in found), default='-')})",
                        **evaluate_set(found, METRICS, HEADLINE, regions=tuple(mr.REGIONS))}
                 for name, found in rows.items()},
    }


CARD = {
    "id": MODEL_ID,
    "name": "Economic regime by region",
    "module": "ml/macro_regime.py",
    "used_by": "Monte Carlo -> Scenarios, \"Economic regime\" (POST /api/ml/macro-regime); a region's regime is "
               "shown only where this card says it beats the baseline",
    "estimates": "Which scenario preset a region's year looks like -- recession, stagflation, bull or base -- "
                 "from its real GDP growth and inflation against the median of its own previous five years",
    "training": {
        "data": "Nothing is trained: each region's cut-offs are the weakest and strongest fifth of its own "
                "earlier years' growth gaps and the highest fifth of its inflation gaps (IMF World Economic "
                "Outlook since 1980), refitted on the first scheduled run of each month (GitHub Actions' "
                "nightly economy-refresh task); this card's reports are in "
                "ml/evaluation/data/regime_history.json (python -m tests.ml_regime_history)",
        "command": None,
        "deterministic": True,
    },
    "limitations": [
        "The IMF's historical forecasts file ends with its 2022 reports, so later years are not tested "
        "until the IMF extends it.",
        "Earlier years' figures are today's, not those known at the time: revisions since then reach a tested "
        "year's cut-offs and its previous-five-year median.",
        "One regime a year from two figures: no quarterly turning points, no markets, no credit conditions. A "
        "downturn that starts late in a year shows the year after.",
        "A fifth of a region's years is a recession by construction: the word means weak against its own "
        "record, not two quarters of falling output.",
        "India's figures are for fiscal years starting in April, which the IMF files under the calendar year "
        "they start in.",
        "The euro area and the world are the IMF's own aggregates, as composed at each report; the euro "
        "area's start in 1992, so its first tested year is later than the others'.",
        "A deal outside the six economies and the euro area is shown the world's regime, which may differ "
        "from its own country's.",
    ],
}
