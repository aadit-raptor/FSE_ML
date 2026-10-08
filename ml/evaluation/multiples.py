"""The multiple predictor's evaluation (ml/multiple_predictor.py; PLAN.md 5.1, 5.4).

Question: does the range the app suggests for an industry's EV/EBITDA
contain what the multiple turned out to be, and is it a better range than
the region's whole market gives?

- **Cases**: every industry with ``MIN_FIRMS`` companies in every Damodaran
  group but the global one (the app never uses it), each year its multiple
  is known, from the archive and the current edition
  (``data/multiple_history.json``, written by ``python -m
  tests.ml_multiple_history``, every industry). A case is the multiple in a
  year, predicted from the multiple ``h`` years earlier: ``h`` = 1 for the
  ``entry`` set (a deal closing the year after the latest edition) and
  ``h`` = 4 to 8 for the ``exit`` set (holds of 3 to 7 years after it).
  Each group is placed in its S&P region (``GROUP_REGION``).
- **Walk-forward, yearly**: for each year from ``FIRST_CUTOFF`` the ranges
  are built only from moves that ended before it, and tested on that
  year's multiples, as the app rebuilds them from whatever is stored. A
  horizon with too few earlier moves gives no range and the case is left
  out (``MIN_PAIRS``), as the app shows none.
- **Baseline**: the region's whole market in the base year -- the same
  percentiles across every industry of the group, whatever the industry.
- **Headline**: the interval score of the central 80% range (Gneiting and
  Raftery 2007): its width, plus ten times how far the multiple fell
  outside it. Lower is better; it rewards a range that is narrow and
  still contains the answer, so neither a wide range nor a narrow one wins
  by itself.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Optional, Sequence

from benchmarks.damodaran import Table
from ml.evaluation.harness import Case, Metric, evaluate_set, mean_abs, metric_specs, share, walk_forward
from ml.multiple_predictor import MIN_PAIRS, QUANTILES, band, fit as fit_line, gap, group_series, market_band

MODEL_ID = "multiples"
HISTORY = Path(__file__).parent / "data" / "multiple_history.json"
HEADLINE_SET = "entry"
# Damodaran's groups in S&P's regions; the global group has none
GROUP_REGION = {"us": "us", "europe": "europe", "japan": "other_developed", "aus_nz_canada": "other_developed",
                "china": "emerging", "india": "emerging", "emerging": "emerging"}
ENTRY_HORIZONS = (1,)
# A deal's hold of 3 to 7 years, after the year to its close
EXIT_HORIZONS = (4, 5, 6, 7, 8)
# The first year tested: the archive starts with 2011, so earlier cutoffs
# would rest on two or three years of moves
FIRST_CUTOFF = 2014
ALPHA = 1 - (QUANTILES[-1] - QUANTILES[0])


def tables() -> dict[str, Table]:
    data = json.loads(HISTORY.read_text(encoding="utf-8"))["tables"]
    return {name: Table(name, date.fromisoformat(t["published"]) if t["published"] else None, t["url"], t["rows"])
            for name, t in data.items()}


def cases(by_group: dict[str, dict], horizons: Sequence[int]) -> list[Case]:
    """Every industry-year with a multiple ``h`` years before it, per horizon."""
    out = []
    for group, by_industry in by_group.items():
        for industry, s in by_industry.items():
            for y, v in s.items():
                for h in horizons:
                    if (y - h) in s:
                        out.append(Case(id=f"{group}.{industry}.{y}.{h}", region=GROUP_REGION[group], year=y,
                                        features={"group": group, "industry": industry, "horizon": h,
                                                  "base_year": y - h, "base": s[y - h],
                                                  "gap": gap(by_industry, s[y - h], y - h)},
                                        truth=v))
    return out


def fit(train: Sequence[Case]) -> dict:
    """The line through the moves by group and horizon, from cases that
    ended before the cutoff (exactly what the app fits on what is stored)."""
    points: dict[tuple, list] = {}
    for c in train:
        f = c.features
        points.setdefault((f["group"], f["horizon"]), []).append((f["gap"], math.log(c.truth / f["base"])))
    return {k: fit_line(v) for k, v in points.items()}


def predict(model: dict, case: Case) -> Optional[dict]:
    f = case.features
    return band(f["base"], f["gap"], model.get((f["group"], f["horizon"])))


def interval_score(cases_: Sequence[Case], preds: Sequence[dict]) -> Optional[float]:
    if not cases_:
        return None
    total = 0.0
    for c, p in zip(cases_, preds):
        y, lo, hi = c.truth, p["low"], p["high"]
        total += (hi - lo) + (2 / ALPHA) * max(lo - y, 0.0) + (2 / ALPHA) * max(y - hi, 0.0)
    return total / len(cases_)


def inside(cases_: Sequence[Case], preds: Sequence[dict]) -> Optional[float]:
    return share([p["low"] <= c.truth <= p["high"] for c, p in zip(cases_, preds)])


def median_error(cases_: Sequence[Case], preds: Sequence[dict]) -> Optional[float]:
    return mean_abs([p["median"] - c.truth for c, p in zip(cases_, preds)])


def width(cases_: Sequence[Case], preds: Sequence[dict]) -> Optional[float]:
    return sum(p["high"] - p["low"] for p in preds) / len(preds) if preds else None


METRICS = (
    Metric("interval_score", "lower", interval_score, 0.05,
           "The central 80% range's width plus ten times how far the multiple fell outside it, in turns "
           "of EBITDA (Gneiting and Raftery's interval score)"),
    Metric("inside", "higher", inside, 0.01,
           "Share of multiples inside the central 80% range (about 0.8 when the range is right)"),
    Metric("median_error", "lower", median_error, 0.05,
           "Mean distance between the suggestion (the range's middle) and the multiple, in turns"),
    Metric("width", "lower", width, 0.05, "Mean width of the range, in turns"),
)
HEADLINE = "interval_score"


def scored(horizons: Sequence[int], t=None) -> list[tuple[Case, dict, dict]]:
    """``(case, range, baseline range)`` for every case the predictor answers."""
    t = t if t is not None else tables()
    by_group = {g: group_series(t, g) for g in GROUP_REGION}
    all_cases = cases(by_group, horizons)
    last = max(c.year for c in all_cases)
    found = walk_forward(all_cases, list(range(FIRST_CUTOFF, last + 1)), fit, predict)
    rows = []
    for c, p in found:
        if p is None:
            continue
        base = market_band(by_group[c.features["group"]], c.features["base_year"])
        if base is not None:
            rows.append((c, p, base))
    return rows


def _set(horizons: Sequence[int], description: str, t) -> dict:
    rows = scored(horizons, t)
    years = sorted({r[0].year for r in rows})
    return {"description": f"{description} ({len(rows)} cases, {years[0]}-{years[-1]})",
            **evaluate_set(rows, METRICS, HEADLINE)}


def evaluate(base: Optional[str] = None) -> dict:   # noqa: ARG001 -- nothing trained to read
    t = tables()
    return {
        "headline_set": HEADLINE_SET,
        "headline_metric": HEADLINE,
        "metrics": metric_specs(METRICS),
        "split": {"kind": "walk_forward", "dated_by": "the year of the multiple predicted",
                  "cutoffs": list(range(FIRST_CUTOFF, max(c.year for c in cases(
                      {g: group_series(t, g) for g in GROUP_REGION}, ENTRY_HORIZONS)) + 1))},
        "baseline": {"id": "market", "description":
                     "The region's whole market: the same percentiles (10th, 50th, 90th) across every "
                     "industry of the group in the base year, whatever the deal's industry"},
        "sets": {
            "entry": _set(ENTRY_HORIZONS, "Each industry's multiple a year after the one it is predicted from "
                          "(the entry range)", t),
            "exit": _set(EXIT_HORIZONS, "Each industry's multiple four to eight years after the one it is "
                         "predicted from (the exit range, holds of three to seven years)", t),
        },
    }


CARD = {
    "id": MODEL_ID,
    "name": "Multiple predictor (entry and exit EV/EBITDA by region)",
    "module": "ml/multiple_predictor.py",
    "used_by": "Deal -> Returns, Multiples (POST /api/ml/multiples); each range shown only in regions where "
               "this card says it beats the baseline",
    "estimates": "The central 80% range and middle of the deal industry's EV/EBITDA in its region a year after "
                 "the latest published figure (entry) and the deal's hold after that (exit)",
    "training": {
        "data": "Nothing is trained: the ranges read Damodaran's industry averages for the deal's region and "
                "his archive of past editions (PLAN.md 4.3, 4.4), refreshed nightly",
        "command": None,
        "deterministic": True,
    },
    "limitations": [
        "Cases are industries' aggregate multiples, not single companies or buyouts: a buyout's price also "
        "reflects its size, growth and control, which the averages don't split out.",
        "The archive starts with 2011 and some groups miss a few editions (Australia, New Zealand and "
        "Canada miss 2011, 2013 and 2014; China 2014 and 2023), so the longest exit horizons are tested on "
        "the newest years only.",
        f"A horizon needs {MIN_PAIRS} earlier industry moves in the group; the early folds of the longest "
        "horizons had fewer and are left out.",
        "Cases from the same year share that year's market: the regions' statistics move together and "
        "count fewer independent years than cases.",
        "Out of time the ranges held about three multiples in four, where four in five is the aim: "
        "industries' multiples moved more after each cutoff than before it, so read the range as a little "
        "narrow.",
        "The averages are listed companies' and are not split by size.",
    ],
}
