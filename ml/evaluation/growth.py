"""The growth calibrator's evaluation (ml/growth_calibrator.py; PLAN.md 5.1, 5.5).

Question: does the growth range the app would set contain how fast a
company's revenue really grew over a hold, as often as it says, and is it
a better range than the economy-wide one in force (PLAN.md 4.4)?

- **Cases**: every company-span the calibrator rests on
  (``growth_calibrator.spans``: SEC filers with revenue of at least
  ``FLOOR_USD_M`` US dollars in the start year, not financial, over holds of
  3 to 7 years), placed in the S&P region of the company's country.
- **Out of time, strictly**: a span is predicted as of its start year
  ``s``, from the spans that had **ended** by then (``end <= s``), one start
  year after another from ``FIRST_START``. The economy's growth the range
  is centred on stands in for the IMF projection the app uses (no past
  projections are free): the country's nominal growth over the
  ``FORECAST_YEARS`` years to ``s``, compounded. A span whose group has
  too few earlier companies has no range and is left out, as the app
  shows none.
- **Baseline**: the range in force since PLAN.md 4.4 -- the same centre,
  and the spread of the country's nominal GDP growth from year to year
  since 2000 (its region's median spread when the country has too short a
  history), as the normal draw's 80%.
- **Headline**: the interval score of each range's central 80% in
  percentage points of yearly growth (Gneiting and Raftery 2007): its
  width plus ten times how far the growth fell outside it. Lower is better.
  Both ranges are the normal draws the simulation makes from the Settings.
"""
from __future__ import annotations

import statistics
from typing import Mapping, Optional, Sequence

from benchmarks.catalogue import region_of
from economy.catalogue import COUNTRIES
from ml.evaluation.harness import Case, Metric, evaluate_set, mean_abs, metric_specs, share
from ml.growth_calibrator import (
    FLOOR_USD_M, GATE_SET, HORIZONS, MIN_COMPANIES, QUANTILES, Z80, Span, band, compound, economy_growth, fit, load,
    lookup, spans,
)

MODEL_ID = "growth"
# The first start year predicted: earlier ones have almost no ended spans to learn from
FIRST_START = 2012
# Years of the economy's growth the centre is read from (the app reads the IMF's projection)
FORECAST_YEARS = 5
# The baseline's spread needs this many years of the economy's growth (benchmarks/risk.py MIN_YEARS)
MIN_YEARS = 6
WINDOW_START = 2000
ALPHA = 1 - (QUANTILES[-1] - QUANTILES[0])


def _stdev_to(series: Mapping[int, float], year: int) -> Optional[float]:
    window = [v for y, v in series.items() if WINDOW_START <= y <= year]
    return statistics.stdev(window) if len(window) >= MIN_YEARS else None


def baseline_spread(nominal: Mapping[str, Mapping[int, float]], country: str, year: int) -> Optional[float]:
    """The economy-wide spread in force (benchmarks/risk.py ``mc_growth_std``),
    as it read in ``year``: the country's own, else its region's (then the
    world's) median of its economies' spreads; in %."""
    own = _stdev_to(nominal.get(country, {}), year)
    if own is not None:
        return own
    region = region_of(country)
    for members in ([c for c in COUNTRIES if region_of(c) == region], list(COUNTRIES)):
        found = [s for c in members if (s := _stdev_to(nominal.get(c, {}), year)) is not None]
        if found:
            return statistics.median(found)
    return None


def scored(data: Optional[Mapping] = None) -> list[tuple[Case, dict, dict]]:
    """``(case, range, baseline range)`` for every span the app would range."""
    data = data if data is not None else load()
    nominal = data["nominal_growth"]
    all_spans = spans(data)
    economies: dict[str, dict] = {}
    rows = []
    for start in range(FIRST_START, max(s.start for s in all_spans) + 1):
        test = [s for s in all_spans if s.start == start]
        if not test:
            continue
        model = fit(s for s in all_spans if s.end <= start)
        for s in test:
            group, _ = lookup(model, s.region, s.sector, s.horizon)
            if group is None:
                continue
            if s.country not in economies:
                economies[s.country] = economy_growth(nominal, s.country)
            centre = compound(economies[s.country], start - FORECAST_YEARS + 1, start)
            spread = baseline_spread(nominal, s.country, start)
            if centre is None or spread is None:
                continue
            b = band(centre, group)
            base_mean, base_std = round(centre * 100, 2), round(spread, 2)
            rows.append((_case(s), {"low": b["draw_low"], "high": b["draw_high"], "centre": b["mean"] / 100},
                         {"low": (base_mean - Z80 * base_std) / 100, "high": (base_mean + Z80 * base_std) / 100,
                          "centre": base_mean / 100}))
    return rows


def _case(s: Span) -> Case:
    return Case(id=f"{s.company}.{s.start}.{s.horizon}", region=s.region, year=s.start,
                features={"sector": s.sector, "horizon": s.horizon, "country": s.country}, truth=s.growth)


def interval_score(cases: Sequence[Case], preds: Sequence[dict]) -> Optional[float]:
    if not cases:
        return None
    total = 0.0
    for c, p in zip(cases, preds):
        y, lo, hi = c.truth, p["low"], p["high"]
        total += (hi - lo) + (2 / ALPHA) * max(lo - y, 0.0) + (2 / ALPHA) * max(y - hi, 0.0)
    return total / len(cases) * 100


def inside(cases: Sequence[Case], preds: Sequence[dict]) -> Optional[float]:
    return share([p["low"] <= c.truth <= p["high"] for c, p in zip(cases, preds)])


def centre_error(cases: Sequence[Case], preds: Sequence[dict]) -> Optional[float]:
    found = mean_abs([p["centre"] - c.truth for c, p in zip(cases, preds)])
    return None if found is None else found * 100


def width(cases: Sequence[Case], preds: Sequence[dict]) -> Optional[float]:
    return sum(p["high"] - p["low"] for p in preds) / len(preds) * 100 if preds else None


METRICS = (
    Metric("interval_score", "lower", interval_score, 0.1,
           "The central 80% range's width plus ten times how far the yearly growth fell outside it, in "
           "percentage points (Gneiting and Raftery's interval score)"),
    Metric("inside", "higher", inside, 0.01,
           "Share of companies whose yearly growth fell inside the central 80% range (about 0.8 when the "
           "range is right)"),
    Metric("centre_error", "lower", centre_error, 0.1,
           "Mean distance between the range's middle (the simulation's mean) and the growth, in points"),
    Metric("width", "lower", width, 0.1, "Mean width of the central 80% range, in percentage points"),
)
HEADLINE = "interval_score"


def evaluate(base: Optional[str] = None) -> dict:   # noqa: ARG001 -- nothing trained to read
    data = load()
    rows = scored(data)
    starts = sorted({r[0].year for r in rows})
    companies = len({r[0].id.split(".")[0] for r in rows})
    return {
        "headline_set": GATE_SET,
        "headline_metric": HEADLINE,
        "metrics": metric_specs(METRICS),
        "split": {"kind": "walk_forward", "dated_by": "the start year of the growth predicted; fitted on spans "
                  "that had ended by then", "cutoffs": starts},
        "baseline": {"id": "economy", "description":
                     "The range in force since PLAN.md 4.4: the country's nominal GDP growth, with the spread of "
                     "that growth from year to year since 2000 (IMF)"},
        "data": {"written_on": data["written_on"], "years": data["years"], "floor_usd_m": FLOOR_USD_M,
                 "min_companies": MIN_COMPANIES, "horizons": list(HORIZONS)},
        "sets": {
            GATE_SET: {"description": f"Each company's yearly revenue growth over holds of three to seven years "
                       f"({len(rows)} spans of {companies} companies, starting {starts[0]}-{starts[-1]})",
                       **evaluate_set(rows, METRICS, HEADLINE)},
        },
    }


CARD = {
    "id": MODEL_ID,
    "name": "Growth calibrator (revenue growth by sector and region)",
    "module": "ml/growth_calibrator.py",
    "used_by": "Monte Carlo -> Scenarios and the rail's Growth group, \"Calibrate from sector and region\" "
               "(POST /api/ml/growth); shown only in regions where this card says it beats the baseline",
    "estimates": "The central 80% range of a company's yearly revenue growth over the deal's hold, as the "
                 "Monte Carlo growth mean and spread (mc_growth_mean, mc_growth_std)",
    "training": {
        "data": "Nothing is trained: the ranges are percentiles of SEC filers' revenue growth over their "
                "country's nominal GDP growth, by S&P region, GICS sector and hold "
                "(ml/evaluation/data/firm_growth.json, python -m tests.ml_firm_growth; ml/growth_ranges.json)",
        "command": None,
        "deterministic": True,
    },
    "limitations": [
        "The companies are SEC filers: almost every U.S. listed company, but outside the U.S. only those "
        "listed or registered in the U.S. (in 2019, about 450 a year), so other regions rest on fewer "
        "companies and may differ from their home markets.",
        "Only companies that kept filing are counted: one that failed, was taken private or was bought before "
        "a span ended has no growth for it, so the low end may be too high.",
        "Revenue includes acquisitions; a buy-and-build plan's growth is in the range, an organic plan's "
        "range is a little wide.",
        f"Companies with under {FLOOR_USD_M:g} million US dollars of revenue in the start year are left out; "
        "sizes above that are not split.",
        "Out of time, the centre stands in for the IMF's projection with the economy's growth over the five "
        "years before; the app uses the projection itself, which this card can't replay.",
        "Spans overlap (one company's three- to seven-year spans share years), so a region counts fewer "
        "independent observations than cases.",
        "A company is placed by its business address and its growth is measured in its reporting currency.",
    ],
}
