"""What the stored series say today (PLAN.md 4.2).

- ``country_figures``: an economy's current figures, each with its period,
  source and how it was arrived at (``basis``);
- ``reference_rates``: each floating-rate benchmark's current level, the
  default a new floating facility starts from;
- ``rebase``: the ECB's euro rates turned into rates against any currency.

A figure counts as **current** only while it is recent enough for its kind
(``MAX_AGE_DAYS``); an old one is still answered, marked not current, and a
reference rate skips to its next candidate rather than default to it.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Optional

from economy.catalogue import (
    BOND_YIELD_SOURCES, COUNTRIES, EURO_AREA, EURO_MEMBERS, REFERENCE_SOURCES, SPREAD_BENCHMARK,
    series_key,
)
from economy.model import FxDay, Series

# How old a figure may be and still be current, from the end of its period.
# Policy rates move in steps and the BIS publishes some weeks late, so they
# get longer; a daily market rate a month and a half old is not today's.
MAX_AGE_DAYS = {"D": 45, "M": 100}
POLICY_RATE_MAX_AGE_DAYS = 120
# The World Bank's latest actual year lags: two years back still counts
ACTUAL_YEARS_BACK = 2
# What makes an economy "current": all three of these
CORE_FIGURES = ("gdp_growth", "inflation", "policy_rate")


@dataclass(frozen=True)
class Figure:
    value: float
    period: str
    source: str
    source_series: str
    url: str
    basis: str          # projection, actual, observed, euro_area, computed, benchmark, policy_rate, interbank_3m
    current: bool


def period_end(period: str) -> Optional[date]:
    """The last day a ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD`` period covers."""
    try:
        if len(period) == 4:
            return date(int(period), 12, 31)
        if len(period) == 7:
            y, m = int(period[:4]), int(period[5:])
            return date(y, m, calendar.monthrange(y, m)[1])
        return date.fromisoformat(period)
    except ValueError:
        return None


def is_current(series: Series, period: str, today: date) -> bool:
    end = period_end(period)
    if end is None:
        return False
    if series.frequency == "A":
        return int(period) >= today.year - ACTUAL_YEARS_BACK
    limit = POLICY_RATE_MAX_AGE_DAYS if series.indicator == "policy_rate" else MAX_AGE_DAYS[series.frequency]
    return (today - end).days <= limit


def _figure(series: Series, period: str, value: float, basis: str, today: date) -> Figure:
    return Figure(value, period, series.source, series.source_series, series.url, basis,
                  is_current(series, period, today))


def _latest(series: Optional[Series], basis: str, today: date) -> Optional[Figure]:
    if series is None or series.latest is None:
        return None
    period, value = series.latest
    return _figure(series, period, value, basis, today)


def _projection(series: Optional[Series], today: date) -> Optional[Figure]:
    """The WEO's figure for the current year (a projection, until the IMF
    publishes the year's outturn)."""
    if series is None:
        return None
    this_year = dict(series.observations).get(str(today.year))
    if this_year is None:
        return None
    return _figure(series, str(today.year), this_year, "projection", today)


def _spread(store: Mapping[str, Series], area: str, today: date) -> Optional[Figure]:
    """A euro member's 10-year yield over Germany's, in the latest month
    both have."""
    own = store.get(series_key("bond_yield_10y", area, "oecd"))
    base = store.get(series_key("bond_yield_10y", SPREAD_BENCHMARK, "oecd"))
    if area not in EURO_MEMBERS or area == SPREAD_BENCHMARK or own is None or base is None:
        return None
    base_values = dict(base.observations)
    common = [(p, v) for p, v in own.observations if p in base_values]
    if not common:
        return None
    period, value = common[-1]
    spread = Series("sovereign_spread", area, "oecd",
                    f"{own.source_series} - {base.source_series}", "M", own.url)
    return _figure(spread, period, round(value - base_values[period], 6), "computed", today)


def country_figures(store: Mapping[str, Series], area: str, today: date) -> dict[str, Figure]:
    """Every figure known for ``area``, by name; missing ones are left out."""
    get = lambda indicator, source: store.get(series_key(indicator, area, source))  # noqa: E731
    found = {
        "gdp_growth": _projection(get("gdp_growth", "imf"), today),
        "gdp_growth_actual": _latest(get("gdp_growth", "worldbank"), "actual", today),
        "inflation": _projection(get("inflation", "imf"), today),
        "inflation_actual": _latest(get("inflation", "worldbank"), "actual", today),
        "short_rate_3m": _latest(get("short_rate_3m", "oecd"), "observed", today),
        "sovereign_spread": _spread(store, area, today),
    }
    if area in EURO_MEMBERS:
        found["policy_rate"] = _latest(store.get(series_key("policy_rate", EURO_AREA, "bis")), "euro_area", today)
    else:
        found["policy_rate"] = _latest(get("policy_rate", "bis"), "observed", today)
    yields = [f for s in BOND_YIELD_SOURCES if (f := _latest(get("bond_yield_10y", s), "observed", today))]
    if yields:
        found["bond_yield_10y"] = next((f for f in yields if f.current), yields[0])
    return {k: v for k, v in found.items() if v is not None}


def economy_is_current(figures: Mapping[str, Figure]) -> bool:
    return all(name in figures and figures[name].current for name in CORE_FIGURES)


def economies(store: Mapping[str, Series], today: date) -> dict[str, dict[str, Figure]]:
    """Every area's figures, countries first, then the euro area."""
    return {area: country_figures(store, area, today) for area in (*COUNTRIES, EURO_AREA)}


def current_economies(store: Mapping[str, Series], today: date) -> list[str]:
    return [c for c in COUNTRIES if economy_is_current(country_figures(store, c, today))]


def reference_rates(store: Mapping[str, Series], today: date) -> dict[str, Figure]:
    """Each benchmark's current level from its first current candidate;
    a benchmark with none is left out (the user types the rate)."""
    out = {}
    for code, candidates in REFERENCE_SOURCES.items():
        for candidate in candidates:
            fig = _latest(store.get(candidate.key), candidate.basis, today)
            if fig is not None and fig.current:
                out[code] = fig
                break
    return out


def rebase(day: FxDay, base: str) -> Optional[dict[str, float]]:
    """Units of each currency per one ``base``; None if the ECB doesn't
    quote ``base``."""
    per_euro = day.rates.get(base)
    if not per_euro:
        return None
    return {currency: rate / per_euro for currency, rate in sorted(day.rates.items())}
