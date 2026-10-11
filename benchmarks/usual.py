"""A user's usual figures, beside the sourced starting figures (PLAN.md 5.9).

The sourced figures (``benchmarks/starting.py``) say what listed companies
in a deal's region and industry look like. This says what **the person
themselves** usually assumes there: for each starting figure, the median of
it across their own saved deals in the same group, with how many deals that
is and the lowest and highest they have used.

A group is S&P's region of the deal's country (as the validation report and
the library split) and the GICS sector of its industry
(``validation.tags.INDUSTRY_SECTOR``); an industry with no sector (the whole
market, diversified) is a group of its own. A deal belongs to a group only
once it has a country, which it gets from "Use sourced figures". Archived
deals count: they are still what the person assumed.

Nothing is trained and nothing is stored: the answer is worked out from the
deals as they are when asked (``db.deals.own_deal_inputs``, which matches
the caller in the same statement as the read), so a deleted deal stops
counting at once and there is no second copy of anyone's figures. It is a
description of the person's own deals, not a prediction, so there is no
model card: the screen says exactly what it is and from how many deals.
"""
from __future__ import annotations

import math
import statistics
from typing import Iterable, Mapping, Optional

from benchmarks.catalogue import ALL_INDUSTRIES_ID
from library.base_rates import sp_region
from validation.tags import INDUSTRY_SECTOR

# The figures suggested: the ones the sourced starting figures fill, so the
# two sit side by side. All are rates, multiples or days, never money, so
# deals of any size and currency compare.
FIELDS = ("growth", "tax", "gross_margin", "opex", "da", "entry_mult", "exit_mult", "capex",
          "ar_days", "inv_days", "ap_days", "nwc", "debt_pct", "senior_pct", "base_rate")
# What a deal that lists its facilities ignores (core/debt.py): its values
# for these are leftovers, not assumptions
PERCENTAGE_DEBT_FIELDS = ("debt_pct", "senior_pct", "base_rate")
DAYS_FIELDS = ("ar_days", "inv_days", "ap_days")
# Deals a figure needs before it is suggested: a median of two is one of them
# or halfway between
MIN_DEALS = 3
# The newest deals read at most (db.deals.own_deal_inputs)
MAX_DEALS = 200


def group_of(country: Optional[str], industry: Optional[str]) -> Optional[tuple[str, str]]:
    """The region and sector a deal counts in; None without a country. The
    sector is the industry's own id when it has no GICS sector."""
    region = sp_region(country)
    if region is None:
        return None
    industry = industry or ALL_INDUSTRIES_ID
    return region, INDUSTRY_SECTOR.get(industry, industry)


def _number(value) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _values(deals: list[Mapping], field: str) -> list[float]:
    listed = field in PERCENTAGE_DEBT_FIELDS
    found = (_number(d.get(field)) for d in deals if not (listed and d.get("tranches")))
    return [v for v in found if v is not None]


def _figure(field: str, values: list[float]) -> dict:
    places = 1 if field in DAYS_FIELDS else 2
    return {"field": field, "value": round(statistics.median(values), places), "deals": len(values),
            "low": round(min(values), places), "high": round(max(values), places)}


def usual_figures(deals: Iterable[Mapping], country: Optional[str], industry: Optional[str]) -> dict:
    """The usual figures for a deal in ``country`` and ``industry``, from
    ``deals`` (one person's saved deal inputs). ``status`` is ``no_country``,
    ``not_enough`` (no figure has ``MIN_DEALS`` deals behind it) or ``ok``."""
    group = group_of(country, industry)
    base = {"region": None, "sector": None, "industry": None, "deals": 0, "min_deals": MIN_DEALS, "figures": []}
    if group is None:
        return {**base, "status": "no_country"}
    mine = [d for d in deals if group_of(d.get("country"), d.get("industry")) == group]
    figures = [_figure(f, v) for f in FIELDS if len(v := _values(mine, f)) >= MIN_DEALS]
    region, sector = group
    known = sector in INDUSTRY_SECTOR.values()
    return {**base, "status": "ok" if figures else "not_enough", "region": region,
            "sector": sector if known else None, "industry": None if known else sector,
            "deals": len(mine), "figures": figures}
