"""The multiple predictor: entry and exit EV/EBITDA ranges by region (PLAN.md 5.4).

A deal's multiples are suggested from **its industry's listed companies in
its own region**, the same Damodaran averages the starting figures use
(PLAN.md 4.3, ``benchmarks/``), and from how those averages have moved,
year by year, in Damodaran's archive (PLAN.md 4.4, ``benchmarks/history.py``).
Nothing is fitted to a handful of deals; nothing is stored but the
published tables, so the ranges follow every refresh.

- **The peer group** is the closest group in the country's chain
  (``benchmarks.catalogue.chain``) whose industry has at least
  ``MIN_FIRMS`` companies in its latest year: the country's own file (the
  US, Japan, China, India), else its region. **Never the global group**, as
  for the deal risk score (5.2): a region with too few companies in the
  industry says "not enough data".
- **The range** for the multiple ``h`` years after the latest published
  year comes from the group's own record of ``h``-year moves, pooled over
  every industry with ``MIN_FIRMS`` companies in both years: the move
  ln(multiple in year y / multiple in year y - h) is fitted on the
  industry's gap from the group's market that year, ln(its multiple / the
  median industry's), by least squares (``Fit``: ``a + b x gap``, ``b``
  negative where expensive industries cheapen and cheap ones catch up),
  and the range is the latest multiple moved by that line plus the 10th,
  50th and 90th percentiles of what the line missed. So it is the central
  80% of what happened, over that many years in that region, to
  industries priced like this one. A horizon with fewer than
  ``MIN_PAIRS`` such moves has no range.
- **Entry** is the year after the latest published one (the January
  edition describes the year before, so a deal closing this year is one
  year on); **exit** is entry plus the deal's hold.
- **Shown only where the card says it beats the baseline** (PLAN.md phase
  5): ``ml/cards/multiples.json`` tests both ranges walk-forward against
  the region's whole market (ml/evaluation/multiples.py), by S&P region;
  elsewhere the screen shows the published figures and "not enough data".
- **Comparables**: the industry's latest multiple in every group (so a
  region can be set beside the others), the region's industries in the
  same GICS sector, and, when the reference library is on (PLAN.md 4.5),
  the approved reference transactions in the same region and sector with
  their entry multiples. With the library off only that last list goes.

The averages are listed companies' (aggregates: large companies weigh
more) and are not split by size, which the answer's ``notes`` say.
"""
from __future__ import annotations

import json
import math
from datetime import date
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from benchmarks import history
from benchmarks.catalogue import ALL_INDUSTRIES_ID, MIN_FIRMS, REGIONS, SOURCE, canonical, chain
from benchmarks.damodaran import Table
from benchmarks.starting import MAX_MULTIPLE
from library import base_rates

CARD = Path(__file__).parent / "cards" / "multiples.json"
# The range's percentiles: its low end, the suggestion and its high end
QUANTILES = (0.1, 0.5, 0.9)
# Pooled industry moves a horizon's range rests on, at least
MIN_PAIRS = 30
DECIMALS = 2
NOTES = ("size_not_split", "listed_company_figures")
# Card sets: which range each tests
SETS = {"entry": "entry", "exit": "exit"}


def usable(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 < value <= MAX_MULTIPLE)


def series(tables: Mapping[str, Table], group: str, industry: str) -> dict[int, float]:
    """The industry's EV/EBITDA by year in ``group``, for years with at least
    ``MIN_FIRMS`` companies and a usable multiple."""
    return {y: f["ev_ebitda"] for y, f in history.industry_years(tables, group, industry).items()
            if f.get("firms", 0) >= MIN_FIRMS and usable(f.get("ev_ebitda"))}


def group_series(tables: Mapping[str, Table], group: str) -> dict[str, dict[int, float]]:
    """Every industry's usable series in ``group`` (not the whole market's)."""
    return {i: s for i in sorted(history.industries_in(tables, group) - {ALL_INDUSTRIES_ID})
            if (s := series(tables, group, i))}


def market_median(by_industry: Mapping[str, Mapping[int, float]], year: int) -> Optional[float]:
    """The median industry's multiple in ``year`` (the group's market)."""
    values = sorted(s[year] for s in by_industry.values() if year in s)
    return quantile(values, 0.5) if values else None


def gap(by_industry: Mapping[str, Mapping[int, float]], value: float, year: int) -> float:
    """ln(``value`` / the market's median in ``year``): how far an industry's
    multiple sits above (positive) or below the group's market."""
    median = market_median(by_industry, year)
    return math.log(value / median) if median else 0.0


def moves(by_industry: Mapping[str, Mapping[int, float]], horizon: int,
          before: Optional[int] = None) -> list[tuple[float, float]]:
    """``(gap in the base year, ln(multiple / multiple horizon years
    earlier))`` pooled over the industries, for target years before
    ``before`` (all when None)."""
    out = []
    for s in by_industry.values():
        for y, v in s.items():
            if (before is None or y < before) and (y - horizon) in s:
                base = s[y - horizon]
                out.append((gap(by_industry, base, y - horizon), math.log(v / base)))
    return out


def quantile(sorted_values: list[float], q: float) -> float:
    """The ``q`` quantile of sorted values, linearly interpolated."""
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


@dataclass(frozen=True)
class Fit:
    """The pooled moves' least-squares line on the gap, and what it missed (sorted)."""
    a: float
    b: float
    residuals: tuple


def fit(points: list[tuple[float, float]]) -> Optional[Fit]:
    """The line through ``(gap, move)`` points; None under ``MIN_PAIRS``."""
    if len(points) < MIN_PAIRS:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in points) / vx if vx > 0 else 0.0
    a = my - b * mx
    return Fit(a, b, tuple(sorted(y - a - b * x for x, y in points)))


def band(base: float, base_gap: float, f: Optional[Fit]) -> Optional[dict]:
    """``base`` moved along the line and by the misses' percentiles."""
    if f is None:
        return None
    centre = f.a + f.b * base_gap
    low, median, high = (base * math.exp(centre + quantile(list(f.residuals), q)) for q in QUANTILES)
    return {"low": low, "median": median, "high": high, "pairs": len(f.residuals)}


def market_band(by_industry: Mapping[str, Mapping[int, float]], year: int) -> Optional[dict]:
    """The baseline: the same percentiles across the group's industries in
    ``year``, whatever the industry (the region's whole market)."""
    values = sorted(s[year] for s in by_industry.values() if year in s)
    if len(values) < 2:
        return None
    low, median, high = (quantile(values, q) for q in QUANTILES)
    return {"low": low, "median": median, "high": high, "pairs": len(values)}


def peer_group(tables: Mapping[str, Table], country: str, industry: str) -> tuple[Optional[str], dict, list[dict]]:
    """The closest group in the country's own region whose industry has
    ``MIN_FIRMS`` companies and a usable multiple in its latest year:
    ``(group, series, skipped)``. The global group is never used."""
    skipped = []
    for area in chain(country):
        if REGIONS[area].level == "global":
            break
        years = history.industry_years(tables, area, industry)
        if not years:
            skipped.append({"area": area, "reason": "missing", "sample": None})
            continue
        latest = years[max(years)]
        if latest.get("firms", 0) < MIN_FIRMS:
            skipped.append({"area": area, "reason": "thin", "sample": latest.get("firms")})
        elif not usable(latest.get("ev_ebitda")):
            skipped.append({"area": area, "reason": "unusable", "sample": latest.get("firms")})
        else:
            return area, series(tables, area, industry), skipped
    return None, {}, skipped


def entry_horizon(latest_year: int, today: date) -> int:
    """Years from the latest published figure to a deal closing this year."""
    return max(1, today.year - latest_year)


def _rounded(b: Optional[dict]) -> Optional[dict]:
    return None if b is None else {**{k: round(b[k], DECIMALS) for k in ("low", "median", "high")},
                                   "pairs": b["pairs"]}


def ranges(tables: Mapping[str, Table], group: str, industry_series: Mapping[int, float], hold: int,
           today: date) -> dict:
    """The entry and exit ranges from the industry's latest multiple in ``group``."""
    latest = max(industry_series)
    base = industry_series[latest]
    pooled = group_series(tables, group)
    base_gap = gap(pooled, base, latest)
    h = entry_horizon(latest, today)
    out = {}
    for name, horizon in (("entry", h), ("exit", h + hold)):
        out[name] = {"horizon": horizon, "year": latest + horizon,
                     "range": _rounded(band(base, base_gap, fit(moves(pooled, horizon))))}
    return {"latest_year": latest, "latest": round(base, DECIMALS), **out}


# ---------------------------------------------------------------------------
# Comparables
# ---------------------------------------------------------------------------
def sector_of(industry: str) -> Optional[str]:
    from validation.tags import INDUSTRY_SECTOR
    return INDUSTRY_SECTOR.get(industry)


def _latest(tables: Mapping[str, Table], group: str, industry: str) -> Optional[dict]:
    years = history.industry_years(tables, group, industry)
    if not years:
        return None
    y = max(years)
    f = years[y]
    return {"year": y, "firms": f.get("firms"), "multiple": round(f["ev_ebitda"], DECIMALS)
            if usable(f.get("ev_ebitda")) else None}


def by_region(tables: Mapping[str, Table], industry: str, used: Optional[str]) -> list[dict]:
    """The industry's latest multiple in every group that has it."""
    out = []
    for area in REGIONS:
        found = _latest(tables, area, industry)
        if found:
            out.append({"group": area, "level": REGIONS[area].level, "used": area == used,
                        "enough": (found["firms"] or 0) >= MIN_FIRMS and found["multiple"] is not None, **found})
    return out


def same_sector(tables: Mapping[str, Table], group: str, industry: str) -> list[dict]:
    """The group's industries in the deal industry's GICS sector with
    ``MIN_FIRMS`` companies, cheapest first."""
    sector = sector_of(industry)
    if not sector:
        return []
    out = []
    for other in sorted(history.industries_in(tables, group) - {ALL_INDUSTRIES_ID}):
        if sector_of(other) != sector:
            continue
        found = _latest(tables, group, other)
        if found and found["multiple"] is not None and (found["firms"] or 0) >= MIN_FIRMS:
            out.append({"industry": other, "name": _name(tables, group, other), "this": other == industry, **found})
    return sorted(out, key=lambda x: (x["multiple"], x["industry"]))


def _name(tables: Mapping[str, Table], group: str, industry: str) -> str:
    for name in (f"multiples.{group}", history.table_name(group), "margins.global"):
        row = tables[name].rows.get(industry) if name in tables else None
        if row and row.get("name"):
            return row["name"]
    return industry


# ---------------------------------------------------------------------------
# The card decides where the ranges are shown
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _card() -> dict:
    return json.loads(CARD.read_text(encoding="utf-8"))


def card_result(set_name: str, region: Optional[str], card: Optional[Mapping] = None) -> dict:
    """The card's verdict for one range (``entry`` or ``exit``) in ``region``,
    with its cases and headline statistic for the model and the baseline."""
    ev = (card or _card())["evaluation"]
    group = ev["sets"][SETS[set_name]]["by_region"].get(region) if region else None
    if not group:
        return {"verdict": "not_enough_data", "cases": 0, "model": None, "baseline": None}
    head = ev["headline_metric"]
    return {"verdict": group["verdict"], "cases": group["cases"],
            "model": (group.get("model") or {}).get(head), "baseline": (group.get("baseline") or {}).get(head)}


def shown(found: Mapping, region: Optional[str], card: Optional[Mapping] = None) -> dict:
    """Each range as the screen may show it: its figures only where the
    card says that range beats the baseline in the deal's region."""
    out = {}
    for name in SETS:
        r = found[name]
        result = card_result(name, region, card)
        visible = result["verdict"] == "beats_baseline" and r["range"] is not None
        out[name] = {"horizon": r["horizon"], "year": r["year"], "shown": visible,
                     "range": r["range"] if visible else None, "card": result}
    return out


# ---------------------------------------------------------------------------
# Everything together
# ---------------------------------------------------------------------------
def predict(country: str, industry: str, hold: int, tables: Mapping[str, Table], today: date,
            library: Optional[Iterable[dict]] = None, ev_usd_m: Optional[float] = None,
            card: Optional[Mapping] = None) -> dict:
    """The whole answer: the peer group's latest multiple, the entry and
    exit ranges where the card allows them, the comparables and, when the
    library is on (``library`` not None), the reference transactions like it."""
    from ml.anomaly_detector import similar_deals
    country = canonical(country)
    industry = industry or ALL_INDUSTRIES_ID
    region = base_rates.sp_region(country)
    world = tables.get("margins.global") or tables.get("multiples.global")
    deals = (similar_deals(library, country, industry, ev_usd_m) if library is not None
             else {"enabled": False, "region": None, "sector": None, "size": None, "deals": []})
    out = {"status": "not_enough_data", "reason": None, "country": country, "industry": industry,
           "industry_name": world.rows[industry]["name"] if world and industry in world.rows else None,
           "region": region, "hold": hold, "min_firms": MIN_FIRMS, "min_pairs": MIN_PAIRS,
           "quantiles": list(QUANTILES), "group": None, "level": None, "firms": None, "latest_year": None,
           "latest": None, "published": None, "url": None, "skipped": [],
           "entry": None, "exit": None, "regions": by_region(tables, industry, None), "sector": [],
           "deals": deals, "notes": list(NOTES), "source": dict(SOURCE)}
    if not country:
        return {**out, "reason": "no_country"}
    group, s, skipped = peer_group(tables, country, industry)
    out["skipped"] = skipped
    if group is None or not s:
        return {**out, "reason": "no_peers"}
    found = ranges(tables, group, s, hold, today)
    current = tables.get(f"multiples.{group}")
    latest_firms = history.industry_years(tables, group, industry)[found["latest_year"]].get("firms")
    return {**out, "status": "ok", "group": group, "level": REGIONS[group].level, "firms": latest_firms,
            "latest_year": found["latest_year"], "latest": found["latest"],
            "published": current.published if current else None,
            "url": current.url if current else history.ARCHIVE_PAGE,
            **shown(found, region, card),
            "regions": by_region(tables, industry, group), "sector": same_sector(tables, group, industry)}
