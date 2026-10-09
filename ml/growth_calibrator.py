"""The growth calibrator: revenue growth ranges by sector and region (PLAN.md 5.5).

The Monte Carlo simulation draws one revenue growth rate for each path and
holds it over the deal's hold (simulation/vectorized_simulation.py), around
the Settings ``mc_growth_mean`` and ``mc_growth_std``. Their sourced values
(PLAN.md 4.4) are the country's nominal GDP growth and the spread of that
growth from year to year: an economy's, which moves far less than one
company's. This module sets them from **how listed companies' revenue
really grew**, by region and sector:

- **The data** (``ml/evaluation/data/firm_growth.json``, written by
  ``python -m tests.ml_firm_growth``): every SEC filer's yearly revenue
  from the SEC's XBRL frames, with its country (business address) and SIC
  code -- the same filings the company search reads (PLAN.md 4.1) --
  plus the ECB's yearly exchange rates and the IMF's growth history.
- **A case** is one company's revenue growth over ``h`` years (``h`` = a
  hold of 3 to 7, ``HORIZONS``), as a yearly rate: (end / start)^(1/h) - 1,
  both ends from one revenue concept in one currency. Companies whose
  revenue in the start year was under ``FLOOR_USD_M`` US dollars (the
  ECB's average rate that year) are left out (much smaller than a buyout
  target), as are financial companies and those without an industry code
  (``sector``).
- **What is measured** is the company's growth **over its country's**
  nominal GDP growth in the same years (IMF; a country the app doesn't
  cover takes its region's median, as the starting figures do,
  benchmarks/starting.py): ``ln((1 + company) / (1 + economy))``. So a
  region can pool currencies with different inflation, and a deal's range
  is centred on its own country's growth.
- **The range** for a deal: the 10th, 50th and 90th percentiles of those
  excesses among the cases of its S&P region and GICS sector over its
  hold (the region's cases of every sector when its sector has fewer than
  ``MIN_COMPANIES`` companies), applied to the country's nominal growth
  this year -- the IMF's projection the starting figures use. The Settings
  are the normal distribution whose central 80% is that range:
  ``mc_growth_mean`` its middle, ``mc_growth_std`` its width over 2 x 1.2816.
- **Shown only where ``ml/cards/growth.json`` says it beats the baseline**
  (the sourced economy-wide range in force since 4.4) in the deal's S&P
  region, and only for the holds the card tested.

The fitted percentiles are ``ml/growth_ranges.json``, rewritten from the
data with the data (``write_ranges``); a test fails when they disagree.
"""
from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from benchmarks.catalogue import canonical, region_of
from economy.catalogue import COUNTRIES
from library.base_rates import SP_REGIONS, sp_region

BASE = Path(__file__).parent
DATA = BASE / "evaluation" / "data" / "firm_growth.json"
RANGES = BASE / "growth_ranges.json"
CARD = BASE / "cards" / "growth.json"
QUANTILES = (0.1, 0.5, 0.9)
# The normal quantile of 0.9: a normal's central 80% is its mean +- Z80 standard deviations
Z80 = 1.2815515655446004
# Holds the card tests and the app answers (years)
HORIZONS = (3, 4, 5, 6, 7)
# Revenue in the start year, at least, in millions of US dollars
FLOOR_USD_M = 50.0
# A sector's range rests on at least this many companies, else the region's
MIN_COMPANIES = 30
DECIMALS = 2
NOTES = ("listed_companies", "survivors_only", "acquisitions_included", "size_floor")
# The card's set the app's gate reads
GATE_SET = "growth"


# ---------------------------------------------------------------------------
# Industry codes
# ---------------------------------------------------------------------------
# SIC codes (U.S. OSHA's SIC Manual, 1987) to the GICS sector their companies
# mostly belong to, by range, first match wins. Financial companies (SIC
# 6000-6499 and 6700-6799, REITs among them) are left out, as the starting
# figures leave out financial industries; so are public administration and
# non-classifiable establishments (9100-9999).
SIC_SECTORS: tuple[tuple[int, int, Optional[str]], ...] = (
    (100, 799, "consumer_staples"),            # agriculture
    (800, 999, "materials"),                   # forestry, fishing
    (1000, 1199, "materials"),                 # metal mining
    (1200, 1399, "energy"),                    # coal, oil and gas
    (1400, 1499, "materials"),                 # other minerals
    (1500, 1799, "industrials"),               # construction
    (2000, 2199, "consumer_staples"),          # food, tobacco
    (2200, 2399, "consumer_discretionary"),    # textiles, apparel
    (2400, 2499, "materials"),                 # lumber
    (2500, 2599, "consumer_discretionary"),    # furniture
    (2600, 2699, "materials"),                 # paper
    (2700, 2799, "communication_services"),    # printing and publishing
    (2830, 2839, "health_care"),               # drugs
    (2840, 2849, "consumer_staples"),          # soaps, cosmetics
    (2800, 2899, "materials"),                 # other chemicals
    (2900, 2999, "energy"),                    # petroleum refining
    (3000, 3099, "materials"),                 # rubber, plastics
    (3100, 3199, "consumer_discretionary"),    # leather
    (3200, 3399, "materials"),                 # stone, glass, primary metals
    (3400, 3499, "industrials"),               # fabricated metal
    (3570, 3579, "information_technology"),    # computers
    (3500, 3599, "industrials"),               # machinery
    (3630, 3639, "consumer_discretionary"),    # household appliances
    (3650, 3659, "consumer_discretionary"),    # household audio and video
    (3600, 3699, "information_technology"),    # electronics, semiconductors
    (3710, 3716, "consumer_discretionary"),    # motor vehicles
    (3750, 3751, "consumer_discretionary"),    # motorcycles, bicycles
    (3700, 3799, "industrials"),               # aircraft, ships, railway
    (3840, 3851, "health_care"),               # medical instruments
    (3870, 3879, "consumer_discretionary"),    # watches
    (3800, 3899, "information_technology"),    # measuring, photographic
    (3900, 3999, "consumer_discretionary"),    # toys, jewellery, sporting goods
    (4800, 4899, "communication_services"),    # communications
    (4950, 4959, "industrials"),               # waste
    (4900, 4999, "utilities"),                 # electricity, gas, water
    (4000, 4799, "industrials"),               # transport
    (5120, 5129, "health_care"),               # drug wholesale
    (5140, 5149, "consumer_staples"),          # grocery wholesale
    (5170, 5179, "energy"),                    # petroleum wholesale
    (5000, 5199, "industrials"),               # other wholesale
    (5400, 5499, "consumer_staples"),          # food stores
    (5912, 5912, "consumer_staples"),          # drug stores
    (5200, 5999, "consumer_discretionary"),    # other retail
    (6000, 6499, None),                        # banks, credit, brokers, insurance
    (6500, 6599, "real_estate"),               # real estate
    (6700, 6799, None),                        # holding and investment offices, REITs
    (7310, 7319, "communication_services"),    # advertising
    (7370, 7379, "information_technology"),    # software and computer services
    (7300, 7399, "industrials"),               # other business services
    (7500, 7599, "industrials"),               # vehicle rental and repair
    (7800, 7899, "communication_services"),    # motion pictures
    (7000, 7999, "consumer_discretionary"),    # hotels, personal services, recreation
    (8000, 8099, "health_care"),               # health services
    (8200, 8299, "consumer_discretionary"),    # education
    (8730, 8731, "health_care"),               # commercial research (mostly biological)
    (8100, 8999, "industrials"),               # legal, engineering, management
    (9100, 9999, None),                        # public administration, non-classifiable
)


def sector(sic: Optional[int]) -> Optional[str]:
    """The GICS sector of a SIC code; None for a financial company, one
    outside every range, or no code."""
    if sic is None:
        return None
    return next((s for lo, hi, s in SIC_SECTORS if lo <= sic <= hi), None)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Span:
    """One company's revenue growth over ``end - start`` years."""
    company: int
    country: str
    region: str
    sector: str
    start: int
    end: int
    growth: float       # yearly, a fraction
    economy: float      # the country's nominal GDP growth, yearly, a fraction
    size_usd_m: float   # revenue in the start year

    @property
    def horizon(self) -> int:
        return self.end - self.start

    @property
    def excess(self) -> float:
        return math.log((1 + self.growth) / (1 + self.economy))


def concept_order() -> dict[str, int]:
    """Revenue concepts by priority, as the deal model maps them."""
    from core.accounting import IFRS_ITEMS, US_GAAP_ITEMS
    names = [f"us-gaap:{c}" for c in US_GAAP_ITEMS["revenue"]] + [f"ifrs-full:{c}" for c in IFRS_ITEMS["revenue"]]
    return {n: i for i, n in enumerate(names)}


def economy_growth(nominal: Mapping[str, Mapping[int, float]], country: str) -> dict[int, float]:
    """A country's nominal growth by year (%): its own, else the median of
    its region's covered economies, else of all of them (the starting
    figures' rule, benchmarks/starting.py)."""
    if country in nominal and nominal[country]:
        return dict(nominal[country])
    region = region_of(country)
    for members in ([c for c in COUNTRIES if region_of(c) == region], list(COUNTRIES)):
        found = [nominal[c] for c in members if nominal.get(c)]
        if found:
            years = sorted({y for s in found for y in s})
            return {y: statistics.median(vals) for y in years if (vals := [s[y] for s in found if y in s])}
    return {}


def compound(by_year: Mapping[int, float], first: int, last: int) -> Optional[float]:
    """The yearly rate (fraction) that compounds to the growth in years
    ``first`` to ``last`` (each in %); None when a year is missing."""
    if last < first:
        return None
    factor = 1.0
    for y in range(first, last + 1):
        if y not in by_year:
            return None
        factor *= 1 + by_year[y] / 100
    return factor ** (1 / (last - first + 1)) - 1


def _pick(series: Mapping[str, Mapping[int, float]], order: Mapping[str, int], start: int, end: int):
    """The highest-priority concept (any currency) with revenue in both years."""
    best = None
    for key, years in series.items():
        concept, currency = key.split("|")
        if start in years and end in years and years[start] > 0 and years[end] > 0:
            rank = (order.get(concept, len(order)), currency)
            if best is None or rank < best[0]:
                best = (rank, currency, years[start], years[end])
    return best


def load(path: Path = DATA) -> dict:
    """The data file with years as numbers."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        **raw,
        "usd_per_unit": {c: {int(y): v for y, v in s.items()} for c, s in raw["usd_per_unit"].items()},
        "nominal_growth": {c: {int(y): v for y, v in s.items()} for c, s in raw["nominal_growth"].items()},
        "companies": [{**f, "series": {k: {int(y): v for y, v in s.items()} for k, s in f["series"].items()}}
                      for f in raw["companies"]],
    }


def spans(data: Mapping, horizons: Sequence[int] = HORIZONS) -> list[Span]:
    """Every company's growth over each horizon, for every start year it has."""
    order = concept_order()
    fx, out = data["usd_per_unit"], []
    economies: dict[str, dict[int, float]] = {}
    for firm in data["companies"]:
        sec = sector(firm.get("sic"))
        country = canonical(firm["country"])
        if sec is None:
            continue
        if country not in economies:
            economies[country] = economy_growth(data["nominal_growth"], country)
        economy = economies[country]
        years = sorted({y for s in firm["series"].values() for y in s})
        for start in years:
            for h in horizons:
                found = _pick(firm["series"], order, start, start + h)
                if found is None:
                    continue
                _, currency, first, last = found
                rate = fx.get(currency, {}).get(start)
                macro = compound(economy, start + 1, start + h)
                if rate is None or macro is None or first * rate < FLOOR_USD_M:
                    continue
                out.append(Span(firm["id"], country, sp_region(country), sec, start, start + h,
                                (last / first) ** (1 / h) - 1, macro, first * rate))
    return out


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------
def quantile(values: Sequence[float], q: float) -> float:
    """The ``q`` quantile of sorted ``values``, interpolated linearly."""
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def fit_group(cases: Iterable[Span]) -> Optional[dict]:
    """The excess's percentiles over ``cases``; None under ``MIN_COMPANIES``."""
    cases = list(cases)
    companies = {c.company for c in cases}
    if len(companies) < MIN_COMPANIES:
        return None
    xs = sorted(c.excess for c in cases)
    return {"quantiles": [round(quantile(xs, q), 6) for q in QUANTILES], "companies": len(companies),
            "cases": len(xs), "first": min(c.start for c in cases), "last": max(c.end for c in cases)}


def fit(cases: Iterable[Span]) -> dict:
    """``{(region, sector or None, horizon): group}`` for every group with
    enough companies; None is the region's every sector."""
    by: dict[tuple, list] = {}
    for c in cases:
        by.setdefault((c.region, c.sector, c.horizon), []).append(c)
        by.setdefault((c.region, None, c.horizon), []).append(c)
    return {k: g for k, v in by.items() if (g := fit_group(v)) is not None}


def lookup(model: Mapping[tuple, dict], region: str, sector_: Optional[str], horizon: int) -> tuple[Optional[dict], Optional[str]]:
    """The group a deal's range rests on, and the sector it stands for
    (None when the region's every sector stands in)."""
    if sector_ and (g := model.get((region, sector_, horizon))):
        return g, sector_
    return model.get((region, None, horizon)), None


def band(anchor: float, group: Mapping) -> dict:
    """The range (yearly growth, fractions) for an economy growing
    ``anchor`` a year, and the Settings whose normal draw has it as its
    central 80% (in %, as Settings hold them)."""
    lo, mid, hi = ((1 + anchor) * math.exp(q) - 1 for q in group["quantiles"])
    mean, std = round((lo + hi) / 2 * 100, DECIMALS), round((hi - lo) / (2 * Z80) * 100, DECIMALS)
    return {"low": lo, "median": mid, "high": hi, "mean": mean, "std": std,
            "draw_low": (mean - Z80 * std) / 100, "draw_high": (mean + Z80 * std) / 100}


def ranges(data: Mapping) -> dict:
    """The fitted percentiles the app serves: every group, on every case."""
    model = fit(spans(data))
    out: dict[str, dict] = {}
    for (region, sec, h), g in sorted(model.items(), key=lambda kv: (kv[0][0], kv[0][1] or "", kv[0][2])):
        out.setdefault(region, {}).setdefault(sec or "all", {})[str(h)] = g
    return {"about": "Written by ml.growth_calibrator.write_ranges from ml/evaluation/data/firm_growth.json "
                     "(python -m tests.ml_firm_growth); do not edit by hand.",
            "data_written_on": data["written_on"], "years": data["years"], "floor_usd_m": FLOOR_USD_M,
            "min_companies": MIN_COMPANIES, "quantiles": list(QUANTILES), "sources": data["sources"],
            "ranges": out}


def write_ranges(path: Path = RANGES) -> dict:
    found = ranges(load())
    path.write_text(json.dumps(found, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return found


@lru_cache(maxsize=1)
def served() -> dict:
    return json.loads(RANGES.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def served_model() -> dict:
    return {(region, None if sec == "all" else sec, int(h)): g
            for region, sectors in served()["ranges"].items()
            for sec, by_h in sectors.items() for h, g in by_h.items()}


# ---------------------------------------------------------------------------
# A deal's answer
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _card() -> dict:
    """The committed card (it changes only with a deploy)."""
    return json.loads(CARD.read_text(encoding="utf-8")) if CARD.exists() else {}


def card_result(region: Optional[str]) -> dict:
    """The card's verdict for the gate set in ``region``."""
    card = _card()
    found = (((card.get("evaluation") or {}).get("sets") or {}).get(GATE_SET) or {}).get("by_region", {}).get(region) or {}
    return {"region": region, "verdict": found.get("verdict", "not_enough_data"), "cases": found.get("cases", 0),
            "model": (found.get("model") or {}).get("interval_score"),
            "baseline": (found.get("baseline") or {}).get("interval_score")}


def _pct(x: float) -> float:
    return round(x * 100, DECIMALS)


def calibrate(country: str, industry: str, hold: int, anchor: Optional[dict]) -> dict:
    """The growth range and Settings for a deal in ``country`` and
    ``industry`` held ``hold`` years. ``anchor`` is the country's nominal
    growth figure the starting figures give (benchmarks/starting.py
    ``growth_figure``, in %), or None when none is stored."""
    from validation.tags import INDUSTRY_SECTOR

    region = sp_region(country) if country else None
    sec = INDUSTRY_SECTOR.get(industry) if industry else None
    meta = served()
    answer = {
        "status": "not_enough_data", "reason": None, "country": country, "industry": industry,
        "region": region, "sector": sec, "group_sector": None, "hold": hold, "horizons": list(HORIZONS),
        "quantiles": list(QUANTILES), "min_companies": MIN_COMPANIES, "floor_usd_m": FLOOR_USD_M,
        "shown": False, "hidden": None, "anchor": anchor, "range": None, "settings": None, "observed": None,
        "card": card_result(region), "years": meta["years"], "data_written_on": meta["data_written_on"],
        "sources": meta["sources"], "notes": list(NOTES),
    }
    if not country:
        return {**answer, "reason": "no_country"}
    if hold not in HORIZONS:
        return {**answer, "hidden": "untested_horizon"}
    group, group_sector = lookup(served_model(), region, sec, hold)
    if group is None:
        return {**answer, "hidden": "not_enough_data"}
    lo, mid, hi = (_pct(math.exp(q) - 1) for q in group["quantiles"])
    answer.update(group_sector=group_sector, observed={
        "low": lo, "median": mid, "high": hi, "companies": group["companies"], "cases": group["cases"],
        "first": group["first"], "last": group["last"]})
    if anchor is None:
        return {**answer, "reason": "no_growth"}
    verdict = answer["card"]["verdict"]
    if verdict != "beats_baseline":
        return {**answer, "status": "ok", "hidden": verdict}
    b = band(anchor["value"] / 100, group)
    return {**answer, "status": "ok", "shown": True,
            "range": {"low": _pct(b["low"]), "median": _pct(b["median"]), "high": _pct(b["high"])},
            "settings": {"mc_growth_mean": b["mean"], "mc_growth_std": b["std"]}}
