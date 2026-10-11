"""Economic regime by region (PLAN.md 5.7).

Which of the Monte Carlo scenario presets the economy a deal sits in looks
like this year -- **recession**, **stagflation**, **bull** or **base** --
for the United States, the euro area, the United Kingdom, Japan, China,
India and the world, each with the year it describes and the day its
figures were read. Nothing is trained: the old four-state hidden Markov
model over US-only FRED series (one of them discontinued in 2022, another
licensed against reproduction) is gone.

- **Data**: the IMF World Economic Outlook's real GDP growth and consumer
  price inflation by year since ``FIRST_YEAR``, through the DataMapper API
  PLAN.md 4.2's connector reads (``economy.connectors``), for the six
  economies and the IMF's own euro area and world aggregates.
- **The measure**: how far a year's growth and inflation sit from the
  **median of the region's own previous ``TRAIL`` years** (``gap``). Levels
  would not do: China's growth has halved since 2010 and India's inflation
  too, so a cut-off on the level would call every recent Chinese year a
  recession. The median, not the mean, so that one crash and its rebound
  (2020 and 2021) do not set what the next years are held to.
- **The rule** (``name``), on cut-offs from the region's earlier years only
  (``thresholds``): a growth gap in the weakest ``SHARE`` of them is a
  **recession**; else an inflation gap in the highest ``SHARE`` is
  **stagflation**; else a growth gap in the strongest ``SHARE`` is
  **bull**; anything else is **base**. A fifth each, as PLAN.md 4.4 chooses
  the presets' own years.
- **Which year** (``regime_year``): the IMF publishes in April and October.
  Read from May, the current year's figure is a same-year projection, which
  the model card tests; read before May it is still last October's
  year-ahead forecast, which nobody should call a regime, so the regime
  shown is last year's (an estimate made in its own October).
- **Refit monthly** (``refresh``): GitHub Actions' nightly
  ``economy-refresh`` task runs it after the economic data (jobs/scheduled.py;
  ``regime-refresh`` runs it alone), and the first run of a calendar month
  reads the figures again, refits every region's cut-offs and stores figures
  and fit together (``regime.fit`` in ``benchmark_tables``). The API answers
  from storage.
- **Shown only where ``ml/cards/regime.json`` says it beats the baseline**
  for the region (``card_result``), and only while the stored fit is under
  ``MAX_AGE_DAYS`` old. The card (ml/evaluation/regime.py) replays what the
  IMF projected for each year in its spring and autumn reports since 2000
  against the final figures; the baseline is "the same as last year".

A deal's country gives its region (``region_of``): one of the six
economies, a euro member the euro area, anywhere else the world, and the
answer says when the world is standing in.
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

BASE = Path(__file__).resolve().parent
CARD = BASE / "cards" / "regime.json"
# The card's headline set: the earliest (weakest) figure ever shown
GATE_SET = "spring"
# The card's set that tests each kind of figure: a projection for the year
# under way is at worst April's; an estimate of the year just ended, read
# before the next April, is the October report's figure for its own year
GATE_SETS = {"projection": "spring", "estimate": "autumn"}
TABLE = "regime.fit"
IMF_PAGE = "https://www.imf.org/external/datamapper/datasets/WEO"
SOURCES = {"imf": {"publisher": "International Monetary Fund, World Economic Outlook", "url": IMF_PAGE}}

# The IMF's first year
FIRST_YEAR = 1980
# A year is measured against the median of this many years before it
TRAIL = 5
# The share of a region's earlier years each regime's cut-off takes (benchmarks/risk.py SHARE)
SHARE = 0.2
# Earlier years with a gap the cut-offs need, at least
MIN_YEARS = 10
# Earlier years listed beside the current one
RECENT = 10
# The month from which the IMF's figure for the current year is a same-year projection
SAME_YEAR_FROM_MONTH = 5
# A stored fit older than this is not shown (three missed monthly refits)
MAX_AGE_DAYS = 100
PLACES = 2

REGIMES = ("recession", "stagflation", "bull", "base")
NOTES = ("own_history", "imf_projection", "annual", "no_markets")


@dataclass(frozen=True)
class Region:
    id: str
    area: Optional[str]     # economy/catalogue.py's code; None for the world
    imf: str                # the DataMapper's code


REGIONS: Mapping[str, Region] = {r.id: r for r in (
    Region("us", "US", "USA"), Region("eurozone", "XM", "EURO"), Region("uk", "GB", "GBR"),
    Region("japan", "JP", "JPN"), Region("china", "CN", "CHN"), Region("india", "IN", "IND"),
    Region("global", None, "WEOWORLD"),
)}
GLOBAL = "global"
# The euro's members (Bulgaria joined in January 2026)
EURO_COUNTRIES = frozenset({
    "AT", "BE", "BG", "CY", "DE", "EE", "ES", "FI", "FR", "GR", "HR", "IE", "IT", "LT", "LU", "LV", "MT", "NL",
    "PT", "SI", "SK", "XM",
})
_BY_AREA = {r.area: r.id for r in REGIONS.values() if r.area}

Series = Mapping[int, tuple]      # year -> (real growth, inflation), in %


class Incomplete(RuntimeError):
    """A refit left a region without a regime, or there was nothing to fit."""


def region_of(country: Optional[str]) -> tuple[str, bool]:
    """The region whose regime a deal in ``country`` is shown, and whether
    it is the country's own (False: the world standing in)."""
    code = (country or "").upper()
    if code in EURO_COUNTRIES:
        return "eurozone", True
    if code in _BY_AREA:
        return _BY_AREA[code], True
    return GLOBAL, False


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------
def norm(series: Series, year: int) -> Optional[tuple[float, float]]:
    """Median growth and median inflation of the ``TRAIL`` years before
    ``year``; None unless every one of them is known."""
    before = [series.get(y) for y in range(year - TRAIL, year)]
    if any(v is None for v in before):
        return None
    return statistics.median(v[0] for v in before), statistics.median(v[1] for v in before)


def gap(series: Series, year: int, figure: Optional[tuple] = None) -> Optional[tuple[float, float]]:
    """``year``'s growth and inflation less its ``norm``, in points.
    ``figure`` stands in for the year's own (a projection made at the time)."""
    own = figure if figure is not None else series.get(year)
    usual = norm(series, year)
    if own is None or usual is None:
        return None
    return own[0] - usual[0], own[1] - usual[1]


def cutoff(values: list[float], top: bool) -> float:
    """The value a new one must reach to join the top (or bottom) ``SHARE``."""
    n = max(1, round(len(values) * SHARE))
    return sorted(values, reverse=top)[n - 1]


def thresholds(series: Series, before: int) -> Optional[dict]:
    """The cut-offs from the gaps of the years before ``before``; None with
    fewer than ``MIN_YEARS`` of them."""
    gaps = [g for y in sorted(series) if y < before and (g := gap(series, y)) is not None]
    if len(gaps) < MIN_YEARS:
        return None
    growth, inflation = [g[0] for g in gaps], [g[1] for g in gaps]
    return {"weak_growth": cutoff(growth, False), "strong_growth": cutoff(growth, True),
            "high_inflation": cutoff(inflation, True), "years": len(gaps)}


def name(gaps: tuple[float, float], cuts: Mapping[str, float]) -> str:
    """The regime of a year with these gaps: weak growth first, then high
    inflation, then strong growth."""
    if gaps[0] <= cuts["weak_growth"]:
        return "recession"
    if gaps[1] >= cuts["high_inflation"]:
        return "stagflation"
    if gaps[0] >= cuts["strong_growth"]:
        return "bull"
    return "base"


def call(series: Series, year: int, figure: Optional[tuple] = None) -> Optional[dict]:
    """``year``'s regime from what was known before it: its figures against
    the previous years' median, on cut-offs from earlier years only. None
    when the history is too short or a figure is missing."""
    own = figure if figure is not None else series.get(year)
    cuts, gaps, usual = thresholds(series, year), gap(series, year, figure), norm(series, year)
    if own is None or cuts is None or gaps is None or usual is None:
        return None
    return {
        "regime": name(gaps, cuts), "year": year,
        "growth": own[0], "inflation": own[1],
        "growth_norm": round(usual[0], PLACES), "inflation_norm": round(usual[1], PLACES),
        "growth_gap": round(gaps[0], PLACES), "inflation_gap": round(gaps[1], PLACES),
        "weak_growth": round(cuts["weak_growth"], PLACES), "strong_growth": round(cuts["strong_growth"], PLACES),
        "high_inflation": round(cuts["high_inflation"], PLACES), "history_years": cuts["years"],
    }


def regime_year(read_on: date) -> int:
    """The year a fit read on ``read_on`` describes: the current one once
    the IMF's April figures are in, last year's before."""
    return read_on.year if read_on.month >= SAME_YEAR_FROM_MONTH else read_on.year - 1


def fit(series: Mapping[str, Series], read_on: date) -> dict:
    """What is stored: each region's figures up to the regime's year, its
    regime (None when it can't be called) and the years before it."""
    year = regime_year(read_on)
    rows = {}
    for region, own in series.items():
        found = call(own, year)
        if found is not None:
            found["basis"] = "projection" if year == read_on.year else "estimate"
            found["first_year"] = min(own)
            found["recent"] = [{"year": y, "regime": past["regime"]} for y in range(year - RECENT, year)
                               if (past := call(own, y)) is not None]
        rows[region] = {"series": {str(y): list(v) for y, v in sorted(own.items()) if y <= year}, "fit": found}
    return rows


# ---------------------------------------------------------------------------
# Reading and the monthly refit
# ---------------------------------------------------------------------------
def read(today: date) -> dict[str, dict[int, tuple]]:
    """Each region's growth and inflation by year, ``FIRST_YEAR`` to this
    year, for the years the IMF gives both (two calls)."""
    from companies import http
    from economy.connectors import IMF_API, IMF_INDICATORS, _number

    found: dict[str, dict[str, dict[int, float]]] = {}
    for indicator, code in IMF_INDICATORS.items():
        data = http.get_json("imf", f"{IMF_API}/{code}") or {}
        values = (data.get("values") or {}).get(code) or {}
        for region in REGIONS.values():
            row = values.get(region.imf) or {}
            found.setdefault(region.id, {})[indicator] = {
                y: v for y in range(FIRST_YEAR, today.year + 1) if (v := _number(row.get(str(y)))) is not None}
    out = {}
    for region, parts in found.items():
        growth, inflation = parts.get("gdp_growth", {}), parts.get("inflation", {})
        out[region] = {y: (growth[y], inflation[y]) for y in sorted(growth) if y in inflation}
    return out


def summary(rows: Mapping[str, dict], read_on: Optional[date], was_read: bool, problem: str = "") -> dict:
    """A refit's flat summary (jobs/scheduled.py)."""
    fits = {r: (rows.get(r) or {}).get("fit") for r in REGIONS}
    dated = {r: f for r, f in fits.items() if f}
    return {
        "read": was_read, "regions": len(REGIONS), "dated": len(dated),
        "year": max((f["year"] for f in dated.values()), default=None),
        "regimes": ",".join(f"{r}:{f['regime']}" for r, f in dated.items()),
        "read_on": read_on.isoformat() if read_on else None, "source_problem": problem,
    }


def refresh(now: Optional[datetime] = None, force: bool = False) -> dict:
    """The scheduled refit: on the first run of a calendar month (or when
    nothing is stored, or ``force``) read the IMF's figures, refit every
    region and store both. A source that fails, or figures that leave a
    region without a regime, leave the stored fit as it was, to be tried
    again the next night; the run fails, so it alerts, when nothing is
    stored or the figures read were incomplete."""
    from api.observability import log_event
    from benchmarks.damodaran import Table
    from companies import http
    from db import benchmarks as store
    from db.engine import is_configured
    if not is_configured():
        return {"skipped_no_database": True}
    now = now or datetime.now(timezone.utc)
    today = now.date()
    stored = store.table(TABLE)
    due = force or stored is None or stored.published is None or (
        (stored.published.year, stored.published.month) != (today.year, today.month))
    problem = ""
    if due:
        try:
            rows = fit(read(today), today)
        except http.SourceError as exc:
            log_event("regime_source_failed", logging.WARNING, source=exc.source, reason=exc.reason)
            problem = exc.reason
        except (AttributeError, TypeError, KeyError, ValueError, IndexError):
            # Valid JSON of a shape this module doesn't know
            log_event("regime_source_failed", logging.WARNING, source="imf", reason="unreadable")
            problem = "unreadable"
        else:
            dated = sum(1 for r in REGIONS if (rows.get(r) or {}).get("fit"))
            if dated < len(REGIONS):
                # Never replace a whole fit with a partial one
                log_event("regime_source_failed", logging.WARNING, source="imf", reason="incomplete")
                problem = f"incomplete:{dated}"
            else:
                stored = Table(TABLE, today, IMF_PAGE, rows)
                store.save([stored], now)
    found = summary(stored.rows if stored else {}, stored.published if stored else None, due and not problem, problem)
    log_event("regime_refreshed", **found)
    if problem.startswith("incomplete"):
        raise Incomplete(f"{problem.split(':')[1]} of {len(REGIONS)} regions have a regime in the figures read; "
                         "the stored fit is unchanged")
    if stored is None:
        raise Incomplete(f"no economic regimes stored (the IMF: {problem})")
    return found


# ---------------------------------------------------------------------------
# A deal's answer
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _card() -> dict:
    """The committed card (it changes only with a deploy)."""
    return json.loads(CARD.read_text(encoding="utf-8")) if CARD.exists() else {}


def card_result(region: str, basis: str = "projection") -> dict:
    """The card's verdict in ``region`` for the set that tests a figure of
    this ``basis`` (``GATE_SETS``)."""
    sets = (_card().get("evaluation") or {}).get("sets") or {}
    found = (sets.get(GATE_SETS[basis]) or {}).get("by_region", {}).get(region) or {}
    return {"region": region, "set": GATE_SETS[basis], "verdict": found.get("verdict", "not_enough_data"), "cases": found.get("cases", 0),
            "model": (found.get("model") or {}).get("accuracy"),
            "baseline": (found.get("baseline") or {}).get("accuracy")}


def _hidden(found: Optional[dict], verdict: str, stale: bool) -> Optional[str]:
    if found is None:
        return "not_enough_data"
    if stale:
        return "stale"
    return None if verdict == "beats_baseline" else verdict


def current(country: Optional[str], rows: Optional[Mapping[str, dict]], read_on: Optional[date],
            today: date) -> dict:
    """The regime for a deal in ``country`` and every region's beside it,
    from the stored fit (``rows``, read on ``read_on``); a regime is given
    only where its card beats the baseline and the fit is recent."""
    region, own = region_of(country)
    answer = {
        "status": "no_data", "country": country or "", "region": region, "own_region": own,
        "shown": False, "hidden": None, "regime": None, "regions": [], "card": card_result(region),
        "read_on": read_on.isoformat() if read_on else None, "trail": TRAIL, "share": SHARE,
        "first_year": FIRST_YEAR, "max_age_days": MAX_AGE_DAYS, "sources": SOURCES, "notes": list(NOTES),
    }
    if not rows or read_on is None:
        return answer
    stale = (today - read_on).days > MAX_AGE_DAYS
    listed = []
    for rid in REGIONS:
        found = (rows.get(rid) or {}).get("fit")
        hidden = _hidden(found, card_result(rid, found["basis"] if found else "projection")["verdict"], stale)
        listed.append({"region": rid, "shown": hidden is None, "hidden": hidden,
                       "regime": found["regime"] if hidden is None else None,
                       "year": found["year"] if found else None, "basis": found["basis"] if found else None})
    mine = (rows.get(region) or {}).get("fit")
    card = card_result(region, mine["basis"]) if mine else answer["card"]
    hidden = _hidden(mine, card["verdict"], stale)
    return {**answer, "card": card, "status": "ok", "shown": hidden is None, "hidden": hidden,
            "regime": mine if hidden is None else None, "regions": listed}
