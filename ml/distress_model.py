"""Distress predictor: the chance a deal defaults, year by year (PLAN.md 5.3).

Nothing is trained. Each year of a deal's model run (or of every simulated
path) is given a **rating band** from two published tables, and the band's
**default rate from S&P's study for the deal's region**:

1. **Coverage**: EBIT / interest through Damodaran's coverage-to-rating table
   (``core.risk_sources.COVERAGE_BANDS``, the table the deal's default risk
   already reads).
2. **Leverage**: debt at the start of the year over the year's EBITDA through
   S&P's Corporate Methodology: Table 17 (standard volatility) gives the
   financial risk profile, 1 minimal to 6 highly leveraged, and Table 3
   combines it with the deal's business risk profile (``business_risk``, a
   deal input, 4 "fair" unless the user says otherwise) into an anchor. Where
   Table 3 lists two anchors ("bbb-/bb+") the weaker is read: the criteria
   choose between them on facts the model doesn't have.
3. The year's band is the **weaker** of the two reads, folded to the letter
   grades S&P's regional tables use (AAA, AA, A, BBB, BB, B, CCC/C).
4. The band's **forward default rate at that age** comes from S&P's 2024
   study, Table 25 for the US, Europe and emerging markets, Table 24 (global)
   for the other developed markets and for a deal with no country:
   ``h = (C(t) - C(t-1)) / (100 - C(t-1))`` from the average cumulative
   default rates ``C``. A deal that stays in one band therefore defaults
   exactly as often as the table says (the card's calibration set); past the
   table's last year the last year's rate is held.

The yearly probability is ``S(t-1) * h(t)`` and the cumulative ``1 - S(t)``
with ``S`` the chance of reaching the end of a year without default; the
simulation's figures are their means over the paths.

**Shown only where its card says it beats the baseline** (phase 5's rule):
``ml/cards/distress.json`` tests it on the sourced reference transactions
against the deal's year-one default risk (``core.risk_warnings.credit_view``).
Elsewhere the probabilities are not sent and the screen says "not enough
data"; the bands, coverage and leverage, read from the deal's own figures
through published tables, are always shown. Every function here works on one
deal's numbers and on arrays of paths alike (numpy only, no ML packages).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from core.risk_sources import COVERAGE_BANDS, SOURCES
from library import base_rates

CARD = Path(__file__).parent / "cards" / "distress.json"

# S&P's regional tables' bands, strongest first; a band is its index here
BANDS = ("AAA", "AA", "A", "BBB", "BB", "B", "CCC/C")
WEAKEST = len(BANDS) - 1

# S&P Corporate Methodology (January 7, 2024), Table 17, standard volatility:
# debt / EBITDA below each bound is the profile beside it ("Minimal: less than
# 1.5", "Modest: 1.5-2", "Intermediate: 2-3", "Significant: 3-4",
# "Aggressive: 4-5"); above 5 is 6, highly leveraged ("greater than 5").
LEVERAGE_PROFILES: tuple[tuple[float, int], ...] = ((1.5, 1), (2.0, 2), (3.0, 3), (4.0, 4), (5.0, 5))
HIGHLY_LEVERAGED = 6
PROFILE_NAMES = ("minimal", "modest", "intermediate", "significant", "aggressive", "highly_leveraged")

# Table 3: the anchor for business risk profile 1 (excellent) .. 6
# (vulnerable), by financial risk profile 1 (minimal) .. 6 (highly leveraged),
# as printed
ANCHORS: dict[int, tuple[str, ...]] = {
    1: ("aaa/aa+", "aa", "a+/a", "a-", "bbb", "bbb-/bb+"),
    2: ("aa/aa-", "a+/a", "a-/bbb+", "bbb", "bb+", "bb"),
    3: ("a/a-", "bbb+", "bbb/bbb-", "bbb-/bb+", "bb", "b+"),
    4: ("bbb/bbb-", "bbb-", "bb+", "bb", "bb-", "b"),
    5: ("bb+", "bb+", "bb", "bb-", "b+", "b/b-"),
    6: ("bb-", "bb-", "bb-/b+", "b+", "b", "b-"),
}
BUSINESS_RISK = ("excellent", "strong", "satisfactory", "fair", "weak", "vulnerable")
DEFAULT_BUSINESS_RISK = 4

SOURCE_IDS = ("sp_corporate_methodology_2024", "damodaran_ratings_2026", "sp_default_study_2024_regions",
              "deal_model")


def band_of(rating: str) -> int:
    """A rating's letter grade ('bb-' -> BB, 'CC' -> CCC/C) as a band index."""
    letters = rating.upper().rstrip("+-")
    return WEAKEST if letters in ("CCC", "CC", "C", "D") else BANDS.index(letters)


def anchor(business: int, profile: int) -> str:
    """Table 3's anchor, the weaker of two where it lists two."""
    return ANCHORS[int(business)][int(profile) - 1].split("/")[-1]


def leverage_profile(leverage):
    """Table 17's financial risk profile (1-6) for debt / EBITDA; a
    non-positive EBITDA (``leverage`` NaN or infinite) is highly leveraged."""
    lev = np.asarray(leverage, dtype=float)
    out = np.full(lev.shape, HIGHLY_LEVERAGED, dtype=int)
    # Exactly 5 is aggressive: highly leveraged is "greater than 5"
    out = np.where(np.isfinite(lev) & (lev <= LEVERAGE_PROFILES[-1][0]), LEVERAGE_PROFILES[-1][1], out)
    for bound, profile in reversed(LEVERAGE_PROFILES[:-1]):
        out = np.where(np.isfinite(lev) & (lev < bound), profile, out)
    return out


# The coverage table's lower bounds and each band's letter grade
_COVERAGE_LOWS = np.array([low for low, _ in COVERAGE_BANDS[1:]])
_COVERAGE_GRADES = np.array([band_of(r) for _, r in COVERAGE_BANDS])


def coverage_band(ebit, interest):
    """Damodaran's coverage band as a letter grade; a year with no interest
    is not held back by coverage (AAA)."""
    ebit, interest = np.asarray(ebit, dtype=float), np.asarray(interest, dtype=float)
    paying = interest > 0
    coverage = np.divide(ebit, interest, out=np.zeros(np.broadcast(ebit, interest).shape), where=paying)
    return np.where(paying, _COVERAGE_GRADES[np.searchsorted(_COVERAGE_LOWS, coverage, side="right")], 0)


def leverage_of(debt, ebitda):
    """Debt / EBITDA; no debt is no leverage whatever the EBITDA, and debt
    against a non-positive EBITDA is infinite (highly leveraged)."""
    debt, ebitda = np.asarray(debt, dtype=float), np.asarray(ebitda, dtype=float)
    leverage = np.divide(debt, ebitda, out=np.full(np.broadcast(debt, ebitda).shape, np.inf), where=ebitda > 0)
    return np.where(debt <= 0, 0.0, leverage)


def leverage_band(debt, ebitda, business: int = DEFAULT_BUSINESS_RISK):
    """The anchor's letter grade for debt / EBITDA at ``business`` risk."""
    leverage = leverage_of(debt, ebitda)
    grades = np.array([band_of(anchor(business, p)) for p in range(1, HIGHLY_LEVERAGED + 1)])
    return grades[leverage_profile(leverage) - 1]


def bands(ebit, ebitda, interest, debt, business: int = DEFAULT_BUSINESS_RISK):
    """Each year's band: the weaker of the coverage and the leverage read."""
    return np.maximum(coverage_band(ebit, interest), leverage_band(debt, ebitda, business))


def table_for(region: Optional[str]) -> str:
    """The block of S&P's study a region reads: its own (Table 25) where the
    study prints one, else the global table (Table 24)."""
    return region if region in base_rates.CUMULATIVE and region != "global" else "global"


@lru_cache(maxsize=None)
def hazards(table: str) -> np.ndarray:
    """Forward default rates (fractions), shape (bands, years printed): the
    chance a company in a band at an age defaults in that year, having
    survived to it."""
    rows = base_rates.CUMULATIVE[table]
    out = []
    for band in BANDS:
        c = np.concatenate([[0.0], np.asarray(rows[band], dtype=float)])
        out.append((c[1:] - c[:-1]) / (100.0 - c[:-1]))
    return np.array(out)


def probabilities(band_path, table: str) -> tuple[np.ndarray, np.ndarray]:
    """Yearly and cumulative default probability for a path of bands (the
    last axis is the year). Past the table's last year its last rate holds."""
    band_path = np.asarray(band_path, dtype=int)
    h = hazards(table)
    ages = np.minimum(np.arange(band_path.shape[-1]), h.shape[1] - 1)
    rate = h[band_path, ages]
    survive = np.cumprod(1.0 - rate, axis=-1)
    before = np.concatenate([np.ones(survive.shape[:-1] + (1,)), survive[..., :-1]], axis=-1)
    return before * rate, 1.0 - survive


# ---------------------------------------------------------------------------
# The card decides where the probabilities are shown
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _card() -> dict:
    return json.loads(CARD.read_text(encoding="utf-8"))


def card_result(region: Optional[str], card: Optional[Mapping] = None) -> dict:
    """The card's headline verdict for ``region`` (S&P's)."""
    ev = (card or _card())["evaluation"]
    group = ev["sets"][ev["headline_set"]]["by_region"].get(region) if region else None
    if not group:
        return {"verdict": "not_enough_data", "cases": 0, "model": None, "baseline": None}
    head = ev["headline_metric"]
    return {"verdict": group["verdict"], "cases": group["cases"],
            "model": (group.get("model") or {}).get(head), "baseline": (group.get("baseline") or {}).get(head)}


def _context(country: str, business: int, card: Optional[Mapping]) -> dict:
    region = base_rates.sp_region(country)
    result = card_result(region, card)
    return {"region": region, "table": table_for(region), "business_risk": int(business),
            "shown": result["verdict"] == "beats_baseline", "card": result,
            "sources": [{"id": s, **SOURCES[s]} for s in SOURCE_IDS]}


def _figure(x: float) -> Optional[float]:
    return float(x) if np.isfinite(x) else None


def deal_view(d, result, card: Optional[Mapping] = None) -> dict:
    """The deal's yearly coverage, leverage, band and (where shown) default
    probability. ``d`` is the deal in millions and ``result`` its model run;
    leases count as the deal values them (``core.risk_warnings.leverage_at_close``)."""
    from core.deal import leases_of
    om, ds = result.operating_model, result.debt_schedule
    terms = leases_of(d)
    ebit = np.asarray(om.ebit, dtype=float)
    interest = np.asarray(om.interest_expense, dtype=float)
    ebitda = np.asarray(om.ebitda, dtype=float) + terms.valuation_addback
    debt = np.asarray(ds.total_beginning_debt, dtype=float) + terms.debt_like
    business = getattr(d, "business_risk", DEFAULT_BUSINESS_RISK)
    ctx = _context(d.country, business, card)
    path = bands(ebit, ebitda, interest, debt, business)
    yearly, cumulative = probabilities(path, ctx["table"])
    cov, lev = coverage_band(ebit, interest), leverage_band(debt, ebitda, business)
    leverage = leverage_of(debt, ebitda)
    shown = ctx["shown"]
    years = []
    for t in range(len(path)):
        years.append({
            "year": t + 1,
            "coverage": _figure(ebit[t] / interest[t]) if interest[t] > 0 else None,
            "leverage": _figure(leverage[t]),
            "coverage_band": BANDS[cov[t]], "leverage_band": BANDS[lev[t]],
            "leverage_profile": int(leverage_profile(leverage[t])),
            "band": BANDS[path[t]],
            "probability": float(yearly[t]) if shown else None,
            "cumulative": float(cumulative[t]) if shown else None,
        })
    return {**ctx, "years": years}


def simulated_view(ebit, ebitda, interest, debt, *, country: str, business: int,
                   card: Optional[Mapping] = None) -> dict:
    """The simulation's version, from (paths, years) arrays: per year the
    share of paths in each band and (where shown) the mean probabilities."""
    ctx = _context(country, business, card)
    path = bands(ebit, ebitda, interest, debt, business)
    yearly, cumulative = probabilities(path, ctx["table"])
    shown = ctx["shown"]
    n = path.shape[0]
    years = []
    for t in range(path.shape[1]):
        counts = np.bincount(path[:, t], minlength=len(BANDS))
        years.append({
            "year": t + 1,
            "band_shares": {b: float(c / n) for b, c in zip(BANDS, counts)},
            "probability": float(yearly[:, t].mean()) if shown else None,
            "cumulative": float(cumulative[:, t].mean()) if shown else None,
        })
    return {**ctx, "years": years}
