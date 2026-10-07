"""Risk warnings computed from the deal and published data (PLAN.md 2.8).

They replace the anomaly detector's written-in statistics ("a 38% historical
distress rate", "only 2 of 9 such deals ..."), which had no source. A warning
here carries **numbers, never sentences**: the figures it shows, each computed
from this deal's own model run or read from a table in ``core/risk_sources.py``,
and the sources behind them. The words live in the web app's catalogue
(``warnings.<id>``), which a test keeps free of written-in numbers.

Four warnings, each raised only when the deal's own figures call for it:

* ``leverage_above_guidance`` -- debt at close above the 6.0x EBITDA that the
  ECB's and the US banking supervisors' leveraged-lending guidance flag;
* ``implied_rating`` -- year-one interest coverage (EBIT / interest) falls in
  a speculative-grade band of Damodaran's coverage-to-rating table, with S&P's
  average cumulative default rate for that rating over the deal's hold;
* ``interest_exceeds_ebitda`` -- a year whose EBITDA does not cover its
  interest;
* ``unfunded_repayment`` -- a year whose scheduled repayments exceed the cash
  the deal has, which the model fills with cash nobody provides (CLAUDE.md
  "Model findings", finding 11).
"""
from __future__ import annotations

from core.deal import leases_of
from core.risk_sources import (
    LEVERAGE_GUIDANCE_SOURCES, LEVERAGE_GUIDANCE_X, SOURCES, cumulative_default_pct,
    is_speculative, rating_for_coverage, study_row,
)

# A shortfall below this (in millions) is rounding in the engine's two-decimal
# money, not a missing payment.
ROUNDING = 0.005

# Money figures a warning carries, for the answer's unit conversion
# (core.deal.DEAL_MONEY_KEYS).
WARNING_MONEY_KEYS = frozenset({"unfunded", "unfunded_total", "repayment_due"})


def _warning(wid: str, figures: dict, sources, labels: dict | None = None) -> dict:
    return {
        "id": wid,
        "figures": {k: float(v) if v is not None else None for k, v in figures.items()},
        "labels": labels or {},
        "sources": [{"id": s, **SOURCES[s]} for s in sources],
    }


def leverage_at_close(d, result) -> float | None:
    """Total debt at close over EBITDA, both as the deal values them: with
    leases on the post-IFRS 16 view the liability is debt and the lease cost
    is added back, as the entry price does."""
    terms = leases_of(d)
    ebitda = terms.operating_ebitda + terms.valuation_addback
    if ebitda <= 0:
        return None
    return (result.debt_schedule.total_beginning_debt[0] + terms.debt_like) / ebitda


def _leverage_warning(d, result):
    leverage = leverage_at_close(d, result)
    if leverage is None or leverage <= LEVERAGE_GUIDANCE_X:
        return None
    return _warning("leverage_above_guidance",
                    {"leverage": leverage, "threshold": LEVERAGE_GUIDANCE_X},
                    LEVERAGE_GUIDANCE_SOURCES)


CREDIT_SOURCES = ("damodaran_ratings_2026", "sp_default_study_2024", "deal_model")


def credit_view(result) -> dict:
    """Year-one interest coverage (EBIT / interest), the rating Damodaran's
    table gives it and S&P's average cumulative default rate for that rating
    over the deal's hold: the coverage and default risk every deal summary
    shows (PLAN.md 4.6). Without interest there is no coverage to read, and
    every figure is None."""
    om = result.operating_model
    years = int(result.params.holding_period)
    interest = om.interest_expense[0]
    out = {"coverage": None, "rating": None, "study_row": None, "band_low": None, "band_high": None,
           "years": years, "default_pct": None, "speculative": None,
           "sources": [{"id": s, **SOURCES[s]} for s in CREDIT_SOURCES]}
    if interest <= 0:
        return out
    coverage = om.ebit[0] / interest
    rating, low, high = rating_for_coverage(coverage)
    return {**out, "coverage": float(coverage), "rating": rating, "study_row": study_row(rating),
            "band_low": None if low == float("-inf") else float(low), "band_high": high,
            "default_pct": cumulative_default_pct(rating, years), "speculative": is_speculative(rating)}


def _rating_warning(result):
    credit = credit_view(result)
    if not credit["speculative"]:
        return None
    return _warning(
        "implied_rating",
        {k: credit[k] for k in ("coverage", "band_low", "band_high", "years", "default_pct")},
        CREDIT_SOURCES,
        {"rating": credit["rating"], "study_row": credit["study_row"]},
    )


def _coverage_warning(result):
    om = result.operating_model
    short = [(t, e / i) for t, (e, i) in enumerate(zip(om.ebitda, om.interest_expense)) if i > 0 and e < i]
    if not short:
        return None
    worst_year, worst = min(short, key=lambda s: s[1])
    return _warning("interest_exceeds_ebitda",
                    {"year": short[0][0] + 1, "worst_year": worst_year + 1,
                     "coverage": worst, "count": len(short)},
                    ("deal_model",))


def unfunded_repayments(result) -> list[float]:
    """Per year, how much of the scheduled repayments neither the year's cash
    flow, the cash above the minimum carried in, nor a revolver draw paid for.

    The debt model's step 7 resets cash to the minimum whatever happened
    (finding 11); this is the amount it conjured to do so.
    """
    ds, cf = result.debt_schedule, result.cash_flow
    minimum = result.params.minimum_cash
    redrawn = [sum(rows[t].redrawn for rows in ds.schedule.values()) for t in range(len(ds.years))]
    gaps = []
    for t in range(len(ds.years)):
        carried = ds.cash_balance[t - 1] - minimum if t else 0.0
        gap = ds.total_mandatory_repayment[t] - cf.levered_fcf[t] - carried - redrawn[t]
        gaps.append(gap if gap > ROUNDING else 0.0)
    return gaps


def _unfunded_warning(result):
    gaps = unfunded_repayments(result)
    years = [t for t, g in enumerate(gaps) if g > 0]
    if not years:
        return None
    first = years[0]
    return _warning("unfunded_repayment",
                    {"year": first + 1, "unfunded": gaps[first],
                     "repayment_due": result.debt_schedule.total_mandatory_repayment[first],
                     "count": len(years), "unfunded_total": sum(gaps)},
                    ("deal_model",))


def risk_warnings(d, result) -> list[dict]:
    """The warnings this deal raises, in the order above. ``d`` is the deal in
    millions (``core.deal.in_millions``) and ``result`` its model run."""
    found = (_leverage_warning(d, result), _rating_warning(result),
             _coverage_warning(result), _unfunded_warning(result))
    return [w for w in found if w is not None]
