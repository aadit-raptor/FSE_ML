"""The lender's view of a deal (PLAN.md 7.5): each facility's exposure,
default probability, loss given default and expected loss, from published
base rates for the deal's region.

* **Default probability** -- the deal's implied rating from its year-one
  interest coverage (``core.risk_warnings.credit_view``, Damodaran's table),
  folded to S&P's letter grade, read in S&P's 2024 study at the deal's hold:
  Table 25 for the US, Europe and emerging markets, Table 24 (global) for
  other developed markets and a deal with no country, the tables the distress
  predictor reads (``ml/distress_model.py`` ``table_for``). A published rate,
  not an estimate: like the summary's default risk, it is always shown.
* **Loss given default** -- Global Credit Data's 2020 report on bank loans to
  large corporates, Table 2, by the facility's seniority: secured loans 22%,
  senior unsecured 26%, subordinated 38%. The report does not split secured
  loans by lien, so a second lien reads the secured row, and the screen says
  so. A shareholder loan is the sponsor's and has no lender's loss.
* **Exposure** -- what the facility can owe at close: its drawn amount, a
  revolver's whole commitment (it is drawn when a borrower is in trouble).

Expected loss = probability x loss given default x exposure; the expected
recovery is what is left of the exposure if it defaults. The tables are
transcribed in ``library/base_rates.py``; nothing here is fitted.
"""
from __future__ import annotations

from typing import Optional

from core.covenants import deal_facilities
from core.risk_warnings import credit_view
from library import base_rates
from ml.distress_model import band_of, table_for

# Global Credit Data's Table 2 row for each seniority
GCD_ROW = {
    "senior_secured": "secured",
    "second_lien": "secured",
    "senior_unsecured": "unsecured_senior",
    "subordinated": "unsecured_subordinated",
}
SOURCE_IDS = ("damodaran_ratings_2026", "sp_default_study_2024_regions", "gcd_lgd_2020", "deal_model")
LENDER_MONEY_KEYS = frozenset({"exposure", "expected_loss", "expected_recovery", "liquidity",
                               "minimum_liquidity"})


def default_pct(rating: Optional[str], table: str, years: int) -> Optional[float]:
    """S&P's average cumulative default rate (%) for ``rating``'s letter grade
    after ``years`` in ``table`` (its last column for a longer hold)."""
    if rating is None:
        return None
    row = base_rates.CUMULATIVE[table][("AAA", "AA", "A", "BBB", "BB", "B", "CCC/C")[band_of(rating)]]
    return float(row[min(max(int(years), 1), len(row)) - 1])


def _sources():
    from core.risk_sources import SOURCES
    out = []
    for s in SOURCE_IDS:
        src = SOURCES.get(s) or {**base_rates.SOURCES[s], "detail": "Table 2: seniority and collateral"}
        out.append({"id": s, **{k: v for k, v in src.items()
                                if k in ("publisher", "title", "published", "url", "detail", "sample")}})
    return out


def expected_loss(d, result) -> dict:
    """Each facility's exposure, default probability, loss given default,
    expected loss and recovery (money in millions)."""
    credit = credit_view(result)
    region = base_rates.sp_region(d.country)
    table = table_for(region)
    years = int(result.params.holding_period)
    pd = default_pct(credit["rating"], table, years)
    rows = []
    for fac, records in zip(deal_facilities(d, result), result.debt_schedule.schedule.values()):
        exposure = fac.commitment if fac.commitment is not None else (records[0].beginning_balance if records else 0.0)
        lender = fac.seniority != "sponsor"
        lgd = base_rates.LGD_BY_SENIORITY[GCD_ROW[fac.seniority]][1] if lender else None
        loss = pd / 100 * lgd / 100 * exposure if lender and pd is not None else None
        rows.append({
            "name": fac.name, "kind": fac.kind, "seniority": fac.seniority, "lender": lender,
            "exposure": float(exposure), "lgd_pct": lgd, "lgd_row": GCD_ROW.get(fac.seniority) if lender else None,
            "default_pct": pd if lender else None,
            "expected_loss": loss,
            "expected_recovery": float(exposure * (1 - lgd / 100)) if lender else None,
        })
    lent = [r for r in rows if r["lender"]]
    total = sum(r["expected_loss"] for r in lent) if lent and pd is not None else None
    exposure = sum(r["exposure"] for r in lent)
    return {
        "rating": credit["rating"], "band": None if credit["rating"] is None else
        ("AAA", "AA", "A", "BBB", "BB", "B", "CCC/C")[band_of(credit["rating"])],
        "region": region, "table": table, "years": years, "default_pct": pd,
        "facilities": rows, "exposure": exposure, "expected_loss": total,
        "loss_rate": total / exposure if total is not None and exposure > 0 else None,
        "sources": _sources(),
    }
