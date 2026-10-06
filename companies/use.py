"""A company's summary figures as a deal's inputs or a forecast's history
(PLAN.md 4.1b).

The summary (companies/items.py) is what every source gives, so these work
the same for a US 10-K, an ESEF report, UK accounts or a Japanese annual
report. Neither invents a figure the filing lacks:

- ``deal_inputs``: the latest year's EBITDA as the standard reports it
  (operating income plus D&A, as the EDGAR answer gives it), the currency,
  and the standard and leases when the deal knows the standard (IFRS or US
  GAAP; core/accounting.py). ``notes`` says what the reader must add.
- ``forecast_history``: every forecasting history row (core/forecasting.py),
  oldest year first. The summary has no cost breakdown, so every operating
  cost is in cost of sales (EBIT and EBITDA are the filing's), the assets
  the summary doesn't name are in PP&E, the liabilities it doesn't name in
  other non-current liabilities, equity in common stock, and whatever lies
  between operating profit and net income in other income. The balance
  sheet balances by construction. A figure the filing doesn't report is 0
  (PP&E never below 0), and the answer's warnings name it.

The sign conventions are the forecast's: costs, interest and tax negative,
D&A and capex positive.
"""
from __future__ import annotations

from typing import Optional

from companies.model import IFRS, US_GAAP, CompanyData

# The standards a deal knows (core/accounting.py); others become "" there
DEAL_STANDARDS = (IFRS, US_GAAP)

# Every row the forecast reads (web/src/lib/forecast.ts HISTORY_GROUPS)
HISTORY_ROWS = (
    "h_rev", "h_cogs", "h_rd", "h_sga", "h_int_inc", "h_int_exp", "h_other", "h_tax", "h_da", "h_sbc",
    "h_cash", "h_ar", "h_inv", "h_ocurr", "h_ppe", "h_nca", "h_lta",
    "h_ap", "h_ocl", "h_def", "h_ltd", "h_ncl", "h_cs", "h_re", "h_oci",
    "h_capex", "h_divs", "h_buybacks",
)


def _plus(x: float) -> float:
    """0.0 for -0.0, so an answer never shows a minus zero."""
    return x + 0.0


def deal_inputs(data: CompanyData) -> Optional[dict]:
    """The latest year as a deal's inputs; None when it has no EBITDA."""
    if not data.years:
        return None
    latest = data.years[-1]
    ebitda = latest.figures.get("ebitda")
    if ebitda is None:
        return None
    notes = []
    known = data.accounting_standard in DEAL_STANDARDS
    lease_cost = latest.figures.get("lease_cost") if known else None
    lease_liability = latest.figures.get("lease_liability") if known else None
    if not known:
        notes.append("standard_not_in_deal")
    elif lease_liability and lease_cost is None:
        notes.append("lease_cost_missing")
    return {
        "ebitda": ebitda,
        "currency": data.currency,
        "unit": data.unit,
        "accounting_standard": data.accounting_standard if known else "",
        "lease_cost": lease_cost or 0.0,
        "lease_liability": lease_liability or 0.0,
        "fiscal_year": latest.fiscal_year,
        "notes": notes,
    }


def _year_rows(f: dict) -> dict:
    def get(name: str) -> float:
        return f.get(name) or 0.0

    revenue, ebit = get("revenue"), get("operating_income")
    interest, tax = get("interest_expense"), get("income_tax_expense")
    cash, receivables, inventories = get("cash_and_equivalents"), get("accounts_receivable"), get("inventories")
    # Never below nothing: a filing without total assets leaves PP&E at 0
    other_assets = max(get("total_assets") - cash - receivables - inventories, 0.0)
    payables, debt, equity = get("accounts_payable"), get("total_debt"), get("total_equity")
    rows = dict.fromkeys(HISTORY_ROWS, 0.0)
    rows.update({
        "h_rev": revenue,
        "h_cogs": -(revenue - ebit),
        "h_int_exp": -interest,
        "h_tax": -tax,
        # Net income = EBIT - interest - tax + other, so other is the rest
        "h_other": get("net_income") - (ebit - interest - tax),
        "h_da": get("depreciation_amortization"),
        "h_cash": cash, "h_ar": receivables, "h_inv": inventories,
        "h_ppe": other_assets,
        "h_ap": payables, "h_ltd": debt,
        "h_ncl": cash + receivables + inventories + other_assets - payables - debt - equity,
        "h_cs": equity,
        "h_capex": get("capital_expenditures"),
    })
    return {k: _plus(v) for k, v in rows.items()}


def forecast_history(data: CompanyData) -> Optional[dict]:
    """Forecasting history rows, oldest year first; None when a year has no
    revenue (the forecast reads every ratio off it)."""
    if not data.years or any(y.figures.get("revenue") is None for y in data.years):
        return None
    years = [_year_rows(y.figures) for y in data.years]
    return {row: [y[row] for y in years] for row in HISTORY_ROWS}
