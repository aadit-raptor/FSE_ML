"""
Fetches financial data from SEC EDGAR XBRL API.
No API key required. Rate limit: 10 requests/second.
Always set User-Agent header per SEC requirements.
"""

import requests
import pandas as pd
import numpy as np
import time
import re
from typing import Optional, Dict, List
from dataclasses import dataclass

EDGAR_BASE   = "https://data.sec.gov"
COMPANY_URL  = f"{EDGAR_BASE}/submissions/CIK{{cik}}.json"
FACTS_URL    = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{{cik}}.json"
SEARCH_URL   = "https://efts.sec.gov/LATEST/search-index"
TICKER_URL   = f"{EDGAR_BASE}/files/company_tickers.json"

HEADERS = {
    "User-Agent": "SimulationModel research@simulationmodel.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov"
}


# XBRL tag mappings: field_name -> list of possible GAAP tags in priority order
TAG_MAP = {
    'revenue': [
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'SalesRevenueNet',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
    ],
    'cost_of_revenue': [
        'CostOfRevenue',
        'CostOfGoodsSold',
        'CostOfGoodsAndServicesSold',
    ],
    'gross_profit': [
        'GrossProfit',
    ],
    'research_and_development': [
        'ResearchAndDevelopmentExpense',
        'ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost',
    ],
    'selling_general_admin': [
        'SellingGeneralAndAdministrativeExpense',
        'GeneralAndAdministrativeExpense',
    ],
    'operating_income': [
        'OperatingIncomeLoss',
    ],
    'interest_expense': [
        'InterestExpense',
        'InterestAndDebtExpense',
    ],
    'interest_income': [
        'InterestAndDividendIncomeOperating',
        'InvestmentIncomeInterest',
    ],
    'income_tax_expense': [
        'IncomeTaxExpenseBenefit',
    ],
    'net_income': [
        'NetIncomeLoss',
        'NetIncomeLossAvailableToCommonStockholdersBasic',
    ],
    'depreciation_amortization': [
        'DepreciationDepletionAndAmortization',
        'DepreciationAndAmortization',
        'Depreciation',
    ],
    'stock_based_compensation': [
        'ShareBasedCompensation',
        'AllocatedShareBasedCompensationExpense',
    ],
    'capital_expenditures': [
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'CapitalExpendituresIncurredButNotYetPaid',
    ],
    'cash_and_equivalents': [
        'CashAndCashEquivalentsAtCarryingValue',
        'CashCashEquivalentsAndShortTermInvestments',
    ],
    'accounts_receivable': [
        'AccountsReceivableNetCurrent',
        'ReceivablesNetCurrent',
    ],
    'inventories': [
        'InventoryNet',
        'InventoryGross',
    ],
    'other_current_assets': [
        'OtherAssetsCurrent',
        'PrepaidExpenseAndOtherAssetsCurrent',
    ],
    'property_plant_equipment': [
        'PropertyPlantAndEquipmentNet',
    ],
    'other_noncurrent_assets': [
        'OtherAssetsNoncurrent',
        'IntangibleAssetsNetExcludingGoodwill',
    ],
    'accounts_payable': [
        'AccountsPayableCurrent',
    ],
    'other_current_liabilities': [
        'OtherLiabilitiesCurrent',
        'AccruedLiabilitiesCurrent',
    ],
    'deferred_revenue': [
        'DeferredRevenueCurrent',
        'ContractWithCustomerLiabilityCurrent',
    ],
    'long_term_debt': [
        'LongTermDebt',
        'LongTermDebtNoncurrent',
        'SeniorNotes',
    ],
    'common_stock_equity': [
        'StockholdersEquity',
        'CommonStockholdersEquity',
    ],
    'retained_earnings': [
        'RetainedEarningsAccumulatedDeficit',
    ],
    'dividends_paid': [
        'PaymentsOfDividendsCommonStock',
        'PaymentsOfDividends',
    ],
    'share_repurchases': [
        'PaymentsForRepurchaseOfCommonStock',
    ],
}


@dataclass
class ExtractedFinancials:
    ticker: str
    company_name: str
    years: List[int]
    data: Dict[str, List[float]]  # field -> [year1_val, year2_val, ...]
    warnings: List[str]


def _get_cik_from_ticker(ticker: str) -> Optional[str]:
    """Look up CIK number for a given stock ticker."""
    try:
        resp = requests.get(TICKER_URL, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        ticker_upper = ticker.upper()
        for _, company_data in data.items():
            if company_data.get('ticker', '').upper() == ticker_upper:
                return str(company_data['cik_str']).zfill(10)
        return None
    except Exception as e:
        raise ValueError(f"Could not find CIK for ticker '{ticker}': {e}")


def _get_annual_values(facts_data: dict, tag: str, 
                        years_wanted: List[int]) -> Optional[List[float]]:
    """
    Extract annual values for a specific XBRL tag.
    Returns values indexed to years_wanted list (None where missing).
    """
    us_gaap = facts_data.get('facts', {}).get('us-gaap', {})
    if tag not in us_gaap:
        return None

    tag_data = us_gaap[tag]
    units = tag_data.get('units', {})

    # Most financial data is in USD
    usd_data = units.get('USD', [])
    if not usd_data:
        return None

    # Filter for annual 10-K filings only
    annual = [
        entry for entry in usd_data
        if entry.get('form') in ('10-K', '10-K/A')
        and entry.get('fp') == 'FY'
        and 'end' in entry
    ]

    if not annual:
        return None

    # Group by fiscal year end date, take the most recent filing for each year
    by_year = {}
    for entry in annual:
        year = int(entry['end'][:4])
        if year not in by_year:
            by_year[year] = entry
        else:
            # Prefer the most recently filed version
            if entry.get('filed', '') > by_year[year].get('filed', ''):
                by_year[year] = entry

    # Build output aligned to years_wanted
    result = []
    for yr in years_wanted:
        if yr in by_year:
            result.append(by_year[yr]['val'] / 1e6)  # Convert to $M
        else:
            result.append(None)

    return result


def fetch_financials(ticker: str, n_years: int = 5) -> ExtractedFinancials:
    """
    Main function: fetch and return the last n_years of annual financials
    for the given ticker symbol.

    Parameters
    ----------
    ticker : str  e.g. 'MCD', 'KO', 'AAPL'
    n_years : int  how many years to fetch (default 5 for Page 7)

    Returns
    -------
    ExtractedFinancials object with all available data
    """
    warnings = []

    # Step 1: Get CIK
    print(f"Looking up CIK for {ticker}...")
    cik = _get_cik_from_ticker(ticker)
    if not cik:
        raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR.")

    # Step 2: Get company name
    time.sleep(0.1)  # Respect rate limit
    company_url = COMPANY_URL.format(cik=cik)
    resp = requests.get(company_url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    company_info = resp.json()
    company_name = company_info.get('name', ticker)

    # Step 3: Determine fiscal years to fetch
    current_year = pd.Timestamp.now().year
    years_wanted = list(range(current_year - n_years, current_year))

    # Step 4: Fetch all financial facts
    print(f"Fetching XBRL data for {company_name} (CIK: {cik})...")
    time.sleep(0.1)
    facts_url = FACTS_URL.format(cik=cik)
    resp = requests.get(facts_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    facts_data = resp.json()

    # Step 5: Extract each field using TAG_MAP priority order
    extracted = {}
    for field_name, tags in TAG_MAP.items():
        for tag in tags:
            values = _get_annual_values(facts_data, tag, years_wanted)
            if values and any(v is not None for v in values):
                extracted[field_name] = values
                break
        if field_name not in extracted:
            warnings.append(f"Could not extract '{field_name}' — will show as 0")
            extracted[field_name] = [0.0] * n_years

    # Step 6: Fill None values with 0 and validate
    for field in extracted:
        extracted[field] = [
            v if v is not None else 0.0 
            for v in extracted[field]
        ]

    # Step 7: Derived calculations where direct tags unavailable
    # Compute gross profit from revenue - COGS if not available directly
    if all(v == 0 for v in extracted.get('gross_profit', [0]*n_years)):
        if any(v != 0 for v in extracted.get('revenue', [0]*n_years)):
            extracted['gross_profit'] = [
                r - c for r, c in zip(
                    extracted['revenue'], 
                    extracted['cost_of_revenue']
                )
            ]

    # Step 8: Validate balance sheet roughly balances (sanity check)
    latest_assets = (
        extracted.get('cash_and_equivalents', [0]*n_years)[-1] +
        extracted.get('accounts_receivable', [0]*n_years)[-1] +
        extracted.get('inventories', [0]*n_years)[-1] +
        extracted.get('property_plant_equipment', [0]*n_years)[-1]
    )
    latest_equity = extracted.get('common_stock_equity', [0]*n_years)[-1]
    if latest_assets > 0 and latest_equity > latest_assets * 2:
        warnings.append("Balance sheet check: equity exceeds total assets — verify data")

    print(f"Successfully extracted {len(extracted)} fields for {company_name}")
    if warnings:
        print(f"Warnings: {warnings}")

    return ExtractedFinancials(
        ticker=ticker,
        company_name=company_name,
        years=years_wanted,
        data=extracted,
        warnings=warnings,
    )


def financials_to_session_state(extracted: ExtractedFinancials) -> dict:
    """
    Convert ExtractedFinancials to the session state keys expected by
    pages/forecasting.py _hist_input_block() function.
    Returns dict of {session_key: value} for st.session_state population.
    """
    d = extracted.data
    n = len(extracted.years)

    def safe(field, default=0.0):
        vals = d.get(field, [default]*n)
        return [v if v is not None else default for v in vals]

    # Map to the hist_ keys used in forecasting.py
    # Each value is a list of n_years floats
    mapping = {}
    for j in range(n):
        mapping[f'hist_h_rev_{j}']       = safe('revenue')[j]
        mapping[f'hist_h_cogs_{j}']      = -abs(safe('cost_of_revenue')[j])
        mapping[f'hist_h_rd_{j}']        = -abs(safe('research_and_development')[j])
        mapping[f'hist_h_sga_{j}']       = -abs(safe('selling_general_admin')[j])
        mapping[f'hist_h_int_exp_{j}']   = -abs(safe('interest_expense')[j])
        mapping[f'hist_h_int_inc_{j}']   = safe('interest_income')[j]
        mapping[f'hist_h_other_{j}']     = 0.0
        mapping[f'hist_h_tax_{j}']       = -abs(safe('income_tax_expense')[j])
        mapping[f'hist_h_da_{j}']        = safe('depreciation_amortization')[j]
        mapping[f'hist_h_sbc_{j}']       = safe('stock_based_compensation')[j]
        # Balance sheet — only use most recent year (j == n-1)
        if j == n - 1:
            mapping['hist_h_cash_0']     = safe('cash_and_equivalents')[j]
            mapping['hist_h_ar_0']       = safe('accounts_receivable')[j]
            mapping['hist_h_inv_0']      = safe('inventories')[j]
            mapping['hist_h_ocurr_0']    = safe('other_current_assets')[j]
            mapping['hist_h_ppe_0']      = safe('property_plant_equipment')[j]
            mapping['hist_h_nca_0']      = safe('other_noncurrent_assets')[j]
            mapping['hist_h_ap_0']       = safe('accounts_payable')[j]
            mapping['hist_h_ocl_0']      = safe('other_current_liabilities')[j]
            mapping['hist_h_def_0']      = safe('deferred_revenue')[j]
            mapping['hist_h_ltd_0']      = safe('long_term_debt')[j]
            mapping['hist_h_cs_0']       = max(
                safe('common_stock_equity')[j] - safe('retained_earnings')[j], 0
            )
            mapping['hist_h_re_0']       = safe('retained_earnings')[j]
            mapping['hist_h_oci_0']      = 0.0
        mapping[f'hist_h_capex_{j}']     = safe('capital_expenditures')[j]
        mapping[f'hist_h_divs_{j}']      = safe('dividends_paid')[j]
        mapping[f'hist_h_buybacks_{j}']  = safe('share_repurchases')[j]

    return mapping