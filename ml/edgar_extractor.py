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
# company_tickers.json is served from www.sec.gov, not data.sec.gov
TICKER_URL   = "https://www.sec.gov/files/company_tickers.json"

HEADERS = {
    "User-Agent": "SimulationModel research@simulationmodel.com",
    "Accept-Encoding": "gzip, deflate",
    # No hardcoded Host: requests derives it per-URL. Pinning it to
    # data.sec.gov routed the www.sec.gov ticker lookup to the wrong vhost.
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
    ],
    # Filers that tag the two halves separately (MSFT); summed into SG&A
    'selling_marketing': ['SellingAndMarketingExpense'],
    'general_admin': ['GeneralAndAdministrativeExpense'],
    'operating_income': [
        'OperatingIncomeLoss',
    ],
    'interest_expense': [
        'InterestExpense',
        'InterestAndDebtExpense',
        'InterestExpenseNonoperating',
        'InterestExpenseDebt',
    ],
    'interest_income': [
        'InterestAndDividendIncomeOperating',
        'InvestmentIncomeInterest',
        'InvestmentIncomeInterestAndDividend',
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
        'DepreciationAmortizationAndAccretionNet',
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
        'AccountsNotesAndLoansReceivableNetCurrent',
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
        'AccountsPayableTradeCurrent',
        'AccountsPayableAndAccruedLiabilitiesCurrent',
    ],
    'other_current_liabilities': [
        'OtherLiabilitiesCurrent',
        'AccruedLiabilitiesCurrent',
    ],
    'deferred_revenue': [
        'DeferredRevenueCurrent',
        'ContractWithCustomerLiabilityCurrent',
    ],
    'debt_total': [                      # includes current maturities
        'LongTermDebt',
        'LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities',
    ],
    'debt_noncurrent': [
        'LongTermDebtNoncurrent',
        'LongTermDebtAndCapitalLeaseObligations',
    ],
    'debt_current': [
        'LongTermDebtCurrent',
        'LongTermDebtAndCapitalLeaseObligationsCurrent',
        'DebtCurrent',
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
    # Reported totals: the balance sheet is reconciled to these
    'total_assets': ['Assets'],
    'total_liabilities': ['Liabilities'],
    'total_current_liabilities': ['LiabilitiesCurrent'],
    'total_liabilities_and_equity': ['LiabilitiesAndStockholdersEquity'],
    'equity_incl_nci': [
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    ],
    'aoci': ['AccumulatedOtherComprehensiveIncomeLossNetOfTax'],
    # For filers with no cost-of-revenue concept (e.g. restaurants)
    'costs_and_expenses': ['CostsAndExpenses'],
}

# Helper / derivation inputs whose absence is normal for many filers, so a
# missing tag is not reported to the user as a data gap.
SUPPORT_FIELDS = {
    'gross_profit', 'operating_income', 'net_income',
    'debt_total', 'debt_noncurrent', 'debt_current',
    'total_liabilities', 'total_current_liabilities',
    'total_liabilities_and_equity', 'equity_incl_nci', 'aoci',
    'costs_and_expenses', 'selling_marketing', 'general_admin',
}


@dataclass
class ExtractedFinancials:
    ticker: str
    company_name: str
    years: List[int]
    data: Dict[str, List[float]]  # field -> [year1_val, year2_val, ...]
    warnings: List[str]
    # Month the filer's fiscal year ends (PLAN.md 2.3a); None if unknown
    fiscal_year_end_month: Optional[int] = None


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
            result.append(by_year[yr]['val'] / 1e6)  # Convert to millions
        else:
            result.append(None)

    return result


def _latest_10k_end(facts_data: dict) -> Optional[str]:
    """Period-end date (YYYY-MM-DD) of the latest 10-K balance sheet."""
    us_gaap = facts_data.get('facts', {}).get('us-gaap', {})
    for tag in ('Assets', 'LiabilitiesAndStockholdersEquity'):
        ends = [e['end'] for e in us_gaap.get(tag, {}).get('units', {}).get('USD', [])
                if e.get('form') in ('10-K', '10-K/A')
                and e.get('fp') == 'FY' and 'end' in e]
        if ends:
            return max(ends)
    return None


# A 52/53-week year ends on a weekday near the month end, sometimes a few
# days into the next month (Apple's 2025-10-04 is its September year)
_FIRST_WEEK = 7


def _fiscal_year_end(facts_data: dict) -> Optional[tuple]:
    """(year, month) the filer's latest fiscal year really ends in, or None.

    Fiscal years are named by the year they end in, so a 52/53-week year
    ending 2025-01-03 is the fiscal year that ended December 2024.
    """
    end = _latest_10k_end(facts_data)
    if end is None:
        return None
    year, month, day = int(end[:4]), int(end[5:7]), int(end[8:10])
    if day <= _FIRST_WEEK:
        year, month = (year - 1, 12) if month == 1 else (year, month - 1)
    return year, month


def _latest_fiscal_year(facts_data: dict) -> Optional[int]:
    """Latest fiscal year (by period-end year) with a 10-K balance sheet."""
    us_gaap = facts_data.get('facts', {}).get('us-gaap', {})
    for tag in ('Assets', 'LiabilitiesAndStockholdersEquity'):
        rows = [e for e in us_gaap.get(tag, {}).get('units', {}).get('USD', [])
                if e.get('form') in ('10-K', '10-K/A')
                and e.get('fp') == 'FY' and 'end' in e]
        if rows:
            return max(int(e['end'][:4]) for e in rows)
    return None


def _reconcile_balance_sheet(x: dict, n: int, warnings: List[str]) -> None:
    """
    Rebuild the balance sheet around the filer's reported totals, in place.

    Tagged line items alone never cover a real balance sheet -- goodwill,
    intangibles, lease and deferred-tax balances have no line in the model --
    and the old equity split, max(equity - retained earnings, 0), clamped away
    treasury stock. So:

      goodwill & other LT assets = total assets - the tagged asset lines
                                  (other non-current assets keeps its own tag,
                                  because the forecast grows it with revenue)
      long-term debt            = all funded debt, current maturities included
                                  (so it bears interest in the forecast)
      other current liabilities = current liabilities - AP - deferred revenue
                                  - current debt
      other non-current liab.   = total liabilities - everything above
      common stock              = total equity - retained earnings - AOCI
                                  (negative when treasury stock is large)

    Assets then equal liabilities plus equity exactly, and each total matches
    the 10-K.
    """
    ltd = [0.0] * n
    lta = [0.0] * n
    ocl = list(x['other_current_liabilities'])
    ncl = [0.0] * n
    cs = [0.0] * n
    negatives = set()

    for j in range(n):
        total_debt, cur_debt = x['debt_total'][j], x['debt_current'][j]
        noncur_debt = x['debt_noncurrent'][j]
        if total_debt:
            if cur_debt and noncur_debt and abs(total_debt - noncur_debt) < 0.5:
                # Some filers' "total" tag equals the non-current figure, i.e.
                # it excludes current maturities (MCD); add them back.
                ltd[j] = total_debt + cur_debt
            else:
                if not cur_debt and noncur_debt:
                    cur_debt = max(total_debt - noncur_debt, 0.0)
                ltd[j] = total_debt
        else:
            ltd[j] = x['debt_noncurrent'][j] + cur_debt

        equity_tag = x['equity_incl_nci'][j] or x['common_stock_equity'][j]
        re, aoci = x['retained_earnings'][j], x['aoci'][j]
        assets = x['total_assets'][j]
        if not assets:
            # No reported total for this year: keep the tagged lines as-is
            cs[j] = equity_tag - re - aoci
            continue

        total_liab = (x['total_liabilities'][j]
                      or (x['total_liabilities_and_equity'][j] or assets) - equity_tag)
        equity = assets - total_liab     # includes NCI / temporary equity

        lta[j] = assets - (x['cash_and_equivalents'][j] + x['accounts_receivable'][j]
                           + x['inventories'][j] + x['other_current_assets'][j]
                           + x['property_plant_equipment'][j]
                           + x['other_noncurrent_assets'][j])
        ap, dr = x['accounts_payable'][j], x['deferred_revenue'][j]
        cur_liab = x['total_current_liabilities'][j]
        if cur_liab:
            ocl[j] = cur_liab - ap - dr - cur_debt
        ncl[j] = total_liab - (ap + ocl[j] + dr + ltd[j])
        cs[j] = equity - re - aoci

        for name, v in (("goodwill & other long-term assets", lta[j]),
                        ("other current liabilities", ocl[j]),
                        ("other non-current liabilities", ncl[j])):
            if v < -0.5:
                negatives.add(name)

    x['long_term_debt'] = ltd
    x['other_longterm_assets'] = lta
    x['other_current_liabilities'] = ocl
    x['other_noncurrent_liabilities'] = ncl
    x['common_stock'] = cs

    if any(x['total_assets']):
        # These lines are now residuals of reported totals, not missing data
        for f in ('other_noncurrent_assets', 'other_current_liabilities'):
            warnings[:] = [w for w in warnings if f"'{f}'" not in w]
    else:
        warnings.append("No total-assets tag reported: balance sheet lines are "
                        "used as tagged and may not balance")
    if not any(ltd):
        warnings.append("Could not extract 'long_term_debt' — will show as 0")
    if negatives:
        warnings.append(f"Derived {', '.join(sorted(negatives))} came out "
                        f"negative; a tag may be double counted — check the "
                        f"balance sheet inputs")


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

    # Step 3: Fetch all financial facts
    print(f"Fetching XBRL data for {company_name} (CIK: {cik})...")
    time.sleep(0.1)
    facts_url = FACTS_URL.format(cik=cik)
    resp = requests.get(facts_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    facts_data = resp.json()

    # Step 4: Fiscal years to fetch, ending at the latest 10-K actually filed.
    # A calendar window (previous n years) missed fiscal years that end
    # mid-year (MSFT's FY ending June 2026 in September 2026) and, early in a
    # year, would request a year not yet filed.
    latest_fy = _latest_fiscal_year(facts_data) or (pd.Timestamp.now().year - 1)
    years_wanted = list(range(latest_fy - n_years + 1, latest_fy + 1))

    # Step 5: Extract each field using TAG_MAP priority order
    # Each year takes the highest-priority tag with a value for THAT year.
    # Taking the first tag with any value in the window gave zeros for recent
    # years whenever a filer switched tags mid-window (KO's debt moved off
    # LongTermDebt after 2023).
    extracted = {}
    for field_name, tags in TAG_MAP.items():
        merged = [None] * n_years
        for tag in tags:
            values = _get_annual_values(facts_data, tag, years_wanted)
            if values:
                merged = [m if m is not None else v for m, v in zip(merged, values)]
            if all(m is not None for m in merged):
                break
        if any(m is not None for m in merged):
            extracted[field_name] = merged
        if field_name not in extracted:
            if field_name not in SUPPORT_FIELDS:
                warnings.append(f"Could not extract '{field_name}' — will show as 0")
            extracted[field_name] = [0.0] * n_years

    # Step 6: Fill None values with 0 and validate
    for field in extracted:
        extracted[field] = [
            v if v is not None else 0.0 
            for v in extracted[field]
        ]

    # Step 7: Derived calculations where direct tags unavailable
    # SG&A: some filers tag selling & marketing and G&A separately instead of
    # a combined line. Taking G&A alone understated MSFT's costs by USD 26.7B.
    combined_sga = [sm + ga for sm, ga in zip(extracted['selling_marketing'],
                                              extracted['general_admin'])]
    if any(v == 0 and c != 0 for v, c in zip(extracted['selling_general_admin'],
                                             combined_sga)):
        extracted['selling_general_admin'] = [
            v or c for v, c in zip(extracted['selling_general_admin'], combined_sga)]
        warnings = [w for w in warnings if "'selling_general_admin'" not in w]

    # Filers with no cost-of-revenue concept (e.g. restaurants) report only
    # total costs and expenses. Derive COGS so that revenue - COGS - R&D - SG&A
    # equals reported operating income.
    if (all(v == 0 for v in extracted['cost_of_revenue'])
            and any(v != 0 for v in extracted['costs_and_expenses'])):
        extracted['cost_of_revenue'] = [
            max(c - sga - rd, 0.0) for c, sga, rd in zip(
                extracted['costs_and_expenses'],
                extracted['selling_general_admin'],
                extracted['research_and_development'])
        ]
        warnings = [w for w in warnings if "'cost_of_revenue'" not in w]
        warnings.append("Cost of revenue derived from total costs and expenses "
                        "less SG&A and R&D (no COGS tag reported): operating "
                        "income is preserved, gross margin is approximate")

    # Reconcile to reported operating income. Operating costs the tags miss
    # (KO's other operating charges, impairments, restructuring) are folded
    # into SG&A so revenue - COGS - R&D - SG&A matches the 10-K.
    folded = []
    for j in range(n_years):
        rev, op_inc = extracted['revenue'][j], extracted['operating_income'][j]
        if not rev or not op_inc:
            continue
        gap = (rev - extracted['cost_of_revenue'][j]
               - extracted['research_and_development'][j]
               - extracted['selling_general_admin'][j]) - op_inc
        if abs(gap) > 0.5 and extracted['selling_general_admin'][j] + gap >= 0:
            extracted['selling_general_admin'][j] += gap
            folded.append((years_wanted[j], gap))
    if folded:
        fy, amt = folded[-1]
        warnings.append(f"Operating costs not separately tagged ({amt:,.0f}M in "
                        f"FY{fy}) folded into SG&A so operating income matches "
                        f"the 10-K")

    # Compute gross profit from revenue - COGS if not available directly
    if all(v == 0 for v in extracted.get('gross_profit', [0]*n_years)):
        if any(v != 0 for v in extracted.get('revenue', [0]*n_years)):
            extracted['gross_profit'] = [
                r - c for r, c in zip(
                    extracted['revenue'], 
                    extracted['cost_of_revenue']
                )
            ]

    # Step 8: Reconcile the balance sheet to the reported totals
    _reconcile_balance_sheet(extracted, n_years, warnings)

    print(f"Successfully extracted {len(extracted)} fields for {company_name}")
    if warnings:
        print(f"Warnings: {warnings}")

    # Label the years as the filer's fiscal years: the lookups above key them
    # by the calendar year the period ends in, one too many for a 52/53-week
    # year that ends in early January
    year_end = _fiscal_year_end(facts_data)
    shift = year_end[0] - latest_fy if year_end and year_end[0] < latest_fy else 0
    return ExtractedFinancials(
        ticker=ticker,
        company_name=company_name,
        years=[y + shift for y in years_wanted],
        data=extracted,
        warnings=warnings,
        fiscal_year_end_month=year_end[1] if year_end else None,
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
        # Balance sheet — one column per year, like the income statement.
        # These were previously written only to the _0 (oldest) column even
        # though the values came from the most recent year, while the model
        # reads its opening balances from the last column -- so forecasts ran
        # on the page's placeholder balance sheet, not the company's.
        mapping[f'hist_h_cash_{j}']      = safe('cash_and_equivalents')[j]
        mapping[f'hist_h_ar_{j}']        = safe('accounts_receivable')[j]
        mapping[f'hist_h_inv_{j}']       = safe('inventories')[j]
        mapping[f'hist_h_ocurr_{j}']     = safe('other_current_assets')[j]
        mapping[f'hist_h_ppe_{j}']       = safe('property_plant_equipment')[j]
        mapping[f'hist_h_nca_{j}']       = safe('other_noncurrent_assets')[j]
        mapping[f'hist_h_lta_{j}']       = safe('other_longterm_assets')[j]
        mapping[f'hist_h_ap_{j}']        = safe('accounts_payable')[j]
        mapping[f'hist_h_ocl_{j}']       = safe('other_current_liabilities')[j]
        mapping[f'hist_h_def_{j}']       = safe('deferred_revenue')[j]
        mapping[f'hist_h_ltd_{j}']       = safe('long_term_debt')[j]
        mapping[f'hist_h_ncl_{j}']       = safe('other_noncurrent_liabilities')[j]
        mapping[f'hist_h_cs_{j}']        = safe('common_stock')[j]
        mapping[f'hist_h_re_{j}']        = safe('retained_earnings')[j]
        mapping[f'hist_h_oci_{j}']       = safe('aoci')[j]
        mapping[f'hist_h_capex_{j}']     = safe('capital_expenditures')[j]
        mapping[f'hist_h_divs_{j}']      = safe('dividends_paid')[j]
        mapping[f'hist_h_buybacks_{j}']  = safe('share_repurchases')[j]

    return mapping