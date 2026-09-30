"""Accounting standards (PLAN.md 2.6): IFRS and US GAAP.

Two things live here, both plain data and arithmetic:

1. **Line items.** Where each model input comes from in a filing, per
   standard: the XBRL concepts of the ``us-gaap`` and ``ifrs-full``
   taxonomies, in priority order (``LINE_ITEMS``), plus the lease lines
   (``LEASE_ITEMS``). ``ml/edgar_extractor.py`` reads filings with them; the
   model never sees a concept name.

2. **Leases.** IFRS 16 puts almost every lease on the balance sheet: rent
   leaves operating costs and comes back as depreciation and interest, so an
   IFRS EBITDA is *before* lease costs. Under US GAAP an operating lease's
   cost stays in operating expenses, so its EBITDA is *after* them. A deal
   says which standard its EBITDA follows, what its leases cost a year and
   what the lease liability is at close, and which view it is priced on:

   - ``pre_ifrs16``: EBITDA after lease costs, leases not counted as debt;
   - ``post_ifrs16``: EBITDA before lease costs, the lease liability counted
     with net debt at entry and exit.

   ``lease_terms`` turns that into three numbers for the engine:

   - the **operating EBITDA** the business earns in cash, always after lease
     costs (an IFRS figure less the lease cost; a US GAAP one as it is), from
     which the operating model grows. Rent is paid in cash under either
     standard, so the cash flows never depend on the view;
   - the **valuation add-back**, the lease cost, added to EBITDA where it is
     multiplied into a value (entry and exit) in the ``post_ifrs16`` view;
   - the **debt-like lease liability**, subtracted with net debt at entry
     and exit in that view.

   With no lease cost and no liability all three change nothing, whatever the
   standard, so every deal that sets none runs exactly as before.

   Simplified, and said so on the screen: the lease cost and the liability
   are held flat over the hold (renewals replace what runs off).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.debt import UnfinanceableStructure

# Ids stored in a deal (``DealInputs.accounting_standard``) and sent by the
# EDGAR answer. "" is a deal that doesn't say. The screen's words for each are
# in web/messages/en.json, keyed by these ids.
IFRS = "ifrs"
US_GAAP = "us_gaap"
STANDARDS = (IFRS, US_GAAP)

PRE_IFRS16 = "pre_ifrs16"
POST_IFRS16 = "post_ifrs16"
LEASE_VIEWS = (PRE_IFRS16, POST_IFRS16)

# XBRL namespace of each standard's taxonomy in SEC company facts
NAMESPACE = {US_GAAP: "us-gaap", IFRS: "ifrs-full"}
# Annual report forms: a 10-K for a US filer; a foreign private issuer files
# a 20-F (or, from Canada, a 40-F), and may file its IFRS accounts on a 10-K
ANNUAL_FORMS = {
    US_GAAP: ("10-K", "10-K/A"),
    IFRS: ("20-F", "20-F/A", "40-F", "40-F/A", "10-K", "10-K/A"),
}

# ---------------------------------------------------------------------------
# Line items: model field -> concepts, highest priority first
# ---------------------------------------------------------------------------
US_GAAP_ITEMS: dict[str, list[str]] = {
    'revenue': [
        'Revenues',
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'SalesRevenueNet',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
    ],
    'cost_of_revenue': ['CostOfRevenue', 'CostOfGoodsSold', 'CostOfGoodsAndServicesSold'],
    'gross_profit': ['GrossProfit'],
    'research_and_development': [
        'ResearchAndDevelopmentExpense',
        'ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost',
    ],
    'selling_general_admin': ['SellingGeneralAndAdministrativeExpense'],
    # Filers that tag the two halves separately (MSFT); summed into SG&A
    'selling_marketing': ['SellingAndMarketingExpense'],
    'general_admin': ['GeneralAndAdministrativeExpense'],
    'operating_income': ['OperatingIncomeLoss'],
    'interest_expense': [
        'InterestExpense', 'InterestAndDebtExpense', 'InterestExpenseNonoperating', 'InterestExpenseDebt',
    ],
    'interest_income': [
        'InterestAndDividendIncomeOperating', 'InvestmentIncomeInterest', 'InvestmentIncomeInterestAndDividend',
    ],
    'income_tax_expense': ['IncomeTaxExpenseBenefit'],
    'net_income': ['NetIncomeLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic'],
    'depreciation_amortization': [
        'DepreciationDepletionAndAmortization', 'DepreciationAndAmortization',
        'DepreciationAmortizationAndAccretionNet', 'Depreciation',
    ],
    'stock_based_compensation': ['ShareBasedCompensation', 'AllocatedShareBasedCompensationExpense'],
    'capital_expenditures': [
        'PaymentsToAcquirePropertyPlantAndEquipment', 'CapitalExpendituresIncurredButNotYetPaid',
    ],
    'cash_and_equivalents': [
        'CashAndCashEquivalentsAtCarryingValue', 'CashCashEquivalentsAndShortTermInvestments',
    ],
    'accounts_receivable': [
        'AccountsReceivableNetCurrent', 'ReceivablesNetCurrent', 'AccountsNotesAndLoansReceivableNetCurrent',
    ],
    'inventories': ['InventoryNet', 'InventoryGross'],
    'other_current_assets': ['OtherAssetsCurrent', 'PrepaidExpenseAndOtherAssetsCurrent'],
    'property_plant_equipment': ['PropertyPlantAndEquipmentNet'],
    'other_noncurrent_assets': ['OtherAssetsNoncurrent', 'IntangibleAssetsNetExcludingGoodwill'],
    'accounts_payable': [
        'AccountsPayableCurrent', 'AccountsPayableTradeCurrent', 'AccountsPayableAndAccruedLiabilitiesCurrent',
    ],
    'other_current_liabilities': ['OtherLiabilitiesCurrent', 'AccruedLiabilitiesCurrent'],
    'deferred_revenue': ['DeferredRevenueCurrent', 'ContractWithCustomerLiabilityCurrent'],
    'debt_total': [                      # includes current maturities
        'LongTermDebt', 'LongTermDebtAndCapitalLeaseObligationsIncludingCurrentMaturities',
    ],
    'debt_noncurrent': ['LongTermDebtNoncurrent', 'LongTermDebtAndCapitalLeaseObligations'],
    'debt_current': [
        'LongTermDebtCurrent', 'LongTermDebtAndCapitalLeaseObligationsCurrent', 'DebtCurrent',
    ],
    'common_stock_equity': ['StockholdersEquity', 'CommonStockholdersEquity'],
    'retained_earnings': ['RetainedEarningsAccumulatedDeficit'],
    'dividends_paid': ['PaymentsOfDividendsCommonStock', 'PaymentsOfDividends'],
    'share_repurchases': ['PaymentsForRepurchaseOfCommonStock'],
    # Reported totals: the balance sheet is reconciled to these
    'total_assets': ['Assets'],
    'total_liabilities': ['Liabilities'],
    'total_current_liabilities': ['LiabilitiesCurrent'],
    'total_liabilities_and_equity': ['LiabilitiesAndStockholdersEquity'],
    'equity_incl_nci': ['StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
    'aoci': ['AccumulatedOtherComprehensiveIncomeLossNetOfTax'],
    # For filers with no cost-of-revenue concept (e.g. restaurants)
    'costs_and_expenses': ['CostsAndExpenses'],
}

# The same fields in the IFRS taxonomy (ifrs-full). IFRS names what US GAAP
# calls interest expense "finance costs", net income "profit or loss",
# accumulated other comprehensive income "other reserves" and receivables and
# payables "trade and other"; the model's fields stay the same.
IFRS_ITEMS: dict[str, list[str]] = {
    'revenue': ['Revenue', 'RevenueFromContractsWithCustomers'],
    'cost_of_revenue': ['CostOfSales'],
    'gross_profit': ['GrossProfit'],
    'research_and_development': ['ResearchAndDevelopmentExpense'],
    'selling_general_admin': ['SellingGeneralAndAdministrativeExpense'],
    # Filers that show selling and administrative costs separately (SAP):
    # summed into SG&A, as for US GAAP
    'selling_marketing': ['SalesAndMarketingExpense', 'SellingExpense', 'DistributionCosts'],
    'general_admin': ['AdministrativeExpense', 'GeneralAndAdministrativeExpense'],
    'operating_income': ['ProfitLossFromOperatingActivities'],
    'interest_expense': ['FinanceCosts', 'InterestExpense'],
    'interest_income': ['FinanceIncome', 'InterestIncomeForFinancialAssetsMeasuredAtAmortisedCost'],
    'income_tax_expense': ['IncomeTaxExpenseContinuingOperations'],
    'net_income': ['ProfitLoss', 'ProfitLossAttributableToOwnersOfParent'],
    # IFRS 16: includes the depreciation of right-of-use assets
    'depreciation_amortization': [
        'DepreciationAndAmortisationExpense', 'AdjustmentsForDepreciationAndAmortisationExpense',
        'DepreciationAmortisationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLoss',
    ],
    'stock_based_compensation': [
        'AdjustmentsForSharebasedPayments',
        'ExpenseFromSharebasedPaymentTransactionsInWhichGoodsOrServicesReceivedDidNotQualifyForRecognitionAsAssets',
    ],
    'capital_expenditures': [
        'PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities',
        'PurchaseOfPropertyPlantAndEquipment',
        'PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAndOtherNoncurrentAssets',
    ],
    'cash_and_equivalents': ['CashAndCashEquivalents'],
    'accounts_receivable': ['TradeAndOtherCurrentReceivables', 'CurrentTradeReceivables', 'TradeReceivables'],
    'inventories': ['Inventories', 'CurrentInventories'],
    'other_current_assets': ['OtherCurrentAssets', 'OtherCurrentNonfinancialAssets'],
    'property_plant_equipment': [
        'PropertyPlantAndEquipment', 'PropertyPlantAndEquipmentIncludingRightofuseAssets',
    ],
    'other_noncurrent_assets': ['IntangibleAssetsOtherThanGoodwill', 'OtherNoncurrentAssets'],
    'accounts_payable': [
        'TradeAndOtherCurrentPayables', 'TradeAndOtherCurrentPayablesToTradeSuppliers',
        'TradeAndOtherPayablesToTradeSuppliers',
    ],
    'other_current_liabilities': ['OtherCurrentLiabilities', 'OtherCurrentNonfinancialLiabilities'],
    'deferred_revenue': ['CurrentContractLiabilities'],
    # Borrowings exclude lease liabilities under IFRS: those are LEASE_ITEMS
    'debt_total': ['Borrowings'],
    'debt_noncurrent': ['LongtermBorrowings', 'NoncurrentPortionOfNoncurrentBorrowings'],
    'debt_current': ['CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings', 'ShorttermBorrowings'],
    'common_stock_equity': ['EquityAttributableToOwnersOfParent'],
    'retained_earnings': ['RetainedEarnings'],
    'dividends_paid': [
        'DividendsPaidClassifiedAsFinancingActivities',
        'DividendsPaidToEquityHoldersOfParentClassifiedAsFinancingActivities', 'DividendsPaid',
    ],
    'share_repurchases': ['PaymentsToAcquireOrRedeemEntitysShares', 'PurchaseOfTreasuryShares'],
    'total_assets': ['Assets'],
    'total_liabilities': ['Liabilities'],
    'total_current_liabilities': ['CurrentLiabilities'],
    'total_liabilities_and_equity': ['EquityAndLiabilities'],
    'equity_incl_nci': ['Equity'],
    'aoci': ['OtherReserves'],
    'costs_and_expenses': [],
}

LINE_ITEMS = {US_GAAP: US_GAAP_ITEMS, IFRS: IFRS_ITEMS}

# Leases (money a year, and the liability at the year end). ``lease_cost`` is
# what the leases cost in cash a year: under IFRS 16 the lease payments
# (principal and interest, shown in financing); under US GAAP the operating
# lease cost, which sits in operating expenses.
LEASE_ITEMS: dict[str, dict[str, list[str]]] = {
    IFRS: {
        'lease_cost': ['PaymentsOfLeaseLiabilitiesClassifiedAsFinancingActivities',
                       'PaymentsOfLeaseLiabilities'],
        'lease_liability': ['LeaseLiabilities'],
        'lease_liability_current': ['CurrentLeaseLiabilities'],
        'lease_liability_noncurrent': ['NoncurrentLeaseLiabilities'],
    },
    US_GAAP: {
        # Filers that don't tag the operating lease cost itself (MCD) tag the
        # rent expense, or at least next year's payments due
        'lease_cost': ['OperatingLeaseCost', 'OperatingLeasePayments', 'LeaseAndRentalExpense',
                       'LesseeOperatingLeaseLiabilityPaymentsDueNextTwelveMonths'],
        'lease_liability': ['OperatingLeaseLiability'],
        'lease_liability_current': ['OperatingLeaseLiabilityCurrent'],
        'lease_liability_noncurrent': ['OperatingLeaseLiabilityNoncurrent'],
    },
}


def lease_figures(items: dict[str, list[float]]) -> dict[str, list[float]]:
    """A filing's lease cost and liability per year, from LEASE_ITEMS'
    fields: the liability is the reported total, or current plus
    non-current where only the parts are tagged."""
    total = items.get('lease_liability') or []
    parts = [c + n for c, n in zip(items.get('lease_liability_current') or [],
                                   items.get('lease_liability_noncurrent') or [])]
    liability = [t or p for t, p in zip(total, parts)] if total and parts else (total or parts)
    return {'lease_cost': list(items.get('lease_cost') or []), 'lease_liability': liability}


# ---------------------------------------------------------------------------
# Leases in a deal
# ---------------------------------------------------------------------------
class InvalidLeases(UnfinanceableStructure):
    """Lease figures the deal can't be run with. A refusal like an
    unfinanceable structure, and handled by the same code (api/main.py,
    jobs/runner.py): its message names the deal's own figures, so it is
    returned to the caller and never logged."""


@dataclass(frozen=True)
class LeaseTerms:
    view: str                    # the view applied: pre_ifrs16 or post_ifrs16
    operating_ebitda: float      # money: EBITDA after lease costs, what the model grows
    valuation_addback: float     # money: added to EBITDA at entry and exit
    debt_like: float             # money: counted with net debt at entry and exit


def default_view(standard: str) -> str:
    """The view a deal's own EBITDA is already in when it names none."""
    return POST_IFRS16 if standard == IFRS else PRE_IFRS16


def lease_terms(ebitda: float, standard: str, view: str, lease_cost: float,
                lease_liability: float) -> LeaseTerms:
    """The engine's numbers for a deal's leases (see the module docstring).

    ``ebitda`` is the deal's EBITDA as its standard reports it. A deal with
    no lease cost and no liability gets back its own EBITDA and two zeros, so
    the engine runs the expressions it always did.
    """
    view = view or default_view(standard)
    if lease_cost < 0 or lease_liability < 0:
        raise InvalidLeases("Lease cost and lease liability can't be negative.")
    operating = ebitda - lease_cost if standard == IFRS else ebitda
    if lease_cost and operating <= 0:
        raise InvalidLeases(
            f"The lease cost ({lease_cost:g}) is not less than the IFRS EBITDA ({ebitda:g}): "
            "EBITDA after lease costs would be zero or negative.")
    post = view == POST_IFRS16
    return LeaseTerms(view=view, operating_ebitda=operating,
                      valuation_addback=lease_cost if post else 0.0,
                      debt_like=lease_liability if post else 0.0)


def valuation_ebitda(terms: LeaseTerms) -> float:
    return terms.operating_ebitda + terms.valuation_addback


def standard_or_none(value: Optional[str]) -> str:
    return value if value in STANDARDS else ""
