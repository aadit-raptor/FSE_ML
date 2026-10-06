"""Which reported concepts give each summary figure, per taxonomy.

The summary is what later tasks need from a company (PLAN.md 4.3: growth,
margins, capex, working capital, tax, leverage) and what a deal or forecast
starts from -- not the whole filing. Money figures are millions of the
filing's currency; ``ebitda`` is operating profit plus depreciation and
amortisation **as the standard reports them** (before lease costs under
IFRS 16), as the EDGAR answer gives it.

Concepts are written ``prefix:Name`` with a canonical prefix per taxonomy
(``canonical_prefix``), whatever prefix a filing declares. An alternative is
a concept, or a ``Sum`` of concepts (a part written ``-prefix:Name`` is
subtracted). Lists run highest priority first.

US GAAP and IFRS reuse ``core.accounting``'s line items, so a company loaded
here and the EDGAR answer read the same concepts. The UK (FRC) and Japanese
(EDINET) lists were checked against the recorded filings in
``tests/fixtures/companies``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from core.accounting import IFRS_ITEMS, LEASE_ITEMS, US_GAAP_ITEMS
from core.accounting import IFRS as CORE_IFRS, US_GAAP as CORE_US_GAAP

SUMMARY_FIELDS = (
    "revenue", "operating_income", "depreciation_amortization", "ebitda", "net_income",
    "income_tax_expense", "interest_expense", "capital_expenditures",
    "cash_and_equivalents", "accounts_receivable", "inventories", "accounts_payable",
    "total_debt", "total_assets", "total_equity", "lease_cost", "lease_liability",
)
# Worked out from other figures, never read directly
DERIVED = frozenset({"ebitda", "total_debt"})
# Amounts over the year (the rest are balances at the year end)
FLOWS = frozenset({
    "revenue", "operating_income", "depreciation_amortization", "net_income", "income_tax_expense",
    "interest_expense", "capital_expenditures", "lease_cost", "lease_interest",
})
# Costs and outflows some taxonomies sign negative (EDINET's cash flows):
# stored as positive amounts. Tax keeps its sign: a credit is real.
ABSOLUTE = frozenset({"depreciation_amortization", "interest_expense", "capital_expenditures", "lease_cost"})


@dataclass(frozen=True)
class Sum:
    """Concepts added together. ``every``: a year counts only when all parts
    are reported (an identity such as assets = capital employed + current
    liabilities); otherwise any part will do and a missing one is nothing
    (borrowings reported line by line)."""
    parts: tuple
    every: bool = False

    def concepts(self) -> tuple:
        return tuple(p.lstrip("-") for p in self.parts)


def concepts_of(alternative) -> tuple:
    return (alternative,) if isinstance(alternative, str) else alternative.concepts()


@dataclass(frozen=True)
class ConceptMap:
    taxonomy: str
    fields: dict
    # Figures this taxonomy often lacks, so a gap is not worth a warning
    optional: frozenset = field(default_factory=frozenset)


def _prefixed(prefix: str, names) -> list:
    return [f"{prefix}:{n}" for n in names]


def _core_map(prefix: str, items: dict, leases: dict) -> dict:
    pick = {
        "revenue": "revenue", "operating_income": "operating_income",
        "depreciation_amortization": "depreciation_amortization", "net_income": "net_income",
        "income_tax_expense": "income_tax_expense", "interest_expense": "interest_expense",
        "capital_expenditures": "capital_expenditures", "cash_and_equivalents": "cash_and_equivalents",
        "accounts_receivable": "accounts_receivable", "inventories": "inventories",
        "accounts_payable": "accounts_payable", "total_assets": "total_assets",
        "total_equity": "equity_incl_nci", "equity_parent": "common_stock_equity",
        "debt_total": "debt_total", "debt_noncurrent": "debt_noncurrent", "debt_current": "debt_current",
    }
    out = {name: _prefixed(prefix, items[src]) for name, src in pick.items()}
    out.update({name: _prefixed(prefix, concepts) for name, concepts in leases.items()})
    return out


_us = _core_map("us-gaap", US_GAAP_ITEMS, LEASE_ITEMS[CORE_US_GAAP])
_us["total_assets"].append(Sum(("us-gaap:AssetsNoncurrent", "us-gaap:AssetsCurrent"), every=True))
_ifrs = _core_map("ifrs-full", IFRS_ITEMS, LEASE_ITEMS[CORE_IFRS])
# Some IFRS balance sheets end at net assets and show no total (Tesco)
_ifrs["total_assets"].append(Sum(("ifrs-full:NoncurrentAssets", "ifrs-full:CurrentAssets"), every=True))


US_GAAP_MAP = ConceptMap("us-gaap", _us, frozenset({"inventories", "lease_cost", "lease_liability"}))
IFRS_MAP = ConceptMap("ifrs-full", _ifrs, frozenset({"inventories", "lease_cost", "lease_liability"}))


# UK: the FRC taxonomy filed with Companies House (FRS 102, FRS 101, FRS 105
# for micro-entities, and IFRS accounts tagged with FRC concepts). Small
# companies may file a balance sheet only; UK formats show no total assets,
# so it is fixed plus current assets, or capital employed plus current
# liabilities (current assets less net current assets).
FRC_MAP = ConceptMap("frc", {
    "revenue": ["frc:TurnoverRevenue", "frc:Revenue"],
    "operating_income": ["frc:OperatingProfitLoss"],
    "depreciation_amortization": [
        "frc:DepreciationAmortisationImpairmentExpense",
        Sum(("frc:DepreciationExpensePropertyPlantEquipment", "frc:AmortisationExpenseIntangibleAssets")),
        Sum(("frc:DepreciationImpairmentExpensePropertyPlantEquipment",
             "frc:AmortisationImpairmentExpenseIntangibleAssets")),
        "frc:IncreaseFromDepreciationChargeForYearPropertyPlantEquipment",
    ],
    "net_income": ["frc:ProfitLoss", "frc:ProfitLossForPeriod"],
    "income_tax_expense": ["frc:TaxTaxCreditOnProfitOrLossOnOrdinaryActivities", "frc:IncomeTaxExpenseCredit"],
    "interest_expense": ["frc:InterestPayableSimilarChargesFinanceCosts", "frc:FinanceCosts"],
    "capital_expenditures": ["frc:PurchasePropertyPlantEquipment",
                             "frc:AdditionsOtherThanThroughBusinessCombinationsPropertyPlantEquipment"],
    "cash_and_equivalents": ["frc:CashBankOnHand", "frc:CashCashEquivalents"],
    "accounts_receivable": ["frc:TradeDebtorsTradeReceivables", "frc:Debtors"],
    "inventories": ["frc:Inventories", "frc:Stocks"],
    "accounts_payable": ["frc:TradeCreditorsTradePayables"],
    "total_assets": [
        "frc:TotalAssets",
        Sum(("frc:FixedAssets", "frc:CurrentAssets"), every=True),
        Sum(("frc:TotalAssetsLessCurrentLiabilities", "frc:CurrentAssets", "-frc:NetCurrentAssetsLiabilities"),
            every=True),
    ],
    "total_equity": ["frc:Equity", "frc:NetAssetsLiabilities"],
    "debt_total": [Sum(("frc:BankBorrowings", "frc:BankBorrowingsOverdrafts", "frc:OtherBorrowings"))],
    "debt_noncurrent": [], "debt_current": [], "equity_parent": [],
    "lease_cost": [], "lease_liability": [],
}, frozenset({"revenue", "operating_income", "capital_expenditures", "inventories", "interest_expense",
              "depreciation_amortization", "income_tax_expense", "net_income", "accounts_payable",
              "lease_cost", "lease_liability", "total_debt", "ebitda"}))


# Japan, EDINET: Japanese GAAP statements (jppfs_cor). Borrowings are
# reported piece by piece and summed. Japanese GAAP keeps operating leases
# off the balance sheet.
JPPFS_MAP = ConceptMap("jppfs", {
    "revenue": ["jppfs:NetSales", "jppfs:OperatingRevenue1", "jppfs:Revenue",
                "jpcrp:NetSalesSummaryOfBusinessResults"],
    "operating_income": ["jppfs:OperatingIncome"],
    "depreciation_amortization": ["jppfs:DepreciationAndAmortizationOpeCF", "jppfs:DepreciationOpeCF"],
    "net_income": ["jppfs:ProfitLoss", "jppfs:ProfitLossAttributableToOwnersOfParent",
                   "jpcrp:ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults"],
    "income_tax_expense": ["jppfs:IncomeTaxes", "jppfs:TotalIncomeTaxes"],
    "interest_expense": ["jppfs:InterestExpensesNOE", "jppfs:InterestExpensesOpeCF"],
    "capital_expenditures": ["jppfs:PurchaseOfPropertyPlantAndEquipmentInvCF",
                             "jppfs:PurchaseOfPropertyPlantAndEquipmentAndIntangibleAssetsInvCF",
                             "jpcrp:CapitalExpendituresOverviewOfCapitalExpendituresEtc"],
    "cash_and_equivalents": ["jppfs:CashAndCashEquivalents", "jppfs:CashAndDeposits",
                             "jpcrp:CashAndCashEquivalentsSummaryOfBusinessResults"],
    "accounts_receivable": ["jppfs:NotesAndAccountsReceivableTradeAndContractAssets",
                            "jppfs:NotesAndAccountsReceivableTrade", "jppfs:AccountsReceivableTrade"],
    "inventories": ["jppfs:Inventories",
                    Sum(("jppfs:MerchandiseAndFinishedGoods", "jppfs:WorkInProcess", "jppfs:RawMaterialsAndSupplies"))],
    "accounts_payable": ["jppfs:NotesAndAccountsPayableTrade", "jppfs:AccountsPayableTrade"],
    "total_assets": ["jppfs:Assets", "jpcrp:TotalAssetsSummaryOfBusinessResults"],
    "total_equity": ["jppfs:NetAssets", "jpcrp:NetAssetsSummaryOfBusinessResults"],
    "debt_total": [Sum(("jppfs:ShortTermLoansPayable", "jppfs:CurrentPortionOfLongTermLoansPayable",
                        "jppfs:CommercialPapersLiabilities", "jppfs:CurrentPortionOfBonds",
                        "jppfs:BondsPayable", "jppfs:LongTermLoansPayable"))],
    "debt_noncurrent": [], "debt_current": [], "equity_parent": [],
    "lease_cost": [], "lease_liability": [],
}, frozenset({"inventories", "lease_cost", "lease_liability", "interest_expense"}))


# Japan, EDINET: IFRS statements (jpigp_cor, the FSA's IFRS taxonomy). Some
# filers report revenue only under a concept of their own (``ext:``; Toyota's
# "total net revenues"); debt is "interest-bearing liabilities" when not
# bonds and borrowings; capex falls back to the report's capital
# expenditure overview, which every annual report states.
JPIGP_MAP = ConceptMap("jpigp", {
    "revenue": ["jpigp:RevenueIFRS", "jpigp:NetSalesIFRS", "jpigp:OperatingRevenueIFRS",
                "jpcrp:RevenueIFRSSummaryOfBusinessResults",
                "ext:TotalNetRevenuesIFRS", "ext:SalesRevenuesIFRS", "ext:OperatingRevenuesIFRS"],
    "operating_income": ["jpigp:OperatingProfitLossIFRS"],
    "depreciation_amortization": ["jpigp:DepreciationAndAmortizationOpeCFIFRS"],
    "net_income": ["jpigp:ProfitLossIFRS", "jpigp:ProfitLossAttributableToOwnersOfParentIFRS",
                   "jpcrp:ProfitLossAttributableToOwnersOfParentIFRSSummaryOfBusinessResults"],
    "income_tax_expense": ["jpigp:IncomeTaxExpenseIFRS"],
    "interest_expense": ["jpigp:FinanceCostsIFRS", "jpigp:InterestExpensesOpeCFIFRS"],
    "capital_expenditures": ["jpigp:PurchaseOfPropertyPlantAndEquipmentInvCFIFRS",
                             "jpcrp:CapitalExpendituresOverviewOfCapitalExpendituresEtc"],
    "cash_and_equivalents": ["jpigp:CashAndCashEquivalentsIFRS",
                             "jpcrp:CashAndCashEquivalentsIFRSSummaryOfBusinessResults"],
    "accounts_receivable": ["jpigp:TradeAndOtherReceivablesCAIFRS"],
    "inventories": ["jpigp:InventoriesCAIFRS"],
    "accounts_payable": ["jpigp:TradeAndOtherPayablesCLIFRS"],
    "total_assets": ["jpigp:AssetsIFRS", "jpcrp:TotalAssetsIFRSSummaryOfBusinessResults"],
    "total_equity": ["jpigp:EquityIFRS"],
    "equity_parent": ["jpigp:EquityAttributableToOwnersOfParentIFRS"],
    "debt_total": [Sum(("jpigp:BondsAndBorrowingsCLIFRS", "jpigp:BondsAndBorrowingsNCLIFRS")),
                   Sum(("jpigp:BorrowingsCLIFRS", "jpigp:BorrowingsNCLIFRS")),
                   Sum(("jpigp:InterestBearingLiabilitiesCLIFRS", "jpigp:InterestBearingLiabilitiesNCLIFRS"))],
    "debt_noncurrent": [], "debt_current": [],
    "lease_cost": ["jpigp:RepaymentsOfLeaseLiabilitiesFinCFIFRS"],
    "lease_liability": [],
    "lease_liability_current": ["jpigp:LeaseLiabilitiesCLIFRS"],
    "lease_liability_noncurrent": ["jpigp:LeaseLiabilitiesNCLIFRS"],
}, frozenset({"inventories", "lease_cost", "lease_liability", "interest_expense"}))

MAPS = {m.taxonomy: m for m in (US_GAAP_MAP, IFRS_MAP, FRC_MAP, JPPFS_MAP, JPIGP_MAP)}


# A filing may declare any prefix for a taxonomy; concepts are compared by
# the namespace it stands for
_NAMESPACES = (
    (re.compile(r"^https?://xbrl\.ifrs\.org/taxonomy/[^/]+/ifrs-full$"), "ifrs-full"),
    (re.compile(r"^https?://fasb\.org/us-gaap/"), "us-gaap"),
    (re.compile(r"^https?://xbrl\.frc\.org\.uk/fr/[^/]+/core$"), "frc"),
    (re.compile(r"^https?://xbrl\.frc\.org\.uk/(cd|reports)/[^/]+/business$"), "frc-bus"),
    (re.compile(r"^https?://xbrl\.frc\.org\.uk/fr/[^/]+/bus$"), "frc-bus"),
)


def canonical_prefix(namespace: str) -> str | None:
    for pattern, prefix in _NAMESPACES:
        if pattern.match(namespace):
            return prefix
    return None
