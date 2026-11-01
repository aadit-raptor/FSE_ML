"""A deal as a live Excel workbook (PLAN.md 7.3a).

One sheet whose cells are working formulas: every line of the deal model --
operating model, income statement, tax rules, levered free cash flow, each
facility's schedule with its cash sweep, PIK, commitment fee and revolver draw,
and the returns -- written from the same resolved parameters the engine runs on
(`core.deal.build_lbo_params`), so the spreadsheet program that opens it works
out the app's answer by itself. Change a blue input cell and every line
follows; nothing is stored but the inputs.

What the engine does and the workbook mirrors, line by line, is
docs/methodology.md section 3 to 7. Two deliberate differences, both said on
the sheet: the workbook does not round money to two decimals of a million
(finding 10), and it needs no convergence loop, because interest is charged on
each year's opening balance and so each year's cells depend only on the year
before (the engine iterates only because it works a whole statement at a time).
The columns are laid out for the deal's hold: a different hold is a new
download.

Money is in the deal's own currency and unit. The engine runs in millions, but
every formula here is the same in any unit (each is a sum, a product with a
rate, or a MAX/MIN against zero or another amount), so the inputs are written
in the deal's unit and the answers come out in it.

`sheets` is the one cell description: the download (`write`) and the Excel
add-in (`GET /api/deals/{id}/model/cells`) both write it, as research/workbook.py
does for sector models, so the two can never hold different cells. The model
is a single sheet, so no formula names another sheet: the add-in may have to
give the sheet a new name when the workbook already has one like it.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Callable, Optional

from openpyxl.utils import get_column_letter

from core.debt import coerce, unique_names
from core.deal import DealInputs, build_lbo_params
from research.workbook import write_sheets

SOURCE = "variater.com"
SHEET = "Deal model"
FIRST_YEAR_COLUMN = 4              # A = label, B = unit, C = input or at close, D.. = years
SAVED_NAME, SAVED_AT = "variater_deal", (1, 2)
SAVED_TEXT = "Variater deal: "

BLUE, GREY, WHITE, GREEN, RED = "0B4F9E", "888888", "FFFFFF", "1D6F42", "B03A2E"
HEAD, SECTION, INPUT_FILL = "1F2A30", "DCE6EA", "EAF2FB"

FORMATS = {
    "money": "#,##0.00;-#,##0.00;\"-\"",
    "percent": "0.00%",
    "multiple": "0.00\"x\"",
    "years": "0",
    "flag": "\"Yes\";\"Yes\";\"No\"",
    "number": "0.0000",
    "text": "@",
}
_LAKH = 100_000
_LOOKS_LIKE_A_CELL = re.compile(r"[A-Za-z]{1,3}[0-9]+|[Rr][0-9]*[Cc][0-9]*")


class WorkbookError(ValueError):
    pass


@dataclass
class _Sheet:
    """The cells of the model sheet as they are placed, and where each row is."""
    years: int
    grouping: str
    app: Callable[[str, Optional[int]], Optional[float]]
    cells: list = field(default_factory=list)
    rows: dict = field(default_factory=dict)
    names: dict = field(default_factory=dict)
    row: int = 0

    # -- addresses --------------------------------------------------------
    @staticmethod
    def column(year: Optional[int]) -> str:
        return "C" if year is None else get_column_letter(FIRST_YEAR_COLUMN + year - 1)

    def at(self, key: str, year: Optional[int] = None) -> str:
        """A row's cell: column C (an input, or the figure at close) or a year's."""
        if key not in self.rows:
            raise WorkbookError("no row %r yet" % key)
        if year is None:
            return "$C$%d" % self.rows[key]
        return "%s%d" % (self.column(year), self.rows[key])

    # -- placing ----------------------------------------------------------
    def put(self, row: int, column: int, **spec) -> None:
        self.cells.append({"row": row, "column": column,
                           **{k: v for k, v in spec.items() if v is not None and v is not False}})

    def text(self, row: int, column: int, text: str, **spec) -> None:
        self.put(row, column, text=str(text), **spec)

    def heading(self, title: str) -> None:
        self.row += 2
        self.text(self.row, 1, title, bold=True, fill=SECTION)
        for column in range(2, FIRST_YEAR_COLUMN + self.years):
            self.put(self.row, column, fill=SECTION)

    def _format(self, kind: str, key: str, year: Optional[int]) -> str:
        if kind == "money" and self.grouping == "lakh":
            value = self.app(key, year)
            if value is not None and abs(value) >= _LAKH:
                from api.routers.export import lakh_pattern    # export imports this module
                return lakh_pattern(value, 2)
        return FORMATS[kind]

    def _start(self, key: str, label: str, unit: str) -> int:
        if key in self.rows:
            raise WorkbookError("two rows named %r" % key)
        if _LOOKS_LIKE_A_CELL.fullmatch(key):
            raise WorkbookError("%r cannot be a name in Excel: it reads as a cell" % key)
        self.row += 1
        self.rows[key] = self.row
        self.text(self.row, 1, label)
        self.text(self.row, 2, unit, color=GREY)
        return self.row

    def value(self, key: str, label: str, unit: str, kind: str, value, *, name: bool = True) -> None:
        """An input in column C: blue, named, the cell a reader types over."""
        row = self._start(key, label, unit)
        if kind == "text":
            self.text(row, 3, value, color=BLUE, fill=INPUT_FILL, align="right")
        else:
            self.put(row, 3, number=float(value), color=BLUE, fill=INPUT_FILL,
                     number_format=self._format(kind, key, None))
        if name:
            self.names[key] = "$C$%d" % row

    def scalar(self, key: str, label: str, unit: str, kind: str, formula: str, *, bold: bool = False) -> None:
        """A formula in column C (a figure at close, or an input worked out from others)."""
        row = self._start(key, label, unit)
        self.put(row, 3, formula=lambda: "=" + formula, number_format=self._format(kind, key, None), bold=bold)
        self.names[key] = "$C$%d" % row

    def yearly(self, key: str, label: str, unit: str, kind: str, formula: Callable[[int], Optional[str]], *,
               bold: bool = False, at_close: Optional[str] = None) -> None:
        """A formula for every year; `formula(year)` gives its text."""
        row = self._start(key, label, unit)
        if at_close is not None:
            self.put(row, 3, formula=lambda: "=" + at_close, number_format=self._format(kind, key, 0), bold=bold)
        for year in range(1, self.years + 1):
            # worked out once every row is placed: a formula may read a row further down
            self.put(row, FIRST_YEAR_COLUMN + year - 1, formula=lambda y=year: "=" + formula(y),
                     number_format=self._format(kind, key, year), bold=bold)

    def resolve(self) -> None:
        for cell in self.cells:
            if callable(cell.get("formula")):
                cell["formula"] = cell["formula"]()

    def inputs_by_year(self, key: str, label: str, unit: str, kind: str, values: list) -> None:
        """A typed figure for every year (a reference rate, a repayment schedule): blue."""
        row = self._start(key, label, unit)
        for year, v in enumerate(values, start=1):
            self.put(row, FIRST_YEAR_COLUMN + year - 1, number=float(v), color=BLUE, fill=INPUT_FILL,
                     number_format=self._format(kind, key, year))


def _sum(cells: list) -> str:
    return "+".join(cells) if cells else "0"


def _app_values(result, factor: float, tranche_names: list) -> Callable[[str, Optional[int]], Optional[float]]:
    """The app's own figure for a row and year (in the deal's unit), where the
    row has one: used only to size lakh grouping, never as a cell's value."""
    op, cf, debt = result.operating_model, result.cash_flow, result.debt_schedule
    yearly = {
        "revenue": op.revenue, "cogs": op.cogs, "gross_profit": op.gross_profit, "opex_row": op.opex,
        "ebit": op.ebit, "da_row": op.da, "ebitda_row": op.ebitda, "interest": op.interest_expense,
        "ebt": op.ebt, "tax": op.taxes, "net_income": op.net_income, "cf_net_income": op.net_income,
        "cf_da": op.da, "capex_row": cf.capex, "nwc_row": cf.delta_nwc, "lfcf": cf.levered_fcf,
        "cash_close": debt.cash_balance, "debt_close": debt.total_ending_debt,
        "mand_total": debt.total_mandatory_repayment, "sweep_total": debt.total_cash_sweep,
        "avail": debt.available_for_sweep,
    }
    for i, name in enumerate(tranche_names, start=1):
        records = debt.schedule.get(name, [])
        yearly["t%d_open" % i] = [r.beginning_balance for r in records]
        yearly["t%d_close" % i] = [r.ending_balance for r in records]
        yearly["t%d_interest" % i] = [r.interest_expense for r in records]
    ret = result.returns
    scalars = {"equity": result.entry_equity, "exit_ev": ret.exit_ev, "net_exit": ret.net_exit_equity,
               "gross_exit": ret.gross_exit_equity, "net_debt_exit": ret.net_debt_at_exit}

    def get(key: str, year: Optional[int]) -> Optional[float]:
        if year:
            values = yearly.get(key)
            if values and year <= len(values):
                return values[year - 1] * factor
            return None
        v = scalars.get(key)
        return None if v is None else v * factor
    return get


def sheets(inputs: DealInputs, cfg, *, title: str, money_label: str, result=None, factor: float = 1.0,
           app_irr: Optional[float] = None, grouping: str = "locale", saved: Optional[str] = None,
           about: tuple = ((), ())) -> list[dict]:
    """The deal as sheets of cells: the model sheet of live formulas and an About sheet.

    `inputs` and `cfg` are in the deal's own unit (not converted to millions);
    `result` is the app's run of the same deal (in millions, `factor` turning
    them into the deal's unit) and `app_irr` its IRR, shown beside the
    workbook's own for a reader to compare. `saved`, a saved deal's id, is
    written beside the title and named `variater_deal`, so the add-in can
    refresh that sheet from the saved deal.
    """
    p = build_lbo_params(inputs, cfg)
    n = int(p.holding_period)
    specs = list(coerce(inputs.tranches))
    app = _app_values(result, factor, [t.name for t in p.capital_structure.tranches] if p.capital_structure
                      else ["Senior Term Loan", "Mezzanine"]) if result is not None else (lambda k, y: None)
    s = _Sheet(years=n, grouping=grouping, app=app)
    m = money_label

    # -- title ---------------------------------------------------------------
    s.text(1, 1, "%s: deal model" % title, bold=True, size=13)
    if saved is not None:
        s.text(*SAVED_AT, SAVED_TEXT + str(saved), color=GREY)
        s.names[SAVED_NAME] = "$B$1"  # an absolute cell address: currency-ok
    s.text(2, 1, "Live formulas: blue cells are inputs to type over, everything in black works itself out. "
                 "Source: " + SOURCE, bold=True, color=GREEN)
    s.text(3, 1, "Money in %s. The columns are laid out for a %d-year hold; another hold needs a new "
                 "download. The app rounds money to two decimals of a million and this sheet does not, "
                 "so very small deals can differ in the last digits." % (m, n))
    s.row = 4
    s.text(s.row, 1, "Year", bold=True, color=WHITE, fill=HEAD)
    s.text(s.row, 2, "Unit", bold=True, color=WHITE, fill=HEAD)
    s.text(s.row, 3, "Input / at close", bold=True, color=WHITE, fill=HEAD, align="right")
    for year in range(1, n + 1):
        s.put(s.row, FIRST_YEAR_COLUMN + year - 1, number=year, bold=True, color=WHITE, fill=HEAD,
              number_format="\"Year \"0", align="right")
    s.rows["year"] = s.row
    year_cell = lambda y: "%s$%d" % (s.column(y), s.rows["year"])      # noqa: E731

    # -- inputs --------------------------------------------------------------
    s.heading("Deal")
    s.value("ebitda", "EBITDA at close (after lease costs)", m, "money", p.entry_ebitda)
    s.value("lease_addback", "Lease cost added back where EBITDA meets a multiple", m, "money",
            p.lease_ebitda_addback)
    s.value("lease_liability", "Lease liability counted with net debt", m, "money", p.lease_liability)
    s.value("entry_multiple", "Entry multiple", "x EBITDA", "multiple", p.entry_multiple)
    s.value("exit_multiple", "Exit multiple", "x EBITDA", "multiple", p.exit_multiple)
    # not an input: the columns are laid out for this hold, so typing another would change the IRR's power alone
    s.value("hold", "Holding period (fixed: the columns are laid out for it)", "years", "years", n)
    s.cells[-1].update(color=GREY)
    s.cells[-1].pop("fill", None)
    s.value("min_cash", "Minimum cash, funded by the sponsor at close", m, "money", p.minimum_cash)
    s.value("income_rate", "Interest earned on minimum cash", "% a year", "percent", 0.005)
    s.value("tx_fee", "Transaction fees", "% of entry EV", "percent", p.transaction_fees_pct)
    s.value("fin_fee", "Financing fees", "% of total debt", "percent", p.financing_fees_pct)
    s.value("other_uses", "Other uses of funds", m, "money", p.other_uses)
    s.value("option_pool", "Management option pool", "% of exit equity", "percent", p.management_option_pool_pct)

    s.heading("Operations")
    s.value("growth", "Revenue growth", "% a year", "percent", p.revenue_growth)
    s.value("gross_margin", "Gross margin", "% of revenue", "percent", p.gross_margin)
    s.value("opex", "Operating costs", "% of revenue", "percent", p.opex_pct)
    s.value("da", "Depreciation and amortisation", "% of revenue", "percent", p.da_pct)
    s.value("capex", "Capital expenditure", "% of revenue", "percent", p.capex_pct)
    s.value("nwc", "Change in working capital", "% of revenue", "percent", p.nwc_pct)

    rules = p.tax_rules
    s.heading("Tax")
    s.value("tax_rate", "Tax rate", "% of taxable profit", "percent", p.tax_rate)
    s.value("limit_kind", "Interest limit: none, ebitda_share or fixed", "rule", "text",
            rules.interest_limit if rules else "none")
    s.value("limit_share", "Interest limit, share of EBITDA", "% of EBITDA", "percent",
            rules.interest_limit_share if rules else 0.30)
    s.value("limit_amount", "Interest limit: allowance (ebitda_share) or cap (fixed)", m, "money",
            rules.interest_limit_amount if rules else 0.0)
    s.value("loss_cf", "Losses carried forward", "yes/no", "flag", 1.0 if rules and rules.loss_carryforward else 0.0)
    s.value("loss_amount", "Profit losses may offset in full each year", m, "money",
            rules.loss_limit_amount if rules else 0.0)
    s.value("loss_share", "Share of the profit above it losses may offset", "%", "percent",
            rules.loss_limit_share if rules else 1.0)
    s.value("min_tax", "Minimum tax on profit before tax", "%", "percent", rules.minimum_tax_rate if rules else 0.0)

    # -- facilities ----------------------------------------------------------
    if specs:
        facilities = _listed(s, specs, m, n)
    else:
        facilities = _percentages(s, p, m, n)

    # -- at close --------------------------------------------------------------
    s.heading("At close")
    s.scalar("entry_ev", "Entry enterprise value", m, "money",
             "(%s+%s)*%s" % (s.at("ebitda"), s.at("lease_addback"), s.at("entry_multiple")), bold=True)
    s.scalar("total_debt", "Debt drawn at close", m, "money", _sum([s.at("t%d_drawn" % i) for i in facilities]))
    s.scalar("tx_fees", "Transaction fees", m, "money", "%s*%s" % (s.at("entry_ev"), s.at("tx_fee")))
    s.scalar("fin_fees", "Financing fees", m, "money", "%s*%s" % (s.at("total_debt"), s.at("fin_fee")))
    s.scalar("tranche_fees", "Facilities' own upfront fees", m, "money",
             _sum(["%s*%s" % (s.at("t%d_amount" % i), s.at("t%d_fee" % i)) for i in facilities]))
    s.scalar("entry_costs", "Entry costs", m, "money", "%s+%s+%s+%s" % (
        s.at("tx_fees"), s.at("fin_fees"), s.at("tranche_fees"), s.at("other_uses")))
    s.scalar("equity", "Sponsor equity", m, "money", "%s+%s+%s-%s-%s" % (
        s.at("entry_ev"), s.at("entry_costs"), s.at("min_cash"), s.at("total_debt"), s.at("lease_liability")),
        bold=True)

    prev = lambda key, y, first: first if y == 1 else s.at(key, y - 1)    # noqa: E731

    # -- operating model -----------------------------------------------------
    s.heading("Operating model")
    s.yearly("revenue", "Revenue", m, "money", lambda y: "%s*(1+%s)" % (
        s.at("revenue") if y == 1 else s.at("revenue", y - 1), s.at("growth")),
        at_close="%s/(%s-%s+%s)" % (s.at("ebitda"), s.at("gross_margin"), s.at("opex"), s.at("da")))
    s.yearly("cogs", "Cost of sales", m, "money", lambda y: "%s*(1-%s)" % (s.at("revenue", y), s.at("gross_margin")))
    s.yearly("gross_profit", "Gross profit", m, "money", lambda y: "%s-%s" % (s.at("revenue", y), s.at("cogs", y)))
    s.yearly("opex_row", "Operating costs", m, "money", lambda y: "%s*%s" % (s.at("revenue", y), s.at("opex")))
    s.yearly("ebit", "EBIT", m, "money", lambda y: "%s-%s" % (s.at("gross_profit", y), s.at("opex_row", y)))
    s.yearly("da_row", "Depreciation and amortisation", m, "money", lambda y: "%s*%s" % (s.at("revenue", y), s.at("da")))
    s.yearly("ebitda_row", "EBITDA", m, "money", lambda y: "%s+%s" % (s.at("ebit", y), s.at("da_row", y)), bold=True)

    # -- income statement (interest from the debt schedule below) ------------
    s.heading("Income statement")
    interest_row = s.row + 1
    s.yearly("interest", "Interest expense (from the facilities below)", m, "money", lambda y: "0")
    s.yearly("income", "Interest income on minimum cash", m, "money",
             lambda y: "%s*%s" % (s.at("min_cash"), s.at("income_rate")))
    s.yearly("ebt", "Profit before tax", m, "money", lambda y: "%s-%s+%s" % (
        s.at("ebit", y), s.at("interest", y), s.at("income", y)))
    tax_row = s.row + 1
    s.yearly("tax", "Tax (from the tax lines below)", m, "money", lambda y: "0")
    s.yearly("net_income", "Net income", m, "money", lambda y: "%s-%s" % (s.at("ebt", y), s.at("tax", y)), bold=True)

    # -- tax lines -------------------------------------------------------------
    s.heading("Tax lines")
    kind = s.at("limit_kind")
    s.yearly("net_interest", "Net interest (expense less income)", m, "money",
             lambda y: "%s-%s" % (s.at("interest", y), s.at("income", y)))
    s.yearly("cap", "Interest the limit allows", m, "money", lambda y: (
        'IF(%s="ebitda_share",MAX(MAX(%s*%s,%s),0),IF(%s="fixed",MAX(%s,0),0))' % (
            kind, s.at("ebitda_row", y), s.at("limit_share"), s.at("limit_amount"), kind, s.at("limit_amount"))))
    claim_row = s.row + 1
    s.yearly("claim", "Interest claimed (this year's and what was carried in)", m, "money", lambda y: "0")
    s.yearly("deductible", "Interest deducted", m, "money", lambda y: 'IF(%s="none",%s,MIN(%s,%s)+MIN(%s,0))' % (
        kind, s.at("net_interest", y), s.at("claim", y), s.at("cap", y), s.at("net_interest", y)))
    s.yearly("carried", "Interest carried forward", m, "money", lambda y: 'IF(%s="none",0,%s-MIN(%s,%s))' % (
        kind, s.at("claim", y), s.at("claim", y), s.at("cap", y)))
    _rewrite(s, claim_row, lambda y: "MAX(%s,0)+%s" % (s.at("net_interest", y), prev("carried", y, "0")))
    s.yearly("profit", "Profit for tax (EBIT less interest deducted)", m, "money",
             lambda y: "%s-%s" % (s.at("ebit", y), s.at("deductible", y)))
    s.yearly("loss_limit", "Losses this profit may absorb", m, "money", lambda y: "MIN(MAX(%s,0),%s)+%s*MAX(MAX(%s,0)-%s,0)" % (
        s.at("profit", y), s.at("loss_amount"), s.at("loss_share"), s.at("profit", y), s.at("loss_amount")))
    used_row = s.row + 1
    s.yearly("losses_used", "Losses used", m, "money", lambda y: "0")
    s.yearly("losses_carried", "Losses carried forward", m, "money", lambda y: "IF(%s=1,%s-%s+MAX(0-%s,0),0)" % (
        s.at("loss_cf"), prev("losses_carried", y, "0"), s.at("losses_used", y), s.at("profit", y)))
    _rewrite(s, used_row, lambda y: "IF(%s=1,MIN(%s,%s),0)" % (
        s.at("loss_cf"), prev("losses_carried", y, "0"), s.at("loss_limit", y)))
    s.yearly("taxable", "Taxable profit", m, "money", lambda y: "MAX(%s,0)-%s" % (s.at("profit", y), s.at("losses_used", y)))
    s.yearly("regular_tax", "Tax at the rate", m, "money", lambda y: "%s*%s" % (s.at("taxable", y), s.at("tax_rate")))
    s.yearly("minimum_tax", "Minimum tax", m, "money", lambda y: "MAX(%s-%s,0)*%s" % (
        s.at("ebit", y), s.at("net_interest", y), s.at("min_tax")))
    _rewrite(s, tax_row, lambda y: "%s+MAX(%s-%s,0)" % (
        s.at("regular_tax", y), s.at("minimum_tax", y), s.at("regular_tax", y)))

    # -- cash flow -------------------------------------------------------------
    s.heading("Levered free cash flow (before repaying debt)")
    s.yearly("cf_net_income", "Net income", m, "money", lambda y: s.at("net_income", y))
    s.yearly("cf_da", "Add back depreciation and amortisation", m, "money", lambda y: s.at("da_row", y))
    pik_row = s.row + 1
    s.yearly("pik_total", "Add back interest accrued, not paid (PIK)", m, "money", lambda y: "0")
    s.yearly("capex_row", "Capital expenditure", m, "money", lambda y: "%s*%s" % (s.at("revenue", y), s.at("capex")))
    s.yearly("nwc_row", "Increase in working capital", m, "money", lambda y: "%s*%s" % (s.at("revenue", y), s.at("nwc")))
    s.yearly("lfcf", "Levered free cash flow", m, "money", lambda y: "%s+%s+%s-%s-%s" % (
        s.at("cf_net_income", y), s.at("cf_da", y), s.at("pik_total", y), s.at("capex_row", y), s.at("nwc_row", y)),
        bold=True)

    # -- cash and the sweep --------------------------------------------------
    s.heading("Cash and the sweep")
    s.yearly("cash_open", "Opening cash", m, "money", lambda y: prev("cash_close", y, s.at("min_cash")))
    s.yearly("cash_in_hand", "Cash in hand (opening cash plus levered FCF)", m, "money",
             lambda y: "%s+%s" % (s.at("cash_open", y), s.at("lfcf", y)))
    mand_row = s.row + 1
    s.yearly("mand_total", "Mandatory repayments", m, "money", lambda y: "0")
    s.yearly("avail", "Cash available to sweep", m, "money", lambda y: "MAX(%s-%s-%s,0)" % (
        s.at("cash_in_hand", y), s.at("mand_total", y), s.at("min_cash")))
    sweep_row = s.row + 1
    s.yearly("sweep_total", "Swept to debt", m, "money", lambda y: "0")
    s.yearly("shortfall", "Shortfall below minimum cash", m, "money", lambda y: "MAX(%s-(%s-%s),0)" % (
        s.at("min_cash"), s.at("cash_in_hand", y), s.at("mand_total", y)))
    draw_row = s.row + 1
    s.yearly("draw_total", "Drawn on revolving facilities to cover it", m, "money", lambda y: "0")
    s.yearly("unfunded", "Shortfall no facility covered (the model assumes it; finding 11)", m, "money",
             lambda y: "MAX(%s-%s,0)" % (s.at("shortfall", y), s.at("draw_total", y)))
    s.yearly("cash_close", "Closing cash (minimum cash plus what no facility could take)", m, "money",
             lambda y: "%s+%s-%s" % (s.at("min_cash"), s.at("avail", y), s.at("sweep_total", y)), bold=True)

    # -- each facility's schedule ----------------------------------------------
    order = list(facilities)
    sweep_order = sorted([i for i in order if facilities[i]["sweep"]], key=lambda i: (facilities[i]["priority"], i))
    for i in order:
        f = facilities[i]
        k = "t%d" % i
        s.heading("Schedule: %s" % f["name"])
        s.yearly(k + "_open", "Opening balance", m, "money",
                 lambda y, k=k: s.at(k + "_drawn") if y == 1 else s.at(k + "_close", y - 1))
        s.yearly(k + "_coupon", "Coupon (opening balance x rate)", m, "money",
                 lambda y, k=k: "%s*%s" % (s.at(k + "_open", y), s.at(k + "_rate", y)))
        s.yearly(k + "_pik_y", "Of which accrued to principal (PIK)", m, "money",
                 lambda y, k=k: "%s*%s" % (s.at(k + "_coupon", y), s.at(k + "_pik")))
        s.yearly(k + "_cfee", "Commitment fee on the undrawn part", m, "money",
                 lambda y, k=k: "IF(%s>0,MAX(%s-%s,0),0)*%s" % (
                     s.at(k + "_undrawn"), s.at(k + "_amount"), s.at(k + "_open", y), s.at(k + "_commit_fee")))
        s.yearly(k + "_interest", "Interest expense", m, "money",
                 lambda y, k=k: "%s+%s" % (s.at(k + "_coupon", y), s.at(k + "_cfee", y)))
        s.yearly(k + "_owed", "Owed before repaying (opening plus PIK)", m, "money",
                 lambda y, k=k: "%s+%s" % (s.at(k + "_open", y), s.at(k + "_pik_y", y)))
        s.yearly(k + "_due", "Repayment the terms set", m, "money",
                 lambda y, k=k, custom=f["custom"]: "IF(%s=1,%s,IF(%s>0,%s*%s,IF(%s=%s,%s,0)))" % (
                     s.at(k + "_custom"), s.at(k + "_sched", y) if custom else "0",
                     s.at(k + "_amort"), s.at(k + "_drawn"), s.at(k + "_amort"),
                     year_cell(y), s.at(k + "_maturity"), s.at(k + "_owed", y)))
        s.yearly(k + "_mand", "Mandatory repayment", m, "money",
                 lambda y, k=k: "MIN(%s,%s)" % (s.at(k + "_due", y), s.at(k + "_owed", y)))
        ahead = sweep_order[:sweep_order.index(i)] if i in sweep_order else []
        s.yearly(k + "_swept", "Cash sweep (swept %s)" % _ordinal(sweep_order, i), m, "money",
                 lambda y, k=k, ahead=ahead: "IF(%s=1,MAX(MIN(%s*%s,%s-%s,%s-%s),0),0)" % (
                     s.at(k + "_sweep"), s.at("avail", y), s.at(k + "_sweep_share"), s.at("avail", y),
                     "(" + _sum([s.at("t%d_swept" % j, y) for j in ahead]) + ")",
                     s.at(k + "_open", y), s.at(k + "_mand", y)))
        s.yearly(k + "_pre", "Balance before any draw", m, "money", lambda y, k=k: "MAX(%s-%s-%s+%s,0)" % (
            s.at(k + "_open", y), s.at(k + "_mand", y), s.at(k + "_swept", y), s.at(k + "_pik_y", y)))
        before = order[:order.index(i)]
        s.yearly(k + "_draw", "Drawn to cover a shortfall", m, "money",
                 lambda y, k=k, before=before: "IF(%s=1,IF(%s>0,MAX(MIN(%s-%s,MAX(%s-%s,0)),0),0),0)" % (
                     s.at(k + "_redraw"), s.at(k + "_undrawn"), s.at("shortfall", y),
                     "(" + _sum([s.at("t%d_draw" % j, y) for j in before]) + ")",
                     s.at(k + "_amount"), s.at(k + "_pre", y)))
        s.yearly(k + "_close", "Closing balance", m, "money",
                 lambda y, k=k: "%s+%s" % (s.at(k + "_pre", y), s.at(k + "_draw", y)), bold=True)

    def total(suffix: str) -> Callable[[int], str]:
        return lambda y: _sum([s.at("t%d_%s" % (i, suffix), y) for i in order])
    _rewrite(s, interest_row, total("interest"))
    _rewrite(s, pik_row, total("pik_y"))
    _rewrite(s, mand_row, total("mand"))
    _rewrite(s, sweep_row, total("swept"))
    _rewrite(s, draw_row, total("draw"))

    # -- returns -------------------------------------------------------------
    s.heading("Returns")
    s.yearly("debt_close", "Debt at the end of the year", m, "money", total("close"))
    last = lambda key: s.at(key, n)       # noqa: E731
    s.scalar("exit_ebitda", "Exit EBITDA (with the lease add-back)", m, "money",
             "%s+%s" % (last("ebitda_row"), s.at("lease_addback")))
    s.scalar("exit_ev", "Exit enterprise value", m, "money", "%s*%s" % (s.at("exit_ebitda"), s.at("exit_multiple")))
    s.scalar("net_debt_exit", "Net debt at exit (with the lease liability)", m, "money",
             "%s-%s+%s" % (last("debt_close"), last("cash_close"), s.at("lease_liability")))
    s.scalar("gross_exit", "Exit equity before the option pool", m, "money",
             "%s-%s" % (s.at("exit_ev"), s.at("net_debt_exit")))
    s.scalar("pool", "Management option pool", m, "money", "%s*%s" % (s.at("gross_exit"), s.at("option_pool")))
    s.scalar("net_exit", "Exit equity", m, "money", "%s-%s" % (s.at("gross_exit"), s.at("pool")), bold=True)
    s.scalar("moic", "MOIC", "x", "multiple", "MAX(%s,0)/%s" % (s.at("net_exit"), s.at("equity")), bold=True)
    s.scalar("irr", "IRR (no interim dividends, so MOIC to the power 1/hold, less 1)", "%", "percent",
             "IF(%s<=0,0-1,%s^(1/%s)-1)" % (s.at("net_exit"), s.at("moic"), s.at("hold")), bold=True)
    if app_irr is not None:
        s.value("app_irr", "IRR in Variater when this file was made", "%", "percent", app_irr, name=False)
        s.cells[-1]["color"] = GREY
        s.cells[-1].pop("fill", None)
        s.scalar("irr_gap", "Difference (this sheet less Variater)", "%", "percent",
                 "%s-%s" % (s.at("irr"), s.at("app_irr")))

    s.resolve()
    widths = {"A": 62, "B": 16, "C": 16, **{get_column_letter(FIRST_YEAR_COLUMN + y): 15 for y in range(n)}}
    model = {"title": SHEET, "cells": s.cells, "widths": widths, "freeze": "D%d" % (s.rows["year"] + 1),
             "names": s.names, "heights": {}, "rows": dict(s.rows)}
    return [model, _about(*about)]


def _rewrite(s: _Sheet, row: int, formula: Callable[[int], str]) -> None:
    """Fill a row placed before the rows its formula reads (interest, tax, totals)."""
    for cell in s.cells:
        if cell["row"] == row and cell["column"] >= FIRST_YEAR_COLUMN:
            cell["formula"] = "=" + formula(cell["column"] - FIRST_YEAR_COLUMN + 1)


def _ordinal(order: list, i: int) -> str:
    if i not in order:
        return "never"
    return {1: "first", 2: "second", 3: "third"}.get(order.index(i) + 1, "%dth" % (order.index(i) + 1))


def _facility_terms(s: _Sheet, i: int, name: str, kind: str, m: str, n: int, *, amount, drawn_pct: float,
                    floating: bool, fixed, floor: float, margin: float, reference: str, references: list,
                    maturity: int, amort, custom: Optional[list], fee: float, commit_fee: float, pik: float,
                    sweep: bool, sweep_share: float, priority: int, redraw: bool) -> dict:
    """One facility's terms: inputs in column C (or formulas, for a percentage deal), its rate every year."""
    k = "t%d" % i
    s.heading("Facility %d: %s" % (i, name))
    s.text(s.row, 2, kind, color=GREY)
    put = lambda key, label, unit, fmt, v: (s.scalar if isinstance(v, str) else s.value)(   # noqa: E731
        k + "_" + key, label, unit, fmt, v)
    put("amount", "Facility size (a revolver's commitment)", m, "money", amount)
    put("drawn_pct", "Drawn at close", "% of the facility", "percent", drawn_pct)
    s.scalar(k + "_drawn", "Drawn at close", m, "money", "%s*%s" % (s.at(k + "_amount"), s.at(k + "_drawn_pct")))
    s.scalar(k + "_undrawn", "Undrawn at close (pays the commitment fee; can be drawn)", m, "money",
             "%s-%s" % (s.at(k + "_amount"), s.at(k + "_drawn")))
    s.value(k + "_pricing", "Pricing: floating or fixed", "rule", "text", "floating" if floating else "fixed")
    put("fixed", "Fixed rate", "% a year", "percent", fixed)
    s.value(k + "_reference", "Reference rate", "benchmark", "text", reference, name=False)
    s.value(k + "_floor", "Floor on the reference rate", "% a year", "percent", floor)
    s.value(k + "_margin", "Margin over the reference rate", "% a year", "percent", margin)
    put("maturity", "Maturity (a bullet is repaid that year)", "years", "years", maturity)
    put("amort", "Amortisation", "% of drawn principal a year", "percent", amort)
    s.value(k + "_custom", "Repaid on its own schedule (the row below)", "yes/no", "flag", 1.0 if custom else 0.0)
    s.value(k + "_fee", "Upfront fee", "% of the facility", "percent", fee)
    s.value(k + "_commit_fee", "Commitment fee", "% a year of the undrawn part", "percent", commit_fee)
    s.value(k + "_pik", "Share of the coupon accrued to principal (PIK)", "% of the coupon", "percent", pik)
    s.value(k + "_sweep", "Takes the cash sweep", "yes/no", "flag", 1.0 if sweep else 0.0)
    s.value(k + "_sweep_share", "Share of the cash available it may take", "%", "percent", sweep_share)
    s.value(k + "_redraw", "Draws to cover a cash shortfall (revolver)", "yes/no", "flag", 1.0 if redraw else 0.0)
    s.inputs_by_year(k + "_ref", "Reference rate", "% a year", "percent", references)
    if custom:
        s.inputs_by_year(k + "_sched", "Repayment schedule", m, "money", (list(custom) + [0.0] * n)[:n])
    s.yearly(k + "_rate", "Rate (floating: the larger of the reference and its floor, plus the margin)",
             "% a year", "percent", lambda y: 'IF(%s="floating",MAX(%s,%s)+%s,%s)' % (
                 s.at(k + "_pricing"), s.at(k + "_ref", y), s.at(k + "_floor"), s.at(k + "_margin"),
                 s.at(k + "_fixed")))
    return {"name": name, "sweep": sweep, "priority": priority, "custom": bool(custom)}


def _listed(s: _Sheet, specs: list, m: str, n: int) -> dict:
    """A deal with a list of facilities (core/debt.py), in the list's order."""
    out = {}
    for i, (spec, name) in enumerate(zip(specs, unique_names(specs)), start=1):
        out[i] = _facility_terms(
            s, i, name, spec.kind, m, n, amount=spec.amount, drawn_pct=spec.drawn_pct / 100,
            floating=spec.floating, fixed=spec.fixed_rate / 100, floor=spec.floor / 100, margin=spec.margin / 100,
            reference=spec.reference_rate, references=[spec.reference_in_year(y) for y in range(1, n + 1)],
            maturity=int(spec.maturity_years), amort=(0.0 if spec.amort_schedule else spec.amort_pct / 100),
            custom=list(spec.amort_schedule) or None, fee=spec.upfront_fee_pct / 100,
            commit_fee=spec.commitment_fee_pct / 100, pik=spec.pik_share / 100, sweep=bool(spec.sweep),
            sweep_share=spec.sweep_share / 100, priority=spec.sweep_priority or i,
            redraw=bool(spec.allow_redraw))
    return out


def _percentages(s: _Sheet, p, m: str, n: int) -> dict:
    """A deal sized by percentages: the senior loan and the mezzanine of
    `build_simple_two_tranche_structure`, sized and priced by formulas of the
    deal's debt share, senior share, base rate and spread."""
    s.heading("Debt as shares of EV")
    s.value("debt_pct", "Debt", "% of entry EV", "percent", p.debt_pct)
    s.value("senior_pct", "Senior loan", "% of debt", "percent", p.senior_pct)
    s.value("base_rate", "Senior rate", "% a year", "percent", p.interest_rate)
    s.value("mezz_spread", "Mezzanine spread over the senior rate", "% a year", "percent", p.mezz_spread)
    s.value("senior_amort", "Senior amortisation", "% of principal a year", "percent", p.senior_amort_pct)
    ev = "((%s+%s)*%s)" % (s.at("ebitda"), s.at("lease_addback"), s.at("entry_multiple"))
    zero = [0.0] * n
    common = dict(drawn_pct=1.0, floating=False, floor=0.0, margin=0.0, reference="custom", references=zero,
                  maturity="%s" % s.at("hold"), custom=None, fee=0.0, commit_fee=0.0, pik=0.0, sweep=True,
                  sweep_share=1.0, redraw=False)
    out = {1: _facility_terms(s, 1, "Senior term loan", "senior", m, n,
                              amount="%s*%s*%s" % (ev, s.at("debt_pct"), s.at("senior_pct")),
                              fixed=s.at("base_rate"), amort=s.at("senior_amort"), priority=1, **common)}
    out[2] = _facility_terms(s, 2, "Mezzanine", "mezzanine", m, n,
                             amount="%s*%s*(1-%s)" % (ev, s.at("debt_pct"), s.at("senior_pct")),
                             fixed="%s+%s" % (s.at("base_rate"), s.at("mezz_spread")), amort=0.0, priority=2,
                             **common)
    return out


def _about(items, values) -> dict:
    cells = []
    for r, (item, value) in enumerate(zip(["Item", *items], ["Value", *values]), start=1):
        head = r == 1
        cells.append({"row": r, "column": 1, "text": str(item), **({"bold": True, "color": WHITE, "fill": HEAD} if head else {})})
        cells.append({"row": r, "column": 2, "text": str(value), **({"bold": True, "color": WHITE, "fill": HEAD} if head else {})})
    return {"title": "About", "cells": cells, "widths": {"A": 28, "B": 60}, "freeze": None, "names": {}, "heights": {}}

write = write_sheets      # the sheets as a workbook of live formulas
