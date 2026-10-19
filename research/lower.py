"""The lower part every sector model shares (PLAN.md R4): costs,
depreciation, other income, interest on borrowings and on leases, associates,
tax, non-controlling interests, a balance sheet rolled from the last year end
and a cash flow that closes it, and the checks of the model's past against the
filing's own totals.

It reads the revenue blocks through two names only: each segment's
``<segment>_rev`` (revenue is their sum) and the row costs follow
(``driver``): a segment's average count of places, for costs per place per
day, or its volume sold, for costs per unit sold (PLAN.md R7). A model with
neither has costs as a share of revenue alone.

**The cost lines** are a model's own (``costs``): model 1 has four
(`COSTS`), a cement model adds freight, a retailer has no power line of its
own. Every model has materials, employees and other expenses; payables are
days of every cost but employees.
"""
from __future__ import annotations

from research.definition import ASSUME, FIGURE, Heading, ModelError, Row
from research.pieces import UNIT, added_up, balance, balance_rate, cost, cost_per_unit, ratio

WORKING_CAPITAL = "(inv[prev] - inv) + (rec[prev] - rec) + (oca[prev] - oca) + (pay - pay[prev]) + (ol - ol[prev])"
PROFIT = "ebitda - dep + othinc - int_debt - int_lease + assoc"
COSTS = ("mat", "emp", "pwr", "oth")
COST_LINES = ("mat", "emp", "pwr", "frt", "oth")     # every cost line a model may have, in the order shown
NEEDED = ("mat", "emp", "oth")                       # and the ones every model has

NOTES = (
    "Deferred tax is left out: the tax charged is taken as the tax paid, so no deferred tax balance moves.",
    "Lease liabilities and other balance sheet lines without a rule of their own are held at their last figure "
    "unless an assumption moves them.",
    "The forecast columns hold whatever assumptions were typed. The model states no view and picks no rating.",
)


def check_costs(costs: tuple) -> None:
    if any(c not in COST_LINES for c in costs) or not set(NEEDED) <= set(costs) or len(set(costs)) != len(costs):
        raise ModelError("a model's cost lines are %s, each once, with %s among them, not %r"
                         % (", ".join(COST_LINES), ", ".join(NEEDED), costs))


def suppliers(costs: tuple) -> str:
    return "(%s)" % " + ".join(c for c in costs if c != "emp")


def rows(revenues: list, driver: tuple | None, costs: tuple = COSTS) -> list:
    """The income statement, balance sheet, cash flow and checks under the revenue lines `revenues`. `driver`
    is (row, "day") for costs per place per day, (row, "unit") for costs per unit sold, or None."""
    check_costs(costs)
    return [*_income(" + ".join(revenues), driver, costs), *_balance_sheet(costs), *_cash_flow(), *_checks()]


def _income(revenue: str, driver: tuple | None, costs_: tuple) -> list:
    on, per = driver or (None, "day")

    def costs(key):
        if per == "unit":
            return cost_per_unit(key, "{%s}" % key, revenue="rev", volume=on)
        return cost(key, "{%s}" % key, revenue="rev", places=on)

    tax = "tax_rate * max(pbt, 0)"
    per_unit = [Row("ebitda_pu", "   EBITDA per {place} sold, {currency}", "rupees",
                    {"all": "div(ebitda * %s, %s)" % (UNIT, on)})] if per == "unit" else []
    return [
        Heading("INCOME STATEMENT"),
        Row("rev", "Revenue from operations", "money",
            {"past": FIGURE, "rest": revenue, "forecast": revenue, "current": added_up("rev")}, bold=True),
        Row("rev_g", "   growth on the year before", "share", {"current": "rev / rev[prev] - 1", "forecast": "rev / rev[prev] - 1"}),
        *[r for key in costs_ for r in costs(key)],
        Row("ebitda", "EBITDA (before other income)", "money", {"all": " - ".join(("rev",) + tuple(costs_))}, bold=True),
        Row("ebitda_m", "   margin", "share", {"all": "div(ebitda, rev)"}),
        *per_unit,
        *balance_rate("dep", "Depreciation and amortisation", "dep_rate",
                      "   a year's charge as a share of fixed assets at the start", balance="fa"),
        *balance_rate("othinc", "Other income and finance income", "oi_yield",
                      "   a year's income as a share of cash and investments at the start", balance="cash"),
        *balance_rate("int_debt", "Interest on borrowings", "debt_rate", "   interest rate on average borrowings",
                      balance="debt", basis="average"),
        *balance_rate("int_lease", "Interest on lease liabilities", "lease_rate",
                      "   interest rate on lease liabilities at the start", balance="lease"),
        Row("assoc", "Share of profit or loss of associates", "money",
            {"past": FIGURE, "rest": ASSUME, "forecast": ASSUME, "current": added_up("assoc")}),
        Row("pbt", "Profit before tax", "money", {"all": PROFIT}, bold=True),
        Row("tax", "Tax", "money", {"past": FIGURE, "rest": tax, "forecast": tax, "current": added_up("tax")}),
        ratio("tax_rate", "   tax as a share of profit before tax", "div(tax, pbt)"),
        Row("pat", "Profit after tax", "money", {"all": "pbt - tax"}, bold=True),
        Row("mi", "Non-controlling interests' share", "money",
            {"past": FIGURE, "rest": "mi_share * pat", "forecast": "mi_share * pat", "current": added_up("mi")}),
        ratio("mi_share", "   as a share of profit after tax", "div(mi, pat)"),
        Row("pat_own", "Profit for the owners of the company", "money", {"all": "pat - mi"}, bold=True),
        Row("shares", "Shares, in the statements' unit of count", "count3", {"past": FIGURE, "current": ASSUME, "forecast": ASSUME}),
        Row("eps", "Earnings per share, {currency}", "rupees2", {"year_end": "div(pat_own, shares)"}),
    ]


def _balance_sheet(costs: tuple) -> list:
    assumed = {"current": ASSUME, "forecast": ASSUME}
    payables = "pay_days * %s / days" % suppliers(costs)
    pay_label = ("Payables, days of materials, power and other expenses" if costs == COSTS
                 else "Payables, days of every cost but employees")
    short = "max(min_cash - cash, 0)"
    return [
        Heading("BALANCE SHEET (year ends; the year in progress is rolled from the last year end)"),
        balance("fa", "Fixed assets (work in progress, intangibles and leased assets included)", "+ capex - dep"),
        balance("gw", "Goodwill"),
        balance("onca", "Investments and other non-current assets", "+ assoc"),
        Row("inv", "Inventories", "money", {"past": FIGURE, "current": "inv_days * mat / days", "forecast": "inv_days * mat / days"}),
        Row("rec", "Trade receivables", "money", {"past": FIGURE, "current": "rec_days * rev / days", "forecast": "rec_days * rev / days"}),
        balance("oca", "Other current assets"),
        Row("cash", "Cash, bank balances and liquid investments", "money", {"past": FIGURE, "current": "cf_close", "forecast": "cf_close"}),
        Row("ta", "Total assets", "money", {"year_end": "fa + gw + onca + inv + rec + oca + cash"}, bold=True),
        balance("eq", "Equity of the owners", "+ pat_own - dividends"),
        balance("nci", "Non-controlling interests", "+ mi"),
        balance("debt", "Borrowings", "+ drawn - repaid"),
        balance("lease", "Lease liabilities", "- lease_red"),
        Row("pay", "Trade payables", "money", {"past": FIGURE, "current": payables, "forecast": payables}),
        balance("ol", "Other liabilities and provisions"),
        Row("tle", "Total equity and liabilities", "money", {"year_end": "eq + nci + debt + lease + pay + ol"}, bold=True),
        Row("chk_bs", "CHECK: assets less equity and liabilities (must be 0)", "check", {"year_end": "ta - tle"}),

        Heading("Balance sheet assumptions"),
        Row("capex", "Capital expenditure in the year", "money", assumed),
        Row("drawn", "Borrowings drawn in the year", "money", assumed),
        Row("repaid", "Borrowings repaid in the year", "money", assumed),
        Row("lease_red", "Lease liabilities paid down in the year", "money", assumed),
        Row("dividends", "Dividends paid in the year", "money", assumed),
        Row("inv_days", "Inventories, days of materials used", "count", {"past_year": "div(inv * days, mat)", **assumed}),
        Row("rec_days", "Receivables, days of revenue", "count", {"past_year": "div(rec * days, rev)", **assumed}),
        Row("pay_days", pay_label, "count", {"past_year": "div(pay * days, %s)" % suppliers(costs), **assumed}),
        Row("min_cash", "The least cash the company wants to hold", "money", assumed),
        Row("short", "Cash still to be found (0 when the plan funds itself)", "money", {"current": short, "forecast": short}),
    ]


def _cash_flow() -> list:
    def year(key, label, formula, bold=False):
        return Row(key, label, "money", {"current": formula, "forecast": formula}, bold=bold)

    multiple = "div(netdebt, ebitda)"
    return [
        Heading("CASH FLOW (forecast years)"),
        year("cf_pat", "Profit after tax", "pat"),
        year("cf_dep", "Depreciation and amortisation", "dep"),
        year("cf_assoc", "Less associates' profit (not cash)", "0 - assoc"),
        year("cf_wc", "Change in working capital", WORKING_CAPITAL),
        year("cf_ops", "Cash from operations", "cf_pat + cf_dep + cf_assoc + cf_wc", bold=True),
        year("cf_capex", "Capital expenditure", "0 - capex"),
        year("cf_debt", "Borrowings drawn less repaid", "drawn - repaid"),
        year("cf_lease", "Lease liabilities paid down", "0 - lease_red"),
        year("cf_div", "Dividends", "0 - dividends"),
        year("cf_change", "Change in cash", "cf_ops + cf_capex + cf_debt + cf_lease + cf_div", bold=True),
        year("cf_open", "Cash at the start", "cash[prev]"),
        year("cf_close", "Cash at the end", "cf_open + cf_change", bold=True),
        Row("netdebt", "Borrowings and leases less cash", "money", {"year_end": "debt + lease - cash"}),
        Row("nd_ebitda", "   as a multiple of the year's EBITDA", "times",
            {"past_year": multiple, "current": multiple, "forecast": multiple}),
    ]


def _checks() -> list:
    return [
        Heading("CHECKS ON THE PAST (the model's lines against the filing's own totals; must be 0)"),
        Row("r_pbt", "Profit before tax, as the filing prints it", "money", {"past": FIGURE}),
        Row("r_pat", "Profit after tax, as the filing prints it", "money", {"past": FIGURE}),
        Row("r_own", "Owners' profit, as the filing prints it", "money", {"past": FIGURE}),
        Row("chk_pbt", "Profit before tax: model less filing", "check", {"past": "pbt - r_pbt"}),
        Row("chk_pat", "Profit after tax: model less filing", "check", {"past": "pat - r_pat"}),
        Row("chk_own", "Owners' profit: model less filing", "check", {"past": "pat_own - r_own"}),
    ]
