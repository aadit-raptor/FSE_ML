"""Model 1, Occupancy (PLAN.md R4): revenue is places x share filled x price
per day. Hospitals (beds, occupancy, revenue per occupied bed) and hotels
(rooms, occupancy, room rate) are two sectors and this one model.

It is the model built by hand in Excel for Yatharth Hospital and Lemon Tree
Hotels (R4 part a), row for row; `tests/test_research_occupancy.py` holds it
to those two workbooks.

**Revenue.** Line 1 is the occupancy block. For the past a company prints
either its places, share filled and price (revenue is worked out) or its
revenue, share filled and price (the places in use are worked out): the
choice `line1_past`. Line 2 is units x revenue per unit a year (a hotel
chain's fees on managed rooms) and line 3 a share of line 1 (food and drink);
a company without them switches the segments `s2` and `s3` off.

**The lower part.** Four costs, each a share of revenue plus an amount per
place per day; depreciation on fixed assets; other income as a yield on cash;
interest on borrowings and, apart, on lease liabilities; associates; tax;
non-controlling interests; a balance sheet rolled from the last year end and a
cash flow that closes it.

**Not vetted.** No practising analyst has read it yet (phase R, rule 4).
"""
from __future__ import annotations

from research.definition import ASSUME, FIGURE, Heading, Model, Pick, Row, check
from research.pieces import (LY, UNIT, YTD, added_up, balance, balance_rate, cost, days_row, periods, ratio,
                             running_count)

LINE1 = "s1_avg * days * s1_occ * s1_price / " + UNIT
LINE2 = "s2_avg * s2_rpu * days / 365 / " + UNIT
REVENUE = "s1_rev + s2_rev + s3_rev"
PROFIT = "ebitda - dep + othinc - int_debt - int_lease + assoc"
WORKING_CAPITAL = "(inv[prev] - inv) + (rec[prev] - rec) + (oca[prev] - oca) + (pay - pay[prev]) + (ol - ol[prev])"
SUPPLIERS = "(mat + pwr + oth)"

NOTES = (
    "Deferred tax is left out: the tax charged is taken as the tax paid, so no deferred tax balance moves.",
    "The year in progress is the part reported plus the rest assumed. The rest is typed, not grown from the part "
    "reported: a hotel's second half is its strong half.",
    "A price per day is in single units of the currency and the statements are in lakh, crore or million, so "
    "revenue and costs per place are divided by the constant at the top of the sheet.",
    "For the past, a company that prints its revenue, share filled and price has its places in use worked out "
    "from them; the count it prints as capacity may be larger than the count its occupancy is measured on.",
    "Lease liabilities and other balance sheet lines without a rule of their own are held at their last figure "
    "unless an assumption moves them.",
    "The forecast columns hold whatever assumptions were typed. The model states no view and picks no rating.",
)

SECTORS = {
    "hospitals": dict(
        currency="Rs", s1="Beds", place="bed", places="Beds", occ="Occupancy",
        price="Revenue per occupied bed per day, Rs", s1_rev="Hospital revenue",
        s2="Second line", s2_units="Units", s2_rpu="Revenue per unit per year, Rs", s2_rev="Second line revenue",
        s3="Third line", s3_share="Other revenue, as a share of hospital revenue", s3_rev="Other revenue",
        mat="Drugs, consumables and implants", emp="Employees", pwr="Power and fuel", oth="Other expenses"),
    "hotels": dict(
        currency="Rs", s1="Rooms", place="room", places="Rooms", occ="Occupancy",
        price="Average room rate, Rs per night", s1_rev="Room revenue",
        s2="Fees on managed rooms", s2_units="Managed rooms", s2_rpu="Fee per managed room per year, Rs",
        s2_rev="Fees from managed hotels",
        s3="Food, drink and other revenue", s3_share="Food, drink and other, as a share of room revenue",
        s3_rev="Food, drink and other revenue",
        mat="Food and beverages consumed", emp="Employees", pwr="Power and fuel", oth="Other expenses"),
}


def _revenue() -> list:
    line1_count = running_count(
        "s1", "{places}", closures=True,
        end={LY: FIGURE, YTD: Pick("line1_end_to_date", {"typed": FIGURE, "average": "s1_avg"})},
        average=Pick("line1_past", {"drivers": FIGURE, "revenue": "s1_rev * %s / (days * s1_occ * s1_price)" % UNIT}))
    return [
        Heading("REVENUE LINE 1: places x share filled x price per day  [{s1}]"),
        *line1_count,
        Row("s1_occ", "{occ}", "share", {
            "past": FIGURE, "rest": ASSUME, "forecast": ASSUME,
            "current": "(s1_occ[ytd] * s1_avg[ytd] * days[ytd] + s1_occ[roy] * s1_avg[roy] * days[roy]) / (s1_avg * days)"}),
        Row("s1_pg", "   assumption: price growth on the year before", "share", {"forecast": ASSUME}),
        Row("s1_price", "{price}", "rupees", {
            "past": FIGURE, "rest": ASSUME, "forecast": "s1_price[prev] * (1 + s1_pg)",
            "current": "s1_rev * %s / (s1_avg * days * s1_occ)" % UNIT}),
        Row("s1_rev", "{s1_rev}", "money", {
            "past": Pick("line1_past", {"drivers": LINE1, "revenue": FIGURE}),
            "rest": LINE1, "forecast": LINE1, "current": added_up("s1_rev")}),

        Heading("REVENUE LINE 2: units x revenue per unit  [{s2}]"),
        *running_count("s2", "{s2_units}", segment="s2"),
        ratio("s2_rpu", "{s2_rpu}", "div(s2_rev * %s, s2_avg * days / 365)" % UNIT, unit="rupees", segment="s2"),
        Row("s2_rev", "{s2_rev}", "money", {"past": FIGURE, "rest": LINE2, "forecast": LINE2, "current": added_up("s2_rev")},
            segment="s2"),

        Heading("REVENUE LINE 3: free formula, a share of line 1  [{s3}]"),
        ratio("s3_share", "{s3_share}", "div(s3_rev, s1_rev)", segment="s3"),
        Row("s3_rev", "{s3_rev}", "money", {
            "past": "rev - s1_rev - s2_rev", "rest": "s3_share * s1_rev", "forecast": "s3_share * s1_rev",
            "current": added_up("s3_rev")}, segment="s3"),
    ]


def _income() -> list:
    def costs(key):
        return cost(key, "{%s}" % key, revenue="rev", places="s1_avg")

    tax = "tax_rate * max(pbt, 0)"
    return [
        Heading("INCOME STATEMENT"),
        Row("rev", "Revenue from operations", "money",
            {"past": FIGURE, "rest": REVENUE, "forecast": REVENUE, "current": added_up("rev")}, bold=True),
        Row("rev_g", "   growth on the year before", "share", {"current": "rev / rev[prev] - 1", "forecast": "rev / rev[prev] - 1"}),
        *costs("mat"), *costs("emp"), *costs("pwr"), *costs("oth"),
        Row("ebitda", "EBITDA (before other income)", "money", {"all": "rev - mat - emp - pwr - oth"}, bold=True),
        Row("ebitda_m", "   margin", "share", {"all": "div(ebitda, rev)"}),
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


def _balance_sheet() -> list:
    assumed = {"current": ASSUME, "forecast": ASSUME}
    payables = "pay_days * %s / days" % SUPPLIERS
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
        Row("pay_days", "Payables, days of materials, power and other expenses", "count",
            {"past_year": "div(pay * days, %s)" % SUPPLIERS, **assumed}),
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


def build(forecast_years: int = 2) -> Model:
    """Model 1 with as many forecast years as asked for; the rows are the same."""
    model = Model(
        id="occupancy", name="Occupancy", periods=periods(forecast_years),
        constants={UNIT: "{currency} in one unit of the statements (prices per day are in {currency})"},
        rows=(Heading("PERIOD"), days_row(), *_revenue(), *_income(), *_balance_sheet(), *_cash_flow(), *_checks()),
        sectors=SECTORS, notes=NOTES, vetted=False)
    check(model)
    return model


MODEL = build()
