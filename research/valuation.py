"""Valuation (PLAN.md R6): a discounted cash flow and a target multiple,
written inside the model definition, so the workbook holds them as formulas
like every other row and the server works them out from the same trees.

**As at the end of last year.** Net debt, non-controlling interests and the
share count are last year's balance sheet: the latest one in the accepted
results (``ly``), so the valuation sits in last year's column. The
assumptions (discount rate, growth after the forecast, target multiple, the
share price and its date) are typed once, in that column too: a roll forward
leaves them where they are.

**The discounted cash flow.** Each year from the current one on, the cash
flow to the firm is EBITDA less tax on operating profit (EBITDA less
depreciation, at the year's tax rate), less capital expenditure, plus the
change in working capital. The current year is discounted one year, the
first forecast year two, and so on; after the last forecast year the cash
flow grows for ever at the growth typed (Gordon's formula), discounted with
that year. The enterprise value less net debt (borrowings and leases less
cash) and non-controlling interests at book is the owners' value, and over
the shares the value per share.

**The target multiple** is the analyst's: EV/EBITDA (the multiple of a
year's EBITDA, less the same net debt and non-controlling interests) or P/E
(the multiple of a year's earnings per share), on the year the analyst
chooses. Each year's value is shown, so the choice is in plain sight.

**The price** is typed with its date; each method's value is set against it
as a percentage. The model states no view: it never words a rating, a call
or a target of its own (`tests/test_research_valuation.py` reads every word
the screen and the workbook can show).
"""
from __future__ import annotations

from research.definition import ASSUME, Heading, Pick, Row
from research.pieces import CY, LY

VALUE_YEAR = "value_year"           # the year the target multiple is applied to
VALUE_MULTIPLE = "value_multiple"   # EV/EBITDA or P/E
MULTIPLES = ("ev_ebitda", "pe")

NOTES = (
    "Valuation is as at the end of last year, the latest balance sheet in the accepted results: net debt, "
    "non-controlling interests (at book value) and the share count are that year end's, and every cash flow from "
    "the current year on is discounted to it, the current year by one year. Compare the price with that in mind.",
    "The cash flow to the firm is EBITDA less tax on operating profit, capital expenditure and the change in "
    "working capital. Other income, associates and investments are left out of the value; cash is in net debt.",
    "After the last forecast year the cash flow grows for ever at the rate typed; a growth rate at or above the "
    "discount rate gives no value.",
)

def rows(forecast_years: int) -> list:
    """The valuation rows for a model with `forecast_years` forecast years (``y1`` ...)."""
    years = [CY] + ["y%d" % n for n in range(1, forecast_years + 1)]
    last = years[-1]
    each = lambda formula: {"current": formula, "forecast": formula}   # noqa: E731
    fcf = "ebitda - tax_rate * max(ebitda - dep, 0) + cf_wc - capex"
    present = " + ".join("v_pv[%s]" % y for y in years) + " + v_tv[%s] * v_df[%s]" % (last, last)
    on_year = Pick(VALUE_YEAR, {y: "v_mv[%s]" % y for y in years[1:] + years[:1]})     # the first forecast year first
    return [
        Heading("VALUATION (as at the end of last year; the analyst's assumptions are typed in last year's column)"),
        Row("v_rate", "Discount rate, a year", "share", {"past_year": ASSUME}),
        Row("v_g", "Growth of the cash flow after the last forecast year, a year", "share", {"past_year": ASSUME}),
        Row("v_fcf", "Cash flow to the firm: EBITDA less tax on operating profit, capital expenditure and the "
                     "change in working capital", "money", each(fcf)),
        Row("v_df", "   discount factor to the end of last year", "factor",
            {"current": "1 / (1 + v_rate[%s])" % LY, "forecast": "v_df[prev] / (1 + v_rate[%s])" % LY}),
        Row("v_pv", "   present value", "money", each("v_fcf * v_df")),
        Row("v_tv", "Value of the cash flows after the last forecast year, at that year's end", "money",
            {last: "v_fcf * (1 + v_g[%s]) / (v_rate[%s] - v_g[%s])" % (LY, LY, LY)}),
        Row("v_ev", "Enterprise value: the present values and the value after the forecast, discounted", "money",
            {LY: present}, bold=True),
        Row("v_eq", "Less net debt and non-controlling interests: the owners' value", "money",
            {LY: "v_ev - netdebt - nci"}),
        Row("v_ps", "Value per share, discounted cash flow, {currency}", "rupees2", {LY: "v_eq / shares"}, bold=True),
        Row("v_mult", "Target multiple", "times", {"past_year": ASSUME}),
        Row("v_mv", "   value per share at the target multiple of this year's figure, {currency}", "rupees2", each(
            Pick(VALUE_MULTIPLE, {
                "ev_ebitda": "(v_mult[%s] * ebitda - netdebt[%s] - nci[%s]) / shares[%s]" % (LY, LY, LY, LY),
                "pe": "v_mult[%s] * eps" % LY}))),
        Row("v_mps", "Value per share, target multiple on the year chosen, {currency}", "rupees2", {LY: on_year},
            bold=True),
        Row("v_price", "Share price typed, {currency}", "rupees2", {"past_year": ASSUME}),
        Row("v_date", "   the date of that price", "date", {"past_year": ASSUME}),
        Row("v_gap_dcf", "Discounted cash flow value against the price", "share", {LY: "v_ps / v_price - 1"}),
        Row("v_gap_mult", "Target multiple value against the price", "share", {LY: "v_mps / v_price - 1"}),
    ]





# ---------------------------------------------------------------------------
# The country's risk-free rate, offered as a starting point (PLAN.md 4.2's stored figures)
# ---------------------------------------------------------------------------
NO_CURRENCY = "no_currency"         # nothing accepted for the company yet, or its results are in several currencies
NO_COUNTRY = "no_country"           # its currency is no one country's we hold figures for (the euro: whose yield?)
NO_DATA = "no_data"                 # no yield stored for that country


def risk_free(currencies, store, today) -> dict:
    """The 10-year government bond yield of the country whose currency the company reports in, as stored by the
    nightly economic data refresh; never a figure fetched for the request. `currencies` are those of the company's
    accepted results. The answer says why when there is none: the analyst then types the rate."""
    from economy import views
    from economy.catalogue import AREAS, COUNTRIES

    found = {c for c in currencies if c}
    if len(found) != 1:
        return {"status": NO_CURRENCY, "currency": None, "country": None}
    currency = found.pop()
    countries = [c for c in COUNTRIES if AREAS[c].currency == currency]
    if len(countries) != 1:
        return {"status": NO_COUNTRY, "currency": currency, "country": None}
    figure = views.country_figures(store, countries[0], today).get("bond_yield_10y")
    if figure is None:
        return {"status": NO_DATA, "currency": currency, "country": countries[0]}
    return {"status": "ok", "currency": currency, "country": countries[0], "value": figure.value / 100,
            "period": figure.period, "source": figure.source, "source_series": figure.source_series,
            "url": figure.url, "current": figure.current}


__all__ = ["MULTIPLES", "NOTES", "NO_COUNTRY", "NO_CURRENCY", "NO_DATA", "VALUE_MULTIPLE", "VALUE_YEAR", "risk_free",
           "rows"]
