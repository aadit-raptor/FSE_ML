"""The pieces every sector model shares (PLAN.md R4, "What the blocks lack").

Each function returns ordinary rows, so the evaluator and the workbook writer
never know a piece was used. They are written for the standard periods of
`periods()`: last year, the year to date, the rest of the year, the current
year (the two added up) and the forecast years.

Money is in the statements' unit and prices are in single units of the
currency, so a price times a count is divided by the constant `unit_rupees`
(100,000 for statements in lakh).
"""
from __future__ import annotations

from research.definition import ASSUME, FIGURE, Period, Row, Rule

LY, YTD, ROY, CY = "ly", "ytd", "roy", "cy"
UNIT = "unit_rupees"


def periods(forecast_years: int = 2) -> tuple:
    """Last year, year to date, rest of year, current year, then the forecast years."""
    out = [Period(LY, "past_year"), Period(YTD, "to_date"), Period(ROY, "rest"), Period(CY, "current", prev=LY)]
    for n in range(1, forecast_years + 1):
        out.append(Period("y%d" % n, "forecast", prev=CY if n == 1 else "y%d" % (n - 1)))
    return tuple(out)


def added_up(key: str) -> str:
    """The current year of a flow: the part reported plus the part assumed."""
    return "%s[%s] + %s[%s]" % (key, YTD, key, ROY)


def days_row(key: str = "days") -> Row:
    """The days in each period: a quarter is not a quarter of 365."""
    return Row(key, "Days in the period", "days", {"past": FIGURE, "rest": ASSUME, "current": added_up(key), "forecast": ASSUME})


def running_count(key: str, label: str, closures: bool = False, days: str = "days", segment: str | None = None,
                  end: dict | None = None, average: Rule = FIGURE) -> list:
    """Beds, rooms, stores: at the start, added, closed, at the end, and the average.

    The forecast opens where the period before ended. The current year's
    average weighs its two parts by their days. `end` and `average` replace
    the past's rules (typed figures by default) where a company prints
    something else.
    """
    closed = " - %s_cls" % key if closures else ""
    forward = "%s_open + %s_add%s" % (key, key, closed)
    mean = "(%s_open + %s_end) / 2" % (key, key)
    rows = [
        Row(key + "_open", label + ": at the start", "count", {"rest": "%s_end[%s]" % (key, YTD), "forecast": "%s_end[prev]" % key}, segment=segment),
        Row(key + "_add", "   added" + ("" if closures else ", net"), "count", {"rest": ASSUME, "forecast": ASSUME}, segment=segment),
    ]
    if closures:
        rows.append(Row(key + "_cls", "   closed", "count", {"rest": ASSUME, "forecast": ASSUME}, segment=segment))
    rows += [
        Row(key + "_end", "   at the end", "count",
            {"past": FIGURE, **(end or {}), "rest": forward, "current": "%s_end[%s]" % (key, ROY), "forecast": forward}, segment=segment),
        Row(key + "_avg", "   average over the period", "count",
            {"past": average, "rest": mean, "forecast": mean,
             "current": "(%s_avg[%s] * %s[%s] + %s_avg[%s] * %s[%s]) / %s" % (key, YTD, days, YTD, key, ROY, days, ROY, days)},
            segment=segment),
    ]
    return rows


def cost(key: str, label: str, revenue: str, places: str | None, place: str = "{place}", days: str = "days") -> list:
    """A cost that is a share of revenue plus an amount per place per day.

    One of the two assumptions is left at nil: food follows revenue, staff and
    power follow rooms. The two "as it came out" rows show every period both
    ways, so the analyst sees which one the past supports. A model that counts
    no places (`places` None) has the share of revenue alone.
    """
    if places is None:
        forward = "%s_sh * %s" % (key, revenue)
        return [
            Row(key, label, "money", {"past": FIGURE, "rest": forward, "current": added_up(key), "forecast": forward}),
            Row(key + "_sh", "   assumption: share of revenue", "share", {"rest": ASSUME, "forecast": ASSUME}),
            Row(key + "_m1", "   as it came out: share of revenue", "share", {"all": "div(%s, %s)" % (key, revenue)}),
        ]
    forward = "%s_sh * %s + %s_pd * %s * %s / %s" % (key, revenue, key, places, days, UNIT)
    return [
        Row(key, label, "money", {"past": FIGURE, "rest": forward, "current": added_up(key), "forecast": forward}),
        Row(key + "_sh", "   assumption: share of revenue", "share", {"rest": ASSUME, "forecast": ASSUME}),
        Row(key + "_pd", "   assumption: {currency} per %s per day" % place, "rupees", {"rest": ASSUME, "forecast": ASSUME}),
        Row(key + "_m1", "   as it came out: share of revenue", "share", {"all": "div(%s, %s)" % (key, revenue)}),
        Row(key + "_m2", "   as it came out: {currency} per %s per day" % place, "rupees",
            {"all": "div(%s * %s, %s * %s)" % (key, UNIT, places, days)}),
    ]


def cost_per_unit(key: str, label: str, revenue: str, volume: str, place: str = "{place}") -> list:
    """A cost that is a share of revenue plus an amount per unit sold (cement: power and fuel, freight per
    tonne). One of the two assumptions is left at nil, as in `cost`."""
    forward = "%s_sh * %s + %s_pu * %s / %s" % (key, revenue, key, volume, UNIT)
    return [
        Row(key, label, "money", {"past": FIGURE, "rest": forward, "current": added_up(key), "forecast": forward}),
        Row(key + "_sh", "   assumption: share of revenue", "share", {"rest": ASSUME, "forecast": ASSUME}),
        Row(key + "_pu", "   assumption: {currency} per %s sold" % place, "rupees", {"rest": ASSUME, "forecast": ASSUME}),
        Row(key + "_m1", "   as it came out: share of revenue", "share", {"all": "div(%s, %s)" % (key, revenue)}),
        Row(key + "_m2", "   as it came out: {currency} per %s sold" % place, "rupees",
            {"all": "div(%s * %s, %s)" % (key, UNIT, volume)}),
    ]


def balance_rate(key: str, label: str, rate_key: str, rate_label: str, balance: str, basis: str = "opening",
                 days: str = "days") -> list:
    """A yearly flow that is a rate on a balance: interest, income on cash, depreciation.

    `basis` is the balance the rate applies to: at the start of the year, or
    the average of the start and the end. The rest of a year in progress takes
    its share of the year by days. The rate is typed for the future and shown
    as it came out for the past (a part year scaled up to a year on the
    opening balance, since a quarter's filing carries no balance sheet).
    """
    if basis not in ("opening", "average"):
        raise ValueError("basis is 'opening' or 'average', not %r" % basis)
    if basis == "opening":
        this_year, later, whole = "%s[%s]" % (balance, LY), "%s[prev]" % balance, "%s[%s]" % (balance, LY)
    else:
        this_year = "(%s[%s] + %s[%s]) / 2" % (balance, LY, balance, CY)
        later = "(%s[prev] + %s) / 2" % (balance, balance)
        whole = "(%s[%s] + %s) / 2" % (balance, LY, balance)
    return [
        Row(key, label, "money", {
            "past": FIGURE, "rest": "%s * %s * %s / 365" % (rate_key, this_year, days),
            "current": added_up(key), "forecast": "%s * %s" % (rate_key, later)}),
        Row(rate_key, rate_label, "share", {
            "to_date": "div(%s * 365 / %s, %s[%s])" % (key, days, balance, LY), "rest": ASSUME,
            "current": "div(%s, %s)" % (key, whole), "forecast": ASSUME}),
    ]


def ratio(key: str, label: str, worked_out: str, unit: str = "share", segment: str | None = None) -> Row:
    """A ratio: worked out in the past, typed for the future, and shown as it came out for the year in progress."""
    return Row(key, label, unit, {"past": worked_out, "rest": ASSUME, "current": worked_out, "forecast": ASSUME}, segment=segment)


def balance(key: str, label: str, movement: str = "", bold: bool = False) -> Row:
    """A balance sheet line that rolls: last year's, plus or minus the year's movement."""
    roll = "%s[prev]%s" % (key, (" " + movement) if movement else "")
    return Row(key, label, "money", {"past": FIGURE, "current": roll, "forecast": roll}, bold=bold)
