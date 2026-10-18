"""The revenue blocks a sector model is made of, one per segment (PLAN.md R4).

A segment is one line of revenue worked out by one block. Every block returns
ordinary rows whose keys start with the segment's own key (``s1_rev``,
``s2_avg``), so blocks never collide with each other or with the shared lower
part, and its revenue is always the row ``<segment>_rev``. The labels' slots
are the segment's too (``{s1_places}``); `words` gives each block's own words
for a segment that a sector names nothing for.

The eight blocks of the plan:

==============  =====================================================  =====================
block           revenue is                                             counts places
==============  =====================================================  =====================
capacity        places x share filled x price per day                  yes
units           units x revenue per unit a year                        yes
price_volume    volume x price per unit                                no
subscribers     subscribers x revenue per subscriber a month, less     yes
                churn
people          headcount x utilisation x billing rate a year          yes
order_book      the order book x the share of it executed in a year    no
growth          last year's grown                                      no
share           a share of another line (a free formula, or the        no
                analyst's own ``formula``)
==============  =====================================================  =====================

The year in progress is the part reported (typed) plus the rest, which each
block works out from its own assumptions; the current year adds the two.
"""
from __future__ import annotations

from research.definition import ASSUME, FIGURE, Heading, ModelError, Pick, Row
from research.pieces import LY, ROY, UNIT, YTD, added_up, ratio, running_count

BLOCKS = ("capacity", "units", "price_volume", "subscribers", "people", "order_book", "growth", "share")
COUNTING = {"capacity", "units", "subscribers", "people"}       # blocks with an average count of places


def words(block: str, seg: str, n: int) -> dict:
    """The words a block uses when the sector names none for this segment."""
    line = "Line %d" % n
    own = {
        "capacity": {"places": "Places", "occ": "Share filled", "price": "Price per place per day"},
        "units": {"units": "Units", "rpu": "Revenue per unit per year"},
        "price_volume": {"vol": "Volume sold", "price": "Price per unit sold"},
        "subscribers": {"subs": "Subscribers", "arpu": "Revenue per subscriber per month"},
        "people": {"heads": "People", "util": "Share of people billed", "rate": "Billing rate per billed person per year"},
        "order_book": {"book": "Order book"},
        "growth": {},
        "share": {"share": "%s revenue, as a share of the line it follows" % line},
    }[block]
    return {seg: line, seg + "_rev": line + " revenue", **{"%s_%s" % (seg, k): v for k, v in own.items()}}


def rows(block: str, seg: str, n: int, *, of: str | None = None, rest_of_revenue: tuple = (), formula: str | None = None) -> list:
    """The rows of one segment. `n` is its place in the model (line 1 cannot be switched off).

    `of` is the segment a `share` follows; `rest_of_revenue` lists the other
    segments when this one is, for the past, total revenue less theirs;
    `formula` replaces a `share` line's forecast with the analyst's own.
    """
    if block not in BLOCKS:
        raise ModelError("there is no block called %r (the blocks are %s)" % (block, ", ".join(BLOCKS)))
    part = None if n == 1 else seg
    make = {"capacity": _capacity, "units": _units, "price_volume": _price_volume, "subscribers": _subscribers,
            "people": _people, "order_book": _order_book, "growth": _growth}.get(block)
    if make is not None:
        return make(seg, n, part)
    return _share(seg, n, part, of, rest_of_revenue, formula)


def _revenue(seg: str, part, forward: str, past=FIGURE) -> Row:
    return Row(seg + "_rev", "{%s_rev}" % seg, "money",
               {"past": past, "rest": forward, "forecast": forward, "current": added_up(seg + "_rev")}, segment=part)


def _capacity(seg: str, n: int, part) -> list:
    """Places x share filled x price per day. For the past a company prints either its places or its revenue
    (the choice ``line<number>_past``, numbered by the segment's key): the other is worked out."""
    line = "%s_avg * days * %s_occ * %s_price / %s" % (seg, seg, seg, UNIT)
    worked_places = "%s_rev * %s / (days * %s_occ * %s_price)" % (seg, UNIT, seg, seg)
    count = running_count(
        seg, "{%s_places}" % seg, closures=True, segment=part,
        end={LY: FIGURE, YTD: Pick("line%s_end_to_date" % seg[1:], {"typed": FIGURE, "average": seg + "_avg"})},
        average=Pick("line%s_past" % seg[1:], {"drivers": FIGURE, "revenue": worked_places}))
    mean = "(%s_occ[%s] * %s_avg[%s] * days[%s] + %s_occ[%s] * %s_avg[%s] * days[%s]) / (%s_avg * days)" % (
        seg, YTD, seg, YTD, YTD, seg, ROY, seg, ROY, ROY, seg)
    return [
        Heading("REVENUE LINE %d: places x share filled x price per day  [{%s}]" % (n, seg)),
        *count,
        Row(seg + "_occ", "{%s_occ}" % seg, "share", {"past": FIGURE, "rest": ASSUME, "forecast": ASSUME, "current": mean},
            segment=part),
        Row(seg + "_pg", "   assumption: price growth on the year before", "share", {"forecast": ASSUME}, segment=part),
        Row(seg + "_price", "{%s_price}" % seg, "rupees", {
            "past": FIGURE, "rest": ASSUME, "forecast": "%s_price[prev] * (1 + %s_pg)" % (seg, seg),
            "current": "%s_rev * %s / (%s_avg * days * %s_occ)" % (seg, UNIT, seg, seg)}, segment=part),
        _revenue(seg, part, line, past=Pick("line%s_past" % seg[1:], {"drivers": line, "revenue": FIGURE})),
    ]


def _units(seg: str, n: int, part) -> list:
    """Units x revenue per unit a year (a hotel chain's managed rooms and its fee on each)."""
    line = "%s_avg * %s_rpu * days / 365 / %s" % (seg, seg, UNIT)
    return [
        Heading("REVENUE LINE %d: units x revenue per unit  [{%s}]" % (n, seg)),
        *running_count(seg, "{%s_units}" % seg, segment=part),
        ratio(seg + "_rpu", "{%s_rpu}" % seg, "div(%s_rev * %s, %s_avg * days / 365)" % (seg, UNIT, seg), unit="rupees", segment=part),
        _revenue(seg, part, line),
    ]


def _price_volume(seg: str, n: int, part) -> list:
    """Volume x price per unit (tonnes of cement and the realisation per tonne). The past's price is worked out
    from the revenue and volume printed; the forecast grows the volume and the price."""
    line = "%s_vol * %s_price / %s" % (seg, seg, UNIT)
    price = "div(%s_rev * %s, %s_vol)" % (seg, UNIT, seg)
    return [
        Heading("REVENUE LINE %d: volume x price  [{%s}]" % (n, seg)),
        Row(seg + "_vol", "{%s_vol}" % seg, "count", {
            "past": FIGURE, "rest": ASSUME, "current": added_up(seg + "_vol"),
            "forecast": "%s_vol[prev] * (1 + %s_vg)" % (seg, seg)}, segment=part),
        Row(seg + "_vg", "   assumption: volume growth on the year before", "share", {"forecast": ASSUME}, segment=part),
        Row(seg + "_price", "{%s_price}" % seg, "rupees", {
            "past": price, "rest": ASSUME, "current": price,
            "forecast": "%s_price[prev] * (1 + %s_pg)" % (seg, seg)}, segment=part),
        Row(seg + "_pg", "   assumption: price growth on the year before", "share", {"forecast": ASSUME}, segment=part),
        _revenue(seg, part, line),
    ]


def _subscribers(seg: str, n: int, part) -> list:
    """Subscribers x revenue per subscriber a month, less churn: a month's churn is a share of the subscribers
    at the start, so a period loses that share for each of its months (days x 12 / 365)."""
    months = "days * 12 / 365"
    forward = "%s_open + %s_add - %s_cls" % (seg, seg, seg)
    mean = "(%s_open + %s_end) / 2" % (seg, seg)
    lost = "%s_churn * %s_open * %s" % (seg, seg, months)
    line = "%s_avg * %s_arpu * %s / %s" % (seg, seg, months, UNIT)
    return [
        Heading("REVENUE LINE %d: subscribers x revenue per subscriber, less churn  [{%s}]" % (n, seg)),
        Row(seg + "_open", "{%s_subs}: at the start" % seg, "count",
            {"rest": "%s_end[%s]" % (seg, YTD), "forecast": "%s_end[prev]" % seg}, segment=part),
        Row(seg + "_add", "   gross additions", "count", {"rest": ASSUME, "forecast": ASSUME}, segment=part),
        Row(seg + "_churn", "   assumption: share of subscribers lost each month", "share",
            {"rest": ASSUME, "forecast": ASSUME}, segment=part),
        Row(seg + "_cls", "   lost to churn", "count", {"rest": lost, "forecast": lost}, segment=part),
        Row(seg + "_end", "   at the end", "count",
            {"past": FIGURE, "rest": forward, "current": "%s_end[%s]" % (seg, ROY), "forecast": forward}, segment=part),
        Row(seg + "_avg", "   average over the period", "count", {
            "past": FIGURE, "rest": mean, "forecast": mean,
            "current": "(%s_avg[%s] * days[%s] + %s_avg[%s] * days[%s]) / days" % (seg, YTD, YTD, seg, ROY, ROY)},
            segment=part),
        ratio(seg + "_arpu", "{%s_arpu}" % seg, "div(%s_rev * %s, %s_avg * %s)" % (seg, UNIT, seg, months),
              unit="rupees", segment=part),
        _revenue(seg, part, line),
    ]


def _people(seg: str, n: int, part) -> list:
    """Headcount x the share billed x a billing rate a year."""
    line = "%s_avg * %s_util * %s_rate * days / 365 / %s" % (seg, seg, seg, UNIT)
    return [
        Heading("REVENUE LINE %d: headcount x utilisation x billing rate  [{%s}]" % (n, seg)),
        *running_count(seg, "{%s_heads}" % seg, segment=part),
        Row(seg + "_util", "{%s_util}" % seg, "share", {"past": FIGURE, "rest": ASSUME, "forecast": ASSUME,
            "current": "(%s_util[%s] * %s_avg[%s] * days[%s] + %s_util[%s] * %s_avg[%s] * days[%s]) / (%s_avg * days)" % (
                seg, YTD, seg, YTD, YTD, seg, ROY, seg, ROY, ROY, seg)}, segment=part),
        ratio(seg + "_rate", "{%s_rate}" % seg, "div(%s_rev * %s, %s_avg * %s_util * days / 365)" % (seg, UNIT, seg, seg),
              unit="rupees", segment=part),
        _revenue(seg, part, line),
    ]


def _order_book(seg: str, n: int, part) -> list:
    """The order book x the share of it executed in a year. Revenue is what is executed; the book at the end is
    the book at the start plus new orders less revenue. The book is in the statements' money."""
    executed = "%s_exec * %s_ob_open * days / 365" % (seg, seg)
    end = "%s_ob_open + %s_ob_in - %s_rev" % (seg, seg, seg)
    return [
        Heading("REVENUE LINE %d: order book x share executed  [{%s}]" % (n, seg)),
        Row(seg + "_ob_open", "{%s_book}: at the start" % seg, "money",
            {"rest": "%s_ob_end[%s]" % (seg, YTD), "forecast": "%s_ob_end[prev]" % seg}, segment=part),
        Row(seg + "_ob_in", "   new orders", "money",
            {"past": FIGURE, "rest": ASSUME, "current": added_up(seg + "_ob_in"), "forecast": ASSUME}, segment=part),
        Row(seg + "_ob_end", "   at the end", "money",
            {"past": FIGURE, "rest": end, "current": "%s_ob_end[%s]" % (seg, ROY), "forecast": end}, segment=part),
        Row(seg + "_exec", "   share of the book at the start executed in a year", "share", {
            "to_date": "div(%s_rev * 365 / days, %s_ob_end[%s])" % (seg, seg, LY), "rest": ASSUME,
            "current": "div(%s_rev, %s_ob_end[%s])" % (seg, seg, LY), "forecast": ASSUME}, segment=part),
        _revenue(seg, part, executed),
    ]


def _growth(seg: str, n: int, part) -> list:
    """Last year's revenue grown. The year in progress takes its growth on last year; its rest is what that
    growth leaves after the part reported."""
    return [
        Heading("REVENUE LINE %d: growth on the year before  [{%s}]" % (n, seg)),
        Row(seg + "_g", "   assumption: growth on the year before", "share", {"current": ASSUME, "forecast": ASSUME},
            segment=part),
        Row(seg + "_rev", "{%s_rev}" % seg, "money", {
            "past": FIGURE, "rest": "%s_rev[%s] * (1 + %s_g[cy]) - %s_rev[%s]" % (seg, LY, seg, seg, YTD),
            "current": added_up(seg + "_rev"), "forecast": "%s_rev[prev] * (1 + %s_g)" % (seg, seg)}, segment=part),
    ]


def _share(seg: str, n: int, part, of: str | None, rest_of_revenue: tuple, formula: str | None) -> list:
    """A free formula: a share of another line (food and drink on room revenue), or the analyst's own."""
    if of is None and formula is None:
        raise ModelError("line %d is a share of another line: say which (`of`)" % n)
    forward = formula or "%s_share * %s_rev" % (seg, of)
    past = " - ".join(["rev"] + ["%s_rev" % s for s in rest_of_revenue]) if rest_of_revenue else FIGURE
    heading = ("free formula, a share of line %s" % of[1:]) if formula is None else "free formula"
    out = [Heading("REVENUE LINE %d: %s  [{%s}]" % (n, heading, seg))]
    if formula is None:
        out.append(ratio(seg + "_share", "{%s_share}" % seg, "div(%s_rev, %s_rev)" % (seg, of), segment=part))
    out.append(_revenue(seg, part, forward, past=past))
    return out
