"""Model 1, Occupancy (PLAN.md R4): revenue is places x share filled x price
per day. Hospitals (beds, occupancy, revenue per occupied bed) and hotels
(rooms, occupancy, room rate) are two sectors and this one model.

It is the model built by hand in Excel for Yatharth Hospital and Lemon Tree
Hotels (R4 part a), row for row; `tests/test_research_occupancy.py` holds it
to those two workbooks.

**Revenue.** Line 1 is the capacity block. For the past a company prints
either its places, share filled and price (revenue is worked out) or its
revenue, share filled and price (the places in use are worked out): the
choice `line1_past`. Line 2 is units x revenue per unit a year (a hotel
chain's fees on managed rooms) and line 3 a share of line 1 (food and drink);
a company without them switches the segments `s2` and `s3` off, or drops
them (`research/compose.py`).

**The lower part** (`research/lower.py`). Four costs, each a share of revenue
plus an amount per place per day; depreciation on fixed assets; other income
as a yield on cash; interest on borrowings and, apart, on lease liabilities;
associates; tax; non-controlling interests; a balance sheet rolled from the
last year end and a cash flow that closes it.

**Not vetted.** No practising analyst has read it yet (phase R, rule 4).
"""
from __future__ import annotations

from research import lower
from research.compose import Segment, Spec, Structure, build as compose

NOTES = (
    lower.NOTES[0],
    "The year in progress is the part reported plus the rest assumed. The rest is typed, not grown from the part "
    "reported: a hotel's second half is its strong half.",
    "A price per day is in single units of the currency and the statements are in lakh, crore or million, so "
    "revenue and costs per place are divided by the constant at the top of the sheet.",
    "For the past, a company that prints its revenue, share filled and price has its places in use worked out "
    "from them; the count it prints as capacity may be larger than the count its occupancy is measured on.",
    lower.NOTES[1],
    lower.NOTES[2],
)

SECTORS = {
    "hospitals": dict(
        currency="Rs", s1="Beds", place="bed", s1_places="Beds", s1_occ="Occupancy",
        s1_price="Revenue per occupied bed per day, Rs", s1_rev="Hospital revenue",
        s2="Second line", s2_units="Units", s2_rpu="Revenue per unit per year, Rs", s2_rev="Second line revenue",
        s3="Third line", s3_share="Other revenue, as a share of hospital revenue", s3_rev="Other revenue",
        mat="Drugs, consumables and implants", emp="Employees", pwr="Power and fuel", oth="Other expenses"),
    "hotels": dict(
        currency="Rs", s1="Rooms", place="room", s1_places="Rooms", s1_occ="Occupancy",
        s1_price="Average room rate, Rs per night", s1_rev="Room revenue",
        s2="Fees on managed rooms", s2_units="Managed rooms", s2_rpu="Fee per managed room per year, Rs",
        s2_rev="Fees from managed hotels",
        s3="Food, drink and other revenue", s3_share="Food, drink and other, as a share of room revenue",
        s3_rev="Food, drink and other revenue",
        mat="Food and beverages consumed", emp="Employees", pwr="Power and fuel", oth="Other expenses"),
}

DEFAULT = Structure("occupancy", (
    Segment("s1", "capacity"),
    Segment("s2", "units"),
    Segment("s3", "share", of="s1", rest_of_revenue=True),
), places="s1", forecast_years=2)

SPEC = Spec(
    id="occupancy", name="Occupancy",
    about="Places x share filled x price per day: hospitals (beds, occupancy, revenue per occupied bed) and hotels "
          "(rooms, occupancy, room rate). Solved by hand for Yatharth Hospital and Lemon Tree Hotels.",
    default=DEFAULT, sectors=SECTORS, notes=NOTES, offered=True, vetted=False)


def build(forecast_years: int = 2):
    """Model 1 with as many forecast years as asked for; the rows are the same."""
    return compose(SPEC, Structure(DEFAULT.model, DEFAULT.segments, DEFAULT.places, forecast_years))


MODEL = build()
