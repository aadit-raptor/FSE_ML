"""Model 3, Stores (PLAN.md R7): revenue is the average count of stores x
the revenue per store per day, and costs are a share of revenue plus an
amount per store per day. Retail and restaurants are two sectors and this one
model.

It is the model solved by hand in Excel for Avenue Supermarts and V-Mart
Retail (`research-samples/r7/` on the Mac); `tests/test_research_stores.py`
holds it to those two workbooks.

**Revenue.** Line 1 is the stores block: stores at the start, opened,
closed, at the end and the average, and the revenue per store per day, worked
out for the past and grown for the future by an assumption analysts read as
like-for-like growth. Line 2 is other operating revenue as a share of line 1,
the rest of the revenue for the past; a company that prints no split switches
it off.

**The lower part** (`research/lower.py`) with three cost lines (goods sold,
employees, other expenses: a retailer's power and rent sit in other expenses
and, under Ind AS 116, in depreciation and interest on leases).

**Not vetted.** No practising analyst has read it yet (phase R, rule 4).
"""
from __future__ import annotations

from research import lower
from research.compose import Segment, Spec, Structure, build as compose

NOTES = (
    lower.NOTES[0],
    "The year in progress is the part reported plus the rest assumed: the rest's stores opened and closed and its "
    "revenue per store per day are typed, since festive months fall in the second half.",
    "A store's revenue per day is in single units of the currency and the statements are in lakh, crore or million, "
    "so revenue and costs per store are divided by the constant at the top of the sheet.",
    "Filings print the stores at the end of a period, not the average; the average of the past is typed (the count "
    "at the start and at the end, halved, unless the company says more). Bigger new stores raise the revenue per "
    "store, so its growth is not the company's like-for-like growth.",
    "Rent is mostly depreciation of right-of-use assets and interest on lease liabilities under Ind AS 116; short "
    "leases and revenue-share rents stay in other expenses.",
    lower.NOTES[1],
    lower.NOTES[2],
)

COSTS = ("mat", "emp", "oth")

SECTORS = {
    "retail": dict(
        currency="Rs", place="store", s1="Stores", s1_places="Stores", s1_price="Revenue per store per day, Rs",
        s1_rev="Revenue from the stores", s2="Other operating revenue",
        s2_share="Other operating revenue, as a share of the stores' revenue", s2_rev="Other operating revenue",
        mat="Cost of goods sold (purchases and change in stocks)", emp="Employees", oth="Other expenses"),
    "restaurants": dict(
        currency="Rs", place="restaurant", s1="Restaurants", s1_places="Restaurants",
        s1_price="Average daily sales per restaurant, Rs", s1_rev="Revenue from the restaurants",
        s2="Other operating revenue", s2_share="Other operating revenue, as a share of the restaurants' revenue",
        s2_rev="Other operating revenue", mat="Food and packaging consumed (purchases and change in stocks)",
        emp="Employees", oth="Other expenses"),
}

DEFAULT = Structure("stores", (
    Segment("s1", "stores"),
    Segment("s2", "share", of="s1", rest_of_revenue=True),
), places="s1", forecast_years=2)

SPEC = Spec(
    id="stores", name="Stores",
    about="Stores x revenue per store per day, costs per store per day: retail and restaurants. Solved by hand "
          "for Avenue Supermarts and V-Mart Retail.",
    default=DEFAULT, sectors=SECTORS, notes=NOTES, offered=True, vetted=False, costs=COSTS)


def build(forecast_years: int = 2):
    return compose(SPEC, Structure(DEFAULT.model, DEFAULT.segments, DEFAULT.places, forecast_years))


MODEL = build()
