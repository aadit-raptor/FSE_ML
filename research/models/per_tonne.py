"""Model 2, Per tonne (PLAN.md R7): revenue is capacity a year x utilisation
x the price per unit sold, and costs are per unit sold. Cement (tonnes of
capacity, tonnes sold, the realisation per tonne), steel and chemicals are
three sectors and this one model.

It is the model solved by hand in Excel for JK Lakshmi Cement and Star
Cement (`research-samples/r7/` on the Mac); `tests/test_research_per_tonne.py`
holds it to those two workbooks.

**Revenue.** Line 1 is the plant block: the capacity at the end of each
period and its average, the volume sold, the utilisation and the price per
unit sold. For the past the volume and the revenue are printed and the
utilisation and the price are worked out; for the future the utilisation is
assumed and the price grows. Line 2 is other operating revenue as a share of
line 1, the rest of the revenue for the past; a company that prints no split
switches it off.

**The lower part** (`research/lower.py`) with five cost lines, freight among
them, each a share of revenue plus an amount per unit sold, and EBITDA per
unit sold beside the margin.

**Not vetted.** No practising analyst has read it yet (phase R, rule 4).
"""
from __future__ import annotations

from research import lower
from research.compose import Segment, Spec, Structure, build as compose

NOTES = (
    lower.NOTES[0],
    "The year in progress is the part reported plus the rest assumed: the rest's utilisation and price are typed, "
    "since a cement company's second half and fourth quarter are its strong ones.",
    "Volumes are in units (tonnes, not lakh tonnes) and prices in single units of the currency; the statements are "
    "in lakh, crore or million, so revenue and costs per unit are divided by the constant at the top of the sheet.",
    "Capacity is a year's capacity; the utilisation of a part year is its volume against the capacity for its days. "
    "The volume a company prints may include what it sells but does not make (clinker, bought-in cement), so the "
    "utilisation is as it comes out, not the plants' own.",
    lower.NOTES[1],
    lower.NOTES[2],
)

COSTS = ("mat", "emp", "pwr", "frt", "oth")

SECTORS = {
    "cement": dict(
        currency="Rs", place="tonne", s1="Cement", s1_cap="Cement capacity, tonnes a year",
        s1_util="Capacity utilisation", s1_vol="Cement and clinker sold, tonnes",
        s1_price="Realisation, Rs per tonne", s1_rev="Cement revenue",
        s2="Other operating revenue", s2_share="Other operating revenue, as a share of cement revenue",
        s2_rev="Other operating revenue", mat="Raw materials, purchases and change in stocks",
        emp="Employees", pwr="Power and fuel", frt="Freight and forwarding", oth="Other expenses"),
    "steel": dict(
        currency="Rs", place="tonne", s1="Steel", s1_cap="Steel capacity, tonnes a year",
        s1_util="Capacity utilisation", s1_vol="Steel delivered, tonnes",
        s1_price="Realisation, Rs per tonne", s1_rev="Steel revenue",
        s2="Other operating revenue", s2_share="Other operating revenue, as a share of steel revenue",
        s2_rev="Other operating revenue", mat="Raw materials, purchases and change in stocks",
        emp="Employees", pwr="Power and fuel", frt="Freight and handling", oth="Other expenses"),
    "chemicals": dict(
        currency="Rs", place="tonne", s1="Products", s1_cap="Capacity, tonnes a year",
        s1_util="Capacity utilisation", s1_vol="Products sold, tonnes",
        s1_price="Realisation, Rs per tonne", s1_rev="Revenue from products",
        s2="Other operating revenue", s2_share="Other operating revenue, as a share of product revenue",
        s2_rev="Other operating revenue", mat="Raw materials, purchases and change in stocks",
        emp="Employees", pwr="Power and fuel", frt="Freight and forwarding", oth="Other expenses"),
}

DEFAULT = Structure("per_tonne", (
    Segment("s1", "plant"),
    Segment("s2", "share", of="s1", rest_of_revenue=True),
), places="s1", forecast_years=2)

SPEC = Spec(
    id="per_tonne", name="Per tonne",
    about="Capacity x utilisation x realisation, costs per tonne sold: cement, steel and chemicals. Solved by hand "
          "for JK Lakshmi Cement and Star Cement.",
    default=DEFAULT, sectors=SECTORS, notes=NOTES, offered=True, vetted=False, costs=COSTS)


def build(forecast_years: int = 2):
    return compose(SPEC, Structure(DEFAULT.model, DEFAULT.segments, DEFAULT.places, forecast_years))


MODEL = build()
