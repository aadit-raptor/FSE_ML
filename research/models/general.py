"""Model 7, General (PLAN.md R4): revenue is each line's last year grown, and
the shared lower part with costs as a share of revenue. Pharma (by
geography) and auto components (by channel) in the sample; anything a more
particular model does not fit.

**Not offered yet** (phase R, rule 4): it is built and tested with the other
blocks, but listed for analysts only once two companies of its kind are
solved by hand, as model 1's were. PLAN.md R7 holds that step.
"""
from __future__ import annotations

from research import lower
from research.compose import Segment, Spec, Structure

NOTES = (
    lower.NOTES[0],
    "The year in progress is the part reported plus the rest: each line's growth for the whole year is assumed, "
    "and the rest is what that growth leaves after the part reported.",
    "Costs are a share of revenue; a line that counts places can be added for costs per place per day.",
    lower.NOTES[1],
    lower.NOTES[2],
)

COSTS = dict(currency="Rs", place="unit", emp="Employees", pwr="Power and fuel", oth="Other expenses")

SECTORS = {
    "pharma": dict(COSTS, s1="India", s1_rev="Revenue in India", s2="Outside India", s2_rev="Revenue outside India",
                   mat="Materials consumed and goods bought for resale"),
    "auto_components": dict(COSTS, s1="Vehicle makers", s1_rev="Revenue from vehicle makers", s2="Replacement market",
                            s2_rev="Revenue from the replacement market", mat="Materials consumed"),
}

DEFAULT = Structure("general", (Segment("s1", "growth"), Segment("s2", "growth")), places=None, forecast_years=2)

SPEC = Spec(
    id="general", name="General",
    about="Each line of revenue grown on the year before, costs a share of revenue: pharma by geography, auto "
          "components by channel. Not yet solved by hand for two companies, so not yet offered.",
    default=DEFAULT, sectors=SECTORS, notes=NOTES, offered=False, vetted=False)
