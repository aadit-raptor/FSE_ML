"""The sector models (PLAN.md R4): each a mechanism, not a sector.

`CATALOGUE` lists every model; the analyst picks a sector, and the sector
chooses the model and its words. Only models solved by hand for two companies
are `offered` (phase R, rule 4).
"""
from research.compose import Catalogue
from research.models import general, occupancy

CATALOGUE = Catalogue({spec.id: spec for spec in (occupancy.SPEC, general.SPEC)})
