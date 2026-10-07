"""Which group a validation case falls in: region, sector, size and era (PLAN.md 4.6).

The buckets are the reference library's (``library/coverage.py``), so the
report and Library -> Coverage count alike: S&P's regions, GICS sectors,
entry enterprise value in US dollars and the entry year's era. A reference
transaction carries all four (``library.references.tags``). A user's deal
carries its country and Damodaran industry (PLAN.md 4.3), mapped here to a
region and a GICS sector; its size is its entry value converted at the ECB's
rate; its era is the year its plan was saved. What a deal doesn't say is
``unknown``.
"""
from __future__ import annotations

from typing import Callable, Optional

from library import base_rates, coverage, references

DIMENSIONS = ("region", "sector", "size", "era")
UNKNOWN = "unknown"

# Damodaran's industries (benchmarks/, January 2026 edition) to the GICS
# sector each one's companies mostly belong to. "all" (the whole market) and
# "diversified" have none.
INDUSTRY_SECTOR: dict[str, str] = {
    "advertising": "communication_services",
    "aerospace_defense": "industrials",
    "air_transport": "industrials",
    "apparel": "consumer_discretionary",
    "auto_truck": "consumer_discretionary",
    "auto_parts": "consumer_discretionary",
    "beverage_alcoholic": "consumer_staples",
    "beverage_soft": "consumer_staples",
    "broadcasting": "communication_services",
    "building_materials": "industrials",
    "business_consumer_services": "industrials",
    "cable_tv": "communication_services",
    "chemical_basic": "materials",
    "chemical_diversified": "materials",
    "chemical_specialty": "materials",
    "coal_related_energy": "energy",
    "computer_services": "information_technology",
    "computers_peripherals": "information_technology",
    "construction_supplies": "materials",
    "drugs_biotechnology": "health_care",
    "drugs_pharmaceutical": "health_care",
    "education": "consumer_discretionary",
    "electrical_equipment": "industrials",
    "electronics_consumer_office": "consumer_discretionary",
    "electronics_general": "information_technology",
    "engineering_construction": "industrials",
    "entertainment": "communication_services",
    "environmental_waste_services": "industrials",
    "farming_agriculture": "consumer_staples",
    "food_processing": "consumer_staples",
    "food_wholesalers": "consumer_staples",
    "furn_home_furnishings": "consumer_discretionary",
    "green_renewable_energy": "utilities",
    "healthcare_products": "health_care",
    "healthcare_support_services": "health_care",
    "heathcare_information_and_technology": "health_care",
    "homebuilding": "consumer_discretionary",
    "hospitals_healthcare_facilities": "health_care",
    "hotel_gaming": "consumer_discretionary",
    "household_products": "consumer_staples",
    "information_services": "industrials",
    "machinery": "industrials",
    "metals_mining": "materials",
    "office_equipment_services": "industrials",
    "oil_gas_integrated": "energy",
    "oil_gas_production_and_exploration": "energy",
    "oil_gas_distribution": "energy",
    "oilfield_svcs_equip": "energy",
    "packaging_container": "materials",
    "paper_forest_products": "materials",
    "power": "utilities",
    "precious_metals": "materials",
    "publishing_newspapers": "communication_services",
    "real_estate_development": "real_estate",
    "real_estate_general_diversified": "real_estate",
    "real_estate_operations_services": "real_estate",
    "recreation": "consumer_discretionary",
    "restaurant_dining": "consumer_discretionary",
    "retail_automotive": "consumer_discretionary",
    "retail_building_supply": "consumer_discretionary",
    "retail_distributors": "consumer_discretionary",
    "retail_general": "consumer_discretionary",
    "retail_grocery_and_food": "consumer_staples",
    "retail_special_lines": "consumer_discretionary",
    "rubber_tires": "consumer_discretionary",
    "semiconductor": "information_technology",
    "semiconductor_equip": "information_technology",
    "shipbuilding_marine": "industrials",
    "shoe": "consumer_discretionary",
    "software_entertainment": "communication_services",
    "software_internet": "information_technology",
    "software_system_application": "information_technology",
    "steel": "materials",
    "telecom_wireless": "communication_services",
    "telecom_equipment": "information_technology",
    "telecom_services": "communication_services",
    "tobacco": "consumer_staples",
    "transportation": "industrials",
    "transportation_railroads": "industrials",
    "trucking": "industrials",
    "utility_general": "utilities",
    "utility_water": "utilities",
}

# Every bucket of each dimension, in the order the report lists them
BUCKETS: dict[str, tuple[str, ...]] = {
    "region": base_rates.SP_REGIONS,
    "sector": references.SECTORS,
    "size": tuple(name for name, *_ in coverage.SIZES),
    "era": tuple(name for name, *_ in coverage.ERAS),
}


def _known(tags: dict) -> dict:
    return {dim: tags.get(dim) or UNKNOWN for dim in DIMENSIONS}


def reference_tags(deal: dict) -> dict:
    """A reference transaction's groups (its outcome is not a split here)."""
    return _known(references.tags(deal))


def deal_tags(*, country: str, industry: str, ev: Optional[float], currency: str, year: int,
              usd_per: Callable[[str], Optional[float]]) -> dict:
    """A user's deal's groups. ``ev`` is the entry enterprise value in
    millions of ``currency``; ``usd_per(currency)`` gives US dollars per one
    unit of it (None when no rate is known, and the size is then unknown)."""
    rate = 1.0 if currency == "USD" else usd_per(currency)
    return _known({
        "region": base_rates.sp_region(country),
        "sector": INDUSTRY_SECTOR.get(industry),
        "size": coverage.size(ev * rate) if ev is not None and rate else None,
        "era": coverage.era(year),
    })
