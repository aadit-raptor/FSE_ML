"""What the reference library covers, in counts (PLAN.md 4.5, Library -> Coverage).

Each collection of deals is counted by the dimensions a balanced library
needs -- region, size, sector, era and outcome -- with every bucket listed,
empty ones too, so a gap shows as a zero rather than as nothing. The sourced
reference transactions (PLAN.md 4.5b, ``library/references.py``) are counted
once two administrators have approved them, with how many await review; the
four inception-era examples are unsourced and counted apart. The base rates
are counted per table.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, Optional

from core.examples import examples
from library import base_rates

REGIONS = base_rates.SP_REGIONS
# Entry year: before the 2000s boom, the boom to the crisis, the crisis and
# its aftermath, low rates, and since the pandemic
ERAS = (("before_2000", None, 1999), ("2000_2007", 2000, 2007), ("2008_2014", 2008, 2014),
        ("2015_2019", 2015, 2019), ("2020_on", 2020, None))
# Enterprise value at entry, US dollar millions
SIZES = (("under_100m", None, 100.0), ("100m_1bn", 100.0, 1_000.0), ("1bn_10bn", 1_000.0, 10_000.0),
         ("over_10bn", 10_000.0, None))
OUTCOMES = ("success", "distress", "held")
DIMENSIONS = ("region", "size", "sector", "era", "outcome")

# The examples' target companies' home countries (their names say the buyer
# and the year; core/backtesting.py records them as "Global" businesses)
EXAMPLE_COUNTRY = {
    "Burger King (3G Capital, 2010)": "US",
    "Hilton Hotels (Blackstone, 2007)": "US",
    "Dell (Silver Lake, 2013)": "US",
    "Freescale Semiconductor (Consortium, 2006)": "US",
}


def _band(value: Optional[float], bands) -> Optional[str]:
    if value is None:
        return None
    for name, low, high in bands:
        if (low is None or value >= low) and (high is None or value < high):
            return name
    return None


def era(year: Optional[int]) -> Optional[str]:
    return _band(year, ((n, lo, None if hi is None else hi + 1) for n, lo, hi in ERAS))


def size(ev_usd_m: Optional[float]) -> Optional[str]:
    return _band(ev_usd_m, SIZES)


def _deal_year(name: str) -> Optional[int]:
    m = re.search(r",\s*(\d{4})\)$", name)
    return int(m.group(1)) if m else None


def example_tags(example: dict) -> dict:
    """An example's bucket in each dimension."""
    plan = example["plan"]
    return {
        "region": base_rates.sp_region(EXAMPLE_COUNTRY.get(example["name"])),
        # The examples are US dollar millions (core/backtesting.py)
        "size": size(plan["ebitda"] * plan["entry_mult"]),
        "sector": example["sector"],
        "era": era(_deal_year(example["name"])),
        "outcome": "success" if example["outcome"] == "SUCCESS" else "distress",
    }


def _counts(tagged: Iterable[dict], sectors: Optional[tuple[str, ...]] = None) -> dict:
    tagged = list(tagged)
    fixed = {"region": REGIONS, "size": tuple(n for n, *_ in SIZES), "era": tuple(n for n, *_ in ERAS),
             "outcome": OUTCOMES, **({"sector": sectors} if sectors else {})}
    out = {}
    for dim in DIMENSIONS:
        seen = Counter(t[dim] for t in tagged if t.get(dim))
        buckets = fixed.get(dim, tuple(sorted(seen)))
        out[dim] = [{"bucket": b, "count": seen.get(b, 0)} for b in buckets]
    return out


def collection(id_: str, tagged: list[dict], *, sourced: bool, sectors: Optional[tuple[str, ...]] = None,
               awaiting_review: Optional[int] = None) -> dict:
    out = {"id": id_, "count": len(tagged), "sourced": sourced, "dimensions": _counts(tagged, sectors)}
    if awaiting_review is not None:
        out["awaiting_review"] = awaiting_review
    return out


def coverage(reference_deals: Iterable[dict] = (), awaiting_review: int = 0) -> dict:
    """``reference_deals`` are the approved reference transactions."""
    from library import references  # it reads this module's buckets
    return {
        "collections": [
            collection("reference_deals", [references.tags(d) for d in reference_deals], sourced=True,
                       sectors=references.SECTORS, awaiting_review=awaiting_review),
            collection("examples", [example_tags(x) for x in examples()], sourced=False),
        ],
        "base_rates": base_rates.coverage(),
    }
