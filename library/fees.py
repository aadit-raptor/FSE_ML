"""Fees and amortisation from the reference transactions' filings (PLAN.md 4.5b).

Three Settings had no free published source: the transaction fees
(``tx_fee_pct``, per cent of the value paid), the financing fees
(``fin_fee_pct``, per cent of the debt raised) and the senior debt's yearly
amortisation (``def_senior_amort``, per cent of the original principal). The
approved reference transactions record them from their filings, so each is
offered as the **median** across the approved deals that give it, with how
many deals that is, their range and their closing years.

Like the sourced Monte Carlo ranges (benchmarks/risk.py), these are offered
Settings: Settings -> Fees shows them beside the current values and "Use
sourced figures" applies them. The factory values stay the illustrative
fallback, so no saved deal and no result moves until the user applies them.
A figure with fewer than ``MIN_DEALS`` deals behind it is not offered.
"""
from __future__ import annotations

from statistics import median

from library import references

# Settings key -> the figure in references.derived that measures it
SETTINGS = {"tx_fee_pct": "tx_fee_pct", "fin_fee_pct": "fin_fee_pct", "def_senior_amort": "def_senior_amort"}
MIN_DEALS = 3


def sourced(deals: list[dict]) -> dict:
    """Each Setting's median across ``deals`` (approved reference
    transactions), or ``None`` where fewer than ``MIN_DEALS`` give it."""
    out = {}
    measured = [(references.derived(d), d) for d in deals]
    for key, measure in SETTINGS.items():
        found = sorted(((m[measure], d) for m, d in measured if m[measure] is not None), key=lambda x: x[0])
        if len(found) < MIN_DEALS:
            out[key] = None
            continue
        values = [v for v, _ in found]
        years = [references.closed_year(d) for _, d in found]
        out[key] = {
            "value": round(median(values), 2),
            "n": len(found),
            "low": values[0],
            "high": values[-1],
            "first_year": min(years),
            "last_year": max(years),
            "deals": sorted(d["key"] for _, d in found),
        }
    return out
