"""The nightly ``validation-report`` task (PLAN.md 4.6, ``jobs/scheduled.py``).

Builds the cases from the approved reference transactions (while the
library is switched on) and from every opted-in deal with an exit, writes
the report and answers a flat summary of counts for the run log.
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Optional

from core.model_version import ENGINE_VERSION
from validation import cases as cases_
from validation import report as report_


def _usd_per():
    from db.economy import fx_on
    from economy.views import rebase

    @lru_cache(maxsize=64)
    def per_usd(day: date) -> Optional[dict]:
        # The plan's day, or the newest stored when the ECB's rates for it
        # are older than the 400 days kept
        found = fx_on(day) or fx_on()
        return None if found is None else rebase(found, "USD")

    def usd_per(currency: str, day: date) -> Optional[float]:
        rates = per_usd(day)
        units = rates.get(currency) if rates else None
        return 1.0 / units if units else None
    return usd_per


def build(now: Optional[datetime] = None) -> tuple[dict, dict]:
    """The report and the run's counts, from what is stored now."""
    from db import references as stored_references
    from db import validation as store
    from library import switch

    now = now or datetime.now(timezone.utc)
    library_on = switch.enabled()
    found: list[cases_.Case] = []
    for deal in stored_references.approved() if library_on else []:
        case = cases_.library_case(deal, today=now.date())
        if case is not None:
            found.append(case)
    skipped: Counter = Counter()
    usd_per = _usd_per()
    contributions = store.contributions()
    for contribution in contributions:
        try:
            found.extend(cases_.contributed_cases(contribution, usd_per=usd_per))
        except cases_.Skipped as reason:
            skipped[str(reason)] += 1
    report = report_.build(found, generated_at=now.isoformat().replace("+00:00", "Z"),
                           engine_version=ENGINE_VERSION, library_included=library_on,
                           fit_until=cases_.fit_until())
    counts = {
        "library_on": library_on,
        "library_cases": sum(1 for c in found if c.origin == "library"),
        "contributions": len(contributions),
        "contributions_used": len(contributions) - sum(skipped.values()),
        **{f"skipped_{reason}": n for reason, n in sorted(skipped.items())},
        "out_of_time_cases": sum(1 for c in found if c.out_of_time),
        "checks": len(report["checks"]),
        "splits": len(report["dimensions"]),
    }
    return report, counts


def run(now: Optional[datetime] = None) -> dict:
    from db import validation as store

    report, counts = build(now)
    return {**counts, "report_id": store.save_report(report, keep=report_.KEEP_REPORTS, now=now)}
