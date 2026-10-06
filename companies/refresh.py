"""The scheduled refresh of company data (``company-refresh``, nightly).

Run by GitHub Actions through the API (jobs/scheduled.py, scheduled.yml),
with a database only. Each run, within ``TIME_BUDGET_S``:

1. **EDINET's index**: reads the day lists filed since the last run (one
   call a day of filings), adding the annual reports to ``edinet_reports``.
   The first runs backfill ``EDINET_BACKFILL_DAYS``, ``EDINET_DAYS_PER_RUN``
   at a time; the summary says ``more`` until done, and the workflow calls
   again (``ops.scheduled task --repeat``).
2. **Stale companies**: reloads up to ``REFRESH_PER_RUN`` companies not
   refreshed for ``REFRESH_AFTER``, oldest first. A failure leaves the
   stored figures as they were and tries again next cycle.
3. **Housekeeping**: prunes old EDINET reports and removes the least
   recently used companies beyond ``db.companies.MAX_COMPANIES``.

Within each source's rules: calls are paced per host (companies/http.py),
and a run makes at most about ``EDINET_DAYS_PER_RUN`` + 3 x
``REFRESH_PER_RUN`` + a few calls in all.
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from companies import edinet, http, sources

REFRESH_AFTER = timedelta(days=14)
REFRESH_PER_RUN = 15
EDINET_DAYS_PER_RUN = 60
EDINET_BACKFILL_DAYS = 400
TIME_BUDGET_S = 100.0
JAPAN = ZoneInfo("Asia/Tokyo")


def use_database_index() -> None:
    """EDINET's index in the database when there is one (called before any
    EDINET work); a process without one keeps its memory index."""
    from db.companies import DatabaseReportIndex
    from db.engine import is_configured
    if is_configured() and not isinstance(edinet.index(), DatabaseReportIndex):
        edinet.use_index(DatabaseReportIndex())


def scan_edinet(today: date, deadline: float) -> dict:
    """Read EDINET's day lists from the cursor to yesterday (Japan time)."""
    index = edinet.index()
    through = index.scanned_through() or today - timedelta(days=EDINET_BACKFILL_DAYS + 1)
    last = today - timedelta(days=1)
    days = added = 0
    day = through
    while day < last and days < EDINET_DAYS_PER_RUN and time.monotonic() < deadline:
        day += timedelta(days=1)
        added += edinet.scan_day(day)
        index.set_scanned_through(day)
        days += 1
    return {"edinet_days_scanned": days, "edinet_reports_added": added,
            "edinet_days_left": max(0, (last - day).days)}


def refresh_stale(now: datetime, deadline: float) -> dict:
    from db import companies as store
    refreshed = failed = 0
    for source, source_id in store.stale(REFRESH_PER_RUN, now - REFRESH_AFTER):
        if time.monotonic() >= deadline:
            break
        try:
            store.save(sources.fetch(source, source_id))
            refreshed += 1
        except (http.SourceError, ValueError):
            store.mark_refreshed(source, source_id)
            failed += 1
    return {"companies_refreshed": refreshed, "refresh_failures": failed}


def run(now: Optional[datetime] = None, budget_s: float = TIME_BUDGET_S) -> dict:
    from db import companies as store
    from db.engine import is_configured
    if not is_configured():
        return {"skipped_no_database": True}
    now = now or datetime.now(timezone.utc)
    deadline = time.monotonic() + budget_s
    use_database_index()
    summary: dict = {}
    if edinet.configured():
        try:
            summary.update(scan_edinet(now.astimezone(JAPAN).date(), deadline))
        except http.SourceError as exc:
            if exc.reason == "refused":      # the key: fail the run, so it alerts
                raise
            summary["edinet_error"] = exc.reason   # an outage: the cursor resumes next run
    else:
        summary["edinet_not_configured"] = True
    summary.update(refresh_stale(now, deadline))
    summary["edinet_reports_pruned"] = edinet.index().prune(now.date())
    summary["companies_evicted"] = store.evict()
    use = store.usage()
    summary.update(companies=use["companies"], company_bytes=use["bytes"], company_budget_warning=use["warning"])
    summary["more"] = summary.get("edinet_days_left", 0) > 0
    return summary
