"""The scheduled refresh of the industry averages (``benchmarks-refresh``, nightly).

Run by GitHub Actions through the API (jobs/scheduled.py, scheduled.yml), on
both environments, with a database only. Damodaran publishes once a year (in
January, with some data sets again mid-year), so a nightly run reads the 41
workbooks only once the stored tables are ``REFRESH_EVERY`` old, and
otherwise answers what is stored: one polite pass a week, about a minute of
paced calls (companies/http.py).

A workbook that fails, or has a shape the reader doesn't know
(``unreadable``), keeps its stored table. The run fails, so it alerts,
once everything it read is stored, when any table needed for a deal's
starting figures is still missing.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from api.observability import log_event
from benchmarks import damodaran
from companies import http

REFRESH_EVERY = timedelta(days=7)
# A workbook that still reads but has lost most of its rows (a changed layout)
# doesn't replace a stored table more than twice its size
MIN_KEPT_FRACTION = 0.5


class Incomplete(RuntimeError):
    """Tables a deal's starting figures need are missing after a refresh."""


def _read() -> tuple[list[damodaran.Table], dict[str, str]]:
    found, problems = [], {}
    for name, call in damodaran.every_table():
        try:
            found.append(call())
        except http.SourceError as exc:
            log_event("benchmark_source_failed", logging.WARNING, table=name, reason=exc.reason)
            problems[name] = exc.reason
        except (damodaran.Unreadable, IndexError, TypeError, ValueError):
            log_event("benchmark_source_failed", logging.WARNING, table=name, reason="unreadable")
            problems[name] = "unreadable"
    return found, problems


def run(now: Optional[datetime] = None, force: bool = False) -> dict:
    from db import benchmarks as store
    from db.engine import is_configured
    if not is_configured():
        return {"skipped_no_database": True}
    now = now or datetime.now(timezone.utc)
    expected = [name for name, _ in damodaran.every_table()]
    stored, oldest = store.all_tables()
    fresh = set(stored) >= set(expected) and oldest is not None and now - oldest < REFRESH_EVERY
    problems: dict[str, str] = {}
    saved = 0
    if force or not fresh:
        found, problems = _read()
        shrunk = [t.name for t in found
                  if t.name in stored and len(t.rows) < MIN_KEPT_FRACTION * len(stored[t.name].rows)]
        problems.update({name: "shrunk" for name in shrunk})
        saved = store.save([t for t in found if t.name not in shrunk])
        stored, _ = store.all_tables()
    absent = sorted(set(expected) - set(stored))
    published = sorted({t.published.isoformat() for t in stored.values() if t.published})
    use = store.usage()
    # A run's summary is flat (jobs/scheduled.py, api/routers/scheduled.py TaskRun)
    summary = {
        "read": not fresh or force, "tables_saved": saved, "tables_stored": len(stored),
        "tables_missing": ",".join(absent),
        "source_problems": ",".join(f"{name}:{reason}" for name, reason in sorted(problems.items())),
        "industries": len(stored["margins.global"].rows) if "margins.global" in stored else 0,
        "published": published[-1] if published else None,
        "benchmark_bytes": use["bytes"], "benchmark_budget_warning": use["warning"],
    }
    log_event("benchmarks_refreshed", **summary)
    if absent:
        raise Incomplete(f"{len(absent)} tables missing after the refresh: {', '.join(absent)}")
    return summary
