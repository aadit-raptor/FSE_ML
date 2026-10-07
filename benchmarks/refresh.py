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

**History** (PLAN.md 4.4, benchmarks/history.py): every run also reads up
to ``history.BATCH`` archived editions not read yet, and the economic
history whenever the current tables are read (or it is missing); a call
that read the current tables leaves the archive to the next call. The first
fill is about 230 workbooks, so the summary says ``more`` while any are
left and the workflow calls again (``--repeat``); after that a run reads
nothing from the archive until a new edition is archived in January.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from api.observability import log_event
from benchmarks import damodaran, history
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


def _history(stored: dict, now: datetime, read_macro: bool, batch: int) -> dict:
    """Read the next batch of archived editions, and the economic history
    when asked; store what was read."""
    from db import benchmarks as store
    today = now.date()
    problems: dict[str, str] = {}
    found = []
    if read_macro:
        try:
            found.append(history.read_macro(today))
        except http.SourceError as exc:
            log_event("benchmark_source_failed", logging.WARNING, table=history.MACRO_TABLE, reason=exc.reason)
            problems[history.MACRO_TABLE] = exc.reason
        except damodaran.Unreadable:
            log_event("benchmark_source_failed", logging.WARNING, table=history.MACRO_TABLE, reason="unreadable")
            problems[history.MACRO_TABLE] = "unreadable"
    before = len(history.to_read(stored[history.READ_TABLE].rows if history.READ_TABLE in stored else {}, today,
                                 retry=False))
    tables, filled = history.fill(stored, today, log=log_event, batch=batch)
    store.save(found + tables, now)
    problems.update({k: v for k, v in filled["problems"].items()})
    return {"problems": problems, "left": filled["left"], "read": before - filled["left"]}


def run(now: Optional[datetime] = None, force: bool = False) -> dict:
    from db import benchmarks as store
    from db.engine import is_configured
    if not is_configured():
        return {"skipped_no_database": True}
    now = now or datetime.now(timezone.utc)
    expected = [name for name, _ in damodaran.every_table()]
    stored, oldest = store.all_tables(history=True)
    fresh = set(stored) >= set(expected) and oldest is not None and now - oldest < REFRESH_EVERY
    problems: dict[str, str] = {}
    saved = 0
    if force or not fresh:
        found, problems = _read()
        shrunk = [t.name for t in found
                  if t.name in stored and len(t.rows) < MIN_KEPT_FRACTION * len(stored[t.name].rows)]
        problems.update({name: "shrunk" for name in shrunk})
        saved = store.save([t for t in found if t.name not in shrunk], now)
    # A call that read the current tables leaves the archive to the next one,
    # so no call runs past the scheduler's two minutes
    read_current = force or not fresh
    hist = _history(stored, now, read_macro=read_current or history.MACRO_TABLE not in stored,
                    batch=0 if read_current else history.BATCH)
    problems.update(hist["problems"])
    stored, _ = store.all_tables(history=True)
    absent = sorted(set(expected) - set(stored))
    published = sorted({t.published.isoformat() for n, t in stored.items() if n in expected and t.published})
    use = store.usage()
    # A run's summary is flat (jobs/scheduled.py, api/routers/scheduled.py TaskRun)
    groups = [g for g in stored if g.startswith("history.") and g not in (history.READ_TABLE, history.MACRO_TABLE)]
    summary = {
        "read": not fresh or force, "tables_saved": saved, "tables_stored": len(set(expected) & set(stored)),
        "tables_missing": ",".join(absent),
        "source_problems": ",".join(f"{name}:{reason}" for name, reason in sorted(problems.items())),
        "industries": len(stored["margins.global"].rows) if "margins.global" in stored else 0,
        "published": published[-1] if published else None,
        "benchmark_bytes": use["bytes"], "benchmark_budget_warning": use["warning"],
        "history_groups": len(groups), "history_files_read": hist["read"], "history_left": hist["left"],
        "macro_economies": len(stored[history.MACRO_TABLE].rows) if history.MACRO_TABLE in stored else 0,
        # Call again while the archive's backfill has files left and this call got through some
        "more": hist["left"] > 0 and (hist["read"] > 0 or read_current),
    }
    log_event("benchmarks_refreshed", **summary)
    if absent:
        raise Incomplete(f"{len(absent)} tables missing after the refresh: {', '.join(absent)}")
    return summary
