"""Scheduled tasks and the log of their runs (PLAN.md 1.9).

GitHub Actions is the scheduler (``.github/workflows/scheduled.yml``). A
scheduled workflow either

- **calls a task here** (``POST /api/scheduled/tasks/{name}``): the API runs
  it and records the run, or
- **runs a script itself** and reports how it went (``POST
  /api/scheduled/runs``), for work that doesn't belong in the API, such as
  the Supabase keep-alive.

Either way the run lands in ``scheduled_runs``, and ``/api/health/jobs``
shows the latest run of each task, so a stopped schedule is visible.

To add a task (data refresh, retraining in later phases): write a function
returning a small dict of counts, decorate it with ``@task("name")``, and add
a step to scheduled.yml. Tasks must finish well within a request (Render's
proxy gives up after a few minutes); anything longer should queue a job.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from sqlalchemy import delete, func, insert, select

from db import audit
from db.engine import connect, transaction
from db.models import ScheduledRun, utc_now
from jobs import config

TASK_NAME = re.compile(r"^[a-z][a-z0-9-]{1,39}$")
# Newest runs kept per task: a nightly task keeps about three months
KEEP_RUNS_PER_TASK = 100
# Tasks run by scripts in the workflows, which report here afterwards
REPORTED_TASKS = frozenset({"supabase-keepalive", "job-drill"})


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    run: Callable[[], dict]


TASKS: dict[str, Task] = {}


def task(name: str, description: str):
    assert TASK_NAME.match(name), name

    def register(fn: Callable[[], dict]) -> Callable[[], dict]:
        TASKS[name] = Task(name, description, fn)
        return fn
    return register


@task("job-maintenance", "Requeue or fail jobs whose runner went silent; apply retention; "
                          "merge old audit edits (PLAN.md 3.3); delete uploaded files past their time "
                          "(PLAN.md 6.1); end the Excel add-in's expired sign-ins (PLAN.md R3); sum yesterday's "
                          "AI use and delete old counts (PLAN.md R9).")
def job_maintenance() -> dict:
    queue = config.get_queue()
    recovered = queue.recover()
    pruned = queue.prune()
    runs = prune_runs()
    compacted = audit.compact()
    from db import addin, ai_usage
    from db.models import utc_now
    yesterday = ai_usage.totals(utc_now().date() - timedelta(days=1))
    summary = {**recovered, **pruned, "scheduled_runs_deleted": runs, "audit_rows_compacted": compacted,
               "addin_sessions_deleted": addin.sweep(), **{
        f"jobs_{status}": n for status, n in queue.counts().items()},
               # what the AI cost yesterday, for everyone together (nobody named): the paid tier's bill
               "ai_requests_yesterday": yesterday["requests"], "ai_pages_yesterday": yesterday["pages"],
               "ai_cost_usd_yesterday": float(yesterday["cost"]), "ai_people_yesterday": yesterday["people"],
               "ai_usage_rows_deleted": ai_usage.sweep()}
    # Uploaded files (PLAN.md 6.1) last: a file store that can't be reached
    # fails the run, so it alerts, after everything else was done
    from uploads import service
    return {**summary, **service.sweep()}


@task("company-refresh", "Refresh stored company figures and EDINET's index of annual reports "
                         "(PLAN.md 4.1, companies/refresh.py).")
def company_refresh() -> dict:
    from companies import refresh
    return refresh.run()


@task("economy-refresh", "Refresh economic data by country, reference rates and the ECB's exchange "
                         "rates (PLAN.md 4.2, economy/refresh.py); refit the economic regimes on the "
                         "first run of a month (PLAN.md 5.7).")
def economy_refresh() -> dict:
    """Economic data, then the economic regimes' monthly refit (PLAN.md
    5.7), which reads the same IMF source: it runs whatever the data refresh
    did, and either failing fails the run, so it alerts."""
    from economy import refresh
    from ml import macro_regime
    try:
        summary = refresh.run()
    finally:
        regimes = macro_regime.refresh()
    return {**summary, **{f"regime_{k}": v for k, v in regimes.items()}}


@task("benchmarks-refresh", "Refresh the published industry averages and country tax rates new "
                            "deals start from (PLAN.md 4.3, benchmarks/refresh.py).")
def benchmarks_refresh() -> dict:
    from benchmarks import refresh
    return refresh.run()


@task("validation-report", "Check the model's predicted ranges and default risk against outcomes, "
                           "anonymised, by region, sector, size and era (PLAN.md 4.6, validation/).")
def validation_report() -> dict:
    from validation import run
    return run.run()


@task("regime-refresh", "Refit each region's economic regime on the first run of a month, from the "
                        "IMF's growth and inflation (PLAN.md 5.7, ml/macro_regime.py). Also run by "
                        "economy-refresh, which is what the workflows schedule.")
def regime_refresh() -> dict:
    from ml import macro_regime
    return macro_regime.refresh()


# ---------------------------------------------------------------------------
# The run log
# ---------------------------------------------------------------------------
def record(task_name: str, status: str, *, trigger: str, workflow: str, run_id: Optional[int],
           started_at: datetime, summary: Optional[dict] = None,
           error: Optional[str] = None) -> dict:
    with transaction() as conn:
        row = conn.execute(insert(ScheduledRun).values(
            task=task_name, status=status, trigger=trigger[:20], workflow=workflow[:80],
            github_run_id=run_id, started_at=started_at, summary=summary or {},
            error=error[:500] if error else None, finished_at=utc_now(),
        ).returning(ScheduledRun.id, ScheduledRun.finished_at)).one()
    return {"id": row.id, "finished_at": row.finished_at}


def prune_runs() -> int:
    ranked = (select(ScheduledRun.id, func.row_number().over(
        partition_by=ScheduledRun.task, order_by=ScheduledRun.started_at.desc()).label("n"))
        .subquery())
    with transaction() as conn:
        return conn.execute(delete(ScheduledRun).where(ScheduledRun.id.in_(
            select(ranked.c.id).where(ranked.c.n > KEEP_RUNS_PER_TASK)))).rowcount


def latest_runs() -> dict[str, dict]:
    """The newest run of each task."""
    newest = (select(ScheduledRun).distinct(ScheduledRun.task)
              .order_by(ScheduledRun.task, ScheduledRun.started_at.desc()))
    with connect() as conn:
        rows = conn.execute(newest).all()
    return {r.task: {"status": r.status, "trigger": r.trigger, "workflow": r.workflow,
                     "github_run_id": r.github_run_id, "summary": r.summary, "error": r.error,
                     "started_at": _iso(r.started_at), "finished_at": _iso(r.finished_at)}
            for r in rows}


def _iso(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else value.isoformat().replace("+00:00", "Z")
