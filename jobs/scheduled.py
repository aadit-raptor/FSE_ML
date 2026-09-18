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
from datetime import datetime
from typing import Callable, Optional

from sqlalchemy import delete, func, insert, select

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


@task("job-maintenance", "Requeue or fail jobs whose runner went silent; apply retention.")
def job_maintenance() -> dict:
    queue = config.get_queue()
    recovered = queue.recover()
    pruned = queue.prune()
    runs = prune_runs()
    return {**recovered, **pruned, "scheduled_runs_deleted": runs, **{
        f"jobs_{status}": n for status, n in queue.counts().items()}}


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
