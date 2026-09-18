"""Endpoints for the scheduler (PLAN.md 1.9): GitHub Actions workflows only.

Every route here needs a GitHub Actions OIDC token from an allowed workflow
on an allowed branch (api/github_oidc.py); a user's session token is refused.

    POST /api/scheduled/tasks/{task}   run a task in the API and record the run
    POST /api/scheduled/runs           record a task the workflow ran itself
    GET  /api/scheduled/runs           the newest run of each task
    POST /api/scheduled/drill          queue the job drill (not in production)
    GET  /api/scheduled/drill/{id}     one drill job: status and summary only
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Dict, Literal, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.github_oidc import Workflow, require_workflow
from api.observability import deploy_environment, log_event
from api.routers.jobs import job_out, job_queue, submit
from api.schemas import MonteCarloJob
from db.models import utc_now
from jobs import drill, scheduled
from jobs.queue import JobNotFound, JobQueue

router = APIRouter(prefix="/scheduled", tags=["scheduled"])

SummaryValue = Union[int, float, bool, str, None]


class TaskRun(BaseModel):
    task: str
    status: Literal["succeeded", "failed"]
    summary: Dict[str, SummaryValue] = {}
    error: Optional[str] = None
    workflow: str
    github_run_id: Optional[int] = None


class ReportedRun(BaseModel):
    task: str = Field(pattern=scheduled.TASK_NAME.pattern)
    status: Literal["succeeded", "failed"]
    started_at: datetime
    summary: Dict[str, SummaryValue] = Field(default_factory=dict, max_length=20)
    error: Optional[str] = Field(None, max_length=500)


class DrillRequest(BaseModel):
    count: int = Field(drill.MAX_DRILL_JOBS, ge=1, le=drill.MAX_DRILL_JOBS)


class DrillStarted(BaseModel):
    jobs: list[str]
    request: dict
    expected_summary: Dict[str, float]


class DrillJob(BaseModel):
    id: str
    status: str
    stage: Optional[str]
    progress: float
    attempts: int
    error: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    summary: Optional[Dict[str, SummaryValue]] = None
    elapsed_ms: Optional[float] = None
    mismatched: list[str] = []


def _run_record(workflow: Workflow, name: str, status: str, started: datetime, summary: dict,
                error: Optional[str]) -> None:
    scheduled.record(name, status, trigger=workflow.event, workflow=workflow.workflow,
                     run_id=workflow.run_id, started_at=started, summary=summary, error=error)


@router.post("/tasks/{task}", response_model=TaskRun)
def run_task(task: str, workflow: Workflow = Depends(require_workflow)):
    """Run a scheduled task now and record the run (a failed task answers 500
    after recording, so the workflow fails and alerts)."""
    entry = scheduled.TASKS.get(task)
    if entry is None:
        raise HTTPException(404, f"No such task. Tasks: {', '.join(sorted(scheduled.TASKS))}")
    started = utc_now()
    try:
        summary, status, error = entry.run(), "succeeded", None
    except Exception as exc:  # noqa: BLE001 - recorded, then reported as a failure
        log_event("scheduled_task_failed", logging.ERROR, exc_info=True, task=task)
        summary, status, error = {}, "failed", f"{type(exc).__name__}: the task failed; see the API logs"
    _run_record(workflow, task, status, started, summary, error)
    log_event("scheduled_task", task=task, status=status, workflow=workflow.workflow,
              trigger=workflow.event)
    if status == "failed":
        raise HTTPException(500, error)
    return {"task": task, "status": status, "summary": summary, "error": error,
            "workflow": workflow.workflow, "github_run_id": workflow.run_id}


@router.post("/runs", response_model=TaskRun, status_code=201)
def report_run(run: ReportedRun, workflow: Workflow = Depends(require_workflow)):
    """Record a task the workflow ran itself (e.g. the Supabase keep-alive)."""
    if run.task not in scheduled.REPORTED_TASKS:
        raise HTTPException(422, f"Only these tasks are reported: {', '.join(sorted(scheduled.REPORTED_TASKS))}")
    _run_record(workflow, run.task, run.status, run.started_at, run.summary, run.error)
    return {"task": run.task, "status": run.status, "summary": run.summary, "error": run.error,
            "workflow": workflow.workflow, "github_run_id": workflow.run_id}


@router.get("/runs")
def get_runs(workflow: Workflow = Depends(require_workflow)):  # noqa: ARG001
    """The newest recorded run of each task."""
    return {"tasks": scheduled.latest_runs()}


@router.post("/drill", response_model=DrillStarted, status_code=202)
def start_drill(body: DrillRequest = DrillRequest(), workflow: Workflow = Depends(require_workflow),
                queue: JobQueue = Depends(job_queue)):
    """Queue ``count`` identical seeded Monte Carlo jobs at once (jobs/drill.py)."""
    if deploy_environment() == "production":
        raise HTTPException(404, "The drill doesn't run in production.")
    job = MonteCarloJob(kind="montecarlo.run", input=drill.DRILL_REQUEST)
    ids = [str(submit(queue, drill.DRILL_OWNER, job, max_active_per_owner=drill.MAX_DRILL_JOBS).id)
           for _ in range(body.count)]
    log_event("job_drill_started", jobs=len(ids), workflow=workflow.workflow)
    return {"jobs": ids, "request": drill.DRILL_REQUEST, "expected_summary": drill.EXPECTED_SUMMARY}


@router.get("/drill/{job_id}", response_model=DrillJob)
def get_drill_job(job_id: uuid.UUID, workflow: Workflow = Depends(require_workflow),  # noqa: ARG001
                  queue: JobQueue = Depends(job_queue)):
    try:
        record = queue.get(drill.DRILL_OWNER, job_id, with_result=True)
    except JobNotFound:
        raise HTTPException(404, "No such drill job.") from None
    out = job_out(record)
    result = out.pop("result") or {}
    summary = result.get("summary")
    if record.status in ("queued", "running"):
        from jobs import config
        config.wake_runner()
    return {**{k: out[k] for k in ("id", "status", "stage", "progress", "attempts", "error",
                                   "started_at", "finished_at")},
            "summary": summary, "elapsed_ms": result.get("elapsed_ms"),
            "mismatched": drill.summary_mismatches(summary) if summary else []}
