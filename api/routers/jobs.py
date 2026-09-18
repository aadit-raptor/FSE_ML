"""Background jobs (PLAN.md 1.9): submit a long run, watch it, fetch the result.

    POST /api/jobs               {"kind": "montecarlo.run", "input": {...}}  -> 202, the job
    GET  /api/jobs/{id}          progress; the result once it succeeded
    GET  /api/jobs               the caller's recent jobs (no results)
    POST /api/jobs/{id}/cancel

``input`` is the body the matching endpoint takes, checked the same way, so a
bad request is refused at once with the same 422. The run itself is the same
code as that endpoint (jobs/kinds.py), so the result is too.

The endpoints only talk to ``jobs.config.get_queue()``: which queue and which
runner are behind it is configuration.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from api.auth import AuthUser, require_user
from api.deps import resolve_settings
from api.schemas import JobList, JobOut, JobSubmit
from jobs import config
from jobs import queue as q
from jobs.queue import ACTIVE, JobNotFound, JobQueue, JobRecord, QueueFull

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Settings are checked before queueing, so a bad one is a 422 now, not a
# failed job later
_SETTINGS_CHECKED = {"montecarlo.run": True, "montecarlo.scenarios": True, "backtesting.run": False}


def job_queue() -> JobQueue:
    """FastAPI dependency: this process's queue (tests may override it)."""
    return config.get_queue()


def _iso(value: Optional[datetime]) -> Optional[str]:
    return None if value is None else value.isoformat().replace("+00:00", "Z")


def job_out(job: JobRecord) -> dict:
    return {
        "id": str(job.id), "kind": job.kind, "status": job.status,
        "progress": round(job.progress, 4), "stage": job.stage, "ahead": job.ahead,
        "attempts": job.attempts, "cancel_requested": job.cancel_requested,
        "created_at": _iso(job.created_at), "started_at": _iso(job.started_at),
        "finished_at": _iso(job.finished_at), "error": job.error, "error_status": job.error_status,
        "result_expired": job.result_expired, "result": job.result,
    }


def _not_found() -> HTTPException:
    return HTTPException(404, "No such job.")


def submit(queue: JobQueue, owner: str, job, *, max_active_per_owner: Optional[int] = None,
           max_queued: Optional[int] = None) -> JobRecord:
    """Queue ``job`` (a ``JobSubmit``) for ``owner`` and wake the runner."""
    check = _SETTINGS_CHECKED.get(job.kind)
    if check is not None:
        resolve_settings(job.input.settings, check_correlations=check)
    try:
        record = queue.submit(
            owner, job.kind, job.input.model_dump(mode="json"),
            max_active_per_owner=max_active_per_owner or q.MAX_ACTIVE_PER_OWNER,
            max_queued=max_queued or q.MAX_QUEUED)
    except QueueFull as exc:
        if exc.owner_limit:
            raise HTTPException(429, str(exc), headers={"Retry-After": "10"}) from None
        raise HTTPException(503, str(exc), headers={"Retry-After": "30"}) from None
    config.wake_runner()
    return record


@router.post("", response_model=JobOut, status_code=202)
def post_job(job: JobSubmit, response: Response, request: Request,
             user: AuthUser = Depends(require_user), queue: JobQueue = Depends(job_queue)):
    """Queue a long run. Poll ``GET /api/jobs/{id}`` for progress and the result."""
    record = submit(queue, user.subject, job)
    response.headers["Location"] = f"{request.url.path}/{record.id}"
    return job_out(record)


@router.get("", response_model=JobList)
def get_jobs(user: AuthUser = Depends(require_user), queue: JobQueue = Depends(job_queue)):
    """The caller's recent jobs, newest first, without their results."""
    jobs = queue.list(user.subject)
    if any(j.status in ACTIVE for j in jobs):
        config.wake_runner()
    return {"jobs": [job_out(j) for j in jobs]}


@router.get("/{job_id}", response_model=JobOut)
def get_job(job_id: uuid.UUID, user: AuthUser = Depends(require_user),
            queue: JobQueue = Depends(job_queue)):
    """A job's progress, and its result once it succeeded."""
    try:
        record = queue.get(user.subject, job_id, with_result=True)
    except JobNotFound:
        raise _not_found() from None
    if record.status in ACTIVE:
        # After a restart nothing runs until someone asks: this is the ask
        config.wake_runner()
    return job_out(record)


@router.post("/{job_id}/cancel", response_model=JobOut)
def cancel_job(job_id: uuid.UUID, user: AuthUser = Depends(require_user),
               queue: JobQueue = Depends(job_queue)):
    """Cancel a job: at once if it is waiting; a running one ends as cancelled
    and its result is dropped."""
    try:
        return job_out(queue.cancel(user.subject, job_id))
    except JobNotFound:
        raise _not_found() from None
