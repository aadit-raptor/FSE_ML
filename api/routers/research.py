"""Equity research (PLAN.md phase R): the results reader (R1).

    GET  /api/research/reader          what the reader is: provider, whether a read needs the
                                       public confirmation, whether it is open, the caller's pages left
    POST /api/research/reads           read an uploaded, clean PDF: 202 and the job
    GET  /api/research/reads/{id}      the job: stage, progress, and the checked statements once done

A read is a background job (``research.read``) under the caller's name; its
result is kept six hours like every job's. Nothing is stored from it until R2
lets the analyst accept a quarter. Refusals say what is wrong as a code in
``detail`` (``reader.error_*`` on screen).
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from api.auth import AuthUser, require_user
from api.limits import READS_ALL, USER_READ_PAGES, USER_READS, effective_limit
from api.observability import deploy_environment
from api.routers.jobs import job_out, job_queue
from api.schemas import ReaderInfo, ReadJobOut, ReadStart
from jobs import config
from jobs.queue import ACTIVE, JobNotFound, JobQueue
from research.reader import gate, pages, providers, service

router = APIRouter(prefix="/research", tags=["research"])


@router.get("/reader", response_model=ReaderInfo)
def get_reader(user: AuthUser = Depends(require_user)):
    """What the screen needs before a read: is it on, must the caller
    confirm the file is public, how many pages are left today."""
    found = providers.get_provider()
    return {"status": "on" if found is not None else "off",
            "provider": getattr(found, "name", "none"),
            "free_tier": bool(getattr(found, "free_tier", providers.free_tier())),
            "confirm_public": bool(getattr(found, "free_tier", True)),
            "open": found is not None and gate.open_to_people(deploy_environment()),
            "measured": gate.measured(),
            "pages_per_day": effective_limit(USER_READ_PAGES), "pages_left": service.pages_left(user.subject),
            "reads_per_day": min(effective_limit(USER_READS), effective_limit(READS_ALL)),
            "max_pages": pages.MAX_PAGES}


def _out(record) -> dict:
    out = job_out(record)
    return {k: out[k] for k in ("id", "status", "progress", "stage", "error", "error_status", "result_expired",
                                "result")}


@router.post("/reads", response_model=ReadJobOut, status_code=202)
def post_read(body: ReadStart, request: Request, response: Response, user: AuthUser = Depends(require_user)):
    """Read a results PDF twice and check it, as a background job."""
    record = service.start(user.subject, body.file_id, public=body.public, quarter_end=body.quarter_end)
    response.headers["Location"] = f"{request.url.path}/{record.id}"
    return _out(record)


@router.get("/reads/{job_id}", response_model=ReadJobOut)
def get_read(job_id: uuid.UUID, user: AuthUser = Depends(require_user), queue: JobQueue = Depends(job_queue)):
    """A read's progress, and the checked statements once it is done. Another
    account's read, or another kind of job, is a 404."""
    try:
        record = queue.get(user.subject, job_id, with_result=True)
    except JobNotFound:
        raise HTTPException(404, "no_read") from None
    if record.kind != service.KIND:
        raise HTTPException(404, "no_read")
    if record.status in ACTIVE:
        config.wake_runner()
    return _out(record)
