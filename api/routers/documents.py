"""A document read for a deal (PLAN.md 6.2).

    GET  /api/documents/reader            what the reader is: on, whether a read needs the public
                                          confirmation, whether it is open, the caller's pages left
    POST /api/documents/reads             read an uploaded, clean PDF: 202 and the job
    GET  /api/documents/reads/{id}        the job: stage, progress, and the values with their pages and quotes
    POST /api/documents/reads/{id}/deal   the deal inputs the accepted values give, and only those

A read is a background job (``document.extract``) under the caller's name; its result is kept six hours like
every job's and nothing is stored from it. The deal inputs are worked out from the job's own result, never from
values the browser sends, so a figure the caller did not accept can't reach a deal. Refusals are codes in
``detail`` (``documents.error_*`` on screen).
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from api.auth import AuthUser, require_user
from api.limits import READS_ALL, USER_DOCUMENT_PAGES, USER_READS, effective_limit
from api.observability import deploy_environment
from api.routers.jobs import job_out, job_queue
from api.schemas import DocumentDealRequest, DocumentDealResponse, DocumentReaderInfo, DocumentReadJobOut, \
    DocumentReadStart
from jobs import config
from jobs.queue import ACTIVE, JobNotFound, JobQueue
from ml import document_reader as reader
from ml import nlp_extractor as nx

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/reader", response_model=DocumentReaderInfo)
def get_reader(user: AuthUser = Depends(require_user)):
    found = reader.provider()
    return {"status": "on" if found is not None else "off",
            "provider": getattr(found, "name", "none"),
            "confirm_public": bool(getattr(found, "free_tier", True)),
            "open": found is not None and reader.open_to_people(deploy_environment()),
            "measured": reader.measured(),
            "pages_per_day": effective_limit(USER_DOCUMENT_PAGES, user.subject),
            "pages_left": reader.pages_left(user.subject),
            "reads_per_day": min(effective_limit(USER_READS, user.subject), effective_limit(READS_ALL, "all")),
            "max_pages": reader.MAX_PAGES}


def _out(record) -> dict:
    out = job_out(record)
    return {k: out[k] for k in ("id", "status", "progress", "stage", "error", "error_status", "result_expired",
                                "result")}


def _job(subject: str, job_id: uuid.UUID, queue: JobQueue):
    try:
        record = queue.get(subject, job_id, with_result=True)
    except JobNotFound:
        raise HTTPException(404, "no_read") from None
    if record.kind != reader.KIND:
        raise HTTPException(404, "no_read")
    return record


@router.post("/reads", response_model=DocumentReadJobOut, status_code=202)
def post_read(body: DocumentReadStart, request: Request, response: Response, user: AuthUser = Depends(require_user)):
    """Read a document twice and check every quote, as a background job."""
    record = reader.start(user.subject, body.file_id, public=body.public)
    response.headers["Location"] = f"{request.url.path}/{record.id}"
    return _out(record)


@router.get("/reads/{job_id}", response_model=DocumentReadJobOut)
def get_read(job_id: uuid.UUID, user: AuthUser = Depends(require_user), queue: JobQueue = Depends(job_queue)):
    """A read's progress, and its values once done. Another account's read, or another kind of job, is a 404."""
    record = _job(user.subject, job_id, queue)
    if record.status in ACTIVE:
        config.wake_runner()
    return _out(record)


@router.post("/reads/{job_id}/deal", response_model=DocumentDealResponse)
def post_deal(job_id: uuid.UUID, body: DocumentDealRequest, user: AuthUser = Depends(require_user),
              queue: JobQueue = Depends(job_queue)):
    """The deal inputs the caller's accepted values give: from the read's own result, only what was accepted."""
    record = _job(user.subject, job_id, queue)
    if record.status != "succeeded":
        raise HTTPException(409, "read_not_done")
    if record.result is None:
        raise HTTPException(410, "read_expired")
    return nx.deal_inputs(record.result, set(body.accepted))
