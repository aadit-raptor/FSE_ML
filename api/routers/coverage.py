"""Equity research coverage (PLAN.md R2): the companies an analyst follows
and the periods they accepted from the results reader.

    GET    /api/research/companies                     the caller's companies, each with its periods
    POST   /api/research/companies                     add one {name}
    DELETE /api/research/companies/{id}                remove it and its periods
    POST   /api/research/companies/{id}/periods        accept a reviewed read: one period per scope
    GET    /api/research/periods/{id}                  a period with every figure, its page and mark
    DELETE /api/research/periods/{id}                  remove a period
    GET    /api/research/periods/{id}/workbook         the checked statements as a workbook

**Nothing is stored until the analyst accepts.** A read (``POST
/api/research/reads``) is a job whose result is kept six hours; accepting
takes that result from the caller's own job -- never figures sent by the
browser -- applies the analyst's corrections, and refuses while an amber
figure is unanswered. Another account's company, period or read is a 404.
Refusals say what is wrong as a code in ``detail`` (``coverage.error_*`` on
screen).
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response

from api.auth import AuthUser, require_user
from api.routers.export import SOURCE, _xlsx_response
from api.routers.jobs import job_queue
from api.schemas import (
    CoverageCompanyIn, CoverageCompanyOut, CoverageList, CoveragePeriodOut, PeriodAccept, PeriodAcceptOut,
)
from db import coverage as rows
from db.engine import is_configured as database_configured
from jobs.queue import JobNotFound, JobQueue
from research import coverage
from research.reader import service as reader

router = APIRouter(prefix="/research", tags=["research"])

NO_DATABASE = "no_database"
NO_ACCOUNT = "no_account"


@contextmanager
def store_errors():
    if not database_configured():
        raise HTTPException(503, NO_DATABASE)
    try:
        yield
    except rows.NotFound:
        raise HTTPException(404, "not_found") from None
    except rows.NoAccount:
        raise HTTPException(409, NO_ACCOUNT) from None
    except rows.Full as exc:
        raise HTTPException(409, exc.code) from None


def _iso(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


def period_summary(p: rows.PeriodRecord) -> dict:
    return {"id": str(p.id), "company_id": str(p.company_id), "period_end": p.period_end.isoformat(),
            "months": p.months, "scope": p.scope, "unit": p.unit, "currency": p.currency,
            "file_id": str(p.file_id) if p.file_id else None, "accepted_at": _iso(p.accepted_at)}


def company_out(c: rows.CompanyRecord, periods: list) -> dict:
    return {"id": str(c.id), "name": c.name, "created_at": _iso(c.created_at), "updated_at": _iso(c.updated_at),
            "periods": [period_summary(p) for p in periods]}


@router.get("/companies", response_model=CoverageList)
def get_companies(user: AuthUser = Depends(require_user)):
    with store_errors():
        found = rows.list_companies(user.subject)
    return {"companies": [company_out(c, periods) for c, periods in found],
            "limits": {"companies": rows.MAX_COMPANIES, "periods": rows.MAX_PERIODS}}


@router.post("/companies", response_model=CoverageCompanyOut, status_code=201)
def post_company(body: CoverageCompanyIn, user: AuthUser = Depends(require_user)):
    name = " ".join(body.name.split())
    if not name or not name.isprintable():
        raise HTTPException(422, "bad_name")
    with store_errors():
        return company_out(rows.create_company(user.subject, name), [])


@router.delete("/companies/{company_id}", status_code=204)
def delete_company(company_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with store_errors():
        rows.delete_company(user.subject, company_id)
    return Response(status_code=204)


def _finished_read(queue: JobQueue, subject: str, read_id: uuid.UUID) -> dict:
    """The caller's own finished read: its result, as the job answered it."""
    try:
        record = queue.get(subject, read_id, with_result=True)
    except JobNotFound:
        raise HTTPException(404, "no_read") from None
    if record.kind != reader.KIND:
        raise HTTPException(404, "no_read")
    if record.status != "succeeded":
        raise HTTPException(409, "read_not_done")
    if record.result is None:
        raise HTTPException(410, "read_expired")
    return record.result


@router.post("/companies/{company_id}/periods", response_model=PeriodAcceptOut, status_code=201)
def post_periods(company_id: uuid.UUID, body: PeriodAccept, user: AuthUser = Depends(require_user),
                 queue: JobQueue = Depends(job_queue)):
    """Accept a reviewed read into the company: every amber figure accepted
    as read or corrected. A period accepted again replaces the one kept."""
    with store_errors():
        rows.get_company(user.subject, company_id)
    result = _finished_read(queue, user.subject, body.read_id)
    review = {"accepted": body.accepted, "corrected": [c.model_dump() for c in body.corrected]}
    try:
        periods = coverage.accept(result, review, unit=body.unit)
    except coverage.Refused as exc:
        raise HTTPException(422, exc.code) from None
    file_id = uuid.UUID(result["file_id"]) if result.get("file_id") else None
    with store_errors():
        saved = rows.save_periods(user.subject, company_id, periods, file_id)
    return {"periods": [period_summary(p) for p in saved]}


@router.get("/periods/{period_id}", response_model=CoveragePeriodOut)
def get_period(period_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with store_errors():
        found = rows.get_period(user.subject, period_id)
    return {**period_summary(found), "company_name": found.company_name, "figures": found.figures,
            "counts": coverage.counts(found.figures)}


@router.delete("/periods/{period_id}", status_code=204)
def delete_period(period_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with store_errors():
        rows.delete_period(user.subject, period_id)
    return Response(status_code=204)


@router.get("/periods/{period_id}/workbook", response_class=Response, responses={
    200: {"content": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {}},
          "description": "The checked statements: subtotals as formulas, each figure's page in a note"}})
def get_workbook(period_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with store_errors():
        found = rows.get_period(user.subject, period_id)
        rows.record_export(user.subject)
    period = {"period_end": found.period_end.isoformat(), "months": found.months, "scope": found.scope,
              "unit": found.unit, "currency": found.currency, "figures": found.figures,
              "accepted_at": _iso(found.accepted_at)}
    data = coverage.workbook_bytes(found.company_name, period, generated=datetime.now(timezone.utc), source=SOURCE)
    return _xlsx_response(data, f"{found.company_name}_{found.period_end.isoformat()}_{found.scope}.xlsx")
