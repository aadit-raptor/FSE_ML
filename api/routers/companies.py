"""Company filings from many countries (PLAN.md 4.1, companies/).

    GET  /api/companies/sources              the sources, their coverage and keys
    GET  /api/companies/search?q=...         by name, LEI, ISIN, ticker or number
    POST /api/companies/load                 read a company's filings and store the summary
    GET  /api/companies/{source}/{id}        a stored company, without calling the source

Search and load call outside sources, so they count as runs (api/limits.py
``RUN_PATHS``). Company data is public and shared: nothing here records who
asked for which company, and logs name the source only.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from api.observability import log_event
from api.schemas import CompanyLoadRequest, CompanyResponse, CompanySearchResponse, CompanySourcesResponse
from companies import http, refresh, sources
from companies.model import CompanyData, CompanyRef
from db import DatabaseUnavailable
from db.engine import is_configured

router = APIRouter(prefix="/companies", tags=["companies"])

REASONS = {
    "not_configured": (503, "{source} isn't set up on this server: it needs its API key (DEPLOY.md "
                            "'Company filings')."),
    "rate_limited": (503, "{source} is limiting requests just now. Try again in a few minutes."),
    "refused": (502, "{source} refused the request. Try again later."),
    "unreachable": (502, "{source} didn't answer. Try again in a few minutes."),
    "unreadable": (502, "{source} sent something that couldn't be read."),
    "response_too_large": (502, "{source} sent more than this server reads at once."),
    "redirect_refused": (502, "{source} sent the request somewhere this server doesn't follow."),
}


def _refused(exc: http.SourceError):
    if isinstance(exc, http.NotFound):
        raise HTTPException(404, f"No such company at {exc.source}.") from None
    status, text = REASONS.get(exc.reason, (502, "{source} failed. Try again later."))
    raise HTTPException(status, text.format(source=exc.source)) from None


def _ref_out(ref: CompanyRef, stored: bool) -> dict:
    return {"source": ref.source, "id": ref.source_id, "name": ref.name, "local_name": ref.local_name,
            "country": ref.country, "identifiers": {k: str(v) for k, v in ref.identifiers.items() if v},
            "loadable": sources.source(ref.source).configured(), "stored": stored}


def _answer(data: CompanyData, refreshed_at, stored: bool) -> dict:
    return {
        "company": _ref_out(data.ref, stored),
        "money": {"currency": data.currency, "unit": data.unit},
        "accounting_standard": data.accounting_standard,
        "fiscal_year_end_month": data.fiscal_year_end_month,
        "years": [{"fiscal_year": y.fiscal_year, "period_end": y.period_end, "figures": y.figures,
                   "filing": {"url": y.filing.url, "filed_on": y.filing.filed_on, "form": y.filing.form,
                              "id": y.filing.id}} for y in data.years],
        "warnings": [{"code": w.code, "field": w.field or None} for w in data.warnings],
        "refreshed_at": refreshed_at,
        "licence": sources.source(data.ref.source).licence,
    }


@router.get("/sources", response_model=CompanySourcesResponse)
def get_sources():
    return {"sources": [{"id": s.id, "coverage": list(s.coverage), "searchable_by": list(s.searchable_by),
                         "needs_key": s.key_env is not None, "configured": s.configured(), "licence": s.licence}
                        for s in sources.SOURCES.values()],
            "fallback": sources.FALLBACK}


@router.get("/search", response_model=CompanySearchResponse)
def get_search(q: str = Query(min_length=2, max_length=100)):
    """Every source at once; one that fails is listed in ``unavailable``."""
    try:
        found = sources.search(q)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    keys = [(r.source, r.source_id) for r in found.results]
    stored = set()
    if is_configured() and keys:
        try:
            from db import companies as store
            stored = store.stored(keys)
        except DatabaseUnavailable:
            stored = set()
    return {"matched": found.matched,
            "results": [_ref_out(r, (r.source, r.source_id) in stored) for r in found.results],
            "unavailable": [{"source": s, "reason": reason} for s, reason in found.unavailable],
            "fallback": sources.FALLBACK}


@router.post("/load", response_model=CompanyResponse)
def post_load(req: CompanyLoadRequest):
    """Read the company's latest filings from its source, store the summary
    (when the server has a database) and answer it."""
    try:
        sources.normalise_id(req.source, req.id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    if req.source == "edinet":
        refresh.use_database_index()
    try:
        data = sources.fetch(req.source, req.id, req.years)
    except http.SourceError as exc:
        log_event("company_source_failed", logging.WARNING, source=req.source, reason=exc.reason)
        _refused(exc)
    refreshed_at, stored = None, False
    if is_configured():
        from db import companies as store
        try:
            refreshed_at, stored = store.save(data), True
        except DatabaseUnavailable:
            log_event("company_not_stored", logging.WARNING, source=req.source)
    return _answer(data, refreshed_at, stored)


@router.get("/{source}/{company_id}", response_model=CompanyResponse)
def get_stored(source: str, company_id: str):
    try:
        company_id = sources.normalise_id(source, company_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    if not is_configured():
        raise HTTPException(404, "Not stored: this server keeps no company data. Load it instead.")
    from db import companies as store
    found = store.get(source, company_id)
    if found is None:
        raise HTTPException(404, "Not stored yet. Load it first.")
    data, refreshed_at = found
    return _answer(data, refreshed_at, True)
