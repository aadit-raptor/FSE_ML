"""Sourced starting figures for a new deal (PLAN.md 4.3).

    GET  /api/benchmarks/industries    every industry the stored averages cover
    GET  /api/benchmarks/starting?country=&industry=&currency=
                                       a new deal's starting figures, each with its source

Both read what the scheduled ``benchmarks-refresh`` and ``economy-refresh``
stored (benchmarks/refresh.py, economy/refresh.py); no request calls an
outside source, so neither is a run (api/limits.py ``RUN_PATHS``). A server
without a database has nothing stored: no industries, and starting figures
all ``missing``, so the deal keeps its own.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.schemas import BenchmarkIndustriesResponse, StartingAssumptionsResponse
from benchmarks import starting
from benchmarks.catalogue import ALL_INDUSTRIES_ID, SOURCE, canonical
from db.engine import is_configured

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


def _stored():
    if not is_configured():
        return {}, None, {}
    from db import benchmarks as store
    from db import economy
    tables, refreshed_at = store.all_tables()
    series, _ = economy.all_series()
    return tables, refreshed_at, series


def _today() -> date:
    return datetime.now(timezone.utc).date()


@router.get("/industries", response_model=BenchmarkIndustriesResponse)
def get_industries():
    tables, refreshed_at, _ = _stored()
    published = tables["margins.global"].published if "margins.global" in tables else None
    return {"industries": starting.industries(tables), "published": published,
            "refreshed_at": refreshed_at, "source": SOURCE}


@router.get("/starting", response_model=StartingAssumptionsResponse)
def get_starting(country: str = Query(pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2"),
                 industry: str = Query(ALL_INDUSTRIES_ID, pattern=r"^[a-z0-9_]{1,60}$",
                                       description="An id from /api/benchmarks/industries"),
                 currency: str = Query(pattern=r"^[A-Z]{3}$", description="ISO 4217")):
    tables, refreshed_at, series = _stored()
    known = {i["id"] for i in starting.industries(tables)}
    if known and industry not in known:
        raise HTTPException(404, "No published averages for that industry.")
    answer = starting.starting_assumptions(canonical(country), industry, currency, tables, series, _today())
    return {**answer, "refreshed_at": refreshed_at}
