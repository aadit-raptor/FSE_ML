"""Economic data by country, reference rates and exchange rates (PLAN.md 4.2).

    GET  /api/economy/countries              each economy's current figures, sourced
    GET  /api/economy/reference-rates        each floating-rate benchmark's current level
    GET  /api/economy/exchange-rates?base=   the ECB's rates, against any currency it quotes

Everything here reads what the nightly ``economy-refresh`` stored
(economy/refresh.py); no request calls an outside source, so none is a run
(api/limits.py ``RUN_PATHS``). A server without a database has nothing
stored and answers empty.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import EconomyResponse, ExchangeRatesResponse, ReferenceRatesResponse
from db.engine import is_configured
from economy import connectors, views
from economy.catalogue import AREAS, CURRENCY_BENCHMARKS, SOURCES

router = APIRouter(prefix="/economy", tags=["economy"])
ECB_FX_PAGE = "https://data.ecb.europa.eu/data/datasets/EXR"


def _stored():
    if not is_configured():
        return {}, None
    from db import economy as store
    return store.all_series()


def _today() -> date:
    return datetime.now(timezone.utc).date()


def _figure(fig: views.Figure) -> dict:
    return dict(fig.__dict__)


@router.get("/countries", response_model=EconomyResponse)
def get_countries():
    series, refreshed_at = _stored()
    today = _today()
    areas = []
    for area, figures in views.economies(series, today).items():
        areas.append({"area": area, "currency": AREAS[area].currency,
                      "current": views.economy_is_current(figures),
                      "figures": {name: _figure(f) for name, f in figures.items()}})
    sources = [{"id": s.id, "name": s.name, "licence": s.licence, "needs_key": s.key_env is not None,
                "configured": s.key_env is None or connectors.fred_configured()} for s in SOURCES.values()]
    return {"as_of": today, "refreshed_at": refreshed_at, "areas": areas, "sources": sources}


@router.get("/reference-rates", response_model=ReferenceRatesResponse)
def get_reference_rates():
    series, refreshed_at = _stored()
    today = _today()
    return {"as_of": today, "refreshed_at": refreshed_at,
            "rates": {code: _figure(f) for code, f in views.reference_rates(series, today).items()},
            "currency_benchmarks": dict(CURRENCY_BENCHMARKS)}


@router.get("/exchange-rates", response_model=ExchangeRatesResponse)
def get_exchange_rates(base: str = Query("EUR", pattern=r"^[A-Z]{3}$"),
                       on: Optional[date] = Query(None, description="The rates of this day, or the last "
                                                                    "publication before it")):
    if not is_configured():
        raise HTTPException(404, "No exchange rates are stored on this server.")
    from db import economy as store
    day = store.fx_on(on)
    if day is None:
        raise HTTPException(404, "No exchange rates are stored for that day yet.")
    rates = views.rebase(day, base)
    if rates is None:
        raise HTTPException(404, f"The ECB doesn't publish a reference rate for {base}.")
    return {"base": base, "published_on": day.day, "rates": rates, "url": ECB_FX_PAGE}
