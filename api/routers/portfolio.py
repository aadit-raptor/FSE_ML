"""Portfolio tracking in any currency (PLAN.md 7.4).

    GET    /api/portfolio?currency=&unit=&on=              owned deals, converted at one day's ECB rates, and totals
    GET    /api/deals/{id}/portfolio                       the deal's holding, quarters, drift and latest reforecast
    PUT    /api/deals/{id}/portfolio                       mark it as owned, or change when it closed or its tolerance
    DELETE /api/deals/{id}/portfolio                       take it out of the portfolio (its quarters stay)
    PUT    /api/deals/{id}/quarters                        its quarterly results (replacing them)
    DELETE /api/deals/{id}/quarters                        clear them
    POST   /api/deals/{id}/reforecast                      reforecast from the last four quarters: 202 and the job
    GET    /api/deals/{id}/reforecast/jobs/{job_id}        the job: stage, progress, the reforecast once done

The arithmetic is core/portfolio.py; storage db/portfolio.py, reached as the deal is (an owner or a team editor
changes a holding, a viewer reads it, anyone else gets 404). The plan is the saved deal run through the deal
model, so a holding's figures move with its inputs as every screen's do. Money comes back in the deal's own
currency and unit, and on the dashboard also in the reporting currency at the ECB's reference rates of the day
asked for (or the last day before it the ECB published), which the answer states. Refusals of quarters are 422
``{"code"}``: the screen words them.
"""
from __future__ import annotations

import uuid
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from api.auth import AuthUser, require_user
from api.deps import resolve_settings
from api.routers.deals import store_errors
from api.routers.economy import ECB_FX_PAGE
from api.routers.jobs import job_out, job_queue, submit_kind
from api.schemas import (
    DealInputsIn, HoldingIn, HoldingOut, MoneyUnit, PortfolioResponse, QuarterlyActuals, ReforecastJobOut,
)
from core import portfolio as pf
from core.deal import DealInputs, in_millions
from core.model_version import stamp
from core.money import in_unit, rescale, to_millions
from core.plan_actual import _returns, _run
from db import portfolio as store
from db.models import utc_now
from jobs import config
from jobs.queue import ACTIVE, JobNotFound, JobQueue

router = APIRouter(tags=["portfolio"])
KIND = "portfolio.reforecast"


def _refused(exc: pf.QuartersRefused) -> HTTPException:
    return HTTPException(422, {"code": exc.code})


def _deal(holding: store.Holding) -> tuple[DealInputs, dict, DealInputsIn]:
    """The saved deal and its Settings, money in millions as the engine runs."""
    inputs = DealInputsIn.model_validate(holding.inputs)
    cfg = resolve_settings(holding.settings)
    deal, cfg = in_millions(DealInputs(**inputs.model_dump()), cfg)
    return deal, cfg, inputs


def _quarters(holding: store.Holding, inputs: DealInputsIn) -> tuple[Optional[tuple], Optional[str]]:
    """The holding's quarters in millions, or why they can't be set against the deal as it is now."""
    saved = holding.quarters
    if not saved:
        return None, None
    if saved["currency"] != inputs.currency:
        return None, "other_currency"
    quarters = pf.quarters_from(saved["quarters"])
    try:
        pf.check_hold(quarters, inputs.hold)
    except pf.QuartersRefused as exc:
        return None, exc.code
    return pf.quarters_in_millions(quarters, saved["unit"]), None


def _reforecast_current(holding: store.Holding) -> bool:
    ref = holding.reforecast
    return bool(ref and holding.quarters_updated_at
                and ref.get("quarters_updated_at") == _iso(holding.quarters_updated_at))


def _iso(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _holding_out(holding: store.Holding) -> dict:
    deal, cfg, inputs = _deal(holding)
    quarters, problem = _quarters(holding, inputs)
    drift = None
    if quarters:
        drift = rescale(pf.drift(pf.plan_path(_run(deal, cfg)), quarters, _tolerance(holding)),
                        in_unit(1.0, inputs.unit), pf.DRIFT_MONEY_KEYS)
    owned = holding.portfolio is not None
    return {"deal_id": str(holding.deal_id), "name": holding.name, "access": holding.access, "owned": owned,
            "since": (holding.portfolio or {}).get("since"), "tolerance": (holding.portfolio or {}).get("tolerance"),
            "quarters": holding.quarters, "quarters_updated_at": holding.quarters_updated_at,
            "quarters_problem": problem, "drift": drift, "reforecast": holding.reforecast,
            "reforecast_current": _reforecast_current(holding), "money": inputs.money()}


def _tolerance(holding: store.Holding) -> float:
    return float((holding.portfolio or {}).get("tolerance", pf.DEFAULT_TOLERANCE))


# ------------------------------------------------------------------------------------------------ one holding
@router.get("/deals/{deal_id}/portfolio", response_model=HoldingOut)
def get_holding(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with store_errors():
        return _holding_out(store.get_holding(user.subject, deal_id))


@router.put("/deals/{deal_id}/portfolio", response_model=HoldingOut)
def put_holding(deal_id: uuid.UUID, body: HoldingIn, user: AuthUser = Depends(require_user)):
    with store_errors():
        try:
            store.set_holding(user.subject, deal_id, body.since, body.tolerance)
        except store.PortfolioFull:
            raise HTTPException(409, {"code": "portfolio_full"}) from None
        return _holding_out(store.get_holding(user.subject, deal_id))


@router.delete("/deals/{deal_id}/portfolio", response_model=HoldingOut)
def delete_holding(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with store_errors():
        store.remove_holding(user.subject, deal_id)
        return _holding_out(store.get_holding(user.subject, deal_id))


@router.put("/deals/{deal_id}/quarters", response_model=HoldingOut)
def put_quarters(deal_id: uuid.UUID, body: QuarterlyActuals, user: AuthUser = Depends(require_user)):
    with store_errors():
        holding = store.get_holding(user.subject, deal_id)          # saving checks the caller may edit
        try:
            pf.check_hold(pf.quarters_from([q.model_dump() for q in body.quarters]),
                          DealInputsIn.model_validate(holding.inputs).hold)
            store.save_quarters(user.subject, deal_id, body.model_dump())
        except pf.QuartersRefused as exc:
            raise _refused(exc) from None
        return _holding_out(store.get_holding(user.subject, deal_id))


@router.delete("/deals/{deal_id}/quarters", response_model=HoldingOut)
def delete_quarters(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with store_errors():
        store.save_quarters(user.subject, deal_id, None)
        return _holding_out(store.get_holding(user.subject, deal_id))


# ------------------------------------------------------------------------------------------------ reforecast
def run_reforecast(owner: str, deal_id: uuid.UUID) -> dict:
    """The job (``portfolio.reforecast``): the deal reforecast from its quarters, kept on the deal and answered,
    money in the deal's unit. The deal is read with the owner's access matched, as by the endpoint."""
    holding = store.get_holding(owner, deal_id, store.EDIT)
    deal, cfg, inputs = _deal(holding)
    quarters, problem = _quarters(holding, inputs)
    if problem or not quarters:
        raise ValueError(problem or "no_quarters")
    answer = rescale(pf.reforecast(deal, cfg, quarters), in_unit(1.0, inputs.unit), pf.REFORECAST_MONEY_KEYS)
    result = {**answer, "deal_id": str(deal_id), "quarters_updated_at": _iso(holding.quarters_updated_at),
              "run_at": _iso(utc_now()), "money": inputs.money().model_dump(),
              "model": stamp(resolve_settings(holding.settings))}
    store.save_reforecast(owner, deal_id, result)
    return result


def _reforecast_job(record) -> dict:
    out = job_out(record)
    return {k: out[k] for k in ("id", "status", "progress", "stage", "error", "error_status", "result_expired",
                                "result")}


@router.post("/deals/{deal_id}/reforecast", response_model=ReforecastJobOut, status_code=202)
def post_reforecast(deal_id: uuid.UUID, request: Request, response: Response,
                    user: AuthUser = Depends(require_user)):
    """Reforecast the deal from its last four quarters, as a background job (it simulates)."""
    with store_errors():
        holding = store.get_holding(user.subject, deal_id, store.EDIT)
        deal, cfg, inputs = _deal(holding)
        quarters, problem = _quarters(holding, inputs)
        if problem:
            raise HTTPException(422, {"code": problem})
        window = pf.ltm(quarters or (), pf.plan_path(_run(deal, cfg)), "ebitda")
        if window is None:
            raise HTTPException(422, {"code": "no_ltm"})
        if window["actual"] <= 0 or window["plan"] <= 0:
            raise HTTPException(422, {"code": "ebitda_not_positive"})
    record = submit_kind(user.subject, KIND, {"owner": user.subject, "deal_id": str(deal_id)})
    response.headers["Location"] = f"{request.url.path}/jobs/{record.id}"
    return _reforecast_job(record)


@router.get("/deals/{deal_id}/reforecast/jobs/{job_id}", response_model=ReforecastJobOut)
def get_reforecast_job(deal_id: uuid.UUID, job_id: uuid.UUID, user: AuthUser = Depends(require_user),
                       queue: JobQueue = Depends(job_queue)):
    """A reforecast's progress. Another account's job, or another kind, is a 404."""
    try:
        record = queue.get(user.subject, job_id, with_result=True)
    except JobNotFound:
        raise HTTPException(404, "no_job") from None
    if record.kind != KIND:
        raise HTTPException(404, "no_job")
    if record.status in ACTIVE:
        config.wake_runner()
    return _reforecast_job(record)


# ------------------------------------------------------------------------------------------------ the dashboard
def _local(holding: store.Holding) -> dict:
    """One holding on the dashboard, money in its own currency and unit."""
    deal, cfg, inputs = _deal(holding)
    run = _run(deal, cfg)
    plan = pf.plan_path(run)
    returns = _returns(run)
    quarters, problem = _quarters(holding, inputs)
    back = in_unit(1.0, inputs.unit)

    def ltm(line):
        window = pf.ltm(quarters, plan, line) if quarters else None
        return None if window is None else window["actual"] * back

    latest = quarters[-1] if quarters else None
    debt = None if latest is None else latest.figures.get("total_debt")
    ref, current = holding.reforecast, _reforecast_current(holding)
    ref_equity = None
    if ref and ref["money"]["currency"] == inputs.currency:
        ref_equity = to_millions(ref["reforecast"]["exit_equity"], ref["money"]["unit"]) * back
    drift = pf.drift(plan, quarters, _tolerance(holding)) if quarters else None
    return {
        "deal_id": str(holding.deal_id), "name": holding.name, "access": holding.access,
        "since": holding.portfolio.get("since"), "money": inputs.money().model_dump(),
        "currency": inputs.currency, "unit": inputs.unit,
        "local": {"entry_ev": (run.params.entry_ebitda + run.params.lease_ebitda_addback) * run.params.entry_multiple
                  * back,
                  "entry_equity": returns["entry_equity"] * back,
                  "plan_exit_equity": returns["exit_equity"] * back,
                  "ltm_revenue": ltm("revenue"), "ltm_ebitda": ltm("ebitda"),
                  "latest_debt": None if debt is None else debt * back,
                  "reforecast_exit_equity": ref_equity},
        "plan_irr": returns["irr"], "plan_moic": returns["moic"],
        "reforecast_irr": ref["reforecast"]["irr"] if ref else None, "reforecast_current": current,
        "as_of": None if latest is None else {"year": latest.year, "quarter": latest.quarter},
        "alerts": rescale(drift["alerts"], back, pf.DRIFT_MONEY_KEYS) if drift else [],
        "quarters_problem": problem,
    }


@router.get("/portfolio", response_model=PortfolioResponse)
def get_portfolio(currency: Optional[str] = Query(None, pattern=r"^[A-Z]{3}$",
                                                  description="The reporting currency; the account's by default"),
                  unit: MoneyUnit = Query("millions"),
                  on: Optional[date] = Query(None, description="The ECB rates of this day, or the last "
                                                               "publication before it; the latest by default"),
                  user: AuthUser = Depends(require_user)):
    from db import economy, users

    with store_errors():
        held = store.holdings(user.subject)
        if currency is None:
            profile = users.get_profile(user.subject)
            currency = profile.preferred_currency if profile else "USD"
        day = economy.fx_on(on)
        rows = [_local(h) for h in held]
    board = pf.board(rows, currency, unit, day.rates if day else {})
    return {"money": {"currency": currency, "unit": unit}, "published_on": day.day if day else None,
            "rates": board["rates"], "rates_url": ECB_FX_PAGE, "missing_rates": board["missing_rates"],
            "holdings": board["holdings"], "totals": board["totals"],
            "alerts": sum(len(r["alerts"]) for r in rows), "model": stamp(None)}


__all__ = ["KIND", "router", "run_reforecast"]
