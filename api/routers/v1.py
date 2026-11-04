"""The public API, version 1 (PLAN.md 7.7; docs/public-api.md).

    GET  /api/v1/me                          the version, the limits and how many webhooks are told
    POST /api/v1/deal/run                    run a deal (the body of POST /api/deal/run)
    POST /api/v1/montecarlo/run              simulate one (the body of POST /api/montecarlo/run)
    GET  /api/v1/deals                       the caller's saved deals (and those shared with their teams)
    GET  /api/v1/deals/{id}                  one, with its inputs and Settings
    POST /api/v1/deals/{id}/run              run a saved deal's working copy with its Settings
    POST /api/v1/jobs                        queue a long run; every webhook is told when it ends
    GET  /api/v1/jobs/{id}                   its progress, then its result
    POST /api/v1/jobs/{id}/cancel
    GET  /api/v1/defaults/industries         market defaults: the industries they cover
    GET  /api/v1/defaults/starting?country=&industry=&currency=
                                             a deal's sourced starting figures for a region and sector
    GET  /api/v1/defaults/risk?country=&industry=&currency=
                                             the simulation's sourced ranges, correlations and scenarios

**Version 1 is the screens' own code.** Each endpoint calls the function
behind the matching site endpoint, with the same request and response models,
so a result here is the result the screen shows (``tests/test_public_api.py``
compares them), stamped with the model version that made it (PLAN.md 3.1). A
change that would break a script is a ``/api/v2``, never an edit here; adding
a field is not a break. Every answer carries ``Variater-API-Version: 1``.

**Who may call:** an API key (``Bearer vk_...``, which reaches nothing else)
or the site's own sign-in. A key's requests count against the key's own
limits (``api.limits.KEY_REQUESTS``, ``KEY_RUNS_DAY``) and the person's.
Saved deals are reached by role as on the site (``db/deals.py``): another
account's deal is a 404.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, Query, Request, Response

from api import limits
from api.auth import AuthUser, require_user
from api.routers import benchmarks, deal, deals, jobs, montecarlo
from api.schemas import (
    BenchmarkIndustriesResponse, DealDetail, DealInputsIn, DealList, DealRunRequest, DealRunResponse, JobSubmit,
    MonteCarloRequest, MonteCarloResponse, RiskAssumptionsResponse, StartingAssumptionsResponse, V1Job, V1Me,
)
from benchmarks.catalogue import ALL_INDUSTRIES_ID
from db.engine import is_configured as database_configured
from jobs.queue import ACTIVE, JobQueue

API_VERSION = "1"


def _version(response: Response) -> None:
    response.headers["Variater-API-Version"] = API_VERSION


router = APIRouter(prefix="/v1", tags=["public api v1"], dependencies=[Depends(_version)])


def _webhook_count(subject: str) -> int:
    if not database_configured():
        return 0
    from db import webhooks
    return len(webhooks.list_webhooks(subject))


@router.get("/me", response_model=V1Me)
def get_me(user: AuthUser = Depends(require_user)):
    """The API version, how you are signed in and the limits that apply to you."""
    return {"api_version": API_VERSION, "via": "key" if user.via_key else "site", "key_id": user.via_key,
            "limits": {"key_requests_per_minute": limits.effective_limit(limits.KEY_REQUESTS),
                       "key_runs_per_day": limits.effective_limit(limits.KEY_RUNS_DAY),
                       "user_runs_per_day": limits.effective_limit(limits.USER_RUNS_DAY, user.subject),
                       "max_simulation_paths": limits.MAX_SIMULATION_PATHS},
            "webhooks": _webhook_count(user.subject)}


@router.post("/deal/run", response_model=DealRunResponse)
def post_deal_run(req: DealRunRequest):
    """Run the LBO model on a deal: the same answer as the site's ``POST /api/deal/run``."""
    return deal.post_run(req)


@router.post("/montecarlo/run", response_model=MonteCarloResponse)
def post_montecarlo_run(req: MonteCarloRequest):
    """Simulate a deal (one simulation at a time on the server): the site's ``POST /api/montecarlo/run``.
    For a long run, queue it as a job instead and let a webhook say when it is done."""
    return montecarlo.post_run(req)


@router.get("/deals", response_model=DealList)
def get_deals(archived: bool = False, user: AuthUser = Depends(require_user)):
    """Your saved deals and those shared with your teams, most recently edited first."""
    return deals.list_deals(archived=archived, user=user)


@router.get("/deals/{deal_id}", response_model=DealDetail)
def get_deal(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """A saved deal's working copy: its inputs and Settings, as last saved."""
    return deals.get_deal(deal_id, user=user)


@router.post("/deals/{deal_id}/run", response_model=DealRunResponse)
def post_saved_deal_run(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """Run a saved deal's working copy with its own Settings: what its screen shows."""
    from db import deals as store
    with deals.store_errors():
        saved = store.get_deal(user.subject, deal_id)
    return deal.post_run(DealRunRequest(inputs=DealInputsIn.model_validate(saved.inputs), settings=saved.settings))


@router.post("/jobs", response_model=V1Job, status_code=202)
def post_job(job: JobSubmit, response: Response, request: Request, user: AuthUser = Depends(require_user),
             queue: JobQueue = Depends(jobs.job_queue)):
    """Queue a long run (``montecarlo.run``, ``montecarlo.scenarios``, ``backtesting.run``,
    ``forecasting.run``). Each of your webhooks is sent a signed ``job.finished`` message when it ends."""
    record = jobs.submit(queue, user.subject, job)
    told = 0
    if database_configured():
        from api import webhooks as sender
        from db import webhooks
        told = webhooks.expect(user.subject, record.id)
        # A quick job may have ended before its messages were written: then its end is told here
        ended = queue.get(user.subject, record.id)
        if told and ended.status not in ACTIVE:
            sender.job_finished(record.id, ended.kind, ended.status, ended.error_status)
    response.headers["Location"] = f"{request.url.path}/{record.id}"
    return {**jobs.job_out(record), "webhooks": told}


@router.get("/jobs/{job_id}", response_model=V1Job)
def get_job(job_id: uuid.UUID, user: AuthUser = Depends(require_user), queue: JobQueue = Depends(jobs.job_queue)):
    """A job's progress, and its result once it succeeded: what the matching endpoint answers."""
    return jobs.get_job(job_id, user=user, queue=queue)


@router.post("/jobs/{job_id}/cancel", response_model=V1Job)
def cancel_job(job_id: uuid.UUID, user: AuthUser = Depends(require_user),
               queue: JobQueue = Depends(jobs.job_queue)):
    """Cancel a job: at once if it is waiting (its webhooks are told), else when its run notices."""
    return jobs.cancel_job(job_id, user=user, queue=queue)


@router.get("/defaults/industries", response_model=BenchmarkIndustriesResponse)
def get_industries():
    """The industries the market defaults cover (Damodaran's published averages)."""
    return benchmarks.get_industries()


_COUNTRY = Query(pattern=r"^[A-Z]{2}$", description="ISO 3166-1 alpha-2")
_INDUSTRY = Query(ALL_INDUSTRIES_ID, pattern=r"^[a-z0-9_]{1,60}$",
                  description="An id from /api/v1/defaults/industries")
_CURRENCY = Query(pattern=r"^[A-Z]{3}$", description="ISO 4217")


@router.get("/defaults/starting", response_model=StartingAssumptionsResponse)
def get_starting(country: str = _COUNTRY, industry: str = _INDUSTRY, currency: str = _CURRENCY):
    """A deal's sourced starting figures for a country and industry, each with its source, sample and date."""
    return benchmarks.get_starting(country=country, industry=industry, currency=currency)


@router.get("/defaults/risk", response_model=RiskAssumptionsResponse)
def get_risk(country: str = _COUNTRY, industry: str = _INDUSTRY, currency: str = _CURRENCY):
    """The simulation's sourced ranges, correlations and scenarios for a country and industry."""
    return benchmarks.get_risk(country=country, industry=industry, currency=currency)


__all__ = ["API_VERSION", "router"]
