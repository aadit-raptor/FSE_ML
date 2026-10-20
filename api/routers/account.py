"""The signed-in person's own account (PLAN.md 1.4).

All about the caller and nobody else: the subject always comes from the
verified token, never from the request body, so one account can't read or
write another's profile.

    GET  /api/account                 the caller's profile
    POST /api/account                 save it
    GET  /api/account/suggestions?country=&industry=&deal_id=
                                      the caller's usual figures for that region and
                                      sector, from their own saved deals (PLAN.md 5.9)
    PUT  /api/account/suggestions     switch them on or off

Suggestions read the database only (one statement), so they are not a run
(api/limits.py ``RUN_PATHS``).
"""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import AuthUser, require_user
from api.schemas import AccountProfile, AccountResponse, AiUsageResponse, SuggestionsResponse, SuggestionsSwitch
from benchmarks import usual
from benchmarks.catalogue import ALL_INDUSTRIES_ID, canonical
from db import deals as deal_store
from db import users as user_store
from db.engine import is_configured as database_configured

router = APIRouter(prefix="/account", tags=["account"])

NO_DATABASE = ("Accounts need a database: set DATABASE_URL "
               "(python -m db.local locally). See CLAUDE.md.")
NO_ACCOUNT = "Finish your account (country, currency, format and time zone) first."


def _require_database() -> None:
    if not database_configured():
        raise HTTPException(status_code=503, detail=NO_DATABASE)


@router.get("", response_model=AccountResponse)
def get_account(user: AuthUser = Depends(require_user)):
    """The caller's account, with ``profile`` null until they've set one."""
    _require_database()
    profile = user_store.get_profile(user.subject)
    return {"subject": user.subject, "profile": profile.as_dict() if profile else None}


@router.post("", response_model=AccountResponse)
def save_account(profile: AccountProfile, user: AuthUser = Depends(require_user)):
    """Save the caller's country, currency, locale and time zone."""
    _require_database()
    try:
        cleaned = user_store.clean(**profile.model_dump())
    except user_store.InvalidProfile as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    saved = user_store.save_profile(user.subject, cleaned)
    return {"subject": user.subject, "profile": saved.as_dict()}


@router.get("/suggestions", response_model=SuggestionsResponse)
def get_suggestions(country: str = Query("", pattern=r"^([A-Z]{2})?$", description="ISO 3166-1 alpha-2"),
                    industry: str = Query(ALL_INDUSTRIES_ID, pattern=r"^[a-z0-9_]{1,60}$",
                                          description="An id from /api/benchmarks/industries"),
                    deal_id: Optional[uuid.UUID] = Query(
                        None, description="The deal on screen, left out of its own suggestion"),
                    user: AuthUser = Depends(require_user)):
    """The caller's usual figures for a deal in ``country`` and ``industry``:
    the median of each starting figure across their own saved deals in the
    same region and sector. Switched off, nothing is read and nothing answered."""
    _require_database()
    enabled, deals = deal_store.own_deal_inputs(user.subject, exclude=deal_id, limit=usual.MAX_DEALS)
    if not enabled:
        return {"enabled": False, "status": "off", "deals": 0, "min_deals": usual.MIN_DEALS, "figures": []}
    return {"enabled": True, **usual.usual_figures(deals, canonical(country), industry)}


@router.put("/suggestions", response_model=SuggestionsSwitch)
def put_suggestions(req: SuggestionsSwitch, user: AuthUser = Depends(require_user)):
    """Switch suggestions of the caller's usual figures on or off."""
    _require_database()
    try:
        return {"enabled": deal_store.set_suggestions(user.subject, req.enabled)}
    except deal_store.NoAccount:
        raise HTTPException(status_code=409, detail=NO_ACCOUNT) from None


@router.get("/ai-usage", response_model=AiUsageResponse)
def get_ai_usage(days: int = Query(31, ge=1, le=400), user: AuthUser = Depends(require_user)):
    """What the caller's results reads and note sentences used, a day at a time (PLAN.md R9): their own rows
    only, counts and cost, never anything from a document."""
    import dataclasses
    import datetime

    from api import plans
    from research import ai_costs

    rows = []
    if database_configured():
        from db import ai_usage

        since = datetime.datetime.now(datetime.timezone.utc).date() - datetime.timedelta(days=days - 1)
        rows = ai_usage.own(user.subject, since)
    total = sum((r["cost"] for r in rows), start=0)
    return {"days": [{**r, "cost": float(r["cost"])} for r in rows], "total_cost": float(total),
            "money": {"currency": ai_costs.CURRENCY}, "plan": dataclasses.asdict(plans.plan_for(user.subject)),
            "prices_source": ai_costs.SOURCE, "prices_checked_on": ai_costs.CHECKED_ON}
