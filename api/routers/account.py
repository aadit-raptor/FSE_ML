"""The signed-in person's own account (PLAN.md 1.4).

Two endpoints, both about the caller and nobody else: the subject always
comes from the verified token, never from the request body, so one account
can't read or write another's profile.
"""
from fastapi import APIRouter, Depends, HTTPException

from api.auth import AuthUser, require_user
from api.schemas import AccountProfile, AccountResponse
from db import users as user_store
from db.engine import is_configured as database_configured

router = APIRouter(prefix="/account", tags=["account"])

NO_DATABASE = ("Accounts need a database: set DATABASE_URL "
               "(python -m db.local locally). See CLAUDE.md.")


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
