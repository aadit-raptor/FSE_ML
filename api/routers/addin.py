"""The Excel add-in's sign-in (PLAN.md R3).

    POST   /api/addin/sessions    a new add-in pass, for a user signed in on the site itself
    GET    /api/addin/session     the pass in use: when it ends
    DELETE /api/addin/session     sign the add-in out: the pass ends

Excel's task pane is a page of this site (``/excel``), but it can't use the
site's sign-in: on Excel on the web it is a frame inside Microsoft's page,
where the browser keeps the site's cookies from it. So the pane opens
Office's dialog on ``/excel/connect``, a window of its own where the site's
sign-in works as anywhere else; signed in, that page asks here for a pass and
hands it to the pane (``Office.context.ui.messageParent``). The pane sends it
as ``Bearer addin:<pass>`` (``api/auth.py``), which reaches only the research
endpoints. A pass can't make another pass: ``POST`` needs the site's own
sign-in.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.security import HTTPAuthorizationCredentials

from api.auth import AuthUser, _bearer, forget_addin_pass, require_user
from api.schemas import AddinPass, AddinSessionOut
from db import addin
from db.engine import is_configured as database_configured

router = APIRouter(prefix="/addin", tags=["addin"])


def _database() -> None:
    if not database_configured():
        raise HTTPException(status_code=503, detail="no_database")


@router.post("/sessions", response_model=AddinPass, status_code=201)
def post_session(user: AuthUser = Depends(require_user)):
    """A pass for the add-in, answered once; only its hash is kept."""
    if user.via_addin:
        raise HTTPException(status_code=403, detail="addin_scope")
    _database()
    try:
        made = addin.create(user.subject)
    except addin.NoAccount:
        raise HTTPException(status_code=409, detail="no_account") from None
    return {"token": made.token, "expires_at": made.expires_at.isoformat(),
            "lifetime_s": int(addin.LIFETIME.total_seconds())}


def _pass(user: AuthUser, credentials: HTTPAuthorizationCredentials | None) -> str:
    if not user.via_addin or credentials is None:
        raise HTTPException(status_code=400, detail="not_addin")
    return credentials.credentials


@router.get("/session", response_model=AddinSessionOut)
def get_session(user: AuthUser = Depends(require_user),
                credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    token = _pass(user, credentials)
    holder = addin.lookup(token)
    if holder is None:
        raise HTTPException(status_code=401, detail="add-in pass ended; sign in again",
                            headers={"WWW-Authenticate": "Bearer"})
    return {"expires_at": holder.expires_at.isoformat()}


@router.delete("/session", status_code=204)
def delete_session(user: AuthUser = Depends(require_user),
                   credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    token = _pass(user, credentials)
    addin.revoke(token)
    forget_addin_pass(token)
    return Response(status_code=204)
