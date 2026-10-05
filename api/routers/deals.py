"""Saved deals, versions and the account's settings (PLAN.md 1.5).

The owner is always the signed-in caller (``require_user``), never something
in the request, and ``db/deals.py`` matches it in every statement: another
account's deal answers 404, exactly like a deal that doesn't exist.

Which model (PLAN.md 3.1): creating a deal or autosaving it runs the deal
model once (no sensitivity grid, a few milliseconds) and stores the model
stamp with the IRR and MOIC; opening or restoring it runs it again and answers
``model_check``, which the deal screens show as "results changed since saved".
"""
import logging
import uuid
from contextlib import contextmanager
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from api.auth import AuthUser, require_user
from api.observability import log_event
from api.schemas import (
    AccountSettings, AuditHistory, DealActuals, DealContent, DealCreate, DealDetail, DealDuplicate,
    DealList, DealPatch, StoredActuals, VersionDetail, VersionList, VersionSave, VersionSummary,
)
from core import model_version
from core.config import resolve_config
from core.deal import DealInputs
from db import audit
from db import deals as store
from db.engine import is_configured as database_configured

router = APIRouter(tags=["saved deals"])

NO_DATABASE = ("Saved deals need a database: set DATABASE_URL "
               "(python -m db.local locally). See CLAUDE.md.")
NO_ACCOUNT = "Finish your account (country, currency, format and time zone) before saving."


@contextmanager
def store_errors():
    """Store errors as HTTP answers."""
    if not database_configured():
        raise HTTPException(status_code=503, detail=NO_DATABASE)
    try:
        yield
    except store.DealNotFound:
        raise HTTPException(status_code=404, detail="Deal not found") from None
    except store.VersionNotFound:
        raise HTTPException(status_code=404, detail="Version not found") from None
    except store.NoAccount:
        raise HTTPException(status_code=409, detail=NO_ACCOUNT) from None
    except store.InvalidDeal as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None


def _iso(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _summary(deal: store.DealRecord) -> dict:
    return {"id": str(deal.id), "name": deal.name, "archived": deal.archived,
            "latest_version": deal.latest_version,
            "created_at": _iso(deal.created_at), "updated_at": _iso(deal.updated_at)}


def _model_now(inputs, settings) -> Optional[dict]:
    """The stamp and results this content gives today, or None when the deal
    model fails on it: saving someone's work never depends on the model, so
    the deal is kept without a stamp (``unknown`` on reopening) and the
    failure is logged by its kind alone -- never the deal's figures."""
    from api.schemas import DealInputsIn

    inputs, settings = store.clean_content(inputs, settings)
    try:
        deal = DealInputs(**DealInputsIn.model_validate(inputs).model_dump())
        return {**model_version.saved_stamp(deal, resolve_config(settings)),
                "content": model_version.content_fingerprint(inputs, settings)}
    except Exception as exc:  # noqa: BLE001 - recorded, and the save goes on
        log_event("model_stamp_failed", logging.WARNING, error=type(exc).__name__)
        return None


def _detail(deal: store.DealRecord, *, check: bool = False) -> dict:
    out = {**_summary(deal), "inputs": deal.inputs, "settings": deal.settings, "model": deal.model}
    if check:
        now = _model_now(deal.inputs, deal.settings)
        out["model_check"] = model_version.compare(deal.model, now) if now else None
    return out


def _version(v: store.VersionRecord, *, content: bool = False) -> dict:
    out = {"number": v.number, "kind": v.kind, "label": v.label, "created_at": _iso(v.created_at)}
    if content:
        out.update(inputs=v.inputs, settings=v.settings, model=v.model)
    return out


# ---------------------------------------------------------------------------
# Deals
# ---------------------------------------------------------------------------
@router.get("/deals", response_model=DealList)
def list_deals(archived: bool = False, user: AuthUser = Depends(require_user)):
    """The caller's deals, most recently edited first. ``archived=true`` includes archived ones."""
    with store_errors():
        deals = store.list_deals(user.subject, include_archived=archived)
    return {"deals": [_summary(d) for d in deals]}


@router.post("/deals", response_model=DealDetail, status_code=201)
def create_deal(req: DealCreate, user: AuthUser = Depends(require_user)):
    """Save a new deal; its first version is created with it."""
    with store_errors():
        inputs = req.inputs.model_dump()
        deal = store.create_deal(user.subject, req.name, inputs, req.settings,
                                 model=_model_now(inputs, req.settings))
    return _detail(deal)


@router.get("/deals/{deal_id}", response_model=DealDetail)
def get_deal(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """Open a deal: its working copy, exactly as last saved, and whether its
    results have changed since (``model_check``)."""
    with store_errors():
        deal = store.get_deal(user.subject, deal_id)
    return _detail(deal, check=True)


@router.patch("/deals/{deal_id}", response_model=DealDetail)
def patch_deal(deal_id: uuid.UUID, req: DealPatch, user: AuthUser = Depends(require_user)):
    """Rename, archive or unarchive."""
    with store_errors():
        deal = store.update_deal(user.subject, deal_id, name=req.name, archived=req.archived)
    return _detail(deal)


@router.delete("/deals/{deal_id}", status_code=204, response_class=Response)
def delete_deal(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """Delete a deal and all its versions, for good."""
    with store_errors():
        store.delete_deal(user.subject, deal_id)
    return Response(status_code=204)


@router.put("/deals/{deal_id}/draft", response_model=DealDetail)
def save_draft(deal_id: uuid.UUID, req: DealContent, user: AuthUser = Depends(require_user)):
    """Autosave the working copy (adds an automatic checkpoint now and then)."""
    with store_errors():
        inputs = req.inputs.model_dump()
        deal = store.save_draft(user.subject, deal_id, inputs, req.settings,
                                model=_model_now(inputs, req.settings))
    return _detail(deal)


@router.post("/deals/{deal_id}/duplicate", response_model=DealDetail, status_code=201)
def duplicate_deal(deal_id: uuid.UUID, req: Optional[DealDuplicate] = None,
                   user: AuthUser = Depends(require_user)):
    """A new deal from this one's working copy (its history stays behind)."""
    with store_errors():
        deal = store.duplicate_deal(user.subject, deal_id, req.name if req else None)
    return _detail(deal)


# ---------------------------------------------------------------------------
# Actuals (PLAN.md 2.7)
# ---------------------------------------------------------------------------
def _actuals(actuals, stamp) -> dict:
    return {"actuals": actuals, "updated_at": _iso(stamp) if stamp else None}


@router.get("/deals/{deal_id}/actuals", response_model=StoredActuals)
def get_actuals(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """What actually happened to the deal, for plan vs actual; null until saved."""
    with store_errors():
        return _actuals(*store.get_actuals(user.subject, deal_id))


@router.put("/deals/{deal_id}/actuals", response_model=StoredActuals)
def put_actuals(deal_id: uuid.UUID, req: DealActuals, user: AuthUser = Depends(require_user)):
    """Save the deal's actual results and exit, replacing what was there.
    Not a version of the deal: its plan and history are untouched."""
    with store_errors():
        return _actuals(*store.save_actuals(user.subject, deal_id, req.model_dump()))


@router.delete("/deals/{deal_id}/actuals", response_model=StoredActuals)
def delete_actuals(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """Forget the deal's actuals."""
    with store_errors():
        return _actuals(*store.save_actuals(user.subject, deal_id, None))


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------
@router.get("/deals/{deal_id}/versions", response_model=VersionList)
def list_versions(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """The deal's history, newest first."""
    with store_errors():
        versions = store.list_versions(user.subject, deal_id)
    return {"versions": [_version(v) for v in versions]}


@router.post("/deals/{deal_id}/versions", response_model=VersionSummary, status_code=201)
def save_version(deal_id: uuid.UUID, req: Optional[VersionSave] = None,
                 user: AuthUser = Depends(require_user)):
    """Keep the working copy as a version (no new row if nothing changed)."""
    with store_errors():
        version = store.save_version(user.subject, deal_id, req.label if req else None)
    return _version(version)


@router.get("/deals/{deal_id}/versions/{number}", response_model=VersionDetail)
def get_version(deal_id: uuid.UUID, number: int, user: AuthUser = Depends(require_user)):
    """One version with its inputs and settings."""
    with store_errors():
        version = store.get_version(user.subject, deal_id, number)
    return _version(version, content=True)


@router.post("/deals/{deal_id}/versions/{number}/restore", response_model=DealDetail)
def restore_version(deal_id: uuid.UUID, number: int, user: AuthUser = Depends(require_user)):
    """Make this version the working copy; unsaved edits are kept as a version
    first. ``model_check`` says whether its results have changed since it was saved."""
    with store_errors():
        deal = store.restore_version(user.subject, deal_id, number)
    return _detail(deal, check=True)


# ---------------------------------------------------------------------------
# Audit history (PLAN.md 3.3): read only; nothing in the API changes an entry
# ---------------------------------------------------------------------------
HistoryLimit = Query(200, ge=1, le=audit.MAX_ENTRIES, description="Newest entries to answer")


def _entry(e: audit.Entry, *, named: bool = False) -> dict:
    out = {"id": e.id, "action": e.action, "at": _iso(e.occurred_at),
           "until": _iso(e.last_at) if e.last_at else None, "count": e.count,
           "deal_id": str(e.deal_id) if e.deal_id else None, **e.detail}
    if named:
        out["deal_name"] = e.deal_name
    return out


@router.get("/deals/{deal_id}/history", response_model=AuditHistory)
def deal_history(deal_id: uuid.UUID, limit: int = HistoryLimit, user: AuthUser = Depends(require_user)):
    """What was done to the deal and when, newest first: one entry per action
    (old edits merged one per day)."""
    with store_errors():
        entries = store.deal_history(user.subject, deal_id, limit=limit)
    return {"entries": [_entry(e) for e in entries]}


@router.get("/account/history", response_model=AuditHistory)
def account_history(limit: int = HistoryLimit, user: AuthUser = Depends(require_user)):
    """Everything the caller did -- to every deal, deleted ones included, and to
    their Settings -- newest first."""
    with store_errors():
        entries = audit.account_history(user.subject, limit=limit)
    return {"entries": [_entry(e, named=True) for e in entries]}


# ---------------------------------------------------------------------------
# Settings that follow the account
# ---------------------------------------------------------------------------
@router.get("/account/settings", response_model=AccountSettings)
def get_account_settings(user: AuthUser = Depends(require_user)):
    """The caller's Settings overrides, the same on every device."""
    with store_errors():
        settings = store.get_settings(user.subject)
    return {"settings": settings}


@router.put("/account/settings", response_model=AccountSettings)
def put_account_settings(req: AccountSettings, user: AuthUser = Depends(require_user)):
    """Replace the caller's Settings overrides."""
    with store_errors():
        settings = store.save_settings(user.subject, req.settings)
    return {"settings": settings}
