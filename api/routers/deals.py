"""Saved deals, versions and the account's settings (PLAN.md 1.5).

The owner is always the signed-in caller (``require_user``), never something
in the request, and ``db/deals.py`` matches it in every statement: another
account's deal answers 404, exactly like a deal that doesn't exist.
"""
import uuid
from contextlib import contextmanager
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response

from api.auth import AuthUser, require_user
from api.schemas import (
    AccountSettings, DealContent, DealCreate, DealDetail, DealDuplicate, DealList, DealPatch,
    VersionDetail, VersionList, VersionSave, VersionSummary,
)
from db import deals as store
from db.engine import is_configured as database_configured

router = APIRouter(tags=["saved deals"])

NO_DATABASE = ("Saved deals need a database: set DATABASE_URL "
               "(python -m db.local locally). See CLAUDE.md.")
NO_ACCOUNT = "Finish your account (country, currency, format and time zone) before saving."


@contextmanager
def _store():
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


def _detail(deal: store.DealRecord) -> dict:
    return {**_summary(deal), "inputs": deal.inputs, "settings": deal.settings}


def _version(v: store.VersionRecord, *, content: bool = False) -> dict:
    out = {"number": v.number, "kind": v.kind, "label": v.label, "created_at": _iso(v.created_at)}
    if content:
        out.update(inputs=v.inputs, settings=v.settings)
    return out


# ---------------------------------------------------------------------------
# Deals
# ---------------------------------------------------------------------------
@router.get("/deals", response_model=DealList)
def list_deals(archived: bool = False, user: AuthUser = Depends(require_user)):
    """The caller's deals, most recently edited first. ``archived=true`` includes archived ones."""
    with _store():
        deals = store.list_deals(user.subject, include_archived=archived)
    return {"deals": [_summary(d) for d in deals]}


@router.post("/deals", response_model=DealDetail, status_code=201)
def create_deal(req: DealCreate, user: AuthUser = Depends(require_user)):
    """Save a new deal; its first version is created with it."""
    with _store():
        deal = store.create_deal(user.subject, req.name, req.inputs.model_dump(), req.settings)
    return _detail(deal)


@router.get("/deals/{deal_id}", response_model=DealDetail)
def get_deal(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """Open a deal: its working copy, exactly as last saved."""
    with _store():
        deal = store.get_deal(user.subject, deal_id)
    return _detail(deal)


@router.patch("/deals/{deal_id}", response_model=DealDetail)
def patch_deal(deal_id: uuid.UUID, req: DealPatch, user: AuthUser = Depends(require_user)):
    """Rename, archive or unarchive."""
    with _store():
        deal = store.update_deal(user.subject, deal_id, name=req.name, archived=req.archived)
    return _detail(deal)


@router.delete("/deals/{deal_id}", status_code=204, response_class=Response)
def delete_deal(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """Delete a deal and all its versions, for good."""
    with _store():
        store.delete_deal(user.subject, deal_id)
    return Response(status_code=204)


@router.put("/deals/{deal_id}/draft", response_model=DealDetail)
def save_draft(deal_id: uuid.UUID, req: DealContent, user: AuthUser = Depends(require_user)):
    """Autosave the working copy (adds an automatic checkpoint now and then)."""
    with _store():
        deal = store.save_draft(user.subject, deal_id, req.inputs.model_dump(), req.settings)
    return _detail(deal)


@router.post("/deals/{deal_id}/duplicate", response_model=DealDetail, status_code=201)
def duplicate_deal(deal_id: uuid.UUID, req: Optional[DealDuplicate] = None,
                   user: AuthUser = Depends(require_user)):
    """A new deal from this one's working copy (its history stays behind)."""
    with _store():
        deal = store.duplicate_deal(user.subject, deal_id, req.name if req else None)
    return _detail(deal)


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------
@router.get("/deals/{deal_id}/versions", response_model=VersionList)
def list_versions(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """The deal's history, newest first."""
    with _store():
        versions = store.list_versions(user.subject, deal_id)
    return {"versions": [_version(v) for v in versions]}


@router.post("/deals/{deal_id}/versions", response_model=VersionSummary, status_code=201)
def save_version(deal_id: uuid.UUID, req: Optional[VersionSave] = None,
                 user: AuthUser = Depends(require_user)):
    """Keep the working copy as a version (no new row if nothing changed)."""
    with _store():
        version = store.save_version(user.subject, deal_id, req.label if req else None)
    return _version(version)


@router.get("/deals/{deal_id}/versions/{number}", response_model=VersionDetail)
def get_version(deal_id: uuid.UUID, number: int, user: AuthUser = Depends(require_user)):
    """One version with its inputs and settings."""
    with _store():
        version = store.get_version(user.subject, deal_id, number)
    return _version(version, content=True)


@router.post("/deals/{deal_id}/versions/{number}/restore", response_model=DealDetail)
def restore_version(deal_id: uuid.UUID, number: int, user: AuthUser = Depends(require_user)):
    """Make this version the working copy; unsaved edits are kept as a version first."""
    with _store():
        deal = store.restore_version(user.subject, deal_id, number)
    return _detail(deal)


# ---------------------------------------------------------------------------
# Settings that follow the account
# ---------------------------------------------------------------------------
@router.get("/account/settings", response_model=AccountSettings)
def get_account_settings(user: AuthUser = Depends(require_user)):
    """The caller's Settings overrides, the same on every device."""
    with _store():
        settings = store.get_settings(user.subject)
    return {"settings": settings}


@router.put("/account/settings", response_model=AccountSettings)
def put_account_settings(req: AccountSettings, user: AuthUser = Depends(require_user)):
    """Replace the caller's Settings overrides."""
    with _store():
        settings = store.save_settings(user.subject, req.settings)
    return {"settings": settings}
