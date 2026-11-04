"""API keys and webhooks, managed on the site (PLAN.md 7.7).

    GET    /api/account/api-keys              the caller's keys (never the keys themselves)
    POST   /api/account/api-keys              {name} -> a new key, answered once
    DELETE /api/account/api-keys/{id}         revoke it: the next request with it is refused
    GET    /api/account/webhooks              the caller's webhooks and how their newest message went
    POST   /api/account/webhooks              {url} -> a new webhook and its signing secret, answered once
    DELETE /api/account/webhooks/{id}         remove it and its messages
    POST   /api/account/webhooks/{id}/ping    send it a signed test message now

Only the site's own sign-in manages them: an API key reaches ``/api/v1`` and
nothing here (``api.auth.KEY_PATH``), and the add-in's pass never reaches
``/api/account``; both are refused here again in case either list ever widens.
Refusals are ``{"code": ...}``; the screen's words are ``developer.problem_<code>``.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Response

from api import webhooks as sender
from api.auth import AuthUser, require_user
from api.schemas import ApiKeyCreate, ApiKeyList, ApiKeyMade, WebhookCreate, WebhookList, WebhookMade, WebhookPing
from db import api_keys, webhooks
from db.engine import is_configured as database_configured

router = APIRouter(prefix="/account", tags=["api keys"])


def _refuse(status: int, code: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"code": code})


def _on_site(user: AuthUser) -> None:
    if not user.on_site:
        raise _refuse(403, "site_only")
    if not database_configured():
        raise _refuse(503, "no_database")


def _iso(value):
    return None if value is None else value.isoformat().replace("+00:00", "Z")


def _key(key: api_keys.KeyRecord) -> dict:
    return {"id": str(key.id), "name": key.name, "hint": key.hint, "created_at": _iso(key.created_at),
            "last_used_at": _iso(key.last_used_at)}


def _webhook(hook: webhooks.WebhookRecord) -> dict:
    return {"id": str(hook.id), "url": hook.url, "created_at": _iso(hook.created_at),
            "last_state": hook.last_state, "last_status": hook.last_status, "last_at": _iso(hook.last_at)}


@router.get("/api-keys", response_model=ApiKeyList)
def get_keys(user: AuthUser = Depends(require_user)):
    _on_site(user)
    return {"keys": [_key(k) for k in api_keys.list_keys(user.subject)], "max": api_keys.MAX_PER_USER}


@router.post("/api-keys", response_model=ApiKeyMade, status_code=201)
def post_key(req: ApiKeyCreate, user: AuthUser = Depends(require_user)):
    """A new key, answered once: only its hash is kept, so it can't be shown again."""
    _on_site(user)
    try:
        made = api_keys.create(user.subject, req.name)
    except ValueError:
        raise _refuse(422, "invalid_name") from None
    except api_keys.NoAccount:
        raise _refuse(409, "no_account") from None
    except api_keys.TooManyKeys:
        raise _refuse(409, "full_keys") from None
    return {**_key(made.key), "token": made.token}


@router.delete("/api-keys/{key_id}", status_code=204)
def delete_key(key_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    _on_site(user)
    try:
        api_keys.revoke(user.subject, key_id)
    except api_keys.KeyNotFound:
        raise _refuse(404, "not_found") from None
    return Response(status_code=204)


@router.get("/webhooks", response_model=WebhookList)
def get_webhooks(user: AuthUser = Depends(require_user)):
    _on_site(user)
    return {"webhooks": [_webhook(h) for h in webhooks.list_webhooks(user.subject)], "max": webhooks.MAX_PER_USER}


@router.post("/webhooks", response_model=WebhookMade, status_code=201)
def post_webhook(req: WebhookCreate, user: AuthUser = Depends(require_user)):
    """A new webhook and its signing secret, answered once."""
    _on_site(user)
    try:
        url = sender.check_url(req.url)
    except sender.Refused as exc:
        raise _refuse(422, exc.code) from None
    try:
        hook, secret = webhooks.add(user.subject, url)
    except webhooks.NoAccount:
        raise _refuse(409, "no_account") from None
    except webhooks.TooManyWebhooks:
        raise _refuse(409, "full_webhooks") from None
    return {**_webhook(hook), "secret": secret}


@router.delete("/webhooks/{webhook_id}", status_code=204)
def delete_webhook(webhook_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    _on_site(user)
    try:
        webhooks.remove(user.subject, webhook_id)
    except webhooks.WebhookNotFound:
        raise _refuse(404, "not_found") from None
    return Response(status_code=204)


@router.post("/webhooks/{webhook_id}/ping", response_model=WebhookPing, status_code=202)
def ping_webhook(webhook_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """Send a signed ``ping`` now (in the background); the list shows how it went."""
    _on_site(user)
    try:
        message = webhooks.ping(user.subject, webhook_id)
    except webhooks.WebhookNotFound:
        raise _refuse(404, "not_found") from None
    sender.kick()
    return {"message_id": str(message)}
