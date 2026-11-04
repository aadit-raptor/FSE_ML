"""Webhooks and their deliveries (PLAN.md 7.7; tables ``webhooks``, ``webhook_deliveries``).

A webhook is an address the API tells when a job submitted through
``/api/v1`` ends. A person has at most ``MAX_PER_USER``. Its signing secret is
``whsec_`` and 32 random bytes in base64 (Standard Webhooks), answered once
when it is added.

The road of a message: a job submitted through ``/api/v1/jobs`` writes one
``waiting`` row per webhook of its owner (``expect``); when the job ends the
runner makes them ``due`` with their message (``job_finished``), and
``api/webhooks.py`` claims and sends them (``claim``, ``settle``), again after
``BACKOFF_S`` on a failure, until ``failed``. A message holds the job's id,
kind and status, never its result or its error text, which may hold a deal's
figures. Rows go after ``KEEP_DAYS`` (``sweep``, in the nightly
job maintenance).

Every statement a person's request makes matches the owner, so another
account's webhook is ``WebhookNotFound`` (a 404 like a missing one).
"""
from __future__ import annotations

import base64
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import delete, func, insert, literal, select, update

from db import audit
from db.engine import connect, transaction
from db.models import Webhook, WebhookDelivery, User, utc_now

MAX_PER_USER = 3
# Seconds before each retry; a message is tried once and then once after each of these
BACKOFF_S = (10, 60, 300)
MAX_ATTEMPTS = 1 + len(BACKOFF_S)
# A claimed message not settled within this long (the API restarted mid-send) is due again
LEASE_S = 120
KEEP_DAYS = 7
SECRET_PREFIX = "whsec_"


class NoAccount(LookupError):
    """The caller has no account row yet."""


class WebhookNotFound(LookupError):
    """No such webhook of the caller's."""


class TooManyWebhooks(Exception):
    """The caller already has ``MAX_PER_USER`` webhooks."""


@dataclass(frozen=True)
class WebhookRecord:
    id: uuid.UUID
    url: str
    created_at: datetime
    # The newest message's state and the HTTP status its last attempt got (0: no answer); None before any
    last_state: Optional[str] = None
    last_status: Optional[int] = None
    last_at: Optional[datetime] = None


@dataclass(frozen=True)
class Claimed:
    id: uuid.UUID
    url: str
    secret: str
    payload: dict
    attempts: int


def new_secret() -> str:
    return SECRET_PREFIX + base64.b64encode(secrets.token_bytes(32)).decode("ascii")


def secret_bytes(secret: str) -> bytes:
    return base64.b64decode(secret[len(SECRET_PREFIX):])


def add(subject: str, url: str, *, now: Optional[datetime] = None) -> tuple[WebhookRecord, str]:
    """Add a webhook (``url`` already checked by ``api.webhooks.check_url``); its secret, answered once."""
    now = now or utc_now()
    secret = new_secret()
    with transaction() as conn:
        owner = conn.execute(select(User.id).where(User.subject == subject).with_for_update()).scalar()
        if owner is None:
            raise NoAccount(subject)
        held = conn.execute(select(func.count()).select_from(Webhook).where(Webhook.owner_id == owner)).scalar()
        if held >= MAX_PER_USER:
            raise TooManyWebhooks()
        row = conn.execute(insert(Webhook).values(owner_id=owner, url=url, secret=secret, created_at=now)
                           .returning(Webhook.id, Webhook.url, Webhook.created_at)).one()
        audit.record(conn, owner, "webhook_added", now=now)
    return WebhookRecord(id=row.id, url=row.url, created_at=row.created_at), secret


def list_webhooks(subject: str) -> list[WebhookRecord]:
    """The caller's webhooks, newest first, each with how its newest message went."""
    newest = (select(WebhookDelivery.state, WebhookDelivery.last_status, WebhookDelivery.created_at)
              .where(WebhookDelivery.webhook_id == Webhook.id, WebhookDelivery.state != "waiting")
              .order_by(WebhookDelivery.created_at.desc()).limit(1).lateral())
    with connect() as conn:
        rows = conn.execute(
            select(Webhook.id, Webhook.url, Webhook.created_at, newest.c.state, newest.c.last_status,
                   newest.c.created_at.label("last_at"))
            .join(User, User.id == Webhook.owner_id)
            .outerjoin(newest, literal(True))
            .where(User.subject == subject)
            .order_by(Webhook.created_at.desc(), Webhook.id)).all()
    return [WebhookRecord(id=r.id, url=r.url, created_at=r.created_at, last_state=r.state,
                          last_status=r.last_status, last_at=r.last_at) for r in rows]


def remove(subject: str, webhook_id: uuid.UUID, *, now: Optional[datetime] = None) -> None:
    """Delete the caller's webhook and its messages: nothing more is sent to it."""
    with transaction() as conn:
        gone = conn.execute(delete(Webhook).where(Webhook.id == webhook_id,
                                                  Webhook.owner_id == audit.subject_id(subject))
                            .returning(Webhook.owner_id)).first()
        if gone is None:
            raise WebhookNotFound(webhook_id)
        audit.record(conn, gone[0], "webhook_removed", now=now)


def ping(subject: str, webhook_id: uuid.UUID, *, now: Optional[datetime] = None) -> uuid.UUID:
    """Queue a ``ping`` message to the caller's webhook, due now. Its id."""
    now = now or utc_now()
    with transaction() as conn:
        found = conn.execute(select(Webhook.id).where(Webhook.id == webhook_id,
                                                      Webhook.owner_id == audit.subject_id(subject))).scalar()
        if found is None:
            raise WebhookNotFound(webhook_id)
        payload = {"type": "ping", "timestamp": _iso(now), "data": {"webhook_id": str(webhook_id)}}
        return conn.execute(insert(WebhookDelivery).values(
            webhook_id=webhook_id, event="ping", state="due", payload=payload, next_at=now, created_at=now)
            .returning(WebhookDelivery.id)).scalar_one()


def expect(subject: str, job_id: uuid.UUID, *, now: Optional[datetime] = None) -> int:
    """A job of the caller's was submitted through ``/api/v1``: one waiting message per webhook. How many."""
    now = now or utc_now()
    with transaction() as conn:
        made = conn.execute(insert(WebhookDelivery).from_select(
            ["webhook_id", "job_id", "event", "state", "created_at"],
            select(Webhook.id, literal(job_id), literal("job.finished"), literal("waiting"), literal(now))
            .where(Webhook.owner_id == audit.subject_id(subject))).returning(WebhookDelivery.id)).all()
    return len(made)


def job_finished(job_id: uuid.UUID, kind: str, status: str, error_status: Optional[int] = None,
                 *, now: Optional[datetime] = None) -> int:
    """The job ended: its waiting messages are due, with their message. How many."""
    now = now or utc_now()
    payload = {"type": "job.finished", "timestamp": _iso(now),
               "data": {"job_id": str(job_id), "kind": kind, "status": status, "error_status": error_status}}
    with transaction() as conn:
        made = conn.execute(update(WebhookDelivery)
                            .where(WebhookDelivery.job_id == job_id, WebhookDelivery.state == "waiting")
                            .values(state="due", payload=payload, next_at=now))
    return int(made.rowcount or 0)


def claim(*, now: Optional[datetime] = None) -> Optional[Claimed]:
    """The oldest due message, leased for ``LEASE_S`` and counted as an attempt (one statement)."""
    now = now or utc_now()
    oldest = (select(WebhookDelivery.id).where(WebhookDelivery.state == "due", WebhookDelivery.next_at <= now)
              .order_by(WebhookDelivery.next_at).limit(1).with_for_update(skip_locked=True).scalar_subquery())
    with transaction() as conn:
        row = conn.execute(
            update(WebhookDelivery)
            .where(WebhookDelivery.id == oldest, WebhookDelivery.webhook_id == Webhook.id)
            .values(next_at=now + timedelta(seconds=LEASE_S), attempts=WebhookDelivery.attempts + 1)
            .returning(WebhookDelivery.id, WebhookDelivery.payload, WebhookDelivery.attempts,
                       Webhook.url, Webhook.secret)).first()
    if row is None:
        return None
    return Claimed(id=row.id, url=row.url, secret=row.secret, payload=row.payload, attempts=row.attempts)


def settle(message: Claimed, status: int, *, now: Optional[datetime] = None) -> str:
    """Record an attempt's answer (``status`` 0: none): delivered, due again later, or failed. The new state."""
    now = now or utc_now()
    if 200 <= status < 300:
        values = {"state": "delivered", "finished_at": now, "next_at": None}
    elif message.attempts >= MAX_ATTEMPTS:
        values = {"state": "failed", "finished_at": now, "next_at": None}
    else:
        values = {"next_at": now + timedelta(seconds=BACKOFF_S[message.attempts - 1])}
    with transaction() as conn:
        conn.execute(update(WebhookDelivery).where(WebhookDelivery.id == message.id)
                     .values(last_status=status, **values))
    return values.get("state", "due")


def next_due() -> Optional[datetime]:
    """When the next due message may be sent, or None when nothing is due."""
    with connect() as conn:
        return conn.execute(select(func.min(WebhookDelivery.next_at)).where(WebhookDelivery.state == "due")).scalar()


def sweep(*, now: Optional[datetime] = None) -> int:
    """Delete messages older than ``KEEP_DAYS``: sent, failed, or waiting on a job that never ended."""
    cutoff = (now or utc_now()) - timedelta(days=KEEP_DAYS)
    with transaction() as conn:
        gone = conn.execute(delete(WebhookDelivery).where(WebhookDelivery.created_at < cutoff))
    return int(gone.rowcount or 0)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


__all__ = ["BACKOFF_S", "Claimed", "KEEP_DAYS", "LEASE_S", "MAX_ATTEMPTS", "MAX_PER_USER", "NoAccount",
           "TooManyWebhooks", "WebhookNotFound", "WebhookRecord", "add", "claim", "expect", "job_finished",
           "list_webhooks", "next_due", "ping", "remove", "secret_bytes", "settle", "sweep"]
