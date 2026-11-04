"""API keys for the public API (PLAN.md 7.7, table ``api_keys``).

A key is ``vk_`` and 43 random URL-safe characters (32 bytes). It is answered
once, to the signed-in person who made it, and only its SHA-256 is stored with
the name they gave it and its last four characters. A person holds at most
``MAX_PER_USER``. Revoking deletes the row; ``lookup`` reads the table on every
request (no cache), so a revoked key is refused by the very next request.

Every statement matches the owner: a key is listed and revoked by the person
who made it and nobody else (another account's key id is ``KeyNotFound``, a
404 like a missing one). Making and revoking each write one audit entry in
the same transaction, without the key's name.
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import delete, func, insert, select, update

from db import audit
from db.engine import connect, transaction
from db.models import ApiKey, User, utc_now

PREFIX = "vk_"
TOKEN_BYTES = 32
MAX_PER_USER = 5
NAME_MAX = 60
# ``last_used_at`` is written at most this often, so a busy script doesn't write on every request
TOUCH_EVERY = timedelta(minutes=1)


class NoAccount(LookupError):
    """The caller has no account row yet (they haven't finished sign-up)."""


class KeyNotFound(LookupError):
    """No such key of the caller's."""


class TooManyKeys(Exception):
    """The caller already holds ``MAX_PER_USER`` keys."""


@dataclass(frozen=True)
class KeyRecord:
    id: uuid.UUID
    name: str
    hint: str
    created_at: datetime
    last_used_at: Optional[datetime]


@dataclass(frozen=True)
class Made:
    key: KeyRecord
    token: str


@dataclass(frozen=True)
class Holder:
    subject: str
    key_id: uuid.UUID


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def looks_like_key(token: str) -> bool:
    body = token[len(PREFIX):] if token.startswith(PREFIX) else ""
    return 40 <= len(body) <= 64 and all(ch.isalnum() or ch in "-_" for ch in body)


def clean_name(name: str) -> str:
    """A key's name: trimmed, 1 to ``NAME_MAX`` characters, no control characters; else ValueError."""
    name = " ".join((name or "").split())
    if not name or len(name) > NAME_MAX or any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise ValueError("invalid_name")
    return name


def _record(row) -> KeyRecord:
    return KeyRecord(id=row.id, name=row.name, hint=row.hint, created_at=row.created_at,
                     last_used_at=row.last_used_at)


_COLUMNS = (ApiKey.id, ApiKey.name, ApiKey.hint, ApiKey.created_at, ApiKey.last_used_at)


def create(subject: str, name: str, *, now: Optional[datetime] = None) -> Made:
    """A new key for ``subject``, answered once."""
    name = clean_name(name)
    now = now or utc_now()
    token = PREFIX + secrets.token_urlsafe(TOKEN_BYTES)
    with transaction() as conn:
        owner = conn.execute(select(User.id).where(User.subject == subject).with_for_update()).scalar()
        if owner is None:
            raise NoAccount(subject)
        held = conn.execute(select(func.count()).select_from(ApiKey).where(ApiKey.owner_id == owner)).scalar()
        if held >= MAX_PER_USER:
            raise TooManyKeys()
        row = conn.execute(insert(ApiKey).values(owner_id=owner, name=name, token_hash=token_hash(token),
                                                 hint=token[-4:], created_at=now).returning(*_COLUMNS)).one()
        audit.record(conn, owner, "api_key_created", now=now)
    return Made(key=_record(row), token=token)


def list_keys(subject: str) -> list[KeyRecord]:
    with connect() as conn:
        rows = conn.execute(select(*_COLUMNS).join(User, User.id == ApiKey.owner_id)
                            .where(User.subject == subject)
                            .order_by(ApiKey.created_at.desc(), ApiKey.id)).all()
    return [_record(r) for r in rows]


def revoke(subject: str, key_id: uuid.UUID, *, now: Optional[datetime] = None) -> None:
    """Delete the caller's key: the next request with it is refused."""
    with transaction() as conn:
        owner = audit.subject_id(subject)
        gone = conn.execute(delete(ApiKey).where(ApiKey.id == key_id, ApiKey.owner_id == owner)
                            .returning(ApiKey.owner_id)).first()
        if gone is None:
            raise KeyNotFound(key_id)
        audit.record(conn, gone[0], "api_key_revoked", now=now)


def lookup(token: str, *, now: Optional[datetime] = None) -> Optional[Holder]:
    """Whose key this is, or None for an unknown, revoked or malformed one. Marks it used (once a minute)."""
    if not looks_like_key(token):
        return None
    now = now or utc_now()
    with connect() as conn:
        row = conn.execute(select(User.subject, ApiKey.id, ApiKey.last_used_at)
                           .join(User, User.id == ApiKey.owner_id)
                           .where(ApiKey.token_hash == token_hash(token))).first()
        if row is None:
            return None
        if row.last_used_at is None or now - row.last_used_at >= TOUCH_EVERY:
            conn.execute(update(ApiKey).where(ApiKey.id == row.id).values(last_used_at=now))
            conn.commit()
    return Holder(subject=row.subject, key_id=row.id)


__all__ = ["Holder", "KeyNotFound", "KeyRecord", "MAX_PER_USER", "Made", "NoAccount", "PREFIX", "TooManyKeys",
           "clean_name", "create", "list_keys", "looks_like_key", "lookup", "revoke", "token_hash"]
