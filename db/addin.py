"""The Excel add-in's sign-ins (PLAN.md R3, table ``addin_sessions``).

A pass is ``addin:`` and 43 random URL-safe characters (32 bytes). It is
answered once, to the signed-in dialog that asked for it, and only its
SHA-256 is stored, so a copy of the table signs nobody in. A pass lasts
``LIFETIME`` from when it was made (a working day; signing in again is one
click in a window that remembers the site's own sign-in), a person holds at
most ``MAX_PER_USER`` (a new one ends the oldest), and signing out deletes
it. Expired rows go with the nightly job maintenance (``sweep``).

Every statement matches what it is given exactly: ``lookup`` finds a row by
the pass's hash and its end still ahead, ``revoke`` deletes by hash.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import delete, insert, select

from db.engine import connect, transaction
from db.models import AddinSession, User, utc_now

PREFIX = "addin:"
LIFETIME = timedelta(hours=12)
MAX_PER_USER = 5
TOKEN_BYTES = 32


class NoAccount(LookupError):
    """The caller has no account row yet (they haven't finished sign-up)."""


@dataclass(frozen=True)
class Pass:
    token: str
    expires_at: datetime


@dataclass(frozen=True)
class Holder:
    subject: str
    expires_at: datetime


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def looks_like_pass(token: str) -> bool:
    body = token[len(PREFIX):] if token.startswith(PREFIX) else ""
    return 40 <= len(body) <= 64 and all(ch.isalnum() or ch in "-_" for ch in body)


def create(subject: str, *, now: Optional[datetime] = None) -> Pass:
    """A new pass for ``subject``; the oldest beyond ``MAX_PER_USER`` and any expired ones end."""
    now = now or utc_now()
    token = PREFIX + secrets.token_urlsafe(TOKEN_BYTES)
    expires = now + LIFETIME
    with transaction() as conn:
        owner = conn.execute(select(User.id).where(User.subject == subject)).scalar()
        if owner is None:
            raise NoAccount(subject)
        conn.execute(delete(AddinSession).where(AddinSession.owner_id == owner, AddinSession.expires_at <= now))
        keep = select(AddinSession.id).where(AddinSession.owner_id == owner) \
            .order_by(AddinSession.created_at.desc(), AddinSession.id.desc()).limit(MAX_PER_USER - 1)
        conn.execute(delete(AddinSession).where(AddinSession.owner_id == owner, AddinSession.id.not_in(keep)))
        conn.execute(insert(AddinSession).values(owner_id=owner, token_hash=token_hash(token),
                                                 created_at=now, expires_at=expires))
    return Pass(token=token, expires_at=expires)


def lookup(token: str, *, now: Optional[datetime] = None) -> Optional[Holder]:
    """Whose pass this is, while it lasts; None for an unknown, ended or malformed one."""
    if not looks_like_pass(token):
        return None
    now = now or utc_now()
    with connect() as conn:
        row = conn.execute(
            select(User.subject, AddinSession.expires_at)
            .join(User, User.id == AddinSession.owner_id)
            .where(AddinSession.token_hash == token_hash(token), AddinSession.expires_at > now)
        ).first()
    return Holder(subject=row[0], expires_at=row[1]) if row else None


def revoke(token: str) -> bool:
    """End a pass (signing out of the add-in). True when there was one."""
    if not looks_like_pass(token):
        return False
    with transaction() as conn:
        gone = conn.execute(delete(AddinSession).where(AddinSession.token_hash == token_hash(token)))
    return bool(gone.rowcount)


def sweep(*, now: Optional[datetime] = None) -> int:
    """Delete every ended pass (nightly job maintenance). How many went."""
    with transaction() as conn:
        gone = conn.execute(delete(AddinSession).where(AddinSession.expires_at <= (now or utc_now())))
    return int(gone.rowcount or 0)


__all__ = ["Holder", "LIFETIME", "MAX_PER_USER", "NoAccount", "PREFIX", "Pass", "create", "looks_like_pass",
           "lookup", "revoke", "sweep", "token_hash"]
