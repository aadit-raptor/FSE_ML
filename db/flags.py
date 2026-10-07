"""Site-wide switches (PLAN.md 4.5): read by anyone, changed by an administrator.

One row per switch in ``app_flags``; a switch with no row has its default. A
change writes the row and one ``library_switched`` audit entry in the same
transaction; switching to the value already stored changes nothing and
writes nothing (db/audit.py's rule).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from db import audit
from db.engine import connect, transaction
from db.models import AppFlag, User, utc_now

# The switches there are, and the audit action a change of each records
FLAGS = {"library": "library_switched"}


class NoAccount(LookupError):
    """The caller has no account row yet (they haven't finished sign-up)."""


@dataclass(frozen=True)
class Flag:
    name: str
    enabled: bool
    updated_at: Optional[datetime]


def get(name: str) -> Optional[Flag]:
    """The stored switch, or None when nobody has set it."""
    if name not in FLAGS:
        raise ValueError(f"unknown switch: {name}")
    with connect() as conn:
        row = conn.execute(select(AppFlag.enabled, AppFlag.updated_at).where(AppFlag.name == name)).first()
    return Flag(name, row.enabled, row.updated_at) if row else None


def put(subject: str, name: str, enabled: bool, *, default: bool,
        now: Optional[datetime] = None) -> Flag:
    """Set the switch; a change is one audit entry for the caller.

    Setting the value already in force writes nothing, not even a row for a
    switch still at its default. The write only happens when the stored value
    differs (``WHERE ... IS DISTINCT FROM``), so of two administrators switching
    at once only the one who changed it records an entry."""
    if name not in FLAGS:
        raise ValueError(f"unknown switch: {name}")
    now = now or utc_now()
    with transaction() as conn:
        user_id = conn.execute(select(User.id).where(User.subject == subject)).scalar()
        if user_id is None:
            raise NoAccount(subject)
        held = conn.execute(select(AppFlag.enabled, AppFlag.updated_at).where(AppFlag.name == name)).first()
        if (held.enabled if held else default) == enabled:
            return Flag(name, enabled, held.updated_at if held else None)
        stmt = insert(AppFlag).values(name=name, enabled=enabled, updated_at=now, updated_by=user_id)
        changed = conn.execute(stmt.on_conflict_do_update(
            index_elements=[AppFlag.name],
            set_={"enabled": enabled, "updated_at": now, "updated_by": user_id},
            where=AppFlag.enabled.is_distinct_from(stmt.excluded.enabled),
        ).returning(AppFlag.name)).first()
        if changed is None:
            # Someone else switched it to this value first
            row = conn.execute(select(AppFlag.updated_at).where(AppFlag.name == name)).first()
            return Flag(name, enabled, row.updated_at if row else None)
        audit.record(conn, user_id, FLAGS[name], now=now, detail={"enabled": enabled})
    return Flag(name, enabled, now)
