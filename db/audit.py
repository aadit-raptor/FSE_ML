"""Audit history: what was done to each deal and to the account's settings (PLAN.md 3.3).

**One entry per action**, written by ``record`` in the same transaction as the
action itself, so an action that fails or is rolled back leaves no entry and
one that succeeds always has one. An action that changes nothing (an autosave
of the same content, a rename to the same name) is not an action and writes
nothing. db/deals.py calls ``record`` for every deal action; the export router
calls ``db.deals.record_export``.

**Append-only.** Nothing here updates or deletes an entry, the API has no
endpoint that could, and its database role may only insert and read them
(migration 0010). ``compact`` is the one exception: the schema owner's
``audit_compact`` merges old ``edited`` entries one per deal and day.

**What an entry holds.** The action, when, who, which deal, and ``detail``:
the names of the fields an edit changed (``settings.<key>`` for a setting),
a version number, the source of a duplicate, or the kind of export. Never a
figure, a deal name or a version label: those stay in the deal, so deleting
the deal deletes them, while its entries remain as a record that it existed.

Staying small (free 0.5 GB): an entry is under about 120 bytes, an edit's
list of fields a few hundred at most; autosave writes one per save that
changes the deal, and after ``COMPACT_AFTER_DAYS`` a day of them is one row.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional

from sqlalchemy import insert, select, text
from sqlalchemy.engine import Connection

from db.engine import connect, transaction
from db.models import AUDIT_ACTIONS, AuditEvent, Deal, User, utc_now

# Edits older than this are merged one per deal and day
COMPACT_AFTER_DAYS = 7
# Entries one history request answers at most
MAX_ENTRIES = 500
EXPORTS = ("workbook", "simulation_sample")
# Keys ``detail`` may hold, and nothing else
DETAIL_KEYS = ("fields", "version", "source_deal", "export")


@dataclass(frozen=True)
class Entry:
    id: int
    action: str
    deal_id: Optional[uuid.UUID]
    detail: dict
    count: int
    occurred_at: datetime
    last_at: Optional[datetime]
    deal_name: Optional[str] = None


def changed_fields(old_inputs: Mapping, new_inputs: Mapping,
                   old_settings: Optional[Mapping] = None,
                   new_settings: Optional[Mapping] = None) -> list[str]:
    """The names of what differs, sorted: input fields as they are, settings
    as ``settings.<key>``. A key present on one side only (stored only when
    set) counts as changed."""
    def differ(old: Mapping, new: Mapping) -> set:
        return {k for k in set(old) | set(new) if old.get(k, _MISSING) != new.get(k, _MISSING)}
    names = differ(old_inputs or {}, new_inputs or {})
    names |= {f"settings.{k}" for k in differ(old_settings or {}, new_settings or {})}
    return sorted(names)


_MISSING = object()


def record(conn: Connection, user: Any, action: str, *, deal_id: Optional[uuid.UUID] = None,
           detail: Optional[Mapping] = None, now: Optional[datetime] = None) -> None:
    """Add one entry inside the caller's transaction. ``user`` is a user id or
    a SQL expression giving one (the caller's subject's id)."""
    if action not in AUDIT_ACTIONS:
        raise ValueError(f"unknown audit action: {action}")
    detail = dict(detail or {})
    unknown = set(detail) - set(DETAIL_KEYS)
    if unknown:
        raise ValueError(f"audit detail can't hold: {sorted(unknown)}")
    if "export" in detail and detail["export"] not in EXPORTS:
        raise ValueError(f"unknown export: {detail['export']}")
    conn.execute(insert(AuditEvent).values(
        user_id=user, deal_id=deal_id, action=action, detail=detail,
        occurred_at=now or utc_now()))


def subject_id(subject: str):
    """The caller's user id, as a subquery for ``record``."""
    return select(User.id).where(User.subject == subject).scalar_subquery()


_ENTRY_COLUMNS = (AuditEvent.id, AuditEvent.action, AuditEvent.deal_id, AuditEvent.detail,
                  AuditEvent.count, AuditEvent.occurred_at, AuditEvent.last_at)


def entry(row, *, named: bool = False) -> Entry:
    return Entry(id=row.id, action=row.action, deal_id=row.deal_id, detail=row.detail or {},
                 count=row.count, occurred_at=row.occurred_at, last_at=row.last_at,
                 deal_name=row.deal_name if named else None)


def newest_first(query, limit: int):
    return query.order_by(AuditEvent.occurred_at.desc(), AuditEvent.id.desc()) \
        .limit(max(1, min(limit, MAX_ENTRIES)))


def account_history(subject: str, *, limit: int = 200) -> list[Entry]:
    """Everything the caller did, newest first, with each deal's current name
    (none once the deal is deleted)."""
    query = (select(*_ENTRY_COLUMNS, Deal.name.label("deal_name"))
             .select_from(AuditEvent).outerjoin(Deal, Deal.id == AuditEvent.deal_id)
             .where(AuditEvent.user_id == subject_id(subject)))
    with connect() as conn:
        found = conn.execute(newest_first(query, limit)).all()
    return [entry(r, named=True) for r in found]


def compact(now: Optional[datetime] = None, *, after_days: int = COMPACT_AFTER_DAYS) -> int:
    """Merge ``edited`` entries older than ``after_days`` one per deal and UTC
    day; returns the rows removed. Runs nightly (jobs/scheduled.py)."""
    cutoff = (now or utc_now()) - timedelta(days=after_days)
    with transaction() as conn:
        return conn.execute(text("SELECT audit_compact(:cutoff)"), {"cutoff": cutoff}).scalar_one()
