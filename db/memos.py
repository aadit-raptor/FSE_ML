"""Investment memos of saved deal versions (PLAN.md 6.3).

One memo per version of a deal: the version's model run as the memo reads it
(ml/memo.py ``trim``) and the linking sentences last written. Every function
matches the caller's subject in the statement that reads or writes, so another
account's deal is ``DealNotFound`` (404), as a missing one is. Each change
writes one audit entry on the deal (``memo_drafted``, ``memo_written``,
``memo_deleted``, with the version's number), none when nothing changed.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import and_, delete, func, insert, select, update

from db import audit
from db.deals import DealNotFound, VersionNotFound, _lock_deal, _owned
from db.engine import connect, transaction
from db.models import Deal, DealMemo, DealVersion, utc_now

# A person's memos at most: a run is a few kilobytes, so this is about storage hygiene, not the free 0.5 GB
MAX_MEMOS = 40


class MemoNotFound(LookupError):
    pass


class MemosFull(Exception):
    pass


@dataclass(frozen=True)
class MemoRecord:
    version: int
    created_at: datetime
    updated_at: datetime
    engine_version: Optional[str] = None
    run: Optional[dict] = None
    sentences: Optional[dict] = None


def _deal_owner(conn, subject: str, deal_id: uuid.UUID) -> int:
    owner = conn.execute(select(Deal.owner_id).where(_owned(subject, deal_id))).scalar()
    if owner is None:
        raise DealNotFound(str(deal_id))
    return owner


def save_memo(subject: str, deal_id: uuid.UUID, version: int, run: dict, *, now: Optional[datetime] = None) -> bool:
    """Keep the version's memo with this run (its sentences cleared: they were written for the old run). True when
    it changed. ``VersionNotFound`` for a version the deal doesn't have; ``MemosFull`` at ``MAX_MEMOS``."""
    now = now or utc_now()
    with transaction() as conn:
        deal = _lock_deal(conn, subject, deal_id)       # one draft of a deal at a time
        owner = conn.execute(select(Deal.owner_id).where(Deal.id == deal.id)).scalar_one()
        if conn.execute(select(DealVersion.number).where(
                DealVersion.deal_id == deal_id, DealVersion.number == version)).first() is None:
            raise VersionNotFound(str(version))
        row = conn.execute(select(DealMemo.id, DealMemo.run, DealMemo.sentences).where(
            DealMemo.deal_id == deal_id, DealMemo.version == version)).first()
        if row is not None and row.run == run:
            return False
        if row is None:
            held = conn.execute(select(func.count()).select_from(DealMemo).where(DealMemo.owner_id == owner)).scalar()
            if held >= MAX_MEMOS:
                raise MemosFull(str(MAX_MEMOS))
            conn.execute(insert(DealMemo).values(id=uuid.uuid4(), deal_id=deal_id, owner_id=owner, version=version,
                                                 run=run, sentences=None, created_at=now, updated_at=now))
        else:
            conn.execute(update(DealMemo).where(DealMemo.id == row.id).values(run=run, sentences=None,
                                                                              updated_at=now))
        audit.record(conn, owner, "memo_drafted", deal_id=deal_id, now=now, detail={"version": version})
        return True


def has_room(subject: str, deal_id: uuid.UUID, version: int) -> bool:
    """Whether drafting this version's memo fits: it has one already (drafting again replaces it), or the person
    holds fewer than ``MAX_MEMOS``."""
    with connect() as conn:
        owner = _deal_owner(conn, subject, deal_id)
        if conn.execute(select(DealMemo.id).where(DealMemo.deal_id == deal_id, DealMemo.version == version,
                                                  DealMemo.owner_id == owner)).first() is not None:
            return True
        return conn.execute(select(func.count()).select_from(DealMemo).where(DealMemo.owner_id == owner)).scalar() \
            < MAX_MEMOS


def get_memo(subject: str, deal_id: uuid.UUID, version: int) -> MemoRecord:
    with connect() as conn:
        _deal_owner(conn, subject, deal_id)
        row = conn.execute(select(DealMemo.version, DealMemo.created_at, DealMemo.updated_at, DealMemo.run,
                                  DealMemo.sentences)
                           .select_from(Deal).join(DealMemo, DealMemo.deal_id == Deal.id)
                           .where(_owned(subject, deal_id), DealMemo.version == version)).first()
    if row is None:
        raise MemoNotFound(str(version))
    return MemoRecord(version=row.version, created_at=row.created_at, updated_at=row.updated_at,
                      engine_version=((row.run or {}).get("model") or {}).get("engine_version"),
                      run=row.run, sentences=row.sentences)


def list_memos(subject: str, deal_id: uuid.UUID) -> list[MemoRecord]:
    """The deal's memos, newest version first, without their runs."""
    engine = DealMemo.run["model"]["engine_version"].astext
    with connect() as conn:
        rows = conn.execute(select(Deal.id, DealMemo.version, DealMemo.created_at, DealMemo.updated_at,
                                   engine.label("engine_version"))
                            .select_from(Deal).outerjoin(DealMemo, DealMemo.deal_id == Deal.id)
                            .where(_owned(subject, deal_id))
                            .order_by(DealMemo.version.desc())).all()
    if not rows:
        raise DealNotFound(str(deal_id))
    return [MemoRecord(version=r.version, created_at=r.created_at, updated_at=r.updated_at,
                       engine_version=r.engine_version) for r in rows if r.version is not None]


def save_sentences(subject: str, deal_id: uuid.UUID, version: int, sentences: dict,
                   *, now: Optional[datetime] = None) -> bool:
    """Keep the memo's linking sentences; True when they changed."""
    now = now or utc_now()
    with transaction() as conn:
        owner = _deal_owner(conn, subject, deal_id)
        row = conn.execute(select(DealMemo.id, DealMemo.sentences).where(
            and_(DealMemo.deal_id == deal_id, DealMemo.version == version, DealMemo.owner_id == owner))
            .with_for_update()).first()
        if row is None:
            raise MemoNotFound(str(version))
        if row.sentences == sentences:
            return False
        conn.execute(update(DealMemo).where(DealMemo.id == row.id).values(sentences=sentences, updated_at=now))
        audit.record(conn, owner, "memo_written", deal_id=deal_id, now=now, detail={"version": version})
        return True


def delete_memo(subject: str, deal_id: uuid.UUID, version: int, *, now: Optional[datetime] = None) -> None:
    now = now or utc_now()
    with transaction() as conn:
        owner = _deal_owner(conn, subject, deal_id)
        gone = conn.execute(delete(DealMemo).where(DealMemo.deal_id == deal_id, DealMemo.version == version,
                                                   DealMemo.owner_id == owner)).rowcount
        if not gone:
            raise MemoNotFound(str(version))
        audit.record(conn, owner, "memo_deleted", deal_id=deal_id, now=now, detail={"version": version})


__all__ = ["DealNotFound", "MAX_MEMOS", "MemoNotFound", "MemoRecord", "MemosFull", "VersionNotFound",
           "delete_memo", "get_memo", "has_room", "list_memos", "save_memo", "save_sentences"]
