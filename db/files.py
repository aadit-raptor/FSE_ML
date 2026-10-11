"""Uploaded files: whose they are and where each stands (PLAN.md 6.1).

**Whose file.** As with deals (db/deals.py), every function a caller reaches
takes the caller's subject, from the verified token, and matches it in the
same statement that reads or writes the row. Someone else's file is
indistinguishable from one that doesn't exist: both raise ``FileNotFound``.
The functions without a subject (``for_scan``, ``finish_scan``, the sweep's)
are the scanner's and the nightly task's, never an endpoint's.

**Room.** ``reserve`` counts what the caller holds and what everyone holds
and inserts the row in one transaction, with the caller's account row
locked, so two uploads started together can't both take the last of a quota.
A reservation counts from the moment it is made, and a row counts until it
is removed, because its bytes are in the store until then.

**Audit history.** An upload that arrives and a deletion by its owner each
write one entry in the same transaction. Entries hold no name and no size.

**Never log a file's name**, as with a deal's.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import and_, delete, func, insert, select, update

from db import audit
from db.engine import connect, transaction
from db.models import StoredFile, User, utc_now
from uploads import rules

# What counts against the quotas: everything but a refused file, whose bytes are gone
HELD = ("pending", "scanning", "clean")


class NoAccount(LookupError):
    """The caller hasn't finished their account, so there is no one to own a file."""


class FileNotFound(LookupError):
    """No such file for this caller (it may exist and belong to someone else)."""


@dataclass(frozen=True)
class FileRecord:
    id: uuid.UUID
    owner_id: int
    name: str
    kind: str
    size: int
    status: str
    reason: Optional[str]
    created_at: datetime
    expires_at: datetime

    @property
    def key(self) -> str:
        from uploads.store import object_key

        return object_key(self.owner_id, self.id)


_COLUMNS = (StoredFile.id, StoredFile.owner_id, StoredFile.name, StoredFile.kind, StoredFile.size,
            StoredFile.status, StoredFile.reason, StoredFile.created_at, StoredFile.expires_at)


def _record(row) -> FileRecord:
    return FileRecord(id=row.id, owner_id=row.owner_id, name=row.name, kind=row.kind, size=row.size,
                      status=row.status, reason=row.reason, created_at=row.created_at,
                      expires_at=row.expires_at)


def _owner_id(subject: str):
    return select(User.id).where(User.subject == subject).scalar_subquery()


def _owned(subject: str, file_id: uuid.UUID):
    return and_(StoredFile.id == file_id, StoredFile.owner_id == _owner_id(subject))


def _held(conn, owner: Optional[int] = None) -> tuple[int, int]:
    """Bytes and files held, by one owner or by everyone. A row counts until
    it is removed, which is when its bytes are: one past its time still
    occupies the store until the sweep (or its owner's next upload) deletes it."""
    query = select(func.coalesce(func.sum(StoredFile.size), 0), func.count()).where(
        StoredFile.status.in_(HELD))
    if owner is not None:
        query = query.where(StoredFile.owner_id == owner)
    used, files = conn.execute(query).one()
    return int(used), int(files)


# ---------------------------------------------------------------------------
# The caller's own files
# ---------------------------------------------------------------------------
def reserve(subject: str, name: str, kind: str, size: int, *, now: Optional[datetime] = None,
            budget: Optional[int] = None) -> FileRecord:
    """Room for an upload of ``size`` bytes, or ``rules.InvalidUpload`` when
    the caller's quota or the store's budget has none."""
    now = now or utc_now()
    with transaction() as conn:
        owner = conn.execute(select(User.id).where(User.subject == subject).with_for_update()).scalar()
        if owner is None:
            raise NoAccount(subject)
        user_bytes, user_files = _held(conn, owner)
        total_bytes, _ = _held(conn)
        in_flight = conn.execute(select(func.count()).where(
            StoredFile.owner_id == owner, StoredFile.status.in_(("pending", "scanning")))).scalar_one()
        pending_bytes = conn.execute(select(func.coalesce(func.sum(StoredFile.size), 0)).where(
            StoredFile.status == "pending")).scalar_one()
        rules.check_room(size, user_bytes=user_bytes, user_files=user_files, total_bytes=total_bytes,
                         budget=budget, in_flight=in_flight, pending_bytes=int(pending_bytes))
        row = conn.execute(insert(StoredFile).values(
            id=uuid.uuid4(), owner_id=owner, name=name, kind=kind, size=size, status="pending",
            created_at=now, expires_at=now + timedelta(seconds=rules.PENDING_S),
        ).returning(*_COLUMNS)).one()
    return _record(row)


def delete_own_expired(subject: str, *, now: Optional[datetime] = None) -> list[FileRecord]:
    """Remove the caller's rows past their time, so the room is theirs again
    without waiting for the nightly sweep; the caller deletes the bytes."""
    now = now or utc_now()
    with transaction() as conn:
        found = conn.execute(delete(StoredFile).where(
            StoredFile.owner_id == _owner_id(subject), StoredFile.expires_at <= now,
        ).returning(*_COLUMNS)).all()
    return [_record(r) for r in found]


def list_files(subject: str, *, now: Optional[datetime] = None) -> tuple[list[FileRecord], dict]:
    """The caller's files, newest first, and what they hold against their limits."""
    now = now or utc_now()
    with connect() as conn:
        rows = conn.execute(select(*_COLUMNS).where(StoredFile.owner_id == _owner_id(subject))
                            .order_by(StoredFile.created_at.desc(), StoredFile.id)).all()
    records = [_record(r) for r in rows]
    held = [r for r in records if r.status in HELD]
    # A row past its time is no longer the caller's to see, but it counts
    # until its bytes are gone
    return [r for r in records if r.expires_at > now], rules.usage(sum(r.size for r in held), len(held))


def get_file(subject: str, file_id: uuid.UUID, *, now: Optional[datetime] = None) -> FileRecord:
    now = now or utc_now()
    with connect() as conn:
        row = conn.execute(select(*_COLUMNS).where(
            _owned(subject, file_id), StoredFile.expires_at > now)).first()
    if row is None:
        raise FileNotFound(str(file_id))
    return _record(row)


def mark_uploaded(subject: str, file_id: uuid.UUID, size: int, *,
                  now: Optional[datetime] = None) -> tuple[FileRecord, bool]:
    """The bytes arrived: ``pending`` becomes ``scanning``, with the size that
    really arrived. The row, and whether this call made the change: only then
    does the caller queue a scan, and only then is the upload recorded, so a
    repeated call is one upload and one scan."""
    now = now or utc_now()
    with transaction() as conn:
        row = conn.execute(update(StoredFile).where(
            _owned(subject, file_id), StoredFile.status == "pending", StoredFile.expires_at > now,
        ).values(status="scanning", size=size, uploaded_at=now, queued_at=now,
                 expires_at=now + timedelta(days=rules.RETENTION_DAYS)).returning(*_COLUMNS)).first()
        if row is not None:
            audit.record(conn, row.owner_id, "file_uploaded", now=now)
            return _record(row), True
        row = conn.execute(select(*_COLUMNS).where(
            _owned(subject, file_id), StoredFile.expires_at > now)).first()
    if row is None:
        raise FileNotFound(str(file_id))
    return _record(row), False


def claim_rescans(subject: str, *, now: Optional[datetime] = None) -> list[FileRecord]:
    """The caller's files whose scan should have finished by now, each marked
    as queued again in the same statement, so two requests arriving together
    queue one scan between them."""
    now = now or utc_now()
    overdue = now - timedelta(seconds=rules.RESCAN_AFTER_S)
    with transaction() as conn:
        found = conn.execute(update(StoredFile).where(
            StoredFile.owner_id == _owner_id(subject), StoredFile.status == "scanning",
            (StoredFile.queued_at.is_(None)) | (StoredFile.queued_at <= overdue),
        ).values(queued_at=now).returning(*_COLUMNS)).all()
    return [_record(r) for r in found]


def refuse(subject: str, file_id: uuid.UUID, reason: str, *, now: Optional[datetime] = None) -> FileRecord:
    """Refuse the caller's upload as it arrives (more bytes than declared, or none)."""
    now = now or utc_now()
    with transaction() as conn:
        row = conn.execute(update(StoredFile).where(
            _owned(subject, file_id), StoredFile.status == "pending",
        ).values(**_refused(reason, now)).returning(*_COLUMNS)).first()
    if row is None:
        raise FileNotFound(str(file_id))
    return _record(row)


def delete_file(subject: str, file_id: uuid.UUID, *, now: Optional[datetime] = None) -> FileRecord:
    """Delete the caller's file's row; the caller then deletes the bytes."""
    with transaction() as conn:
        row = conn.execute(delete(StoredFile).where(_owned(subject, file_id))
                           .returning(*_COLUMNS)).first()
        if row is None:
            raise FileNotFound(str(file_id))
        audit.record(conn, row.owner_id, "file_deleted", now=now)
    return _record(row)


# ---------------------------------------------------------------------------
# The scanner's and the sweep's
# ---------------------------------------------------------------------------
def _refused(reason: str, now: datetime) -> dict:
    if reason not in rules.REASONS:
        raise ValueError(f"unknown reason: {reason}")
    return {"status": "refused", "reason": reason, "scanned_at": now,
            "expires_at": now + timedelta(days=rules.REFUSED_KEEP_DAYS)}


def for_scan(file_id: uuid.UUID) -> Optional[FileRecord]:
    """The file a scan job was queued for, if it still waits for one."""
    with connect() as conn:
        row = conn.execute(select(*_COLUMNS).where(
            StoredFile.id == file_id, StoredFile.status == "scanning")).first()
    return None if row is None else _record(row)


def finish_scan(file_id: uuid.UUID, *, clean: bool, reason: Optional[str] = None,
                size: Optional[int] = None, now: Optional[datetime] = None) -> bool:
    """Record the scanner's verdict; False when the file is no longer waiting
    for one (deleted meanwhile, or another runner got there first)."""
    now = now or utc_now()
    values = {"status": "clean", "scanned_at": now} if clean else _refused(reason or "scan_failed", now)
    if clean and size:
        values["size"] = size
    with transaction() as conn:
        done = conn.execute(update(StoredFile).where(
            StoredFile.id == file_id, StoredFile.status == "scanning").values(**values)).rowcount
    return bool(done)


def not_queued(file_id: uuid.UUID) -> None:
    """The scan could not be queued (the queue was full), or could not read
    the file: the next time its owner looks, it is queued again."""
    with transaction() as conn:
        conn.execute(update(StoredFile).where(
            StoredFile.id == file_id, StoredFile.status == "scanning").values(queued_at=None))


def exists(file_id: uuid.UUID) -> bool:
    with connect() as conn:
        return conn.execute(select(StoredFile.id).where(StoredFile.id == file_id)).first() is not None


def refuse_stale_scans(*, now: Optional[datetime] = None) -> list[FileRecord]:
    """Files whose scan never finished: refused, never trusted."""
    now = now or utc_now()
    cutoff = now - timedelta(seconds=rules.SCAN_STALE_S)
    with transaction() as conn:
        rows = conn.execute(update(StoredFile).where(
            StoredFile.status == "scanning", StoredFile.uploaded_at < cutoff,
        ).values(**_refused("scan_failed", now)).returning(*_COLUMNS)).all()
    return [_record(r) for r in rows]


def delete_expired(*, now: Optional[datetime] = None, limit: int = 1000) -> list[FileRecord]:
    """Remove rows past ``expires_at``; the caller deletes their bytes."""
    now = now or utc_now()
    due = select(StoredFile.id).where(StoredFile.expires_at <= now).limit(limit)
    with transaction() as conn:
        rows = conn.execute(delete(StoredFile).where(StoredFile.id.in_(due)).returning(*_COLUMNS)).all()
    return [_record(r) for r in rows]


def stored_keys() -> set[str]:
    """Every object name a row accounts for."""
    from uploads.store import object_key

    with connect() as conn:
        rows = conn.execute(select(StoredFile.owner_id, StoredFile.id)).all()
    return {object_key(r.owner_id, r.id) for r in rows}


def totals() -> dict:
    with connect() as conn:
        used, files = _held(conn)
    return {"bytes": used, "files": files}
