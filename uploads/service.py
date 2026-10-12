"""The steps of an upload, a download and a deletion, and the nightly sweep.

1. ``begin``: the name, kind and size are checked and room is reserved
   (``db/files.py``).
2. The bytes arrive at the API (``api/routers/files.py``), which streams them
   to a file in ``spool()`` and counts them as they come; more than was
   declared, and the upload is cut off and refused. ``arrived`` then puts
   the file in the store and queues its scan. **Only this server ever writes
   to the store**: no link to upload with is handed out, so nothing is stored
   that was not counted, and nothing can be changed after its scan.
3. ``scan_file``, in the job runner: the file is scanned in a process of its
   own (``uploads/scan.py`` ``scan_path``) and marked clean, or refused and
   its bytes deleted. The copy the upload left in the spool is used when it
   is still there (the same machine); otherwise it is fetched from the store.
4. ``link``: a signed download link, for a clean file and its owner only.

Until step 3 says clean, nothing can download the file. The scan takes the
one slot the simulations share (``jobs/kinds.py``), so its memory never adds
to a simulation's on the free server.

**A scan is never lost, and an outage never refuses a file.** A file whose
scan could not be queued (a full queue), could not read the store, or whose
job was lost stays ``scanning``; the next time its owner looks at it
(``wake_scans``) a scan that should have finished by now is queued again. A
second job for a file already scanned finds nothing to do.

``sweep`` runs nightly inside the ``job-maintenance`` task: scans that never
finished are refused, rows past their time are removed with their bytes, and
bytes no row accounts for (a failed delete, a deleted account) are removed.
"""
from __future__ import annotations

import logging
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from api.observability import log_event
from db import files as rows
from db.files import FileRecord
from uploads import rules
from uploads.scan import FAILED, Verdict, scan_path
from uploads.store import FileStore, NotStored, StorageError, TooLarge, get_store

SCAN_KIND = "upload.scan"
# Scans are queued under this owner, not the person's: they are the system's
# work, and must not use up the four runs a person may have waiting
SCAN_OWNER = "system:uploads"
MAX_SCANS_QUEUED = 30
# A file left in the spool this long belongs to no upload any more
SPOOL_KEEP_S = 3600


class NotSetUp(RuntimeError):
    """No file store is configured on this server."""


class NotReady(RuntimeError):
    """The file is not clean (yet), so it can't be downloaded."""


class NotExpected(RuntimeError):
    """The file is not waiting for its bytes (they arrived already)."""


def store() -> FileStore:
    found = get_store()
    if found is None:
        raise NotSetUp()
    return found


def enabled() -> bool:
    return get_store() is not None


def spool() -> Path:
    """Where an upload is written as it arrives, and read by its scan."""
    folder = Path(tempfile.gettempdir()) / "fse-uploads"
    folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    return folder


def spool_bytes() -> int:
    total = 0
    for path in spool().iterdir():
        try:
            total += path.stat().st_size
        except OSError:
            pass
    return total


def spooled(file_id: uuid.UUID) -> Path:
    return spool() / str(uuid.UUID(str(file_id)))


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def arriving(file_id: uuid.UUID) -> Path:
    """A spool file of this request's own for an upload on its way in. Two
    requests sending the same file never write the same bytes on disk: the
    one whose bytes are stored is the one whose copy is scanned."""
    return spool() / f"{uuid.UUID(str(file_id))}.{uuid.uuid4().hex}.part"


def discard(path: Path) -> None:
    """Drop what an interrupted or refused upload left in the spool."""
    _discard(path)


def _delete_bytes(files: FileStore, keys: list[str]) -> None:
    """Delete objects now if the store allows; what it keeps is the sweep's."""
    try:
        files.delete(keys)
    except StorageError as exc:
        log_event("upload_delete_deferred", logging.WARNING, error=type(exc).__name__)


# How long the answer of ``status`` is reused: it is public, and each fresh
# one is a call to the store
STATUS_CACHE_S = 300.0
_status: list = [0.0, None, None]


def status(*, now: Optional[float] = None) -> dict:
    """Whether uploads can work on this server, for ``/api/health/files``:
    which store, what **sort** of key it was given (never the key), and
    whether its bucket is there and private. Nothing about any file or
    person. Cached, and never queries the database."""
    now = time.time() if now is None else now
    files = get_store()
    if _status[1] is not None and _status[2] is files and now - _status[0] < STATUS_CACHE_S:
        return _status[1]
    if files is None:
        answer = {"status": "off", "store": "none"}
    else:
        answer = {"status": "ok", "store": files.name, "key": str(getattr(files, "key_kind", "none"))}
        try:
            if hasattr(files, "ensure_bucket"):
                files.ensure_bucket()
        except StorageError as exc:
            answer.update(status="error", reason=str(exc), detail=str(getattr(exc, "detail", ""))[:200])
    answer["checked_at"] = datetime.fromtimestamp(now, timezone.utc).isoformat().replace("+00:00", "Z")
    _status[:] = [now, answer, files]
    return answer


def begin(subject: str, name: object, size: int, *, now: Optional[datetime] = None) -> FileRecord:
    files = store()
    name = rules.clean_name(name)
    kind = rules.kind_of(name)
    size = rules.check_size(size)
    # Rows of the caller's past their time go first, with their bytes, so the
    # room they took is counted as free only once it really is
    gone = rows.delete_own_expired(subject, now=now)
    if gone:
        _delete_bytes(files, [r.key for r in gone if r.status != "refused"])
    return rows.reserve(subject, name, kind.id, size, now=now)


def expecting(subject: str, file_id: uuid.UUID) -> FileRecord:
    """The caller's file, if it is still waiting for its bytes."""
    store()
    record = rows.get_file(subject, file_id)
    if record.status != "pending":
        raise NotExpected()
    return record


def refuse_arrival(subject: str, file_id: uuid.UUID, reason: str) -> None:
    """The bytes that arrived were not what was declared."""
    try:
        rows.refuse(subject, file_id, reason)
    except rows.FileNotFound:
        pass


def arrived(subject: str, record: FileRecord, path: Path, size: int, *,
            now: Optional[datetime] = None) -> FileRecord:
    """The upload is at ``path``, ``size`` bytes of it: store it and queue
    its scan. A name is stored once, so of two requests sending the same
    file one is stored and the other fails here. If the store can't take
    it, the room stays reserved and the caller may send it again."""
    files = store()
    try:
        files.put(record.key, path, rules.KINDS[record.kind].content_type)
        try:
            record, changed = rows.mark_uploaded(subject, record.id, size, now=now)
        except BaseException:
            # The row is gone (deleted while the bytes were on their way) or
            # could not be written: bytes with no row are in nobody's quota,
            # so they do not stay. This request stored them, so they are its
            # to remove; a name is stored once, so they are nobody else's
            _delete_bytes(files, [record.key])
            raise
        if changed and spool_bytes() + size <= rules.SPOOL_MAX_BYTES:
            path.replace(spooled(record.id))      # the scan reads exactly what was stored
    finally:
        _discard(path)
    if changed:
        _queue_scan(record)
    return record


def _queue_scan(record: FileRecord) -> None:
    """Queue the file's scan. A full queue is not the caller's problem: the
    file waits as ``scanning`` and is queued the next time its owner looks."""
    from jobs import config
    from jobs.queue import QueueFull

    try:
        config.get_queue().submit(SCAN_OWNER, SCAN_KIND, {"file_id": str(record.id)},
                                  max_active_per_owner=MAX_SCANS_QUEUED, max_queued=MAX_SCANS_QUEUED + 40)
    except QueueFull:
        rows.not_queued(record.id)
        return
    config.wake_runner()


def _verdict(files: FileStore, record: FileRecord) -> Optional[Verdict]:
    """The scanner's verdict on the file, or None when the store could not be
    reached to read it."""
    path = spooled(record.id)
    try:
        if not path.is_file() or path.stat().st_size != record.size:
            try:
                files.fetch(record.key, path, rules.MAX_FILE_BYTES)
            except TooLarge:
                return Verdict(False, "too_large")
            except NotStored:
                return FAILED
            except StorageError as exc:
                log_event("upload_scan_deferred", logging.WARNING, error=type(exc).__name__)
                return None
        return scan_path(path, record.kind)
    finally:
        _discard(path)


def scan_file(file_id: uuid.UUID, *, now: Optional[datetime] = None) -> dict:
    """Scan one uploaded file and record the verdict. What the job answers."""
    record = rows.for_scan(file_id)
    if record is None:
        _discard(spooled(file_id))
        return {"status": "gone", "reason": None}
    files = store()
    t0 = time.perf_counter()
    verdict = _verdict(files, record)
    if verdict is None:
        # The store can't be reached just now. That says nothing about the
        # file, so it is neither passed nor refused: it is scanned again
        rows.not_queued(file_id)
        return {"status": "retry", "reason": None}
    recorded = rows.finish_scan(file_id, clean=verdict.clean, reason=verdict.reason, now=now)
    if not verdict.clean or (not recorded and not rows.exists(file_id)):
        # Refused, or deleted by its owner while it was being scanned
        _delete_bytes(files, [record.key])
    log_event("upload_scanned", kind=record.kind, clean=verdict.clean, reason=verdict.reason,
              megabytes=round(record.size / rules.MB, 1), duration_ms=round((time.perf_counter() - t0) * 1000))
    return {"status": "clean" if verdict.clean else "refused", "reason": verdict.reason}


def link(subject: str, file_id: uuid.UUID, *, now: Optional[datetime] = None) -> str:
    files = store()
    record = rows.get_file(subject, file_id, now=now)
    if record.status != "clean":
        raise NotReady()
    return files.download_url(record.key, record.name, rules.LINK_S)


def remove(subject: str, file_id: uuid.UUID, *, now: Optional[datetime] = None) -> None:
    """Delete the caller's file: the row first, so it is gone for them
    whatever the store does; bytes a failed delete leaves are the sweep's."""
    record = rows.delete_file(subject, file_id, now=now)
    _discard(spooled(file_id))
    found = get_store()
    if found is not None:
        _delete_bytes(found, [record.key])


def wake_scans(subject: str, records: list[FileRecord]) -> None:
    """After a restart nothing runs until someone asks, and a scan can be
    lost or put off: looking at a file that is still being checked wakes the
    runner and queues again any scan that should have finished by now."""
    if not any(r.status == "scanning" for r in records):
        return
    from jobs import config

    for overdue in rows.claim_rescans(subject):
        _queue_scan(overdue)
    config.wake_runner()


def clear_spool(*, now: Optional[float] = None) -> int:
    """Remove spool files no upload or scan is using any more."""
    cutoff, removed = (time.time() if now is None else now) - SPOOL_KEEP_S, 0
    for path in spool().iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            pass
    return removed


def sweep(*, now: Optional[datetime] = None) -> dict:
    """The nightly clean-up. A flat summary of counts (jobs/scheduled.py)."""
    files = get_store()
    if files is None:
        return {"files_store": "none"}
    stale = rows.refuse_stale_scans(now=now)
    expired = rows.delete_expired(now=now)
    files.delete([r.key for r in stale] + [r.key for r in expired if r.status != "refused"])
    # Objects first, rows second: a file stored in between has its row by
    # the time the rows are read, so it is never taken for an orphan
    stored = files.objects()
    known = rows.stored_keys()
    orphans = [o.key for o in stored if o.key not in known]
    files.delete(orphans)
    clear_spool()
    held = rows.totals()
    budget = rules.total_budget_bytes()
    return {"files_store": files.name, "files_scans_abandoned": len(stale),
            "files_expired": len(expired), "files_orphans_deleted": len(orphans),
            "files_held": held["files"], "files_bytes": held["bytes"], "files_budget_bytes": budget,
            "files_budget_used": round(held["bytes"] / budget, 4)}
