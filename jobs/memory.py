"""A job queue held in this process's memory (local runs and tests).

Same behaviour as ``jobs.database.DatabaseQueue``, minus surviving a restart:
the API uses it only when there is no database (``jobs.config``).
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from db.models import utc_now
from jobs import queue as q
from jobs.queue import JobNotFound, JobRecord, QueueFull


@dataclass
class _Row:
    id: uuid.UUID
    owner: str
    kind: str
    payload: Optional[dict]
    created_at: datetime
    status: str = q.QUEUED
    progress: float = 0.0
    stage: Optional[str] = None
    attempts: int = 0
    cancel_requested: bool = False
    worker: Optional[str] = None
    started_at: Optional[datetime] = None
    heartbeat_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    error_status: Optional[int] = None
    result: Any = None
    result_expired: bool = False
    seq: int = field(default=0)


class MemoryQueue:
    name = "memory"

    def __init__(self, clock=utc_now):
        self._rows: dict[uuid.UUID, _Row] = {}
        self._lock = threading.Lock()
        self._seq = 0
        self._clock = clock

    # -- helpers ------------------------------------------------------------
    def _record(self, row: _Row, *, with_payload=False, with_result=False) -> JobRecord:
        ahead = None
        if row.status == q.QUEUED:
            ahead = sum(1 for r in self._rows.values() if r.status == q.QUEUED and r.seq < row.seq)
        return JobRecord(
            id=row.id, owner=row.owner, kind=row.kind, status=row.status, progress=row.progress,
            stage=row.stage, attempts=row.attempts, created_at=row.created_at,
            started_at=row.started_at, finished_at=row.finished_at, error=row.error,
            error_status=row.error_status, cancel_requested=row.cancel_requested, ahead=ahead,
            payload=_copy(row.payload) if with_payload else None,
            result=_copy(row.result) if with_result else None, result_expired=row.result_expired)

    def _owned(self, owner: str, job_id: uuid.UUID) -> _Row:
        row = self._rows.get(job_id)
        if row is None or row.owner != owner:
            raise JobNotFound(str(job_id))
        return row

    # -- owner side ---------------------------------------------------------
    def submit(self, owner, kind, payload, *, max_active_per_owner=q.MAX_ACTIVE_PER_OWNER,
               max_queued=q.MAX_QUEUED) -> JobRecord:
        with self._lock:
            active = [r for r in self._rows.values() if r.status in q.ACTIVE]
            if max_active_per_owner is not None and \
                    sum(1 for r in active if r.owner == owner) >= max_active_per_owner:
                raise QueueFull(q.owner_message(max_active_per_owner), owner_limit=True)
            if len(active) >= max_queued:
                raise QueueFull(q.BUSY_MESSAGE, owner_limit=False)
            self._seq += 1
            row = _Row(id=uuid.uuid4(), owner=owner, kind=kind, payload=_copy(payload),
                       created_at=self._clock(), seq=self._seq)
            self._rows[row.id] = row
            return self._record(row)

    def get(self, owner, job_id, *, with_result=False) -> JobRecord:
        with self._lock:
            return self._record(self._owned(owner, job_id), with_result=with_result)

    def list(self, owner, *, limit=20) -> list[JobRecord]:
        with self._lock:
            rows = sorted((r for r in self._rows.values() if r.owner == owner),
                          key=lambda r: r.seq, reverse=True)[:limit]
            return [self._record(r) for r in rows]

    def cancel(self, owner, job_id) -> JobRecord:
        with self._lock:
            row = self._owned(owner, job_id)
            if row.status == q.QUEUED:
                row.status, row.finished_at, row.payload = q.CANCELLED, self._clock(), None
                row.error = q.CANCELLED_MESSAGE
            elif row.status == q.RUNNING:
                row.cancel_requested = True
            return self._record(row)

    # -- runner side --------------------------------------------------------
    def claim(self, worker) -> Optional[JobRecord]:
        with self._lock:
            queued = [r for r in self._rows.values() if r.status == q.QUEUED]
            if not queued:
                return None
            row = min(queued, key=lambda r: r.seq)
            now = self._clock()
            row.status, row.worker, row.attempts = q.RUNNING, worker, row.attempts + 1
            row.started_at = row.heartbeat_at = now
            row.progress, row.stage = 0.0, None
            return self._record(row, with_payload=True)

    def report(self, job_id, worker, *, progress=None, stage=None) -> bool:
        with self._lock:
            row = self._rows.get(job_id)
            if row is None or row.status != q.RUNNING or row.worker != worker:
                return False
            row.heartbeat_at = self._clock()
            if progress is not None:
                row.progress = min(1.0, max(0.0, float(progress)))
            if stage is not None:
                row.stage = stage[:80]
            return not row.cancel_requested

    def finish(self, job_id, worker, *, result=None, error=None, error_status=None) -> Optional[str]:
        stored, size = (None, 0) if error is not None else q.encoded(result)
        with self._lock:
            row = self._rows.get(job_id)
            if row is None or row.status != q.RUNNING or row.worker != worker:
                return None
            row.finished_at, row.payload, row.worker = self._clock(), None, None
            if row.cancel_requested:
                row.status, row.error = q.CANCELLED, q.CANCELLED_MESSAGE
            elif error is None and size > q.MAX_RESULT_BYTES:
                row.status, row.error_status = q.FAILED, 500
                row.error = q.RESULT_TOO_LARGE.format(size=size, limit=q.MAX_RESULT_BYTES)
            elif error is None:
                row.status, row.result, row.progress, row.stage = q.SUCCEEDED, stored, 1.0, None
            else:
                row.status, row.error, row.error_status = q.FAILED, error[:500], error_status
            return row.status

    def recover(self, *, lease_s=q.LEASE_S, max_attempts=q.MAX_ATTEMPTS) -> dict:
        requeued = failed = cancelled = 0
        with self._lock:
            now = self._clock()
            for row in self._rows.values():
                if row.status != q.RUNNING or row.heartbeat_at is None or \
                        (now - row.heartbeat_at).total_seconds() <= lease_s:
                    continue
                row.worker = None
                if row.cancel_requested:
                    row.status, row.finished_at, row.payload = q.CANCELLED, now, None
                    row.error = q.CANCELLED_MESSAGE
                    cancelled += 1
                elif row.attempts >= max_attempts:
                    row.status, row.finished_at, row.payload = q.FAILED, now, None
                    row.error, row.error_status = q.FAILED_AFTER_RESTARTS.format(n=row.attempts), 503
                    failed += 1
                else:
                    row.status, row.progress, row.stage = q.QUEUED, 0.0, None
                    requeued += 1
        return {"requeued": requeued, "failed": failed, "cancelled": cancelled}

    def prune(self) -> dict:
        results = deleted = 0
        with self._lock:
            now = self._clock()
            keep_after = now - timedelta(seconds=q.RESULT_KEEP_S)
            by_owner: dict[str, list[_Row]] = {}
            for row in self._rows.values():
                if row.result is not None:
                    by_owner.setdefault(row.owner, []).append(row)
            for rows in by_owner.values():
                rows.sort(key=lambda r: r.seq, reverse=True)
                for i, row in enumerate(rows):
                    if i >= q.KEEP_RESULTS_PER_OWNER or row.finished_at < keep_after:
                        row.result, row.result_expired = None, True
                        results += 1
            cutoff = now - timedelta(seconds=q.KEEP_JOBS_S)
            for job_id in [r.id for r in self._rows.values()
                           if r.status in q.FINISHED and r.finished_at < cutoff]:
                del self._rows[job_id]
                deleted += 1
        return {"results_dropped": results, "jobs_deleted": deleted}

    def counts(self) -> dict:
        with self._lock:
            out = {s: 0 for s in (q.QUEUED, q.RUNNING, q.SUCCEEDED, q.FAILED, q.CANCELLED)}
            for row in self._rows.values():
                out[row.status] += 1
            return out

    # -- tests --------------------------------------------------------------
    def age_heartbeat(self, job_id: uuid.UUID, seconds: float) -> None:
        """Pretend the runner of ``job_id`` went silent ``seconds`` ago."""
        with self._lock:
            row = self._rows[job_id]
            row.heartbeat_at = self._clock() - timedelta(seconds=seconds)

    def age_finished(self, job_id: uuid.UUID, seconds: float) -> None:
        with self._lock:
            row = self._rows[job_id]
            row.finished_at = self._clock() - timedelta(seconds=seconds)


def _copy(value: Any) -> Any:
    return None if value is None else json.loads(json.dumps(value))
