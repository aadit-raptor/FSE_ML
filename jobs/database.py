"""The job queue in the database: the ``jobs`` table (db/models.py ``Job``).

Every statement is a single round trip that works through Neon's pooler
(PgBouncer, transaction mode): no advisory locks, no LISTEN, no session state.
Taking a job is one ``UPDATE ... WHERE id = (SELECT ... FOR UPDATE SKIP
LOCKED)``, so any number of runners -- threads here, or dedicated workers in
phase 12 -- can share the table without taking the same job twice.

Times come from the database clock (``now()``), not the caller's, so
heartbeats compare correctly whichever machine wrote them.

The per-owner and whole-queue limits are counted in the same transaction as
the insert; two submits at the same instant can overshoot by one, which the
limits allow for.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from sqlalchemy import String, and_, cast, delete, func, insert, literal, or_, select, text, update

from db.engine import connect, transaction
from db.models import Job
from jobs import queue as q
from jobs.queue import JobNotFound, JobRecord, QueueFull

_SUMMARY = [Job.id, Job.owner, Job.kind, Job.status, Job.progress, Job.stage, Job.attempts,
            Job.created_at, Job.started_at, Job.finished_at, Job.error, Job.error_status,
            Job.cancel_requested]


def _ahead(created_at_col):
    """Queued jobs submitted before this one."""
    other = Job.__table__.alias("other")
    return (select(func.count()).select_from(other)
            .where(other.c.status == q.QUEUED, other.c.created_at < created_at_col)
            .scalar_subquery())


def _record(row, *, payload=None, result=None) -> JobRecord:
    m = row._mapping
    return JobRecord(
        id=m["id"], owner=m["owner"], kind=m["kind"], status=m["status"],
        progress=float(m["progress"]), stage=m["stage"], attempts=m["attempts"],
        created_at=m["created_at"], started_at=m["started_at"], finished_at=m["finished_at"],
        error=m["error"], error_status=m["error_status"], cancel_requested=m["cancel_requested"],
        ahead=m.get("ahead") if m["status"] == q.QUEUED else None,
        payload=payload, result=result,
        result_expired=bool(m.get("result_expired", False)))


class DatabaseQueue:
    name = "database"

    # -- owner side ---------------------------------------------------------
    def submit(self, owner, kind, payload, *, max_active_per_owner=q.MAX_ACTIVE_PER_OWNER,
               max_queued=q.MAX_QUEUED) -> JobRecord:
        with transaction() as conn:
            active = conn.execute(
                select(func.count(), func.count().filter(Job.owner == owner))
                .where(Job.status.in_(q.ACTIVE))).one()
            if max_active_per_owner is not None and active[1] >= max_active_per_owner:
                raise QueueFull(q.owner_message(max_active_per_owner), owner_limit=True)
            if active[0] >= max_queued:
                raise QueueFull(q.BUSY_MESSAGE, owner_limit=False)
            row = conn.execute(
                insert(Job).values(id=uuid.uuid4(), owner=owner, kind=kind, payload=payload)
                .returning(*_SUMMARY, _ahead(Job.created_at).label("ahead"))).one()
            return _record(row)

    def get(self, owner, job_id, *, with_result=False) -> JobRecord:
        cols = [*_SUMMARY, _ahead(Job.created_at).label("ahead"), self._expired()]
        if with_result:
            cols.append(Job.result)
        with connect() as conn:
            row = conn.execute(select(*cols).where(Job.id == job_id, Job.owner == owner)).first()
        if row is None:
            raise JobNotFound(str(job_id))
        return _record(row, result=row._mapping["result"] if with_result else None)

    def list(self, owner, *, limit=20) -> list[JobRecord]:
        with connect() as conn:
            rows = conn.execute(
                select(*_SUMMARY, _ahead(Job.created_at).label("ahead"), self._expired())
                .where(Job.owner == owner).order_by(Job.created_at.desc()).limit(limit)).all()
        return [_record(r) for r in rows]

    def cancel(self, owner, job_id) -> JobRecord:
        with transaction() as conn:
            conn.execute(
                update(Job).where(Job.id == job_id, Job.owner == owner, Job.status == q.QUEUED)
                .values(status=q.CANCELLED, finished_at=func.now(), payload=None,
                        error=q.CANCELLED_MESSAGE))
            conn.execute(
                update(Job).where(Job.id == job_id, Job.owner == owner, Job.status == q.RUNNING)
                .values(cancel_requested=True))
        return self.get(owner, job_id)

    @staticmethod
    def _expired():
        return and_(Job.status == q.SUCCEEDED, Job.result.is_(None)).label("result_expired")

    # -- runner side --------------------------------------------------------
    def claim(self, worker) -> Optional[JobRecord]:
        oldest = (select(Job.id).where(Job.status == q.QUEUED)
                  .order_by(Job.created_at, Job.id).limit(1)
                  .with_for_update(skip_locked=True).scalar_subquery())
        with transaction() as conn:
            row = conn.execute(
                update(Job).where(Job.id == oldest)
                .values(status=q.RUNNING, worker=worker, attempts=Job.attempts + 1,
                        started_at=func.now(), heartbeat_at=func.now(), progress=0, stage=None)
                .returning(*_SUMMARY, Job.payload)).first()
        if row is None:
            return None
        return _record(row, payload=row._mapping["payload"])

    def report(self, job_id, worker, *, progress=None, stage=None) -> bool:
        values: dict[str, Any] = {"heartbeat_at": func.now()}
        if progress is not None:
            values["progress"] = min(1.0, max(0.0, float(progress)))
        if stage is not None:
            values["stage"] = stage[:80]
        with transaction() as conn:
            row = conn.execute(
                update(Job).where(Job.id == job_id, Job.worker == worker, Job.status == q.RUNNING)
                .values(**values).returning(Job.cancel_requested)).first()
        return row is not None and not row.cancel_requested

    def finish(self, job_id, worker, *, result=None, error=None, error_status=None) -> Optional[str]:
        values: dict[str, Any]
        if error is None:
            stored, size = q.encoded(result)
            if size > q.MAX_RESULT_BYTES:
                error, error_status = q.RESULT_TOO_LARGE.format(size=size, limit=q.MAX_RESULT_BYTES), 500
        if error is None:
            values = {"status": q.SUCCEEDED, "result": stored, "progress": 1.0, "stage": None}
        else:
            values = {"status": q.FAILED, "error": error[:500], "error_status": error_status}
        mine = and_(Job.id == job_id, Job.worker == worker, Job.status == q.RUNNING)
        ended = {"finished_at": func.now(), "payload": None, "worker": None}
        with transaction() as conn:
            # Cancelled while it ran: the result is dropped
            row = conn.execute(
                update(Job).where(mine, Job.cancel_requested)
                .values(status=q.CANCELLED, error=q.CANCELLED_MESSAGE, **ended)
                .returning(Job.status)).first()
            if row is None:
                row = conn.execute(update(Job).where(mine).values(**values, **ended)
                                   .returning(Job.status)).first()
        return None if row is None else row.status

    def recover(self, *, lease_s=q.LEASE_S, max_attempts=q.MAX_ATTEMPTS) -> dict:
        stale = and_(Job.status == q.RUNNING,
                     Job.heartbeat_at < func.now() - text(f"interval '{float(lease_s)} seconds'"))
        with transaction() as conn:
            cancelled = conn.execute(
                update(Job).where(stale, Job.cancel_requested)
                .values(status=q.CANCELLED, finished_at=func.now(), payload=None, worker=None,
                        error=q.CANCELLED_MESSAGE)).rowcount
            failed = conn.execute(
                update(Job).where(stale, Job.attempts >= max_attempts)
                .values(status=q.FAILED, finished_at=func.now(), payload=None, worker=None,
                        error_status=503,
                        error=func.replace(literal(q.FAILED_AFTER_RESTARTS), "{n}",
                                           cast(Job.attempts, String)))).rowcount
            requeued = conn.execute(
                update(Job).where(stale)
                .values(status=q.QUEUED, worker=None, progress=0, stage=None)).rowcount
        return {"requeued": requeued, "failed": failed, "cancelled": cancelled}

    def prune(self) -> dict:
        with transaction() as conn:
            ranked = (select(Job.id, func.row_number().over(
                partition_by=Job.owner, order_by=Job.finished_at.desc()).label("n"))
                .where(Job.result.is_not(None)).subquery())
            results = conn.execute(
                update(Job).where(Job.result.is_not(None), or_(
                    Job.finished_at < func.now() - text(f"interval '{q.RESULT_KEEP_S} seconds'"),
                    Job.id.in_(select(ranked.c.id).where(ranked.c.n > q.KEEP_RESULTS_PER_OWNER))))
                .values(result=None)).rowcount
            deleted = conn.execute(
                delete(Job).where(Job.status.in_(q.FINISHED),
                                  Job.finished_at < func.now() - text(f"interval '{q.KEEP_JOBS_S} seconds'"))
            ).rowcount
        return {"results_dropped": results, "jobs_deleted": deleted}

    def counts(self) -> dict:
        out = {s: 0 for s in (q.QUEUED, q.RUNNING, q.SUCCEEDED, q.FAILED, q.CANCELLED)}
        with connect() as conn:
            for status, n in conn.execute(select(Job.status, func.count()).group_by(Job.status)):
                out[status] = n
        return out

    # -- tests --------------------------------------------------------------
    def age_heartbeat(self, job_id: uuid.UUID, seconds: float) -> None:
        with transaction() as conn:
            conn.execute(update(Job).where(Job.id == job_id).values(
                heartbeat_at=func.now() - text(f"interval '{float(seconds)} seconds'")))

    def age_finished(self, job_id: uuid.UUID, seconds: float) -> None:
        with transaction() as conn:
            conn.execute(update(Job).where(Job.id == job_id).values(
                finished_at=func.now() - text(f"interval '{float(seconds)} seconds'")))
