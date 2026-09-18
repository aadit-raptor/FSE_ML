"""The job runner: takes jobs from the queue and runs them.

On the free plan it is a thread inside the API (``jobs.config``), because
Render's free tier has no separate worker machines. It is built so that it
costs nothing while there is nothing to do:

- it starts when a job is submitted or someone checks on an unfinished one
  (``wake``), never at start-up -- the API opens no database connection until
  a request needs one, so a sleeping Neon stays asleep;
- it stops after ``idle_exit_s`` with nothing left to run.

After a restart, the first check on an unfinished job wakes the runner, which
puts jobs whose previous runner went silent back in the queue (``recover``).

**One simulation at a time.** A simulation job takes the same slot the direct
simulation endpoints use (api/limits.py), so the free server's 512 MB never
holds two big runs; everything else the API does -- health checks, the deal
model, saving -- goes on answering meanwhile.

**Phase 12** runs the same loop in dedicated workers (``python -m
jobs.worker``, ``FSE_JOB_RUNNER=external``) against the database queue. No
endpoint changes.
"""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from typing import Callable, Optional

from fastapi import HTTPException
from pydantic import ValidationError

from api.limits import holding_simulation_slot
from api.observability import log_event, model_run_listener
from jobs import queue as q
from jobs.kinds import KINDS, JobKind
from jobs.queue import JobQueue, JobRecord

UNEXPECTED = ("The run failed unexpectedly. It has been reported; try again, or change the "
              "inputs if it keeps failing.")
WAITING_FOR_SLOT = "Waiting for another simulation to finish"


def worker_name() -> str:
    return f"{socket.gethostname()[:40]}:{os.getpid()}:{uuid.uuid4().hex[:6]}"


def _message(exc: ValidationError) -> str:
    first = exc.errors()[0]
    where = ".".join(str(p) for p in first.get("loc", ())) or "input"
    return f"{where}: {first.get('msg', 'invalid')}"


class Runner:
    def __init__(self, queue: Callable[[], JobQueue], *, kinds: Optional[dict[str, JobKind]] = None,
                 poll_s: float = 1.0, idle_exit_s: Optional[float] = 60.0,
                 heartbeat_s: float = 15.0, maintenance_s: float = 10.0,
                 lease_s: float = q.LEASE_S, max_attempts: int = q.MAX_ATTEMPTS):
        self.queue = queue
        self.kinds = KINDS if kinds is None else kinds
        self.worker = worker_name()
        self.poll_s, self.idle_exit_s = poll_s, idle_exit_s
        self.heartbeat_s, self.maintenance_s = heartbeat_s, maintenance_s
        self.lease_s, self.max_attempts = lease_s, max_attempts
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # -- the thread ---------------------------------------------------------
    def wake(self) -> None:
        """Start the loop if it isn't running, and have it look now."""
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._stop.clear()
                self._thread = threading.Thread(target=self.loop, name="fse-job-runner", daemon=True)
                self._thread.start()
        self._wake.set()

    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def loop(self) -> None:
        """Run jobs until there are none left for ``idle_exit_s`` (forever if None)."""
        log_event("job_runner_started", worker=self.worker)
        idle_since = time.monotonic()
        last_maintenance = 0.0
        while not self._stop.is_set():
            try:
                queue = self.queue()
                if time.monotonic() - last_maintenance >= self.maintenance_s:
                    last_maintenance = time.monotonic()
                    self.maintain(queue)
                if self.run_next(queue) is not None:
                    idle_since = time.monotonic()
                    continue
                if self.idle_exit_s is not None and time.monotonic() - idle_since >= self.idle_exit_s:
                    counts = queue.counts()
                    if counts[q.QUEUED] == 0 and counts[q.RUNNING] == 0:
                        break
                    idle_since = time.monotonic()   # someone else's job: keep watching it
            except Exception as exc:  # noqa: BLE001 - the loop must outlive a bad moment
                log_event("job_runner_error", logging.WARNING, error=type(exc).__name__)
                self._stop.wait(5.0)
            self._wake.wait(self.poll_s)
            self._wake.clear()
        log_event("job_runner_stopped", worker=self.worker)

    def maintain(self, queue: JobQueue) -> None:
        recovered = queue.recover(lease_s=self.lease_s, max_attempts=self.max_attempts)
        if any(recovered.values()):
            log_event("jobs_recovered", **recovered)

    # -- one job ------------------------------------------------------------
    def run_next(self, queue: Optional[JobQueue] = None) -> Optional[str]:
        """Run the oldest queued job to the end. Its final status, or None if none."""
        queue = queue or self.queue()
        job = queue.claim(self.worker)
        if job is None:
            return None
        return self.execute(queue, job)

    def execute(self, queue: JobQueue, job: JobRecord) -> Optional[str]:
        t0 = time.perf_counter()
        kind = self.kinds.get(job.kind)
        result = error = status_code = None
        stop_heartbeat = threading.Event()
        heartbeat = threading.Thread(target=self._beat, args=(queue, job, stop_heartbeat),
                                     name="fse-job-heartbeat", daemon=True)
        heartbeat.start()
        try:
            if kind is None:
                raise LookupError(f"unknown job kind {job.kind!r}")
            result = self._run(queue, job, kind)
            q.encoded(result)   # a result that can't be stored fails here, as a failed job
        except ValidationError as exc:
            error, status_code = _message(exc), 422
        except HTTPException as exc:
            error, status_code = str(exc.detail), exc.status_code
        except Exception:  # noqa: BLE001 - reported, and the job says it failed
            log_event("job_failed", logging.ERROR, exc_info=True, job=str(job.id), kind=job.kind)
            error, status_code = UNEXPECTED, 500
        finally:
            stop_heartbeat.set()
        if error is not None:
            status = queue.finish(job.id, self.worker, error=error, error_status=status_code)
        else:
            status = queue.finish(job.id, self.worker, result=result)
        log_event("job_finished", job=str(job.id), kind=job.kind, status=status,
                  attempt=job.attempts, duration_ms=round((time.perf_counter() - t0) * 1000, 2))
        return status

    def _run(self, queue: JobQueue, job: JobRecord, kind: JobKind):
        def listener(name: str, event: str) -> None:
            stage = kind.stages.get(name)
            if stage is None:
                return
            if event == "start":
                queue.report(job.id, self.worker, stage=stage.started)
            else:
                queue.report(job.id, self.worker, progress=stage.done, stage=stage.after)

        request = kind.parse(job.payload or {})     # bad input fails before waiting
        token = model_run_listener.set(listener)
        try:
            if not kind.simulation:
                return kind.execute(request)
            with holding_simulation_slot(timeout=0) as got:
                if got:
                    return kind.execute(request)
            queue.report(job.id, self.worker, stage=WAITING_FOR_SLOT)
            with holding_simulation_slot():
                queue.report(job.id, self.worker, stage="Starting")
                return kind.execute(request)
        finally:
            model_run_listener.reset(token)

    def _beat(self, queue: JobQueue, job: JobRecord, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_s):
            try:
                queue.report(job.id, self.worker)
            except Exception as exc:  # noqa: BLE001 - a missed beat is recovered later
                log_event("job_heartbeat_failed", logging.WARNING, error=type(exc).__name__)
