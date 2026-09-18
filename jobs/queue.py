"""The job queue interface, and the rules every implementation follows.

A **job** is a long run someone started and watches: submit, get an id,
poll its progress, fetch the result. The API's endpoints and the runner only
ever talk to a ``JobQueue``; which one is used is configuration
(``jobs.config``), so phase 12 can move the work to dedicated workers without
touching an endpoint:

- ``jobs.database.DatabaseQueue``: the ``jobs`` table in Neon. Survives
  restarts; any number of runners can share it (``FOR UPDATE SKIP LOCKED``).
- ``jobs.memory.MemoryQueue``: a dict in this process, for local runs without
  a database and for tests. Jobs are lost on restart.

**Lifecycle.** ``queued`` → ``running`` → ``succeeded`` | ``failed`` |
``cancelled``. A running job's runner sends a heartbeat; when it stops for
``LEASE_S`` (the server restarted, or Render put it to sleep) ``recover``
puts the job back in the queue -- runs are deterministic for a seed, so
running it again *is* resuming it -- or, after ``MAX_ATTEMPTS``, fails it
with a message saying so.

**Staying small** (Neon free has 0.5 GB): a job's inputs are dropped when it
ends; its result after ``RESULT_KEEP_S``, or sooner once the owner has more
than ``KEEP_RESULTS_PER_OWNER`` newer ones; the row itself after
``KEEP_JOBS_S``. A result over ``MAX_RESULT_BYTES`` is refused.

**Privacy.** Owner-scoped like saved deals: every read matches the owner, so
another account's job answers "not found". Inputs and results never go to
logs.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Optional, Protocol

QUEUED, RUNNING, SUCCEEDED, FAILED, CANCELLED = "queued", "running", "succeeded", "failed", "cancelled"
ACTIVE = (QUEUED, RUNNING)
FINISHED = (SUCCEEDED, FAILED, CANCELLED)

# A running job whose heartbeat is older than this is presumed lost
LEASE_S = 90.0
# Runs of one job before it is failed instead of retried
MAX_ATTEMPTS = 3
# Unfinished jobs one account may have: Monte Carlo submits two at once
MAX_ACTIVE_PER_OWNER = 4
# Unfinished jobs in the whole queue: past this, new ones are refused (503).
# With one simulation at a time on the free server, the last in line would
# wait several minutes already.
MAX_QUEUED = 40
# Results are kept this long after a job ends (the screen fetches it at once)
RESULT_KEEP_S = 6 * 3600
KEEP_RESULTS_PER_OWNER = 10
# Finished jobs are deleted after this
KEEP_JOBS_S = 7 * 86_400
# The largest result stored (a Monte Carlo result is ~0.3 MB)
MAX_RESULT_BYTES = 2_000_000

SYSTEM_OWNER_PREFIX = "system:"


class JobNotFound(LookupError):
    """No such job for this owner (it may exist and belong to someone else)."""


class QueueFull(RuntimeError):
    """Too many unfinished jobs, for this owner or for the whole queue."""

    def __init__(self, message: str, *, owner_limit: bool):
        super().__init__(message)
        self.owner_limit = owner_limit


@dataclass(frozen=True)
class JobRecord:
    id: uuid.UUID
    owner: str
    kind: str
    status: str
    progress: float
    stage: Optional[str]
    attempts: int
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error: Optional[str] = None
    error_status: Optional[int] = None
    cancel_requested: bool = False
    # Queued jobs ahead of this one (queued jobs only)
    ahead: Optional[int] = None
    payload: Optional[dict] = None
    result: Optional[Any] = None
    # A finished job whose result was pruned
    result_expired: bool = False

    def without_contents(self) -> "JobRecord":
        return replace(self, payload=None, result=None)


class JobQueue(Protocol):
    """Where jobs wait and what runners take them from."""

    #: Shown by /api/health/jobs ("database", "memory")
    name: str

    # -- the owner's side (the API endpoints) --------------------------------
    def submit(self, owner: str, kind: str, payload: dict, *,
               max_active_per_owner: Optional[int] = MAX_ACTIVE_PER_OWNER,
               max_queued: int = MAX_QUEUED) -> JobRecord: ...

    def get(self, owner: str, job_id: uuid.UUID, *, with_result: bool = False) -> JobRecord: ...

    def list(self, owner: str, *, limit: int = 20) -> list[JobRecord]: ...

    def cancel(self, owner: str, job_id: uuid.UUID) -> JobRecord: ...

    # -- the runner's side --------------------------------------------------
    def claim(self, worker: str) -> Optional[JobRecord]:
        """The oldest queued job, now running for ``worker`` (with its payload)."""

    def report(self, job_id: uuid.UUID, worker: str, *, progress: Optional[float] = None,
               stage: Optional[str] = None) -> bool:
        """Heartbeat, and progress if given. False when ``worker`` no longer owns
        the job (it was recovered elsewhere) or its owner cancelled it."""

    def finish(self, job_id: uuid.UUID, worker: str, *, result: Any = None,
               error: Optional[str] = None, error_status: Optional[int] = None) -> Optional[str]:
        """End a running job: succeeded with ``result``, or failed with ``error``.

        A job cancelled while running ends ``cancelled`` and its result is
        dropped. Returns the final status, or None if ``worker`` had lost it."""

    def recover(self, *, lease_s: float = LEASE_S, max_attempts: int = MAX_ATTEMPTS) -> dict:
        """Requeue (or fail) running jobs whose heartbeat stopped. Counts."""

    def prune(self) -> dict:
        """Apply the retention rules above. Counts of what was removed."""

    def counts(self) -> dict:
        """Jobs per status (for the health check and the drill)."""


FAILED_AFTER_RESTARTS = ("The server restarted {n} times while this job was running, so it was "
                         "stopped. Run it again; if it keeps happening, try fewer paths.")
CANCELLED_MESSAGE = "Cancelled."
RESULT_TOO_LARGE = ("The result was too large to keep ({size:,} bytes; the limit is {limit:,}). "
                    "Try fewer paths or scatter points.")
BUSY_MESSAGE = "The server has too many runs waiting right now. Try again in a minute."


def owner_message(limit: int) -> str:
    return (f"You already have {limit} runs waiting or running. Wait for one to finish, "
            "or cancel one, then try again.")


def encoded(result: Any) -> tuple[Any, int]:
    """``result`` as it will be stored (plain JSON), and its size in bytes."""
    raw = json.dumps(result, separators=(",", ":"), allow_nan=False)
    return json.loads(raw), len(raw.encode())
