"""Which queue and which runner this process uses: configuration, not code.

- ``FSE_JOB_QUEUE``: ``database`` (the ``jobs`` table; the default whenever
  ``DATABASE_URL`` is set) or ``memory`` (the default without a database:
  local runs and tests; jobs don't survive a restart).
- ``FSE_JOB_RUNNER``: ``api`` (the default: a thread inside the API runs the
  jobs, as the free plan requires) or ``external`` (phase 12: the API only
  queues, and ``python -m jobs.worker`` processes elsewhere run them).

Tests swap the queue with ``use_queue``; the endpoints never know which one
they have.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

from db.engine import is_configured as database_configured
from jobs.queue import JobQueue

QUEUES = ("database", "memory")
RUNNERS = ("api", "external")

_lock = threading.Lock()
_queue: Optional[JobQueue] = None
_runner = None


def queue_mode() -> str:
    mode = (os.environ.get("FSE_JOB_QUEUE") or "").strip().lower()
    if mode:
        if mode not in QUEUES:
            raise ValueError(f"FSE_JOB_QUEUE must be one of {', '.join(QUEUES)}, not {mode!r}")
        return mode
    return "database" if database_configured() else "memory"


def runner_mode() -> str:
    mode = (os.environ.get("FSE_JOB_RUNNER") or "api").strip().lower()
    if mode not in RUNNERS:
        raise ValueError(f"FSE_JOB_RUNNER must be one of {', '.join(RUNNERS)}, not {mode!r}")
    return mode


def build_queue(mode: Optional[str] = None) -> JobQueue:
    mode = mode or queue_mode()
    if mode == "database":
        from jobs.database import DatabaseQueue
        return DatabaseQueue()
    from jobs.memory import MemoryQueue
    return MemoryQueue()


def get_queue() -> JobQueue:
    """This process's queue (one shared instance, so a memory queue is shared too)."""
    global _queue
    with _lock:
        if _queue is None:
            _queue = build_queue()
        return _queue


def get_runner():
    """The in-API runner (built on first use)."""
    global _runner
    with _lock:
        if _runner is None:
            from jobs.runner import Runner
            _runner = Runner(get_queue)
        return _runner


def wake_runner() -> None:
    """Have the in-API runner look at the queue (nothing when workers run elsewhere)."""
    if runner_mode() == "api":
        get_runner().wake()


def runner_alive() -> bool:
    return _runner is not None and _runner.alive()


def use_queue(queue: Optional[JobQueue]) -> None:
    """Replace the queue (tests; None goes back to the configured one) and
    stop the runner, so the next wake builds one on the new queue."""
    global _queue, _runner
    with _lock:
        runner, _runner = _runner, None
        _queue = queue
    if runner is not None:
        runner.stop()
