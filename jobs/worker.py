"""A dedicated job worker, for when jobs move off the API (phase 12).

    FSE_JOB_RUNNER=external     # on the API: it only queues
    python -m jobs.worker       # on each worker machine (DATABASE_URL set)

Runs the same loop as the in-API runner (jobs/runner.py) against the
database queue, without stopping when idle. Any number can run at once: the
queue hands each job to one of them. Not used on the free plan, which has no
worker machines; ``tests/test_jobs.py`` runs it once to keep it working.
"""
from __future__ import annotations

import signal
import sys
from typing import Optional

from api.observability import configure_logging
from jobs.config import build_queue
from jobs.runner import Runner


def main(argv: Optional[list[str]] = None, *, once: bool = False) -> int:
    configure_logging()
    queue = build_queue("database")
    runner = Runner(lambda: queue, idle_exit_s=None, poll_s=2.0)
    if once:
        runner.maintain(queue)
        while runner.run_next(queue) is not None:
            pass
        return 0
    signal.signal(signal.SIGTERM, lambda *_: runner.stop(timeout=0))
    runner.loop()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
