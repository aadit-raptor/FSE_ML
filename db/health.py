"""Database status for the health endpoints, and the storage-usage check.

Two levels, because of Neon's free compute hours (100 a month, compute
suspends after 5 minutes idle):

- ``summary()`` goes in ``/api/health``. It never touches the database:
  Render and the uptime monitor call that endpoint often, and a query each
  time would keep the compute awake all month.
- ``check()`` serves ``/api/health/database``. It connects (waiting for the
  compute to wake), brings the schema up to date, measures the database size
  against the free 0.5 GB and records the reading. A result is reused for
  ``CHECK_CACHE_S`` so repeated calls don't keep the compute awake. It runs
  daily from ``live.yml`` (production) and after each staging deploy
  (``staging.yml``).

At ``STORAGE_WARN_FRACTION`` of the limit the check reports a warning, logs
``database_storage_warning`` and sends one Sentry warning a day; the
workflows fail on it and raise a Better Stack incident.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import timedelta
from typing import Optional

import sentry_sdk
from sqlalchemy import delete, func, insert, select
from sqlalchemy.exc import SQLAlchemyError

from api.observability import log_event, utc_now_iso
from db import migrate
from db.engine import DatabaseUnavailable, get_engine, is_configured, open_with_retries
from db.models import StorageCheck, utc_now

# Neon free plan: 0.5 GB of storage per project
FREE_STORAGE_BYTES = 512 * 1024 * 1024
STORAGE_WARN_FRACTION = 0.8
CHECK_CACHE_S = 600.0
KEEP_READINGS_DAYS = 90
ALERT_EVERY_S = 24 * 3600.0

_lock = threading.Lock()
_cache: dict = {"monotonic": None, "result": None}
_last_alert = [None]


def summary() -> dict:
    """Database status from what this process last saw; no query."""
    if not is_configured():
        return {"configured": False, "status": "not_configured"}
    last = _cache["result"]
    return {
        "configured": True,
        "status": last["status"] if last else "unchecked",
        "checked_at": last["checked_at"] if last else None,
        "migrations": migrate.state["status"],
    }


def storage_report(database_bytes: int, limit_bytes: int = None) -> dict:
    limit = FREE_STORAGE_BYTES if limit_bytes is None else limit_bytes
    used = database_bytes / limit
    return {
        "database_bytes": database_bytes,
        "limit_bytes": limit,
        "used_fraction": round(used, 4),
        "warning": used >= STORAGE_WARN_FRACTION,
    }


def _alert_storage(report: dict, environment: str) -> None:
    log_event("database_storage_warning", logging.WARNING, environment=environment,
              database_bytes=report["database_bytes"], limit_bytes=report["limit_bytes"],
              used_fraction=report["used_fraction"])
    now = time.monotonic()
    if _last_alert[0] is None or now - _last_alert[0] >= ALERT_EVERY_S:
        _last_alert[0] = now
        sentry_sdk.capture_message(
            f"Database storage at {report['used_fraction']:.0%} of the free "
            f"{report['limit_bytes'] // (1024 * 1024)} MB ({environment})", level="warning")


def check(environment: str, *, force: bool = False) -> dict:
    """Connect, migrate if needed, measure storage and record it."""
    if not is_configured():
        return {"configured": False, "status": "not_configured"}
    with _lock:
        cached = _cache["result"]
        if (not force and cached is not None and cached["status"] == "ok"
                and time.monotonic() - _cache["monotonic"] < CHECK_CACHE_S):
            return {**cached, "cached": True}

        t0 = time.perf_counter()
        result = {"configured": True, "checked_at": utc_now_iso(), "cached": False}
        try:
            migrate.ensure_migrated()
            conn, attempts = open_with_retries(get_engine())
            with conn:
                with conn.begin():
                    size = conn.execute(select(func.pg_database_size(func.current_database()))).scalar_one()
                    report = storage_report(int(size))
                    conn.execute(insert(StorageCheck).values(
                        environment=environment, database_bytes=report["database_bytes"],
                        limit_bytes=report["limit_bytes"]))
                    conn.execute(delete(StorageCheck).where(
                        StorageCheck.checked_at < utc_now() - timedelta(days=KEEP_READINGS_DAYS)))
                    revision = migrate.current_revision(conn)
            head = migrate.head_revision()
            result.update(
                status="ok",
                connect_attempts=attempts,
                latency_ms=round((time.perf_counter() - t0) * 1000, 1),
                migrations={"revision": revision, "head": head,
                            "status": "current" if revision == head else "behind"},
                storage=report,
            )
            if report["warning"]:
                _alert_storage(report, environment)
        except (DatabaseUnavailable, SQLAlchemyError) as exc:
            # The error's type only: messages can include the host or query values
            orig: Optional[BaseException] = getattr(exc, "orig", None) or exc.__cause__
            result.update(status="error", error=type(exc).__name__,
                          cause=type(orig).__name__ if orig else None,
                          migrations={"status": migrate.state["status"]},
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1))
            log_event("database_check_failed", logging.ERROR, error=result["error"], cause=result["cause"])
        _cache.update(monotonic=time.monotonic(), result=result)
        return result


def reset() -> None:
    """Forget cached results and alerts (tests)."""
    with _lock:
        _cache.update(monotonic=None, result=None)
        _last_alert[0] = None
