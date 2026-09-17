"""Connections to the database, built for Neon's free plan.

Neon suspends the compute after 5 minutes idle and wakes it on the next
connection. Waking takes from under a second to a few seconds, and every
connection open at suspend time is closed by the server. So:

- pooled connections are checked before use (``pool_pre_ping``) and recycled
  before Neon's idle cut-off, so a dead one is replaced, never handed out;
- opening a connection is retried with backoff (``RETRY_DELAYS_S``) while the
  compute wakes; only if every attempt fails does the caller get
  ``DatabaseUnavailable`` (the API answers 503 and the rest keeps working);
- work inside an open connection is not retried: a transaction cut off
  half-way fails loudly rather than running twice.

The app uses the pooled URL Neon gives (``-pooler`` host, PgBouncer).
Migrations use the direct URL (``direct_url``), which Neon recommends for
schema changes.

Least privilege (PLAN.md 1.7): ``DATABASE_URL`` can be a role that only reads
and writes rows (``fse_app``, migration 0005); the schema owner's URL then
goes in ``DATABASE_MIGRATION_URL``, used for migrations only. Without it,
migrations use ``DATABASE_URL`` as before.

Encrypted connections: a deployed copy (production or staging) always uses
TLS. A URL without ``sslmode`` gets ``sslmode=require``; one that would allow
plain text (``disable``, ``allow``, ``prefer``) is refused.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import DBAPIError, OperationalError

from api.observability import deploy_environment, log_event

# Waits between connection attempts: 7 attempts over ~15 s rides out a Neon
# cold start (typically well under 5 s) without holding a request forever
RETRY_DELAYS_S: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
CONNECT_TIMEOUT_S = 10
# Recycle before Neon's pooler/compute drops idle connections (5 min idle)
POOL_RECYCLE_S = 240


class DatabaseUnavailable(RuntimeError):
    """The database could not be reached after every retry (or isn't configured)."""


def database_url() -> Optional[str]:
    """``DATABASE_URL`` from the environment, or None when unset/blank."""
    return (os.environ.get("DATABASE_URL") or "").strip() or None


def migration_url() -> Optional[str]:
    """``DATABASE_MIGRATION_URL`` (the schema owner) when set, else ``DATABASE_URL``."""
    return (os.environ.get("DATABASE_MIGRATION_URL") or "").strip() or database_url()


def is_configured() -> bool:
    return database_url() is not None


# libpq modes that never fall back to plain text
TLS_SSLMODES = ("require", "verify-ca", "verify-full")
DEPLOYED_ENVIRONMENTS = ("production", "staging")
LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")


def require_tls(raw: str, environment: Optional[str] = None) -> str:
    """``raw`` with TLS enforced when this is a deployed copy.

    Locally and in CI (a Postgres without TLS) the URL is returned unchanged,
    as is a loopback host, where traffic never leaves the machine.
    """
    environment = deploy_environment() if environment is None else environment
    if environment not in DEPLOYED_ENVIRONMENTS:
        return raw
    parts = urlsplit(raw)
    if parts.hostname in LOOPBACK_HOSTS:
        return raw
    query = parse_qsl(parts.query, keep_blank_values=True)
    modes = [v for k, v in query if k == "sslmode"]
    if not modes:
        query.append(("sslmode", "require"))
        return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))
    if any(mode not in TLS_SSLMODES for mode in modes):
        log_event("database_tls_refused", logging.ERROR, sslmode=modes[-1][:20])
        raise DatabaseUnavailable(
            f"the database URL must use TLS (sslmode={'|'.join(TLS_SSLMODES)}), not sslmode={modes[-1]}")
    return raw


def sqlalchemy_url(raw: str) -> str:
    """A Postgres URL as given by Neon/Heroku-style hosts, for SQLAlchemy + psycopg 3.

    ``postgres://`` and ``postgresql://`` become ``postgresql+psycopg://``;
    query parameters (``sslmode``, ``channel_binding``) are kept for libpq,
    and a deployed copy gets TLS enforced (``require_tls``).
    """
    parts = urlsplit(require_tls(raw))
    scheme = parts.scheme.lower()
    if scheme in ("postgres", "postgresql"):
        scheme = "postgresql+psycopg"
    elif not scheme.startswith("postgresql+"):
        raise ValueError("DATABASE_URL must be a postgres:// or postgresql:// URL")
    return urlunsplit((scheme, parts.netloc, parts.path, parts.query, parts.fragment))


def direct_url(raw: str) -> str:
    """The non-pooled URL for a Neon pooled URL (drops ``-pooler`` from the endpoint host).

    Other hosts are returned unchanged.
    """
    parts = urlsplit(raw)
    host = parts.hostname or ""
    if not host.endswith(".neon.tech"):
        return raw
    endpoint, _, rest = host.partition(".")
    if not endpoint.endswith("-pooler"):
        return raw
    new_host = endpoint[: -len("-pooler")] + "." + rest
    netloc = parts.netloc.replace(host, new_host, 1)
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def redacted(raw: str) -> str:
    """Host and database only, for logs (never the user or password)."""
    parts = urlsplit(raw)
    return f"{parts.hostname}{parts.path}"


_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()


def connect_args() -> dict:
    return {
        "connect_timeout": CONNECT_TIMEOUT_S,
        # PgBouncer in transaction mode: don't rely on server-side prepared statements
        "prepare_threshold": None,
        "application_name": "fse-api",
    }


def get_engine(url: Optional[str] = None) -> Engine:
    """The shared engine for ``url`` (default ``DATABASE_URL``)."""
    raw = url or database_url()
    if raw is None:
        raise DatabaseUnavailable("DATABASE_URL is not set")
    with _engines_lock:
        engine = _engines.get(raw)
        if engine is None:
            engine = create_engine(
                sqlalchemy_url(raw),
                # Small pool: Render free has one small instance, Neon free a
                # modest connection limit behind the pooler
                pool_size=3, max_overflow=2, pool_timeout=10,
                pool_pre_ping=True, pool_recycle=POOL_RECYCLE_S,
                connect_args=connect_args(),
            )
            _count_connects(engine)
            _engines[raw] = engine
        return engine


def dispose_engines() -> None:
    """Close every pool (tests; also safe at shutdown)."""
    with _engines_lock:
        for engine in _engines.values():
            engine.dispose()
        _engines.clear()


# New physical connections opened, per engine; tests use it to prove a
# health check doesn't touch the database
connects_opened: dict[int, int] = {}


def _count_connects(engine: Engine) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_conn, record):  # noqa: ARG001
        connects_opened[id(engine)] = connects_opened.get(id(engine), 0) + 1


def _is_connection_error(exc: BaseException) -> bool:
    if isinstance(exc, OperationalError):
        return True
    return isinstance(exc, DBAPIError) and bool(exc.connection_invalidated)


def open_with_retries(engine: Engine, delays: Optional[tuple[float, ...]] = None) -> tuple[Connection, int]:
    """Open a checked connection, retrying while the database wakes.

    Returns the connection and how many attempts it took.
    """
    delays = RETRY_DELAYS_S if delays is None else delays
    attempt = 0
    while True:
        attempt += 1
        try:
            return engine.connect(), attempt
        except Exception as exc:  # noqa: BLE001 - narrowed below
            if not _is_connection_error(exc):
                raise
            if attempt > len(delays):
                log_event("database_unavailable", logging.ERROR, attempts=attempt,
                          error=type(getattr(exc, "orig", None) or exc).__name__)
                raise DatabaseUnavailable(
                    f"database unreachable after {attempt} attempts") from exc
            wait = delays[attempt - 1]
            log_event("database_connect_retry", logging.WARNING, attempt=attempt, wait_s=wait,
                      error=type(getattr(exc, "orig", None) or exc).__name__)
            time.sleep(wait)


@contextmanager
def connect(*, migrate: bool = True) -> Iterator[Connection]:
    """A connection from the pool, waiting for Neon to wake if needed.

    The first use in a process brings the schema up to date (``db.migrate``),
    so a free instance that restarts on every wake-up doesn't wake the
    database just to check migrations.
    """
    engine = get_engine()
    if migrate:
        from db.migrate import ensure_migrated
        ensure_migrated()
    conn, _ = open_with_retries(engine)
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction() -> Iterator[Connection]:
    """``connect()`` inside a transaction: committed on success, rolled back on error."""
    with connect() as conn:
        with conn.begin():
            yield conn
