"""Schema migrations (Alembic), run automatically and from the command line.

The API brings the schema up to date the first time a request needs the
database in each process (``ensure_migrated``), not at start-up: the free
Render instance restarts on every wake-up, and checking migrations then would
wake the Neon compute too. Migrations take a Postgres advisory lock, so two
instances starting together can't run them twice.

Migrations connect as ``DATABASE_MIGRATION_URL`` when it is set (the schema
owner, while ``DATABASE_URL`` is the restricted app role; see db/engine.py),
otherwise as ``DATABASE_URL``.

Command line (``DATABASE_URL`` must be set; see CLAUDE.md "Adding a table")::

    python -m db.migrate upgrade              # to the latest revision
    python -m db.migrate downgrade -1         # one step back (or a revision id, or base)
    python -m db.migrate current
    python -m db.migrate revision -m "add deals"   # autogenerate from db/models.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from api.observability import log_event, utc_now_iso
from db.engine import (
    DatabaseUnavailable, connect_args, database_url, direct_url, migration_url, open_with_retries,
    sqlalchemy_url,
)

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
# Postgres advisory lock key for migrations ("FSE1")
MIGRATION_LOCK_ID = 0x46534531


def alembic_config() -> Config:
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    # 0002_add_deals.py (set_main_option needs % doubled)
    cfg.set_main_option("file_template", "%%(rev)s_%%(slug)s")
    return cfg


def next_revision_id() -> str:
    """Sequential ids (0001, 0002...) so the order is obvious in the folder."""
    script = ScriptDirectory.from_config(alembic_config())
    numbers = [int(r.revision) for r in script.walk_revisions() if r.revision.isdigit()]
    return f"{max(numbers, default=0) + 1:04d}"


def head_revision() -> str:
    return ScriptDirectory.from_config(alembic_config()).get_current_head()


def current_revision(conn) -> Optional[str]:
    return MigrationContext.configure(conn).get_current_revision()


def _run(action, url: Optional[str], *, lock: bool = True):
    """Run ``action(cfg, conn)`` on a direct connection inside one transaction."""
    raw = url or migration_url()
    if raw is None:
        raise DatabaseUnavailable("DATABASE_URL is not set")
    engine = create_engine(sqlalchemy_url(direct_url(raw)), poolclass=NullPool,
                           connect_args=connect_args())
    try:
        conn, _ = open_with_retries(engine)
        with conn:
            with conn.begin():
                if lock:
                    conn.execute(text("SELECT pg_advisory_xact_lock(:k)"), {"k": MIGRATION_LOCK_ID})
                cfg = alembic_config()
                cfg.attributes["connection"] = conn
                return action(cfg, conn)
    finally:
        engine.dispose()


def upgrade(url: Optional[str] = None, revision: str = "head") -> tuple[Optional[str], Optional[str]]:
    """Upgrade; returns (revision before, revision after)."""
    def action(cfg, conn):
        before = current_revision(conn)
        command.upgrade(cfg, revision)
        return before, current_revision(conn)
    return _run(action, url)


def downgrade(url: Optional[str] = None, revision: str = "-1") -> tuple[Optional[str], Optional[str]]:
    def action(cfg, conn):
        before = current_revision(conn)
        command.downgrade(cfg, revision)
        return before, current_revision(conn)
    return _run(action, url)


def current(url: Optional[str] = None) -> Optional[str]:
    return _run(lambda cfg, conn: current_revision(conn), url, lock=False)


# ---------------------------------------------------------------------------
# Automatic migration on first use
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_migrated: set[str] = set()
# What /api/health reports without touching the database
state: dict = {"status": "unchecked", "revision": None, "checked_at": None}


def ensure_migrated(url: Optional[str] = None) -> None:
    """Bring the schema to the latest revision once per process and URL.

    ``url`` names the database; without it that is ``DATABASE_URL``, migrated
    through ``DATABASE_MIGRATION_URL`` when set.
    """
    raw = url or database_url()
    if raw is None or raw in _migrated:
        return
    with _lock:
        if raw in _migrated:
            return
        try:
            before, after = upgrade(url or migration_url())
        except Exception as exc:
            state.update(status="failed", checked_at=utc_now_iso())
            log_event("migrations_failed", logging.ERROR, error=type(exc).__name__)
            if isinstance(exc, DatabaseUnavailable):
                raise
            raise DatabaseUnavailable("database migrations failed") from exc
        _migrated.add(raw)
        state.update(status="current", revision=after, checked_at=utc_now_iso())
        if before != after:
            log_event("migrations_applied", revision_from=before, revision_to=after)


def reset_state() -> None:
    """Forget which databases were migrated (tests)."""
    with _lock:
        _migrated.clear()
        state.update(status="unchecked", revision=None, checked_at=None)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m db.migrate", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upgrade")
    up.add_argument("revision", nargs="?", default="head")
    down = sub.add_parser("downgrade")
    down.add_argument("revision", nargs="?", default="-1")
    sub.add_parser("current")
    rev = sub.add_parser("revision", help="autogenerate a migration from db/models.py")
    rev.add_argument("-m", "--message", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "upgrade":
        before, after = upgrade(revision=args.revision)
        print(f"upgraded {before} -> {after}")
    elif args.cmd == "downgrade":
        before, after = downgrade(revision=args.revision)
        print(f"downgraded {before} -> {after}")
    elif args.cmd == "current":
        print(current() or "(empty database)")
    elif args.cmd == "revision":
        # Autogenerate compares the models with an up-to-date database
        rev_id = next_revision_id()
        _run(lambda cfg, conn: command.revision(cfg, message=args.message, autogenerate=True,
                                                rev_id=rev_id), None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
