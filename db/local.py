"""A local Postgres for development and tests, with no Docker or installer.

Uses the ``pgserver`` pip package (Postgres binaries in a wheel, Windows,
macOS and Linux). Data lives in ``.localdb/`` (git-ignored)::

    pip install pgserver
    python -m db.local          # start (or reuse) it; prints the variables to set
    python -m db.local stop

Then, in the same shell (PowerShell: ``$env:DATABASE_URL="..."``)::

    DATABASE_URL=...        # the API's database (migrated on first use)
    TEST_DATABASE_URL=...   # tests create and drop their own databases through it

Any other Postgres 14+ works too (e.g. ``docker run -p 5432:5432
-e POSTGRES_PASSWORD=postgres postgres:18``); CI uses that image, the major
version Neon runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine, text

DATA_DIR = Path(__file__).resolve().parents[1] / ".localdb"
DEV_DATABASE = "fse"


def start():
    try:
        import pgserver
    except ImportError:
        sys.exit("pgserver isn't installed: pip install pgserver")
    server = pgserver.get_server(DATA_DIR, cleanup_mode=None)
    admin_url = server.get_uri()
    engine = create_engine(admin_url.replace("postgresql://", "postgresql+psycopg://", 1),
                           isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        exists = conn.execute(text("SELECT 1 FROM pg_database WHERE datname = :d"),
                              {"d": DEV_DATABASE}).scalar()
        if not exists:
            conn.execute(text(f'CREATE DATABASE "{DEV_DATABASE}"'))
    engine.dispose()
    return server, admin_url


def main(argv: list[str]) -> int:
    if argv and argv[0] == "stop":
        import pgserver
        pgserver.get_server(DATA_DIR, cleanup_mode="stop")
        print("stopped")
        return 0
    _, admin_url = start()
    dev_url = admin_url.rsplit("/", 1)[0] + "/" + DEV_DATABASE
    print(f"DATABASE_URL={dev_url}")
    print(f"TEST_DATABASE_URL={admin_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
