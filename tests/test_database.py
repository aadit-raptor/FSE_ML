"""Database layer (PLAN.md 1.3): URLs, column rules, migrations both ways,
health endpoints, the storage warning and recovery after Neon scales to zero.

Tests that need Postgres take the ``fresh_db`` fixture (tests/conftest.py):
an empty database of their own on ``TEST_DATABASE_URL``. Without that
variable they skip, except in CI, where ``FSE_REQUIRE_DB=1`` turns a skip into
a failure.
"""
import io
import json
import logging
import math
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from fastapi.testclient import TestClient
from sqlalchemy import (
    Column, DateTime, Float, Integer, MetaData, String, Table, create_engine, inspect, select, text,
)
from sqlalchemy.engine import make_url

import api.main as api_main
from api import observability
from db import engine as db_engine
from db import health as db_health
from db import migrate
from db.models import Base, CurrencyCode, MoneyAmount, StorageCheck, UTCDateTime, check_conventions

client = TestClient(api_main.app)


# ---------------------------------------------------------------------------
# No database needed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ("postgres://u:p@host:5432/db", "postgresql+psycopg://u:p@host:5432/db"),
    ("postgresql://u:p@ep-cool-1234-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require",
     "postgresql+psycopg://u:p@ep-cool-1234-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"),
    ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
])
def test_sqlalchemy_url(raw, expected):
    assert db_engine.sqlalchemy_url(raw) == expected


def test_sqlalchemy_url_rejects_other_databases():
    with pytest.raises(ValueError):
        db_engine.sqlalchemy_url("mysql://u:p@host/db")


def test_direct_url_drops_neon_pooler_only():
    pooled = "postgresql://u:pw-pooler@ep-cool-1234-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require"
    assert db_engine.direct_url(pooled) == \
        "postgresql://u:pw-pooler@ep-cool-1234.us-east-2.aws.neon.tech/neondb?sslmode=require"
    assert db_engine.direct_url("postgresql://u:p@db-pooler.example.com/x") == \
        "postgresql://u:p@db-pooler.example.com/x"
    assert db_engine.redacted(pooled) == "ep-cool-1234-pooler.us-east-2.aws.neon.tech/neondb"


def test_models_follow_the_column_rules():
    assert check_conventions(Base.metadata) == []


def test_column_rules_catch_local_times_and_money_without_currency():
    md = MetaData()
    Table("bad", md,
          Column("id", Integer, primary_key=True),
          Column("created", DateTime()),                     # no time zone
          Column("price_amount", MoneyAmount()),             # no price_currency
          Column("fee_amount", MoneyAmount()),
          Column("fee_currency", String(3)),                 # not CurrencyCode
          Column("debt", MoneyAmount()),                     # not named *_amount
          Column("cash_amount", Float()),                    # float money
          Column("ok_at", UTCDateTime()),
          Column("equity_amount", MoneyAmount(), nullable=False),
          Column("equity_currency", CurrencyCode(), nullable=False))
    problems = check_conventions(md)
    assert problems == [
        "bad.created: date-time columns must use UTCDateTime",
        "bad.price_amount: money needs a currency column price_currency",
        "bad.fee_currency: must use CurrencyCode",
        "bad.debt: money columns are named <name>_amount",
        "bad.cash_amount: money amounts must use MoneyAmount",
    ]


def test_column_types_refuse_naive_times_bad_currencies_and_float_money():
    utc = UTCDateTime()
    with pytest.raises(ValueError):
        utc.process_bind_param(datetime(2026, 1, 1, 12, 0), None)
    paris = timezone(timedelta(hours=2))
    stored = utc.process_bind_param(datetime(2026, 6, 1, 14, 0, tzinfo=paris), None)
    assert stored == datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc) and stored.utcoffset() == timedelta(0)
    for bad in ("usd", "US", "EURO", 840):
        with pytest.raises(ValueError):
            CurrencyCode().process_bind_param(bad, None)
    assert CurrencyCode().process_bind_param("JPY", None) == "JPY"
    with pytest.raises(ValueError):
        MoneyAmount().process_bind_param(1.1, None)
    assert MoneyAmount().process_bind_param(Decimal("1.10"), None) == Decimal("1.10")


@pytest.mark.parametrize("database_bytes, warning", [
    (0, False),
    (math.ceil(db_health.FREE_STORAGE_BYTES * 0.8) - 1, False),
    (math.ceil(db_health.FREE_STORAGE_BYTES * 0.8), True),
    (db_health.FREE_STORAGE_BYTES, True),
])
def test_storage_warning_starts_at_80_percent_of_the_free_limit(database_bytes, warning):
    assert db_health.FREE_STORAGE_BYTES == 512 * 1024 * 1024
    report = db_health.storage_report(database_bytes)
    assert report["warning"] is warning
    assert report["limit_bytes"] == 512 * 1024 * 1024


def test_storage_warning_includes_exactly_80_percent():
    assert db_health.storage_report(800, limit_bytes=1000)["warning"] is True
    assert db_health.storage_report(799, limit_bytes=1000)["warning"] is False


def test_health_without_a_database(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    db_health.reset()
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["database"] == {"configured": False, "status": "not_configured"}
    resp = client.get("/api/health/database")
    assert resp.status_code == 200 and resp.json()["status"] == "not_configured"


# ---------------------------------------------------------------------------
# Postgres
# ---------------------------------------------------------------------------
def tables(url):
    engine = create_engine(db_engine.sqlalchemy_url(url))
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_migrations_run_forwards_and_backwards(fresh_db):
    assert tables(fresh_db) == set()
    head = migrate.head_revision()
    assert migrate.upgrade(fresh_db) == (None, head)
    assert tables(fresh_db) == {"alembic_version", "storage_checks", "users", "deals", "deal_versions"}
    assert migrate.current(fresh_db) == head

    # Every revision back to an empty schema, one step at a time
    while migrate.current(fresh_db) is not None:
        before, after = migrate.downgrade(fresh_db, "-1")
        assert after != before
    assert tables(fresh_db) == {"alembic_version"}

    assert migrate.upgrade(fresh_db) == (None, head)
    assert "storage_checks" in tables(fresh_db)


def test_migrations_match_the_models(fresh_db):
    """A model change without a migration (or the reverse) fails here."""
    migrate.upgrade(fresh_db)
    engine = create_engine(db_engine.sqlalchemy_url(fresh_db))
    try:
        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn, opts={"compare_type": True}),
                                    Base.metadata)
    finally:
        engine.dispose()
    assert diff == []


def test_times_round_trip_in_utc(fresh_db):
    migrate.upgrade(fresh_db)
    tokyo = timezone(timedelta(hours=9))
    written = datetime(2026, 9, 16, 9, 30, tzinfo=tokyo)
    with db_engine.transaction() as conn:
        conn.execute(StorageCheck.__table__.insert().values(
            checked_at=written, environment="test", database_bytes=1, limit_bytes=2))
    with db_engine.connect() as conn:
        read = conn.execute(select(StorageCheck.checked_at)).scalar_one()
    assert read == written and read.utcoffset() == timedelta(0) and read.hour == 0


def test_health_database_migrates_on_first_use_and_records_storage(fresh_db, monkeypatch):
    monkeypatch.setenv("FSE_ENV", "staging")
    # /api/health reports the database without connecting to it
    before = sum(db_engine.connects_opened.values())
    body = client.get("/api/health").json()
    assert body["database"] == {"configured": True, "status": "unchecked", "checked_at": None,
                                "migrations": "unchecked"}
    assert sum(db_engine.connects_opened.values()) == before
    assert tables(fresh_db) == set()

    resp = client.get("/api/health/database")
    assert resp.status_code == 200, resp.text
    result = resp.json()
    assert result["status"] == "ok" and result["cached"] is False
    assert result["migrations"] == {"revision": migrate.head_revision(), "head": migrate.head_revision(),
                                    "status": "current"}
    storage = result["storage"]
    assert storage["database_bytes"] > 1_000_000  # a real pg_database_size, not a placeholder
    assert storage["limit_bytes"] == 512 * 1024 * 1024 and storage["warning"] is False

    with db_engine.connect() as conn:
        rows = conn.execute(select(StorageCheck.environment, StorageCheck.database_bytes)).all()
    assert rows == [("staging", storage["database_bytes"])]

    # Repeat calls reuse the reading instead of keeping the compute awake
    again = client.get("/api/health/database").json()
    assert again["cached"] is True and again["storage"] == storage
    with db_engine.connect() as conn:
        assert conn.execute(select(StorageCheck.id)).all().__len__() == 1

    summary = client.get("/api/health").json()["database"]
    assert summary["status"] == "ok" and summary["migrations"] == "current"


def test_old_storage_readings_are_pruned(fresh_db):
    migrate.upgrade(fresh_db)
    old = datetime.now(timezone.utc) - timedelta(days=db_health.KEEP_READINGS_DAYS + 1)
    recent = datetime.now(timezone.utc) - timedelta(days=1)
    with db_engine.transaction() as conn:
        for when in (old, recent):
            conn.execute(StorageCheck.__table__.insert().values(
                checked_at=when, environment="test", database_bytes=1, limit_bytes=2))
    assert db_health.check("test")["status"] == "ok"
    with db_engine.connect() as conn:
        kept = conn.execute(select(StorageCheck.checked_at).order_by(StorageCheck.checked_at)).scalars().all()
    assert len(kept) == 2 and kept[0] == recent


def test_storage_warning_is_logged_and_sent_once(fresh_db, monkeypatch):
    monkeypatch.setattr(db_health, "FREE_STORAGE_BYTES", 1024 * 1024)  # the test database is bigger
    sent = []
    monkeypatch.setattr(db_health.sentry_sdk, "capture_message", lambda msg, level: sent.append((msg, level)))
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(observability.JsonFormatter())
    observability.logger.addHandler(handler)
    try:
        first = client.get("/api/health/database").json()
        db_health.check("staging", force=True)
    finally:
        observability.logger.removeHandler(handler)
    assert first["storage"]["warning"] is True and first["storage"]["used_fraction"] > 1
    warnings = [json.loads(line) for line in stream.getvalue().splitlines()
                if json.loads(line)["event"] == "database_storage_warning"]
    assert len(warnings) == 2 and warnings[0]["level"] == "warning"
    assert warnings[0]["database_bytes"] == first["storage"]["database_bytes"]
    # one Sentry warning a day, however often the check runs
    assert len(sent) == 1 and sent[0][1] == "warning" and "free 1 MB" in sent[0][0]


def test_database_down_gives_503_and_the_model_keeps_working(monkeypatch):
    port = free_port()  # nothing listens here
    monkeypatch.setenv("DATABASE_URL", f"postgresql://u:secret-pw@127.0.0.1:{port}/nope")
    monkeypatch.setattr(db_engine, "RETRY_DELAYS_S", (0.01, 0.01))
    db_health.reset()
    migrate.reset_state()
    try:
        resp = client.get("/api/health/database")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "error" and body["error"] == "DatabaseUnavailable"
        assert "secret-pw" not in resp.text and str(port) not in resp.text
        health = client.get("/api/health")
        assert health.status_code == 200 and health.json()["database"]["status"] == "error"
        deal = client.post("/api/deal/run", json={})
        assert deal.status_code == 200 and round(deal.json()["returns"]["irr"], 4) == 0.2116
    finally:
        db_engine.dispose_engines()
        db_health.reset()
        migrate.reset_state()


# ---------------------------------------------------------------------------
# Neon scaling to zero
# ---------------------------------------------------------------------------
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SuspendingProxy:
    """A TCP proxy in front of Postgres that behaves like Neon's compute
    suspending: ``suspend()`` cuts every open connection and turns new ones
    away until ``wake_after`` seconds have passed."""

    def __init__(self, target_host, target_port):
        self.target = (target_host, target_port)
        self.listener = socket.socket()
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(32)
        self.port = self.listener.getsockname()[1]
        self.asleep_until = 0.0
        self.refused = 0
        self.sockets: list[socket.socket] = []
        self.lock = threading.Lock()
        self.closed = False
        threading.Thread(target=self._accept, daemon=True).start()

    def suspend(self, wake_after: float):
        with self.lock:
            self.asleep_until = time.monotonic() + wake_after
            for s in self.sockets:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                s.close()
            self.sockets.clear()

    def close(self):
        self.closed = True
        self.suspend(0)
        self.listener.close()

    def _accept(self):
        while not self.closed:
            try:
                client_sock, _ = self.listener.accept()
            except OSError:
                return
            if time.monotonic() < self.asleep_until:
                self.refused += 1
                client_sock.close()
                continue
            upstream = socket.create_connection(self.target)
            with self.lock:
                self.sockets += [client_sock, upstream]
            for a, b in ((client_sock, upstream), (upstream, client_sock)):
                threading.Thread(target=self._pipe, args=(a, b), daemon=True).start()

    @staticmethod
    def _pipe(src, dst):
        try:
            while data := src.recv(65536):
                dst.sendall(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.close()
                except OSError:
                    pass


@pytest.fixture
def proxied_db(fresh_db, monkeypatch):
    real = make_url(db_engine.sqlalchemy_url(fresh_db))
    proxy = SuspendingProxy(real.host, real.port or 5432)
    url = real.set(host="127.0.0.1", port=proxy.port).render_as_string(hide_password=False) \
        .replace("postgresql+psycopg://", "postgresql://", 1)
    monkeypatch.setenv("DATABASE_URL", url)
    yield proxy
    db_engine.dispose_engines()
    proxy.close()


def test_api_recovers_after_the_database_scales_to_zero(proxied_db, monkeypatch):
    first = client.get("/api/health/database")
    assert first.status_code == 200 and first.json()["connect_attempts"] == 1

    # The compute suspends: pooled connections die and it takes ~0.5 s to wake
    proxied_db.suspend(wake_after=0.5)
    monkeypatch.setattr(db_health, "CHECK_CACHE_S", 0.0)
    resp = client.get("/api/health/database")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "ok" and body["connect_attempts"] > 1
    assert proxied_db.refused >= 1
    assert body["storage"]["database_bytes"] > 1_000_000

    # Later requests reuse healthy connections again
    again = client.get("/api/health/database").json()
    assert again["status"] == "ok" and again["connect_attempts"] == 1
    with db_engine.connect() as conn:
        assert len(conn.execute(select(StorageCheck.id)).all()) == 3


def test_a_database_that_stays_asleep_fails_cleanly_then_recovers(proxied_db, monkeypatch):
    assert client.get("/api/health/database").status_code == 200
    monkeypatch.setattr(db_health, "CHECK_CACHE_S", 0.0)
    proxied_db.suspend(wake_after=60)  # longer than every retry
    down = client.get("/api/health/database")
    assert down.status_code == 503 and down.json()["error"] == "DatabaseUnavailable"
    assert client.get("/api/health").json()["database"]["status"] == "error"

    proxied_db.asleep_until = 0.0  # wakes
    up = client.get("/api/health/database")
    assert up.status_code == 200 and up.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# The deploy check used by live.yml and staging.yml
# ---------------------------------------------------------------------------
def test_deploy_check_flags_errors_stale_migrations_and_storage():
    from ops.check_database import problems

    healthy = {"status": "ok", "migrations": {"status": "current", "revision": "0001", "head": "0001"},
               "storage": {"warning": False, "used_fraction": 0.02, "limit_bytes": 512 * 1024 * 1024}}
    assert problems(healthy) == []
    assert problems({"status": "error", "error": "DatabaseUnavailable"}) == \
        ["database status error (DatabaseUnavailable)"]
    assert problems({"status": "not_configured"}) == ["database status not_configured (not_configured)"]
    behind = {**healthy, "migrations": {"status": "behind", "revision": "0001", "head": "0002"}}
    assert problems(behind) == ["migrations behind: at 0001, latest 0002"]
    full = {**healthy, "storage": {"warning": True, "used_fraction": 0.83, "limit_bytes": 512 * 1024 * 1024}}
    assert problems(full) == ["storage at 83% of the free 512 MB: plan cleanup or phase 12.2"]
