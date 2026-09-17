"""Backups and recovery (PLAN.md 1.8).

The encryption format, the backup stores, the rotation policy and — with a
real Postgres (``fresh_db``) — a real round trip: a saved deal is backed up,
encrypted, restored into a different database and read back through the API,
and its IRR must come out identical.

Tests that need Postgres take ``fresh_db`` (tests/conftest.py) and skip
without ``TEST_DATABASE_URL``, except in CI where ``FSE_REQUIRE_DB=1`` turns a
skip into a failure.
"""
from __future__ import annotations

import io
import json
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from api.auth import AuthUser, require_user
from api.main import app
from db import engine as db_engine
from db import health as db_health
from db import migrate
from ops import backup, backup_store, encryption
from ops.backup_store import (
    MAX_UPLOAD_BYTES, LocalStore, StorageError, StoredObject, SupabaseStore, store_from_env,
)

client = TestClient(app)

# Not a key: a plain sentence, comfortably over the 32-character minimum
SECRET = "the tests unlock their own backups with this sentence"
PROFILE = {"country": "DE", "preferred_currency": "EUR", "locale": "de-DE",
           "time_zone": "Europe/Berlin"}
# A deal nothing like the defaults, with a setting that moves its IRR
INPUTS = {"ebitda": 240.0, "entry_mult": 9.0, "exit_mult": 10.5, "hold": 6, "growth": 7.5,
          "gross_margin": 44.0, "opex": 17.0, "tax": 29.0, "da": 3.5, "debt_pct": 55.0,
          "senior_pct": 75.0, "base_rate": 5.25, "mezz_spread": 4.5, "capex": 3.0, "nwc": 0.5,
          "mincash": 15.0, "wsp_mode": False, "ar_days": 45.0, "inv_days": 30.0, "ap_days": 60.0}
SETTINGS = {"tx_fee_pct": 3.0}


# ---------------------------------------------------------------------------
# Encryption
# ---------------------------------------------------------------------------
def seal(data: bytes, secret: str = SECRET, chunk_bytes: int = encryption.CHUNK_BYTES) -> bytes:
    out = io.BytesIO()
    encryption.encrypt_stream(io.BytesIO(data), out, secret, chunk_bytes=chunk_bytes)
    return out.getvalue()


def open_sealed(sealed: bytes, secret: str = SECRET) -> bytes:
    out = io.BytesIO()
    encryption.decrypt_stream(io.BytesIO(sealed), out, secret)
    return out.getvalue()


@pytest.mark.parametrize("size", [0, 1, 1000, 3 * 4096 + 17])
def test_encryption_round_trip(size):
    data = os.urandom(size)
    sealed = seal(data, chunk_bytes=4096)
    assert open_sealed(sealed) == data


def test_the_encrypted_file_does_not_contain_the_plaintext():
    data = b"owner_id,inputs,settings" + b"SENSITIVE-DEAL-CONTENT" * 200
    sealed = seal(data)
    assert b"SENSITIVE-DEAL-CONTENT" not in sealed
    assert sealed.startswith(encryption.MAGIC)


def test_the_same_backup_encrypts_differently_every_time():
    """A fresh salt and nonce each run: two backups of the same database can't
    be told apart or attacked together."""
    data = b"x" * 5000
    assert seal(data) != seal(data)
    assert open_sealed(seal(data)) == open_sealed(seal(data)) == data


def test_reported_checksum_is_of_the_encrypted_file():
    import hashlib

    out = io.BytesIO()
    stats = encryption.encrypt_stream(io.BytesIO(b"y" * 9000), out, SECRET, chunk_bytes=4096)
    assert stats["sha256"] == hashlib.sha256(out.getvalue()).hexdigest()
    assert stats["encrypted_bytes"] == len(out.getvalue())
    assert stats["plaintext_bytes"] == 9000 and stats["chunks"] == 3


def test_another_key_cannot_open_it():
    sealed = seal(b"deal data" * 100)
    with pytest.raises(encryption.BackupCorrupt):
        open_sealed(sealed, SECRET[:-1] + "Z")


def test_an_altered_byte_is_caught():
    sealed = bytearray(seal(b"deal data" * 500, chunk_bytes=1024))
    sealed[len(sealed) // 2] ^= 0x01
    with pytest.raises(encryption.BackupCorrupt):
        open_sealed(bytes(sealed))


def test_a_truncated_backup_is_caught():
    sealed = seal(b"deal data" * 500, chunk_bytes=1024)
    with pytest.raises(encryption.BackupCorrupt, match="truncated"):
        open_sealed(sealed[: len(sealed) - 200])


def test_dropping_whole_chunks_from_the_end_is_caught():
    """Cutting at a chunk boundary leaves a file that parses: the last chunk's
    tag is what proves the file ends where it should."""
    data = os.urandom(4096 * 4)
    sealed = seal(data, chunk_bytes=4096)
    one_chunk = 4 + 4096 + encryption.TAG_BYTES
    with pytest.raises(encryption.BackupCorrupt, match="truncated"):
        open_sealed(sealed[: encryption.HEADER_BYTES + one_chunk * 2])


def test_extra_data_after_the_last_chunk_is_caught():
    with pytest.raises(encryption.BackupCorrupt, match="extra data"):
        open_sealed(seal(b"abc") + b"junk")


def test_reordered_chunks_are_caught():
    data = os.urandom(4096 * 3)
    sealed = seal(data, chunk_bytes=4096)
    one = 4 + 4096 + encryption.TAG_BYTES
    head = encryption.HEADER_BYTES
    first, second = sealed[head:head + one], sealed[head + one:head + 2 * one]
    swapped = sealed[:head] + second + first + sealed[head + 2 * one:]
    with pytest.raises(encryption.BackupCorrupt):
        open_sealed(swapped)


def test_a_file_that_is_not_a_backup_is_refused():
    with pytest.raises(encryption.BackupCorrupt, match="not an FSE backup"):
        open_sealed(b"PGDMP" + os.urandom(200))


def test_the_key_must_be_set_and_long_enough(monkeypatch):
    monkeypatch.delenv(encryption.KEY_VARIABLE, raising=False)
    with pytest.raises(encryption.BackupKeyError, match=encryption.KEY_VARIABLE):
        encryption.secret_from_env()
    monkeypatch.setenv(encryption.KEY_VARIABLE, "short")
    with pytest.raises(encryption.BackupKeyError, match="32 characters"):
        encryption.secret_from_env()
    monkeypatch.setenv(encryption.KEY_VARIABLE, f"  {SECRET}  ")
    assert encryption.secret_from_env() == SECRET


def test_a_weak_key_is_refused_even_when_passed_directly():
    with pytest.raises(encryption.BackupKeyError):
        seal(b"x", secret="password123")


# ---------------------------------------------------------------------------
# Postgres tools: passwords and versions
# ---------------------------------------------------------------------------
def test_the_password_never_reaches_the_command_line():
    url = "postgresql://fse_api:p%40ss%3Aword@ep-cool-1234.aws.neon.tech/neondb?sslmode=require"
    safe, password = backup.split_password(url)
    assert password == "p@ss:word"
    assert "p%40ss" not in safe and "p@ss" not in safe
    assert safe == "postgresql://fse_api@ep-cool-1234.aws.neon.tech/neondb?sslmode=require"


def test_urls_without_a_password_are_left_alone():
    for url in ("postgresql://postgres@localhost:5432/fse", "postgresql://localhost:5432/fse"):
        assert backup.split_password(url) == (url, None)


def test_the_tool_environment_carries_the_password(monkeypatch):
    monkeypatch.delenv("PGPASSWORD", raising=False)
    safe, env = backup._tool_env("postgresql://u:secret-pw@host/db")
    assert env["PGPASSWORD"] == "secret-pw"
    assert "secret-pw" not in safe


def test_a_client_older_than_the_server_is_refused(monkeypatch):
    monkeypatch.setattr(backup, "tool_candidates", lambda name: ["/old/pg_dump", "/new/pg_dump"])
    monkeypatch.setattr(backup, "tool_version", lambda path: 16 if "old" in path else 17)
    assert backup.pg_tool("pg_dump", 17) == "/new/pg_dump"
    assert backup.pg_tool("pg_dump", 16) == "/old/pg_dump"     # the first that is new enough
    with pytest.raises(backup.BackupError, match="postgresql-client-18"):
        backup.pg_tool("pg_dump", 18)


def test_the_tools_are_found_on_this_machine():
    """Whatever this machine has (PATH, a Debian per-version directory or the
    pgserver package), the finder reports a usable pg_dump and pg_restore."""
    for name in ("pg_dump", "pg_restore"):
        candidates = backup.tool_candidates(name)
        assert candidates, f"no {name} found; install postgresql-client or pgserver"
        assert any(backup.tool_version(path) for path in candidates)


# ---------------------------------------------------------------------------
# Names and rotation
# ---------------------------------------------------------------------------
def test_backup_names_carry_the_time_they_were_taken():
    when = datetime(2026, 9, 18, 2, 30, 0, tzinfo=timezone.utc)
    name = backup.backup_name("production", when)
    assert name == "production/2026-09-18T023000Z"
    assert backup.taken_at(name) == when
    assert backup.taken_at("production/not-a-time") is None


def test_an_odd_environment_name_is_refused():
    for bad in ("", "../production", "Production/x"):
        with pytest.raises(backup.BackupError):
            backup.backup_name(bad, datetime.now(timezone.utc))


# One a night for 500 nights leaves this many under the daily/weekly/monthly policy
EXPECTED_KEPT = 24


def stored(name: str, size: int) -> StoredObject:
    return StoredObject(name=name, size=size, modified=None)


def test_backups_pairs_each_dump_with_its_manifest():
    found = backup.backups([
        stored("production/2026-09-18T023000Z.dump.enc", 1000),
        stored("production/2026-09-18T023000Z.json", 300),
        stored("production/2026-09-17T023000Z.dump.enc", 900),
        stored("production/README.txt", 10),          # not a backup: ignored
        stored("production/2026-13-99T999999Z.dump.enc", 10),
    ])
    assert [(b.name, b.size) for b in found] == [
        ("production/2026-09-18T023000Z", 1300),
        ("production/2026-09-17T023000Z", 900),
    ]


def nightly(days: int, size: int = 1024 * 1024,
            end: datetime = datetime(2026, 9, 18, 2, 30, tzinfo=timezone.utc)):
    """One backup a night for ``days`` nights, ending at ``end``."""
    return [backup.BackupFile(backup.backup_name("production", end - timedelta(days=d)),
                              end - timedelta(days=d), size) for d in range(days)]


def test_rotation_keeps_days_then_weeks_then_months():
    end = datetime(2026, 9, 18, 2, 30, tzinfo=timezone.utc)
    kept, dropped = backup.rotation_plan(nightly(500))
    assert len(kept) + len(dropped) == 500
    when = [b.taken_at for b in kept]
    assert when == sorted(when, reverse=True) and when[0] == end        # the newest, always
    # every one of the last 7 nights
    assert {d.date() for d in when} >= {(end - timedelta(days=d)).date() for d in range(7)}
    # one for each of the last 8 weeks and each of the last 12 months
    assert {d.isocalendar()[:2] for d in when} >=         {(end - timedelta(weeks=w)).isocalendar()[:2] for w in range(8)}
    months = {divmod(end.year * 12 + end.month - 1 - m, 12) for m in range(12)}
    assert {divmod(d.year * 12 + d.month - 1, 12) for d in when} >= months
    assert min(when) >= end - timedelta(days=366)                       # nothing older is kept
    assert len(kept) == EXPECTED_KEPT


def test_rotation_keeps_one_backup_a_day():
    """Two runs the same day (the nightly job plus a manual one) leave the
    newer one; the day still counts once towards the seven kept."""
    end = datetime(2026, 9, 18, 2, 30, tzinfo=timezone.utc)
    extra = end + timedelta(hours=18)
    twice = nightly(3) + [backup.BackupFile(backup.backup_name("production", extra), extra, 10)]
    kept, dropped = backup.rotation_plan(twice)
    assert [b.name for b in dropped] == [backup.backup_name("production", end)]
    assert [b.name for b in kept] == [backup.backup_name("production", extra),
                                      backup.backup_name("production", end - timedelta(days=1)),
                                      backup.backup_name("production", end - timedelta(days=2))]


def test_rotation_stays_inside_the_storage_budget():
    kept, dropped = backup.rotation_plan(nightly(400, size=30 * 1024 * 1024),
                                         budget_bytes=200 * 1024 * 1024)
    assert sum(b.size for b in kept) <= 200 * 1024 * 1024
    assert len(kept) == 6 and len(dropped) == 394
    assert kept[0].taken_at > kept[-1].taken_at                # oldest go first


def test_the_newest_backup_is_never_rotated_out():
    """Even one backup larger than the whole budget is kept: having something
    to restore beats staying inside a free-plan limit."""
    kept, dropped = backup.rotation_plan(nightly(3, size=900 * 1024 * 1024),
                                         budget_bytes=100 * 1024 * 1024)
    assert len(kept) == 1 and kept[0].taken_at.date() == datetime(2026, 9, 18).date()
    assert len(dropped) == 2


def test_nothing_to_rotate_when_there_are_no_backups():
    assert backup.rotation_plan([]) == ([], [])


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------
def test_local_store_round_trip(tmp_path):
    store = LocalStore(tmp_path / "store")
    source = tmp_path / "dump.enc"
    source.write_bytes(b"encrypted")
    store.put("production/2026-09-18T023000Z.dump.enc", source)
    store.put("production/2026-09-18T023000Z.json", source, "application/json")
    listed = store.list("production")
    assert [o.name for o in listed] == ["production/2026-09-18T023000Z.dump.enc",
                                        "production/2026-09-18T023000Z.json"]
    assert listed[0].size == len(b"encrypted")
    back = tmp_path / "back.enc"
    store.get("production/2026-09-18T023000Z.dump.enc", back)
    assert back.read_bytes() == b"encrypted"
    store.delete([o.name for o in listed])
    assert store.list("production") == []
    with pytest.raises(StorageError, match="no backup named"):
        store.get("production/2026-09-18T023000Z.dump.enc", back)


def test_local_store_refuses_to_escape_its_directory(tmp_path):
    store = LocalStore(tmp_path / "store")
    source = tmp_path / "x"
    source.write_bytes(b"x")
    with pytest.raises(StorageError, match="outside"):
        store.put("../escaped.enc", source)


def test_the_store_is_chosen_by_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("FSE_BACKUP_DIR", raising=False)
    with pytest.raises(StorageError, match="SUPABASE_URL"):
        store_from_env()
    monkeypatch.setenv("FSE_BACKUP_DIR", str(tmp_path))
    assert isinstance(store_from_env(), LocalStore)
    monkeypatch.setenv("SUPABASE_URL", "https://abc.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-role-key")
    monkeypatch.setenv("SUPABASE_BACKUP_BUCKET", "fse-backups")
    store = store_from_env()
    assert isinstance(store, SupabaseStore) and store.bucket == "fse-backups"


class FakeSupabase(BaseHTTPRequestHandler):
    """Enough of Supabase Storage's REST API to prove the client speaks it."""

    objects: dict = {}
    seen: list = []

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def _record(self):
        FakeSupabase.seen.append((self.command, self.path,
                                  {k.lower(): v for k, v in self.headers.items()}))

    def _send(self, code: int, payload: bytes, content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        body = self._body()
        self._record()
        if self.headers.get("Authorization") != "Bearer service-role-key":
            return self._send(401, b'{"error":"unauthorized"}')
        if self.path.startswith("/storage/v1/object/list/"):
            prefix = json.loads(body)["prefix"]
            listing = [{"name": name.split("/", 1)[1], "metadata": {"size": len(data)},
                        "updated_at": "2026-09-18T02:30:00.000Z"}
                       for name, data in sorted(FakeSupabase.objects.items())
                       if name.startswith(prefix + "/")]
            return self._send(200, json.dumps(listing).encode())
        FakeSupabase.objects[self.path.split("/storage/v1/object/backups/", 1)[1]] = body
        self._send(200, b'{"Key":"ok"}')

    def do_GET(self):
        self._record()
        name = self.path.split("/storage/v1/object/backups/", 1)[1]
        if name not in FakeSupabase.objects:
            return self._send(404, b'{"error":"Object not found"}')
        self._send(200, FakeSupabase.objects[name], "application/octet-stream")

    def do_DELETE(self):
        body = self._body()
        self._record()
        for name in json.loads(body)["prefixes"]:
            FakeSupabase.objects.pop(name, None)
        self._send(200, b"[]")

    def log_message(self, *args):
        pass


@pytest.fixture
def supabase():
    FakeSupabase.objects, FakeSupabase.seen = {}, []
    server = HTTPServer(("127.0.0.1", 0), FakeSupabase)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield SupabaseStore(f"http://127.0.0.1:{server.server_port}", "service-role-key")
    server.shutdown()
    server.server_close()


def test_supabase_store_round_trip(supabase, tmp_path):
    source = tmp_path / "dump.enc"
    source.write_bytes(b"encrypted bytes")
    supabase.put("production/2026-09-18T023000Z.dump.enc", source)
    supabase.put("production/2026-09-17T023000Z.dump.enc", source)
    listed = supabase.list("production")
    assert [o.name for o in listed] == ["production/2026-09-17T023000Z.dump.enc",
                                        "production/2026-09-18T023000Z.dump.enc"]
    assert listed[0].size == len(b"encrypted bytes")
    assert listed[0].modified == datetime(2026, 9, 18, 2, 30, tzinfo=timezone.utc)
    back = tmp_path / "back.enc"
    supabase.get("production/2026-09-18T023000Z.dump.enc", back)
    assert back.read_bytes() == b"encrypted bytes"
    supabase.delete(["production/2026-09-17T023000Z.dump.enc"])
    assert [o.name for o in supabase.list("production")] == \
        ["production/2026-09-18T023000Z.dump.enc"]
    # Every call carried the key as a bearer token, and none put it in the URL
    assert all(headers.get("authorization") == "Bearer service-role-key"
               for _, _, headers in FakeSupabase.seen)
    assert not any("service-role-key" in path for _, path, _ in FakeSupabase.seen)


def test_supabase_upload_is_an_upsert(supabase, tmp_path):
    source = tmp_path / "dump.enc"
    source.write_bytes(b"first")
    supabase.put("production/x.dump.enc", source)
    source.write_bytes(b"second")
    supabase.put("production/x.dump.enc", source)
    assert FakeSupabase.objects["production/x.dump.enc"] == b"second"
    assert all(headers.get("x-upsert") == "true"
               for method, path, headers in FakeSupabase.seen if method == "POST")


def test_a_supabase_failure_never_shows_the_key(supabase, tmp_path):
    missing = tmp_path / "back.enc"
    with pytest.raises(StorageError) as caught:
        supabase.get("production/missing.dump.enc", missing)
    assert "404" in str(caught.value) and "service-role-key" not in str(caught.value)


def test_a_backup_too_big_for_the_free_plan_is_refused_before_uploading(supabase, tmp_path,
                                                                        monkeypatch):
    """Nothing is sent: a 50 MB upload is refused by Supabase's free plan, so
    the message has to say so rather than fail half-way."""
    assert MAX_UPLOAD_BYTES == 45 * 1024 * 1024
    monkeypatch.setattr(backup_store, "MAX_UPLOAD_BYTES", 8)
    big = tmp_path / "big.enc"
    big.write_bytes(b"too many bytes")
    with pytest.raises(StorageError, match="in one upload"):
        supabase.put("production/big.dump.enc", big)
    assert FakeSupabase.seen == []


# ---------------------------------------------------------------------------
# The manifest says nothing about the deals
# ---------------------------------------------------------------------------
def test_the_manifest_holds_no_deal_contents(monkeypatch):
    monkeypatch.setattr(backup, "server_major", lambda url: 17)
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    note = backup.manifest("production", "production/2026-09-18T023000Z",
                           {"encrypted_bytes": 10, "plaintext_bytes": 20, "chunks": 1,
                            "sha256": "f" * 64}, "postgresql://u:p@host/db")
    assert set(note) == {"name", "environment", "created_at", "format", "encrypted_bytes",
                         "plaintext_bytes", "chunks", "sha256", "server_major", "commit"}
    assert "p@host" not in json.dumps(note)


# ---------------------------------------------------------------------------
# A real database: back up, restore elsewhere, same answers
# ---------------------------------------------------------------------------
@pytest.fixture
def sign_in():
    def as_user(subject: str):
        app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)
        assert client.post("/api/account", json=PROFILE).status_code == 200
    yield as_user


@contextmanager
def empty_database(admin_url: str):
    """A second, empty database on the same server, dropped afterwards."""
    name = f"fse_restore_{uuid.uuid4().hex[:10]}"
    admin = create_engine(db_engine.sqlalchemy_url(admin_url), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(db_engine.sqlalchemy_url(admin_url)).set(database=name) \
        .render_as_string(hide_password=False).replace("postgresql+psycopg://", "postgresql://", 1)
    try:
        yield url
    finally:
        db_engine.dispose_engines()
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        admin.dispose()


def irr_now() -> float:
    resp = client.post("/api/deal/run", json={"inputs": INPUTS, "settings": SETTINGS})
    assert resp.status_code == 200, resp.text
    return resp.json()["returns"]["irr"]


def use_database(url: str, monkeypatch):
    """Point the app at ``url`` as if it had started with it."""
    db_engine.dispose_engines()
    monkeypatch.setenv("DATABASE_URL", url)
    db_health.reset()
    migrate.reset_state()


def test_a_restored_backup_gives_back_the_deal_and_its_irr(fresh_db, admin_url, tmp_path,
                                                           sign_in, monkeypatch):
    """The heart of PLAN.md 1.8: a saved deal survives dump, encryption,
    storage, decryption and restore into another database, and the model gives
    exactly the same answer from the restored copy."""
    sign_in("user:backup-owner")
    created = client.post("/api/deals", json={"name": "Project Vault", "inputs": INPUTS,
                                              "settings": SETTINGS})
    assert created.status_code == 201, created.text
    deal_id = created.json()["id"]
    before = irr_now()

    store = LocalStore(tmp_path / "store")
    note = backup.make_backup(fresh_db, "test", store, SECRET, log=lambda msg: None)
    assert note["plaintext_bytes"] > 0
    assert [o.name for o in store.list("test")] == [note["name"] + backup.DUMP_SUFFIX,
                                                    note["name"] + backup.MANIFEST_SUFFIX]

    with empty_database(admin_url) as restored_url:
        plain = tmp_path / "dump"
        backup.fetch(store, note["name"], SECRET, plain, log=lambda msg: None)
        backup.restore(restored_url, plain)
        use_database(restored_url, monkeypatch)

        reopened = client.get(f"/api/deals/{deal_id}")
        assert reopened.status_code == 200, reopened.text
        assert reopened.json()["name"] == "Project Vault"
        assert reopened.json()["inputs"] == INPUTS and reopened.json()["settings"] == SETTINGS
        assert irr_now() == before
        # The least-privilege role (migration 0005) can still read the copy
        with db_engine.connect() as conn:
            assert conn.execute(text(
                "SELECT has_table_privilege('fse_app', 'deals', 'SELECT')")).scalar() is True
            assert conn.execute(text(
                "SELECT has_table_privilege('fse_app', 'deals', 'TRUNCATE')")).scalar() is False
    use_database(fresh_db, monkeypatch)


def test_the_drill_restores_checks_and_cleans_up(fresh_db, admin_url, tmp_path, sign_in):
    """The monthly drill: restore the newest backup into a throwaway database
    on the same server, prove every saved deal gives identical results, drop it."""
    sign_in("user:drill")
    assert client.post("/api/deals", json={"name": "Drill deal", "inputs": INPUTS,
                                           "settings": SETTINGS}).status_code == 201
    store = LocalStore(tmp_path / "store")
    backup.make_backup(fresh_db, "test", store, SECRET, log=lambda msg: None)

    report = backup.drill(store, "test", admin_url, SECRET, source_url=fresh_db,
                          log=lambda msg: None)
    assert report["deals"] == 1 and report["counts"]["deals"] == 1
    assert report["counts"]["migration"] == migrate.head_revision()
    admin = create_engine(db_engine.sqlalchemy_url(admin_url), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        left = conn.execute(text("SELECT count(*) FROM pg_database WHERE datname LIKE 'drill_%'")
                            ).scalar()
    admin.dispose()
    assert left == 0, "the drill left a database behind"


def test_the_drill_notices_a_backup_that_does_not_match(fresh_db, admin_url, tmp_path, sign_in):
    """Proof the comparison means something: change the deal after the backup
    and the drill must refuse it."""
    sign_in("user:drill-mismatch")
    created = client.post("/api/deals", json={"name": "Drill deal", "inputs": INPUTS,
                                              "settings": SETTINGS})
    deal_id = created.json()["id"]
    store = LocalStore(tmp_path / "store")
    backup.make_backup(fresh_db, "test", store, SECRET, log=lambda msg: None)

    moved = {**INPUTS, "exit_mult": 12.0}
    assert client.put(f"/api/deals/{deal_id}/draft",
                      json={"inputs": moved, "settings": SETTINGS}).status_code == 200
    with pytest.raises(backup.BackupError, match="differs from the source"):
        backup.drill(store, "test", admin_url, SECRET, source_url=fresh_db, log=lambda msg: None)


def test_rotation_deletes_old_backups_from_a_real_store(fresh_db, tmp_path):
    """Twelve nights of real backups in a real store: rotation leaves the
    seven the policy keeps, deletes both files of each of the others, and what
    is left still decrypts."""
    store = LocalStore(tmp_path / "store")
    when = datetime(2026, 9, 18, 2, 30, tzinfo=timezone.utc)
    note = backup.make_backup(fresh_db, "test", store, SECRET, now=when, log=lambda msg: None)
    root = tmp_path / "store"
    for days in range(1, 12):                     # the same dump, as if taken each night
        name = backup.backup_name("test", when - timedelta(days=days))
        for suffix in (backup.DUMP_SUFFIX, backup.MANIFEST_SUFFIX):
            (root / (name + suffix)).write_bytes((root / (note["name"] + suffix)).read_bytes())
    assert len(backup.backups(store.list("test"))) == 12

    dropped = backup.rotate(store, "test", log=lambda msg: None)
    left = [b.name for b in backup.backups(store.list("test"))]
    assert left == [backup.backup_name("test", when - timedelta(days=d)) for d in range(7)]
    assert [b.name for b in dropped] ==         [backup.backup_name("test", when - timedelta(days=d)) for d in range(7, 12)]
    names = {o.name for o in store.list("test")}
    for item in dropped:
        assert item.name + backup.DUMP_SUFFIX not in names
        assert item.name + backup.MANIFEST_SUFFIX not in names
    stats = backup.fetch(store, left[-1], SECRET, tmp_path / "oldest", log=lambda msg: None)
    assert stats["sha256"] == note["sha256"]


def test_the_newest_backup_is_found_and_verified(fresh_db, tmp_path):
    store = LocalStore(tmp_path / "store")
    when = datetime(2026, 9, 18, 2, 30, tzinfo=timezone.utc)
    old = backup.make_backup(fresh_db, "test", store, SECRET, now=when - timedelta(days=1),
                             log=lambda msg: None)
    new = backup.make_backup(fresh_db, "test", store, SECRET, now=when, log=lambda msg: None)
    assert backup.latest(store, "test").name == new["name"] != old["name"]
    stats = backup.fetch(store, new["name"], SECRET, tmp_path / "out", log=lambda msg: None)
    assert stats["sha256"] == new["sha256"]
    assert (tmp_path / "out").read_bytes()[:5] == b"PGDMP"               # a real pg_dump archive
    with pytest.raises(backup.BackupError, match="no backup"):
        backup.latest(store, "staging")


def test_a_backup_that_storage_damaged_is_refused(fresh_db, tmp_path):
    """The manifest's checksum catches a file that changed in storage, before
    anything is restored from it."""
    store = LocalStore(tmp_path / "store")
    note = backup.make_backup(fresh_db, "test", store, SECRET, log=lambda msg: None)
    path = tmp_path / "store" / (note["name"] + backup.DUMP_SUFFIX)
    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 0x01
    path.write_bytes(bytes(damaged))
    with pytest.raises((backup.BackupError, encryption.BackupCorrupt)):
        backup.fetch(store, note["name"], SECRET, tmp_path / "out", log=lambda msg: None)
