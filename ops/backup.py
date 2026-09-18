"""Database backups, restore and the restore drill (PLAN.md 1.8).

    python -m ops.backup run     --environment production        # dump, encrypt, upload, rotate
    python -m ops.backup list    --environment production
    python -m ops.backup verify  --environment production        # download the newest and decrypt it
    python -m ops.backup restore --name production/2026-09-18T023000Z --into "$URL"
    python -m ops.backup drill   --environment production --target "$STAGING_OWNER_URL"

**What a backup is:** ``pg_dump -Fc`` (the compressed custom format, so single
tables can be restored too), encrypted with AES-256-GCM (``ops/encryption.py``)
and put in a private Supabase Storage bucket (``ops/backup_store.py``) as
``<environment>/<when>.dump.enc``, beside a small ``.json`` manifest — sizes,
checksum, versions, commit, and what the dump held: the migration revision,
how many deals, and a one-way fingerprint of their model results, all read
inside the snapshot ``pg_dump`` itself reads. **Never anything from a deal.**

**Recent mistakes don't need this.** Neon keeps a restore window on the free
plan: restoring a branch to a point in time in the console is faster and
loses nothing. These backups are for what that can't fix — the Neon project
itself gone, or a bad change noticed weeks later. DEPLOY.md "Backups and
recovery" is the procedure.

**Rotation** keeps the last 7 days, one a week for 8 weeks and one a month
for 12 months, and then drops the oldest until the total fits the storage
budget, so the free 1 GB is never reached.

**The drill** (monthly, and after any change here) restores the newest backup
into a throwaway database on the staging branch, checks that every saved deal
produces exactly the model result the manifest recorded when the dump was
taken, and drops it again. It compares with the backup, not with today's live
database — deals change every day — and needs no production credentials. It
prints counts and whether they match, never a deal's contents.

The tools (``pg_dump``/``pg_restore``) are found automatically and must be at
least the server's major version; passwords go to them in the environment,
never on the command line.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence
from urllib.parse import unquote, urlsplit, urlunsplit

from ops import encryption
from ops.backup_store import BackupStore, StorageError, StoredObject, store_from_env

DUMP_SUFFIX = ".dump.enc"
MANIFEST_SUFFIX = ".json"
NAME_FORMAT = "%Y-%m-%dT%H%M%SZ"
NAME_RE = re.compile(r"^(?P<environment>[a-z0-9_-]+)/(?P<when>\d{4}-\d{2}-\d{2}T\d{6}Z)$")

# Supabase's free plan gives 1 GB. Stop at 700 MB so a backup never fails for
# want of room and there is space for the uploads PLAN.md 6.1 will add.
STORAGE_BUDGET_BYTES = 700 * 1024 * 1024
KEEP_DAILY = 7
KEEP_WEEKLY = 8
KEEP_MONTHLY = 12


class BackupError(RuntimeError):
    """The backup or restore could not be completed."""


# ---------------------------------------------------------------------------
# Postgres tools
# ---------------------------------------------------------------------------
def split_password(url: str) -> tuple[str, Optional[str]]:
    """``url`` without its password, and the password.

    The password goes to ``pg_dump`` in ``PGPASSWORD``: anything on the
    command line is visible to every other process on the machine.
    """
    parts = urlsplit(url)
    userinfo, at, host = parts.netloc.rpartition("@")
    if not at or ":" not in userinfo:
        return url, None
    user, _, password = userinfo.partition(":")
    netloc = f"{user}@{host}" if user else host
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment)), \
        unquote(password)


def _tool_env(url: str) -> tuple[str, dict]:
    safe, password = split_password(url)
    env = dict(os.environ)
    if password is not None:
        env["PGPASSWORD"] = password
    return safe, env


def server_major(url: str) -> int:
    """The server's major version (18 for Postgres 18.1)."""
    import psycopg

    try:
        with psycopg.connect(url, connect_timeout=30) as conn:
            number = int(conn.execute("SHOW server_version_num").fetchone()[0])
    except Exception as e:                      # the message can quote the URL; it never does here
        raise BackupError(f"could not read the server version: {type(e).__name__}") from None
    return number // 10000


def tool_candidates(name: str) -> list[str]:
    """Where ``pg_dump``/``pg_restore`` might be, newest first.

    A packaged Postgres (``pgserver``, used by ``python -m db.local``) and the
    per-version directories a Debian/Ubuntu runner installs into are searched
    as well as ``PATH``, so neither a laptop nor CI needs anything set.
    """
    exe = f"{name}.exe" if platform.system() == "Windows" else name
    found: list[str] = []
    chosen = (os.environ.get("FSE_PG_BIN") or "").strip()
    if chosen:
        found.append(str(Path(chosen) / exe))
    on_path = shutil.which(name)
    if on_path:
        found.append(on_path)
    versioned = sorted(glob.glob(f"/usr/lib/postgresql/*/bin/{name}"),
                       key=lambda p: int(re.search(r"postgresql/(\d+)/", p).group(1)), reverse=True)
    found.extend(versioned)
    try:
        import pgserver

        found.append(str(Path(pgserver.__file__).parent / "pginstall" / "bin" / exe))
    except ImportError:
        pass
    return [f for f in dict.fromkeys(found) if Path(f).exists()]


def tool_version(path: str) -> Optional[int]:
    """The major version a ``pg_dump``/``pg_restore`` binary reports."""
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60)
    except OSError:
        return None
    match = re.search(r"\s(\d+)[.\s]", out.stdout or "")
    return int(match.group(1)) if match else None


def pg_tool(name: str, needed_major: int) -> str:
    """A ``name`` binary at least as new as the server.

    An older ``pg_dump`` refuses a newer server outright, and an older
    ``pg_restore`` can't read what a newer one wrote, so this fails early
    with a message that says what to install.
    """
    candidates = tool_candidates(name)
    for path in candidates:
        version = tool_version(path)
        if version is not None and version >= needed_major:
            return path
    have = ", ".join(f"{p} ({tool_version(p)})" for p in candidates) or "none found"
    raise BackupError(
        f"need {name} {needed_major} or newer for this server; found: {have}. "
        f"Install the matching client (Ubuntu: postgresql-client-{needed_major}) "
        f"or set FSE_PG_BIN to its bin directory.")


def dump(url: str, target: Path, *, snapshot: Optional[str] = None) -> None:
    """``pg_dump -Fc`` of the whole database into ``target``.

    Ownership is left out (``--no-owner``) so a restore works as whatever role
    is doing the restoring. Grants are kept in the file — they cost nothing
    and record what the rights were — but ``restore`` leaves them out when
    replaying, because a managed Postgres mixes its own platform grants in
    with ours. ``snapshot`` makes it read an already-exported snapshot
    instead of taking its own (``dump_with_checks``).
    """
    safe, env = _tool_env(url)
    tool = pg_tool("pg_dump", server_major(url))
    command = [tool, "--format=custom", "--compress=9", "--no-owner", "--file", str(target)]
    if snapshot:
        command.append(f"--snapshot={snapshot}")
    result = subprocess.run(command + [safe], env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise BackupError(f"pg_dump failed: {(result.stderr or '').strip()[:500]}")


def dump_with_checks(url: str, target: Path, *, log=print) -> dict:
    """Dump the database and describe what went into the dump.

    The drill has to answer "did this backup restore what it held?", not "does
    it match the live database?" — deals change every day, so comparing a
    month-old backup with today's data would cry wolf every time. So the
    counts and the fingerprint are read **inside the same snapshot pg_dump
    reads** (``pg_export_snapshot``), and travel in the manifest.

    A connection that can't share a snapshot (a pooler in the way) falls back
    to a plain dump and a read straight after it, and says so.
    """
    import psycopg

    with psycopg.connect(url, connect_timeout=30) as conn:
        conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        snapshot = conn.execute("SELECT pg_export_snapshot()").fetchone()[0]
        shared = True
        try:
            dump(url, target, snapshot=snapshot)
        except BackupError as e:
            if "snapshot" not in str(e).lower():
                raise
            log(f"::warning::this connection cannot share a snapshot with pg_dump "
                f"({str(e)[:120]}); dumping without one")
            shared = False
            conn.rollback()
            dump(url, target)
        counts, rows = _deal_rows(conn)
        conn.rollback()
    return {"snapshot": shared, "deals": len(rows), "migration": counts["migration"],
            "fingerprint": _fingerprint(rows)}


def restore(url: str, source: Path, *, clean: bool = False, with_grants: bool = False) -> None:
    """Restore a decrypted dump into the database ``url`` points at.

    All or nothing (``--single-transaction --exit-on-error``): a restore that
    hits a problem leaves the target as it was rather than half-filled.

    **Grants are left out by default** (``--no-privileges``). A dump from a
    managed Postgres carries the platform's own grants, and the role doing
    the restore is not allowed to replay them: on Neon the dump contains
    ``ALTER DEFAULT PRIVILEGES FOR ROLE cloud_admin … TO neon_superuser``,
    which only Neon's superuser may run, and pg_dump puts it in the same
    entry as our grants, so it can't be filtered out. Keeping privileges
    therefore aborts the whole restore. The app's own rights are re-applied
    afterwards from migration 0005 — DEPLOY.md "Restoring" has the step.
    ``with_grants`` keeps them, for a dump from a cluster you own outright.
    """
    safe, env = _tool_env(url)
    tool = pg_tool("pg_restore", server_major(url))
    command = [tool, "--dbname", safe, "--no-owner", "--single-transaction", "--exit-on-error"]
    if clean:
        command += ["--clean", "--if-exists"]
    if not with_grants:
        command += ["--no-privileges"]
    result = subprocess.run(command + [str(source)], env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise BackupError(f"pg_restore failed: {(result.stderr or '').strip()[:1000]}")


# ---------------------------------------------------------------------------
# Names, manifests and rotation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BackupFile:
    """One backup in the store: its name without suffix, when it was taken and
    how much room it and its manifest use."""

    name: str
    taken_at: datetime
    size: int


def backup_name(environment: str, when: datetime) -> str:
    if not re.fullmatch(r"[a-z0-9_-]+", environment or ""):
        raise BackupError("environment must be a short name such as production or staging")
    return f"{environment}/{when.astimezone(timezone.utc).strftime(NAME_FORMAT)}"


def taken_at(name: str) -> Optional[datetime]:
    match = NAME_RE.match(name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group("when"), NAME_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:                        # the right shape, not a real time
        return None


def backups(objects: Iterable[StoredObject]) -> list[BackupFile]:
    """The backups among stored objects, newest first, sized with their manifests."""
    sizes: dict[str, int] = {}
    times: dict[str, datetime] = {}
    for item in objects:
        for suffix in (DUMP_SUFFIX, MANIFEST_SUFFIX):
            if item.name.endswith(suffix):
                stem = item.name[: -len(suffix)]
                when = taken_at(stem)
                if when is None:
                    continue
                sizes[stem] = sizes.get(stem, 0) + item.size
                times[stem] = when
                break
    return sorted((BackupFile(name, times[name], sizes[name]) for name in sizes),
                  key=lambda b: b.taken_at, reverse=True)


def rotation_plan(files: Sequence[BackupFile], *, keep_daily: int = KEEP_DAILY,
                  keep_weekly: int = KEEP_WEEKLY, keep_monthly: int = KEEP_MONTHLY,
                  budget_bytes: int = STORAGE_BUDGET_BYTES) -> tuple[list[BackupFile], list[BackupFile]]:
    """Which backups to keep and which to delete (newest first in both lists).

    Daily, then weekly, then monthly, so recent mistakes have fine-grained
    cover and old ones still have some. Whatever survives that is then trimmed
    oldest-first until it fits ``budget_bytes``; the newest backup is never
    dropped, however large it is.
    """
    ordered = sorted(files, key=lambda b: b.taken_at, reverse=True)
    keep: set[str] = set()
    for count, period in ((keep_daily, lambda d: d.date()),
                          (keep_weekly, lambda d: d.isocalendar()[:2]),
                          (keep_monthly, lambda d: (d.year, d.month))):
        seen: list = []
        for item in ordered:                      # newest first: the first in a period wins
            key = period(item.taken_at)
            if key in seen:
                continue
            if len(seen) >= count:
                break
            seen.append(key)
            keep.add(item.name)

    kept = [b for b in ordered if b.name in keep]
    dropped = [b for b in ordered if b.name not in keep]
    total = sum(b.size for b in kept)
    while total > budget_bytes and len(kept) > 1:
        oldest = kept.pop()                       # never the newest: the list is newest first
        total -= oldest.size
        dropped.append(oldest)
    return kept, sorted(dropped, key=lambda b: b.taken_at, reverse=True)


def manifest(environment: str, name: str, stats: dict, checks: dict, server: int) -> dict:
    """What is stored, unencrypted, beside the backup.

    Sizes, a checksum, versions, and what the dump held: the migration
    revision, how many deals, and the one-way fingerprint of their model
    results the drill checks the restored copy against. **No deal contents**,
    and nothing a fingerprint could be turned back into.
    """
    return {
        "name": name,
        "environment": environment,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "format": f"pg_dump custom, AES-256-GCM v{encryption.VERSION}",
        "encrypted_bytes": stats["encrypted_bytes"],
        "plaintext_bytes": stats["plaintext_bytes"],
        "chunks": stats["chunks"],
        "sha256": stats["sha256"],
        "server_major": server,
        "commit": (os.environ.get("GITHUB_SHA") or "")[:40] or None,
        "migration": checks["migration"],
        "deals": checks["deals"],
        "fingerprint": checks["fingerprint"],
        "one_snapshot": checks["snapshot"],
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
def make_backup(url: str, environment: str, store: BackupStore, secret: str,
                *, now: Optional[datetime] = None, log=print) -> dict:
    """Dump, encrypt, upload and rotate. Returns the manifest."""
    name = backup_name(environment, now or datetime.now(timezone.utc))
    server = server_major(url)
    with tempfile.TemporaryDirectory(prefix="fse-backup-") as work:
        plain = Path(work) / "dump"
        sealed = Path(work) / "dump.enc"
        checks = dump_with_checks(url, plain, log=log)
        with open(plain, "rb") as source, open(sealed, "wb") as target:
            stats = encryption.encrypt_stream(source, target, secret)
        plain.unlink()
        note = manifest(environment, name, stats, checks, server)
        (Path(work) / "manifest.json").write_text(json.dumps(note, indent=2), encoding="utf-8")
        store.put(name + DUMP_SUFFIX, sealed)
        store.put(name + MANIFEST_SUFFIX, Path(work) / "manifest.json", "application/json")
    log(f"backed up {environment}: {stats['plaintext_bytes'] / 1024 / 1024:.1f} MB dumped "
        f"({checks['deals']} deals at migration {checks['migration']}), "
        f"{stats['encrypted_bytes'] / 1024 / 1024:.1f} MB encrypted, sha256 {stats['sha256'][:16]}... "
        f"-> {name}{DUMP_SUFFIX} in {store.describe}")
    return note


def rotate(store: BackupStore, environment: str, *, budget_bytes: int = STORAGE_BUDGET_BYTES,
           log=print) -> list[BackupFile]:
    """Delete the backups the policy no longer keeps. Returns what went."""
    kept, dropped = rotation_plan(backups(store.list(environment)), budget_bytes=budget_bytes)
    for item in dropped:
        store.delete([item.name + DUMP_SUFFIX, item.name + MANIFEST_SUFFIX])
    total = sum(b.size for b in kept)
    log(f"rotation: {len(kept)} backups kept ({total / 1024 / 1024:.1f} MB of the "
        f"{budget_bytes / 1024 / 1024:.0f} MB budget), {len(dropped)} deleted"
        + (": " + ", ".join(b.name for b in dropped) if dropped else ""))
    return dropped


def latest(store: BackupStore, environment: str) -> BackupFile:
    found = backups(store.list(environment))
    if not found:
        raise BackupError(f"no backup for {environment} in {store.describe}")
    return found[0]


def fetch(store: BackupStore, name: str, secret: str, target: Path, *, log=print) -> dict:
    """Download a backup, check it against its manifest and decrypt it.

    Returns what was read, with the manifest under ``manifest``.
    """
    with tempfile.TemporaryDirectory(prefix="fse-restore-") as work:
        sealed = Path(work) / "dump.enc"
        store.get(name + DUMP_SUFFIX, sealed)
        note = {}
        try:
            note_path = Path(work) / "manifest.json"
            store.get(name + MANIFEST_SUFFIX, note_path)
            note = json.loads(note_path.read_text(encoding="utf-8"))
        except (StorageError, json.JSONDecodeError):
            log(f"::warning::{name} has no readable manifest; decrypting anyway")
        with open(sealed, "rb") as source, open(target, "wb") as out:
            stats = encryption.decrypt_stream(source, out, secret)
    if note.get("sha256") and note["sha256"] != stats["sha256"]:
        raise BackupError(f"{name} does not match its manifest checksum: storage returned "
                          f"{stats['sha256'][:16]}..., the manifest says {note['sha256'][:16]}...")
    log(f"{name}: {stats['encrypted_bytes'] / 1024 / 1024:.1f} MB downloaded, decrypted to "
        f"{stats['plaintext_bytes'] / 1024 / 1024:.1f} MB"
        + (", checksum matches the manifest" if note.get("sha256") else ""))
    return {**stats, "manifest": note}


# ---------------------------------------------------------------------------
# The restore drill
# ---------------------------------------------------------------------------
def _deal_rows(conn) -> tuple[dict, list]:
    """The migration revision, the table counts and every deal's stored inputs.

    Read in one transaction and handed back at once, so the caller can close
    the transaction before spending time on the model.
    """
    counts = {"migration": conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]}
    for table in ("users", "deals", "deal_versions"):
        counts[table] = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return counts, conn.execute("SELECT id, inputs, settings FROM deals ORDER BY id").fetchall()


def _fingerprint(rows) -> str:
    """One checksum over every deal's model result.

    The real deal model is run on each deal's stored inputs and settings;
    identical fingerprints mean identical answers. It is one-way, so it can
    travel in the manifest beside the backup without carrying a deal with it.
    """
    from core.config import resolve_config
    from core.deal import DealInputs, run_deal

    digest = hashlib.sha256()
    for deal_id, inputs, settings in rows:
        result = run_deal(DealInputs(**inputs), resolve_config(settings or {}))
        digest.update(f"{deal_id}:{result.returns.irr:.12g}:{result.returns.moic:.12g}\n".encode())
    return digest.hexdigest()


def deal_results(url: str) -> dict:
    """What one database holds: counts, the migration revision and the
    fingerprint of every saved deal's model result."""
    import psycopg

    with psycopg.connect(url, connect_timeout=30) as conn:
        counts, rows = _deal_rows(conn)
    return {"counts": counts, "deals": len(rows), "fingerprint": _fingerprint(rows)}


def drill(store: BackupStore, environment: str, target_url: str, secret: str, *,
          name: Optional[str] = None, log=print) -> dict:
    """Restore a backup into a throwaway database and check what came back.

    The target is a **new** database on the server ``target_url`` points at
    (the staging branch), created and dropped here, so a drill never touches
    data anyone is using. What the restored copy holds is compared with what
    the manifest recorded **when the dump was taken** — not with the live
    database, which has moved on since.
    """
    import psycopg

    from db.engine import direct_url

    name = name or latest(store, environment).name
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    database = f"drill_{stamp}"
    admin = direct_url(target_url)

    with tempfile.TemporaryDirectory(prefix="fse-drill-") as work:
        plain = Path(work) / "dump"
        note = fetch(store, name, secret, plain, log=log)["manifest"]
        with psycopg.connect(admin, connect_timeout=30, autocommit=True) as conn:
            conn.execute(f'CREATE DATABASE "{database}"')
        try:
            restored_url = _with_database(admin, database)
            restore(restored_url, plain)
            found = deal_results(restored_url)
        finally:
            if not database.startswith("drill_"):     # belt and braces: only ever a drill copy
                raise BackupError(f"refusing to drop {database}")
            with psycopg.connect(admin, connect_timeout=30, autocommit=True) as conn:
                conn.execute(f'DROP DATABASE IF EXISTS "{database}" WITH (FORCE)')

    log(f"restored {name} into {database}: {found['counts']} "
        f"({found['deals']} deals re-run, fingerprint {found['fingerprint'][:16]}...), then dropped it")
    expected = {key: note.get(key) for key in ("deals", "migration", "fingerprint")}
    if expected["fingerprint"] is None:
        log(f"::warning::{name} records no fingerprint, so there is nothing to compare the "
            "restored copy with; it restored and every deal re-ran")
    elif (expected["fingerprint"] != found["fingerprint"] or expected["deals"] != found["deals"]
          or expected["migration"] != found["counts"]["migration"]):
        difference = (
            f"{found['deals']} deals at migration {found['counts']['migration']} (fingerprint "
            f"{found['fingerprint'][:16]}...) against the manifest's {expected['deals']} at "
            f"{expected['migration']} ({str(expected['fingerprint'])[:16]}...)")
        if note.get("one_snapshot") is False:
            # The manifest was read after the dump rather than from its snapshot,
            # so a deal saved in between explains this without anything being wrong
            log(f"::warning::{name} restored, but it does not match its manifest: {difference}. "
                "That backup was taken without a shared snapshot, so a deal saved while it ran "
                "would look like this.")
        else:
            raise BackupError(f"the restored copy is not what was backed up: {difference}")
    else:
        log(f"every one of the {found['deals']} saved deals gives the same model result as when "
            "the backup was taken")
    return {"name": name, "database": database, **found}


def _with_database(url: str, database: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, f"/{database}", parts.query, parts.fragment))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def _url(args, *names: str) -> str:
    for value in (args.database_url, *(os.environ.get(n) for n in names)):
        if value and value.strip():
            return value.strip()
    raise BackupError(f"set {' or '.join(names)} (or pass --database-url)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--database-url", help="the database to back up (default: BACKUP_DATABASE_URL)")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "list", "verify"):
        p = sub.add_parser(command)
        p.add_argument("--environment", required=True)
    sub.choices["run"].add_argument("--no-rotate", action="store_true")
    sub.choices["verify"].add_argument("--name")
    p = sub.add_parser("restore")
    p.add_argument("--name", required=True)
    p.add_argument("--into", required=True, help="the database URL to restore into")
    p.add_argument("--clean", action="store_true", help="replace what is already there")
    p.add_argument("--with-grants", action="store_true",
                   help="restore the source's GRANTs too (only works from a cluster you own "
                        "outright; a managed one's platform grants can't be replayed)")
    p = sub.add_parser("drill")
    p.add_argument("--environment", required=True, help="whose backup to restore")
    p.add_argument("--target", help="a database URL on the server to restore into "
                                    "(default: BACKUP_STAGING_DATABASE_URL)")
    p.add_argument("--name", help="a particular backup (default: the newest)")
    args = parser.parse_args(argv)

    try:
        store = store_from_env()
        if args.command == "list":
            found = backups(store.list(args.environment))
            kept, dropped = rotation_plan(found)
            names = {b.name for b in kept}
            for item in found:
                print(f"{item.name}  {item.size / 1024 / 1024:8.2f} MB  "
                      f"{'keep' if item.name in names else 'rotate out'}")
            print(f"{len(found)} backups in {store.describe}, "
                  f"{sum(b.size for b in found) / 1024 / 1024:.1f} MB")
            return 0

        secret = encryption.secret_from_env()
        if args.command == "run":
            make_backup(_url(args, "BACKUP_DATABASE_URL", "DATABASE_MIGRATION_URL", "DATABASE_URL"),
                        args.environment, store, secret)
            if not args.no_rotate:
                rotate(store, args.environment)
        elif args.command == "verify":
            name = args.name or latest(store, args.environment).name
            with tempfile.TemporaryDirectory(prefix="fse-verify-") as work:
                fetch(store, name, secret, Path(work) / "dump")
        elif args.command == "restore":
            with tempfile.TemporaryDirectory(prefix="fse-restore-") as work:
                plain = Path(work) / "dump"
                fetch(store, args.name, secret, plain)
                restore(args.into, plain, clean=args.clean, with_grants=args.with_grants)
            print(f"restored {args.name}")
        else:
            target = args.target or (os.environ.get("BACKUP_STAGING_DATABASE_URL") or "").strip()
            if not target:
                raise BackupError("set BACKUP_STAGING_DATABASE_URL (or pass --target): the "
                                  "database server the drill restores into")
            drill(store, args.environment, target, secret, name=args.name)
    except (BackupError, StorageError, encryption.BackupKeyError, encryption.BackupCorrupt) as e:
        print(f"::error::{e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
