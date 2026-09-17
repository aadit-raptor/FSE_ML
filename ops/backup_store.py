"""Where encrypted backups are kept (PLAN.md 1.8).

Two stores behind one small interface, chosen by environment variables
(``store_from_env``), so moving to another service in phase 12 is
configuration rather than new code:

- **Supabase Storage** (free: 1 GB) — a **private** bucket, reached with the
  service-role key. This is where the nightly workflow puts backups.
- **A local directory** — for the restore drill on a laptop and for the tests,
  which exercise the same code paths as the real thing.

Not GitHub Actions artifacts: on a public repository anyone can download
them, and a database backup, even encrypted, is not something to hand out.

Standard library only (``urllib``), like the other ops scripts.
"""
from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Protocol

# Supabase's free plan takes 50 MB in one upload (bigger files need resumable
# uploads, phase 12). Stop well before that with a message that says so.
MAX_UPLOAD_BYTES = 45 * 1024 * 1024
DEFAULT_BUCKET = "backups"
HTTP_TIMEOUT_S = 120


class StorageError(RuntimeError):
    """The backup store could not be reached, or refused the request."""


@dataclass(frozen=True)
class StoredObject:
    """One file in the store."""

    name: str          # path inside the bucket, e.g. "production/2026-09-18T023000Z.dump.enc"
    size: int
    modified: Optional[datetime]


class BackupStore(Protocol):
    """What a place to keep backups has to do. Four calls, nothing else."""

    def put(self, name: str, source: Path, content_type: str) -> None: ...

    def get(self, name: str, target: Path) -> None: ...

    def list(self, prefix: str) -> list[StoredObject]: ...

    def delete(self, names: Iterable[str]) -> None: ...

    @property
    def describe(self) -> str: ...


# ---------------------------------------------------------------------------
# A local directory
# ---------------------------------------------------------------------------
class LocalStore:
    """Backups in a directory. Used by the drill on a laptop and by the tests."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def describe(self) -> str:
        return f"directory {self.root}"

    def _path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if not str(path).startswith(str(self.root.resolve())):
            raise StorageError(f"{name} would write outside {self.root}")
        return path

    def put(self, name: str, source: Path, content_type: str = "application/octet-stream") -> None:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, path)

    def get(self, name: str, target: Path) -> None:
        path = self._path(name)
        if not path.exists():
            raise StorageError(f"no backup named {name} in {self.root}")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)

    def list(self, prefix: str) -> list[StoredObject]:
        base = self.root / prefix
        if not base.exists():
            return []
        found = []
        for path in sorted(base.rglob("*")):
            if path.is_file():
                stat = path.stat()
                found.append(StoredObject(
                    name=path.relative_to(self.root).as_posix(), size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime, timezone.utc)))
        return found

    def delete(self, names: Iterable[str]) -> None:
        for name in names:
            self._path(name).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Supabase Storage
# ---------------------------------------------------------------------------
class SupabaseStore:
    """A private Supabase Storage bucket, over its REST API.

    The service-role key is a bearer token: it is never printed, never put in
    a URL and never included in an error message.
    """

    def __init__(self, url: str, key: str, bucket: str = DEFAULT_BUCKET,
                 opener=urllib.request.urlopen):
        if not url or not key:
            raise StorageError("Supabase needs both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        self.url = url.rstrip("/")
        self._key = key
        self.bucket = bucket
        self._opener = opener

    @property
    def describe(self) -> str:
        return f"Supabase bucket {self.bucket} at {self.url}"

    def _request(self, method: str, path: str, *, body: Optional[bytes] = None,
                 content_type: Optional[str] = None, headers: Optional[dict] = None):
        request = urllib.request.Request(f"{self.url}{path}", data=body, method=method)
        request.add_header("Authorization", f"Bearer {self._key}")
        if content_type:
            request.add_header("Content-Type", content_type)
        for header, value in (headers or {}).items():
            request.add_header(header, value)
        try:
            return self._opener(request, timeout=HTTP_TIMEOUT_S)
        except urllib.error.HTTPError as e:      # the message may quote the request; not the key
            detail = e.read(2000).decode("utf-8", "replace")
            raise StorageError(f"Supabase {method} {path} -> {e.code}: {detail}") from None
        except urllib.error.URLError as e:
            raise StorageError(f"could not reach Supabase Storage: {e.reason}") from None

    def put(self, name: str, source: Path, content_type: str = "application/octet-stream") -> None:
        size = Path(source).stat().st_size
        if size > MAX_UPLOAD_BYTES:
            raise StorageError(
                f"{name} is {size / 1024 / 1024:.0f} MB; Supabase's free plan takes "
                f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB in one upload. Split the backup or move "
                "to resumable uploads (PLAN.md 12.2).")
        body = Path(source).read_bytes()
        with self._request("POST", f"/storage/v1/object/{self.bucket}/{name}", body=body,
                           content_type=content_type, headers={"x-upsert": "true"}) as response:
            response.read()

    def get(self, name: str, target: Path) -> None:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        with self._request("GET", f"/storage/v1/object/{self.bucket}/{name}") as response:
            with open(target, "wb") as out:
                shutil.copyfileobj(response, out)

    def list(self, prefix: str) -> list[StoredObject]:
        found: list[StoredObject] = []
        offset = 0
        while True:
            body = json.dumps({"prefix": prefix.rstrip("/"), "limit": 100, "offset": offset,
                               "sortBy": {"column": "name", "order": "asc"}}).encode()
            with self._request("POST", f"/storage/v1/object/list/{self.bucket}", body=body,
                               content_type="application/json") as response:
                page = json.load(response)
            for item in page:
                metadata = item.get("metadata") or {}
                if not metadata:      # a folder placeholder, not a file
                    continue
                found.append(StoredObject(
                    name=f"{prefix.rstrip('/')}/{item['name']}", size=int(metadata.get("size") or 0),
                    modified=_parse_time(item.get("updated_at") or item.get("created_at"))))
            if len(page) < 100:
                return found
            offset += len(page)

    def delete(self, names: Iterable[str]) -> None:
        names = list(names)
        if not names:
            return
        body = json.dumps({"prefixes": names}).encode()
        with self._request("DELETE", f"/storage/v1/object/{self.bucket}", body=body,
                           content_type="application/json") as response:
            response.read()


def _parse_time(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def store_from_env(opener=urllib.request.urlopen) -> BackupStore:
    """The store the environment describes: Supabase when its keys are set,
    otherwise the directory in ``FSE_BACKUP_DIR``."""
    url = (os.environ.get("SUPABASE_URL") or "").strip()
    key = (os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if url and key:
        bucket = (os.environ.get("SUPABASE_BACKUP_BUCKET") or DEFAULT_BUCKET).strip()
        return SupabaseStore(url, key, bucket, opener=opener)
    directory = (os.environ.get("FSE_BACKUP_DIR") or "").strip()
    if directory:
        return LocalStore(Path(directory))
    raise StorageError(
        "no backup store configured: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY "
        "(DEPLOY.md 'Backups and recovery'), or FSE_BACKUP_DIR for a local copy")


def is_configured() -> bool:
    """Whether a store is set up, without building one (used by the workflow)."""
    try:
        store_from_env()
    except StorageError:
        return False
    return True


__all__ = ["BackupStore", "LocalStore", "StorageError", "StoredObject", "SupabaseStore",
           "is_configured", "store_from_env", "MAX_UPLOAD_BYTES"]
