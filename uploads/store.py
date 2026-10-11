"""Where uploaded files are kept: a private bucket only this server can reach.

**Nobody but this server writes to it.** A file arrives at the API, which
streams it to disk and puts it in the store itself (``uploads/service.py``):
no upload link is ever handed to a browser, so nothing can be stored that
the API did not count, and nothing can be swapped after it was scanned. To
download, the owner of a clean file is given a **signed link** that works
for ``rules.LINK_S`` seconds, made here with the store's secret key.

Two stores behind one interface, chosen by the environment (``get_store``):

- **Supabase Storage** (``SUPABASE_UPLOADS_URL`` and ``SUPABASE_UPLOADS_KEY``):
  a private bucket, created and kept private by ``ensure_bucket``, with the
  size limit and the allowed types set on the bucket too.
- **A local directory** (``FSE_UPLOADS_DIR``): for local runs and the tests.
  Its download links point back at this API and are signed with a key made
  at start-up; it is refused in production, like the development sign-in.

Object names are ``u<user id>/<file id>``: nothing personal, nothing from the
file. The key, the links and the file's name are never logged, and a
``StorageError``'s message never contains any of them. Redirects are never
followed: the key travels in headers, and must not follow one elsewhere.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Protocol

from uploads.rules import KINDS, MAX_FILE_BYTES

HTTP_TIMEOUT_S = 30
LOCAL_PATH = "/api/files/local/"
KEY = re.compile(r"^u\d{1,19}/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class StorageError(RuntimeError):
    """The store could not be reached, or refused the request."""


class NotStored(LookupError):
    """No such object (it was never uploaded, or is already deleted)."""


class TooLarge(ValueError):
    """More bytes than the caller agreed to read."""


@dataclass(frozen=True)
class StoredObject:
    key: str
    size: int


class FileStore(Protocol):
    name: str

    def put(self, key: str, source: Path, content_type: str) -> None: ...

    def download_url(self, key: str, filename: str, expires_s: int) -> str: ...

    def size(self, key: str) -> Optional[int]: ...

    def fetch(self, key: str, target: Path, limit: int) -> int: ...

    def delete(self, keys: Iterable[str]) -> None: ...

    def objects(self) -> list[StoredObject]: ...


def object_key(owner_id: int, file_id) -> str:
    return f"u{int(owner_id)}/{file_id}"


def _checked(key: str) -> str:
    if not isinstance(key, str) or not KEY.fullmatch(key):
        raise StorageError("not an upload's object name")
    return key


def _copy(source, target: Path, limit: int) -> int:
    """Copy at most ``limit`` bytes into ``target``; ``TooLarge`` past that."""
    copied = 0
    with open(target, "wb") as out:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                return copied
            copied += len(chunk)
            if copied > limit:
                raise TooLarge("more than was agreed")
            out.write(chunk)


# ---------------------------------------------------------------------------
# Supabase Storage
# ---------------------------------------------------------------------------
class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None       # the 3xx then surfaces as an HTTPError, and so as a StorageError


_open = urllib.request.build_opener(_NoRedirects).open


class SupabaseFiles:
    """A private Supabase Storage bucket, over its REST API (standard library
    only, as ops/backup_store.py). The key goes in the ``Authorization`` and
    ``apikey`` headers, never in an address or an error."""

    name = "supabase"

    def __init__(self, url: str, key: str, bucket: str, opener=_open):
        if not url.startswith("https://"):
            raise StorageError("SUPABASE_UPLOADS_URL must be an https address")
        self.url = url.rstrip("/")
        self.origin = "https://" + urllib.parse.urlsplit(self.url).netloc
        self._key = key
        self.bucket = bucket
        self._opener = opener
        self._ensured = False
        self._lock = threading.Lock()

    def _request(self, method: str, path: str, body: Optional[dict] = None, *, data=None,
                 headers: Optional[dict] = None):
        request = urllib.request.Request(f"{self.url}/storage/v1{path}", method=method,
                                         data=data if body is None else json.dumps(body).encode())
        request.add_header("Authorization", f"Bearer {self._key}")
        request.add_header("apikey", self._key)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        for name, value in (headers or {}).items():
            request.add_header(name, value)
        try:
            return self._opener(request, timeout=HTTP_TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            exc.close()
            if exc.code in (400, 404):
                raise NotStored(f"storage answered {exc.code}") from None
            raise StorageError(f"storage answered {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise StorageError("storage could not be reached") from None

    def _json(self, method: str, path: str, body: Optional[dict] = None):
        with self._request(method, path, body) as response:
            try:
                return json.load(response)
            except ValueError:
                raise StorageError("storage answered something unreadable") from None

    @staticmethod
    def _signed(answer, field: str) -> str:
        """The signed path in an answer: ours to trust only if it is a path."""
        path = answer.get(field) if isinstance(answer, dict) else None
        if not isinstance(path, str) or not path.startswith("/object/") or "token=" not in path:
            raise StorageError("storage answered without a signed link")
        return path

    def ensure_bucket(self) -> None:
        """Make the bucket if it is missing, and put it right if it is public
        or its limits differ. Once per process."""
        with self._lock:
            if self._ensured:
                return
            wanted = {"public": False, "file_size_limit": MAX_FILE_BYTES,
                      "allowed_mime_types": sorted(k.content_type for k in KINDS.values())}
            try:
                found = self._json("GET", f"/bucket/{self.bucket}")
            except NotStored:
                found = None
            try:
                if found is None:
                    self._json("POST", "/bucket", {"id": self.bucket, "name": self.bucket, **wanted})
                elif any(found.get(k) != v for k, v in wanted.items()
                         if k != "allowed_mime_types") or \
                        sorted(found.get("allowed_mime_types") or []) != wanted["allowed_mime_types"]:
                    self._json("PUT", f"/bucket/{self.bucket}", wanted)
            except NotStored:
                raise StorageError("the uploads bucket could not be made") from None
            self._ensured = True

    def put(self, key: str, source: Path, content_type: str) -> None:
        """Store the file at ``source``. Never over an object already there:
        a name is used once."""
        self.ensure_bucket()
        size = Path(source).stat().st_size
        with open(source, "rb") as data:
            try:
                with self._request("POST", f"/object/{self.bucket}/{_checked(key)}", data=data, headers={
                        "Content-Type": content_type, "Content-Length": str(size),
                        "x-upsert": "false"}) as response:
                    response.read()
            except NotStored:
                raise StorageError("the file could not be stored") from None

    def download_url(self, key: str, filename: str, expires_s: int) -> str:
        signed = self._json("POST", f"/object/sign/{self.bucket}/{_checked(key)}",
                            {"expiresIn": int(expires_s)})
        return (f"{self.url}/storage/v1{self._signed(signed, 'signedURL')}"
                f"&download={urllib.parse.quote(filename, safe='')}")

    def _list(self, prefix: str) -> list[dict]:
        found, offset = [], 0
        while True:
            page = self._json("POST", f"/object/list/{self.bucket}", {
                "prefix": prefix, "limit": 100, "offset": offset,
                "sortBy": {"column": "name", "order": "asc"}})
            found += page
            if len(page) < 100:
                return found
            offset += len(page)

    def size(self, key: str) -> Optional[int]:
        folder, name = _checked(key).split("/")
        for item in self._list(folder):
            if item.get("name") == name and item.get("metadata"):
                return int(item["metadata"].get("size") or 0)
        return None

    def fetch(self, key: str, target: Path, limit: int) -> int:
        with self._request("GET", f"/object/{self.bucket}/{_checked(key)}") as response:
            return _copy(response, target, limit)

    def delete(self, keys: Iterable[str]) -> None:
        keys = [_checked(k) for k in keys]
        for i in range(0, len(keys), 100):
            try:
                with self._request("DELETE", f"/object/{self.bucket}",
                                   {"prefixes": keys[i:i + 100]}) as response:
                    response.read()
            except NotStored:
                pass

    def objects(self) -> list[StoredObject]:
        self.ensure_bucket()
        found = []
        for folder in self._list(""):
            if folder.get("metadata") or not re.fullmatch(r"u\d{1,19}", folder.get("name") or ""):
                continue
            for item in self._list(folder["name"]):
                if item.get("metadata"):
                    found.append(StoredObject(f"{folder['name']}/{item['name']}",
                                              int(item["metadata"].get("size") or 0)))
        return found


# ---------------------------------------------------------------------------
# A local directory (local runs and tests)
# ---------------------------------------------------------------------------
class BadToken(ValueError):
    """A local link that is not ours, was changed, or is past its time."""


class LocalFiles:
    """Files in a directory, with links that point back at this API
    (``api/routers/files.py`` serves them while this store is in use)."""

    name = "local"

    def __init__(self, root: Path, secret: Optional[bytes] = None):
        self.root = Path(root)
        self._secret = secret or secrets.token_bytes(32)

    def _path(self, key: str) -> Path:
        return self.root / _checked(key)

    def _token(self, payload: dict) -> str:
        body = base64.urlsafe_b64encode(json.dumps(payload, sort_keys=True).encode()).rstrip(b"=")
        mark = hmac.new(self._secret, body, hashlib.sha256).hexdigest()
        return f"{body.decode()}.{mark}"

    def open_token(self, token: str, operation: str, *, now: Optional[float] = None) -> dict:
        body, _, mark = token.partition(".")
        expected = hmac.new(self._secret, body.encode("utf-8", "replace"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(mark.encode("utf-8", "replace"), expected.encode()):
            raise BadToken(operation)
        try:
            payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        except (ValueError, TypeError):
            raise BadToken(operation) from None
        if not isinstance(payload, dict) or payload.get("op") != operation \
                or not isinstance(payload.get("exp"), (int, float)) \
                or payload["exp"] < (time.time() if now is None else now):
            raise BadToken(operation)
        _checked(payload.get("key", ""))
        return payload

    def download_url(self, key: str, filename: str, expires_s: int) -> str:
        return LOCAL_PATH + self._token({"op": "get", "key": _checked(key), "name": filename,
                                         "exp": time.time() + expires_s})

    def put(self, key: str, source: Path, content_type: str = "") -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(source, "rb") as data, open(path, "xb") as out:
                shutil.copyfileobj(data, out)
        except FileExistsError:
            raise StorageError("the file could not be stored") from None

    def size(self, key: str) -> Optional[int]:
        path = self._path(key)
        return path.stat().st_size if path.is_file() else None

    def fetch(self, key: str, target: Path, limit: int) -> int:
        path = self._path(key)
        if not path.is_file():
            raise NotStored("no such object")
        with open(path, "rb") as source:
            return _copy(source, target, limit)

    def delete(self, keys: Iterable[str]) -> None:
        for key in keys:
            self._path(key).unlink(missing_ok=True)

    def objects(self) -> list[StoredObject]:
        if not self.root.is_dir():
            return []
        return [StoredObject(p.relative_to(self.root).as_posix(), p.stat().st_size)
                for p in sorted(self.root.glob("u*/*")) if p.is_file()]

    def clear(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Which store
# ---------------------------------------------------------------------------
_store: list = [None, None]      # the environment it was built from, the store
_store_lock = threading.Lock()


def _environment() -> str:
    from api.observability import deploy_environment

    return deploy_environment()


def default_bucket(environment: str) -> str:
    return "uploads" if environment == "production" else f"uploads-{environment}"


def _describe(environ) -> tuple:
    return tuple((environ.get(name) or "").strip() for name in (
        "SUPABASE_UPLOADS_URL", "SUPABASE_UPLOADS_KEY", "SUPABASE_UPLOADS_BUCKET", "FSE_UPLOADS_DIR"))


def get_store() -> Optional[FileStore]:
    """This process's store, or None when uploads are not set up."""
    wanted = (*_describe(os.environ), _environment())
    with _store_lock:
        if _store[0] != wanted:
            _store[0], _store[1] = wanted, _build(*wanted)
        return _store[1]


def _build(url: str, key: str, bucket: str, directory: str, environment: str) -> Optional[FileStore]:
    if url and key:
        return SupabaseFiles(url, key, bucket or default_bucket(environment))
    if directory and environment != "production":
        return LocalFiles(Path(directory))
    return None


def local_store() -> Optional[LocalFiles]:
    store = get_store()
    return store if isinstance(store, LocalFiles) else None


def reset() -> None:
    with _store_lock:
        _store[0] = _store[1] = None


__all__ = ["BadToken", "FileStore", "LOCAL_PATH", "LocalFiles", "NotStored", "StorageError", "StoredObject",
           "SupabaseFiles", "TooLarge", "default_bucket", "get_store", "local_store",
           "object_key", "reset"]
