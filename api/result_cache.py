"""Cached model results (PLAN.md 8.1).

A seeded Monte Carlo run asked again with the same inputs answers what it
answered before, so the second time the answer comes from a cache instead of
the simulation: in milliseconds, without waiting for the one simulation slot.

**What a result is cached by.** The kind of run, the caller, the whole request
(its seed included) and the model stamp (``core/model_version.py``): engine
version, the commit the API was built from, the fingerprint of the resolved
Settings and of the published data. A new engine version, a new release or a
new data edition therefore misses; nothing has to be cleared. A run without a
seed draws new paths each time and is never cached, nor is a system caller's
(``system:drill`` exists to load the server).

**Nothing readable leaves the process.** Both the store's key and the key the
answer is encrypted with (AES-256-GCM) are HMAC-SHA-256 of that material under
different labels, keyed with a secret only the server holds (``_pepper``), and
the material itself is kept nowhere. Upstash and the table hold a hash and
ciphertext only: reading an entry takes the exact inputs, the caller's user id
and the server's secret, so one account can't read another's results, a copy
of a store can't be tested against guessed inputs, and no key or value can
tell a log anything. Without a known commit (locally), the build in the key is
a per-process token, so a code change can never be answered from an old entry.

**Where it lives.** First a small cache in this process (``MEMORY_BUDGET_BYTES``),
which costs nothing; then Upstash Redis while it answers and is within its
budget; otherwise the ``result_cache`` table (db/result_cache.py), held to
``DATABASE_BUDGET_BYTES``. A store that fails is skipped for
``STORE_RETRY_S``; a cache failure never fails a run.

**Budget.** Upstash's free plan allows 500,000 commands and 10 GB of traffic a
month and holds 256 MB. Each environment may spend ``DAILY_COMMAND_BUDGET``
commands and ``DAILY_BYTE_BUDGET`` bytes a day on the cache, counted with the
usage counters' shared daily counters (api/usage.py), so the count holds across
restarts; a process reads today's count once (two commands) before its first
call. Past either budget, the cache uses the database until the next UTC day.
With the counters' own budgets the daily ceilings stay under the free month
(``api.usage.budget_report``), and entries expire after ``TTL_S``, so at most a
day's bytes are held at once.
"""
from __future__ import annotations

import base64
import functools
import hashlib
import hmac
import inspect
import json
import logging
import os
import threading
import time
import zlib
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

import requests
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from fastapi import Depends
from pydantic import BaseModel, TypeAdapter

from api.auth import AuthUser, require_user
from api.observability import log_event

FORMAT = 1                          # part of every key: a new layout misses old entries
TTL_S = 6 * 3600                    # as long as a job's result is kept (jobs/queue.py)
MAX_VALUE_BYTES = 512_000           # a Monte Carlo answer is about 100 KB compressed
MEMORY_BUDGET_BYTES = 16_000_000
DATABASE_BUDGET_BYTES = 32_000_000
STORE_RETRY_S = 300.0
HTTP_TIMEOUT_S = 3.0
# Commands and bytes a day each environment may spend in Upstash on the cache.
# Commands: 31 x (3,000 + 1,000 + 500) = 139,500 a month at most, beside the
# counters' 341,000. Bytes (sent and received, base64): 31 x 180 MB = 5.6 GB of
# the free 10 GB, and at most 180 MB stored at once of the free 256 MB.
DAILY_COMMAND_BUDGET = {"production": 3_000, "staging": 1_000}
OTHER_DAILY_COMMAND_BUDGET = 500
DAILY_BYTE_BUDGET = {"production": 120_000_000, "staging": 40_000_000}
OTHER_DAILY_BYTE_BUDGET = 20_000_000
SYSTEM_OWNER_PREFIX = "system:"
# The server's own secrets: either one turns the hashes below into keyed ones (``_pepper``)
PEPPER_SOURCES = ("FSE_CACHE_KEY", "DATABASE_URL", "UPSTASH_REDIS_REST_TOKEN", "CLERK_SECRET_KEY",
                  "SUPABASE_UPLOADS_KEY", "GEMINI_API_KEY")
# Stands in for the commit where none is known (locally, a container without Render's variable):
# the cache then lasts one process, since this process's code may differ from the last one's
_PROCESS_BUILD = os.urandom(16).hex()
# An answer the run cut short is not kept: the next ask may get the whole one
INCOMPLETE: dict[str, Callable[[Any], bool]] = {
    # explanations are null when the run was too slow to explain (api/limits.py EXPLAIN_RUN_LIMIT_S)
    "montecarlo.run": lambda answer: isinstance(answer, dict) and answer.get("explanations") is None,
}


class CacheError(RuntimeError):
    """A store could not answer."""


def utc_day(now: float) -> str:
    return datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d")


def _day_end(now: float) -> float:
    day = datetime.fromtimestamp(now, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return (day + timedelta(days=1)).timestamp()


# ---------------------------------------------------------------------------
# Keys and values
# ---------------------------------------------------------------------------
def _pepper() -> bytes:
    """A secret only this server holds, so that someone who can read a store (or a
    backup of the table) can't test guessed inputs against it: ``FSE_CACHE_KEY`` when
    set, else the server's own secrets together. Changing any of them misses every
    entry, which is all a cache needs."""
    if os.environ.get("FSE_CACHE_KEY"):
        return hashlib.sha256(b"fse-result-cache:key\0" + os.environ["FSE_CACHE_KEY"].encode()).digest()
    parts = [f"{name}={os.environ.get(name) or ''}" for name in PEPPER_SOURCES[1:]]
    return hashlib.sha256("\0".join(["fse-result-cache:pepper", *parts]).encode("utf-8")).digest()


class Entry:
    """Where one request's answer is kept, and the key it is sealed with."""

    def __init__(self, material: bytes, pepper: Optional[bytes] = None):
        pepper = _pepper() if pepper is None else pepper
        self.key = hmac.new(pepper, b"lookup\0" + material, hashlib.sha256).hexdigest()
        self._secret = hmac.new(pepper, b"seal\0" + material, hashlib.sha256).digest()

    def seal(self, answer: Any) -> bytes:
        nonce = os.urandom(12)
        plain = zlib.compress(json.dumps(answer, separators=(",", ":")).encode("utf-8"), 6)
        return nonce + AESGCM(self._secret).encrypt(nonce, plain, self.key.encode("ascii"))

    def open(self, blob: bytes) -> Any:
        """The answer, or CacheError for anything that wasn't sealed by this entry."""
        try:
            plain = AESGCM(self._secret).decrypt(blob[:12], blob[12:], self.key.encode("ascii"))
            return json.loads(zlib.decompress(plain))
        except (InvalidTag, ValueError, zlib.error):
            raise CacheError("cached value unreadable") from None


def entry_for(kind: str, owner: str, request: BaseModel, model: dict) -> Entry:
    """The entry for ``request`` of ``kind`` by ``owner`` under the model stamp ``model``."""
    material = json.dumps({
        "format": FORMAT, "kind": kind, "owner": owner,
        "request": request.model_dump(mode="json"),
        "model": {k: model.get(k) for k in ("engine_version", "settings_fingerprint", "data_fingerprint")},
        "build": model.get("commit") or _PROCESS_BUILD,
    }, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return Entry(material.encode("utf-8"))


def _stamp_of(request: BaseModel) -> Optional[dict]:
    """The model stamp a run of ``request`` will carry, or None when its
    Settings are refused (the run then answers the refusal itself)."""
    from api.deps import resolve_settings
    from core.model_version import stamp

    try:
        return stamp(resolve_settings(request.settings, check_correlations=True))
    except Exception:  # noqa: BLE001 - the endpoint gives the reason
        return None


# ---------------------------------------------------------------------------
# Stores
# ---------------------------------------------------------------------------
class MemoryStore:
    """The newest entries in this process, within a byte budget."""

    name = "memory"

    def __init__(self, budget_bytes: int = MEMORY_BUDGET_BYTES, clock: Callable[[], float] = time.time):
        self.budget_bytes = budget_bytes
        self._clock = clock
        self._lock = threading.Lock()
        self._items: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
        self._size = 0

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            if item[0] <= self._clock():
                self._drop(key)
                return None
            self._items.move_to_end(key)
            return item[1]

    def put(self, key: str, value: bytes, ttl_s: int) -> None:
        if len(value) > self.budget_bytes:
            return
        with self._lock:
            self._drop(key)
            self._items[key] = (self._clock() + ttl_s, value)
            self._size += len(value)
            while self._size > self.budget_bytes:
                self._drop(next(iter(self._items)))

    def _drop(self, key: str) -> None:
        item = self._items.pop(key, None)
        if item is not None:
            self._size -= len(item[1])

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._size = 0


class UpstashCache:
    """Entries in Upstash Redis over its REST API, within the daily budgets.

    Commands and bytes are counted on the usage counters' shared daily
    counters; the first call of a process each day reads what is counted so
    far, so a restart doesn't start the day again."""

    name = "upstash"

    def __init__(self, url: str, token: str, *, prefix: str, daily_commands: int, daily_bytes: int,
                 counters=None, session=None, clock: Callable[[], float] = time.time):
        self.url = url.rstrip("/")
        self._token = token
        self.prefix = prefix
        self.daily_commands = daily_commands
        self.daily_bytes = daily_bytes
        self._counters = counters
        self._session = session or requests
        self._clock = clock
        self._lock = threading.Lock()
        self._read_day: Optional[str] = None

    # -- the budget -----------------------------------------------------------
    def _usage(self):
        if self._counters is not None:
            return self._counters
        from api import usage
        return usage.counters()

    def _checks(self, now: float):
        from api.usage import Check
        day, end = utc_day(now), _day_end(now)
        return (Check(f"cache-commands:{day}", self.daily_commands, end, shared=True),
                Check(f"cache-bytes:{day}", self.daily_bytes, end, shared=True))

    def spent(self) -> dict:
        commands, size = self._checks(self._clock())
        counters = self._usage()
        return {"commands": counters.usage(commands.key), "bytes": counters.usage(size.key)}

    def over_budget(self) -> bool:
        spent = self.spent()
        return spent["commands"] >= self.daily_commands or spent["bytes"] >= self.daily_bytes

    def _spend(self, commands: int, size: int) -> None:
        c, b = self._checks(self._clock())
        counters = self._usage()
        counters.add(c, commands)
        if size:
            counters.add(b, size)

    def _read_today(self) -> None:
        """Once a day per process: what the shared counters already hold for today."""
        now = self._clock()
        day = utc_day(now)
        if self._read_day == day:
            return
        c, b = self._checks(now)
        result = self._pipeline([["GET", self.prefix + c.key], ["GET", self.prefix + b.key]])
        with self._lock:
            self._read_day = day        # only once read: a failed read is tried again
        counters = self._usage()
        for check, value in zip((c, b), result):
            try:
                counters.seed(check, int(value or 0))
            except (TypeError, ValueError):
                pass
        self._spend(2, 0)

    # -- calls ----------------------------------------------------------------
    def _post(self, path: str, body: list) -> Any:
        try:
            resp = self._session.post(self.url + path, json=body, timeout=HTTP_TIMEOUT_S,
                                      headers={"Authorization": f"Bearer {self._token}"})
            answer = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise CacheError(f"Upstash unreachable ({type(exc).__name__})") from None
        if resp.status_code != 200 or (isinstance(answer, dict) and "error" in answer):
            raise CacheError(f"Upstash refused the command (HTTP {resp.status_code})")
        return answer

    def _pipeline(self, commands: list) -> list:
        answer = self._post("/pipeline", commands)
        if not isinstance(answer, list) or len(answer) != len(commands) or any(
                not isinstance(a, dict) or "error" in a for a in answer):
            raise CacheError("Upstash returned an unexpected result")
        return [a.get("result") for a in answer]

    def get(self, key: str) -> Optional[bytes]:
        self._read_today()
        if self.over_budget():
            raise CacheError("Upstash cache budget reached")
        answer = self._post("", ["GET", self.prefix + "result:" + key])
        value = answer.get("result") if isinstance(answer, dict) else None
        self._spend(1, len(value) if isinstance(value, str) else 0)
        if not isinstance(value, str):
            return None
        try:
            return base64.b64decode(value, validate=True)
        except ValueError:
            return None

    def put(self, key: str, value: bytes, ttl_s: int) -> None:
        self._read_today()
        text = base64.b64encode(value).decode("ascii")
        if self.over_budget():
            raise CacheError("Upstash cache budget reached")
        self._post("", ["SET", self.prefix + "result:" + key, text, "EX", str(int(ttl_s))])
        self._spend(1, len(text))


class DatabaseCache:
    """Entries in the ``result_cache`` table (the fallback)."""

    name = "database"

    def __init__(self, budget_bytes: int = DATABASE_BUDGET_BYTES):
        self.budget_bytes = budget_bytes

    def get(self, key: str) -> Optional[bytes]:
        from db import DatabaseUnavailable
        from db import result_cache
        try:
            return result_cache.get(key)
        except DatabaseUnavailable as exc:
            raise CacheError(str(exc)) from None

    def put(self, key: str, value: bytes, ttl_s: int) -> None:
        from db import DatabaseUnavailable
        from db import result_cache
        try:
            result_cache.put(key, value, ttl_s, self.budget_bytes)
        except DatabaseUnavailable as exc:
            raise CacheError(str(exc)) from None


def configured_stores(environment: str) -> list:
    """Upstash first when its URL and token are set, then the database when set."""
    stores: list = []
    url = (os.environ.get("UPSTASH_REDIS_REST_URL") or "").strip()
    token = (os.environ.get("UPSTASH_REDIS_REST_TOKEN") or "").strip()
    if url and token:
        stores.append(UpstashCache(
            url, token, prefix=f"fse:{environment}:",
            daily_commands=DAILY_COMMAND_BUDGET.get(environment, OTHER_DAILY_COMMAND_BUDGET),
            daily_bytes=DAILY_BYTE_BUDGET.get(environment, OTHER_DAILY_BYTE_BUDGET)))
    from db.engine import is_configured
    if is_configured():
        stores.append(DatabaseCache())
    return stores


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------
class ResultCache:
    """The memory layer, then the first shared store that can be used."""

    def __init__(self, stores: Optional[list] = None, *, memory: Optional[MemoryStore] = None,
                 enabled: bool = True, clock: Callable[[], float] = time.time):
        self.enabled = enabled
        self.memory = memory if memory is not None else MemoryStore(clock=clock)
        self._stores = list(stores or [])
        self._clock = clock
        self._failed_until: dict[str, float] = {}
        self.hits = self.misses = self.writes = 0

    @property
    def stores(self) -> list:
        return list(self._stores)

    def _usable(self) -> list:
        now = self._clock()
        out = []
        for store in self._stores:
            if self._failed_until.get(store.name, 0) > now:
                continue
            if isinstance(store, UpstashCache):
                try:
                    if store.over_budget():
                        continue
                except Exception:  # noqa: BLE001 - counted below like any failure
                    continue
            out.append(store)
        return out

    def _failed(self, store, exc: Exception, op: str) -> None:
        self._failed_until[store.name] = self._clock() + STORE_RETRY_S
        log_event("result_cache_failed", logging.WARNING, store=store.name, op=op, error=type(exc).__name__)

    def get(self, entry: Entry) -> Optional[Any]:
        blob = self.memory.get(entry.key)
        if blob is None:
            for store in self._usable():
                try:
                    blob = store.get(entry.key)
                except Exception as exc:  # noqa: BLE001 - try the next store
                    self._failed(store, exc, "get")
                    continue
                if blob is not None:
                    self.memory.put(entry.key, blob, TTL_S)
                break       # the first store that answers is the cache: a miss there is a miss
        if blob is None:
            self.misses += 1
            return None
        try:
            answer = entry.open(blob)
        except CacheError:
            self.misses += 1
            return None
        self.hits += 1
        return answer

    def put(self, entry: Entry, answer: Any) -> None:
        blob = entry.seal(answer)
        if len(blob) > MAX_VALUE_BYTES:
            return
        self.memory.put(entry.key, blob, TTL_S)
        for store in self._usable():
            try:
                store.put(entry.key, blob, TTL_S)
            except Exception as exc:  # noqa: BLE001 - try the next store
                self._failed(store, exc, "put")
                continue
            self.writes += 1
            return

    # -- a run ----------------------------------------------------------------
    def entry(self, kind: str, owner: Optional[str], request: BaseModel) -> Optional[Entry]:
        """The entry for this run, or None when it isn't cached: off, no caller,
        a system caller, no seed, or Settings the run will refuse."""
        if not self.enabled or not owner or owner.startswith(SYSTEM_OWNER_PREFIX):
            return None
        if getattr(request, "seed", None) is None:
            return None
        model = _stamp_of(request)
        return entry_for(kind, owner, request, model) if model is not None else None

    def run(self, kind: str, owner: Optional[str], request: BaseModel, response: type[BaseModel],
            compute: Callable[[], Any]) -> Any:
        """The cached answer, or ``compute()``'s, kept for next time as the JSON the endpoint answers."""
        entry = self.entry(kind, owner, request)
        if entry is None:
            return compute()
        hit = self.get(entry)
        if hit is not None:
            return hit
        answer = compute()
        try:
            adapter = TypeAdapter(response)
            if INCOMPLETE.get(kind, lambda _a: False)(answer):
                return answer
            self.put(entry, adapter.dump_python(adapter.validate_python(answer), mode="json"))
        except Exception as exc:  # noqa: BLE001 - a cache never fails a run
            log_event("result_cache_failed", logging.WARNING, store="all", op="put", error=type(exc).__name__)
        return answer

    def status(self) -> dict:
        """What the public ``/api/health/limits`` shows: whether it is on and where
        it keeps answers, nothing about how much it is used."""
        usable = self._usable()
        return {"enabled": self.enabled, "store": usable[0].name if usable else "memory"}

    def spent_today(self) -> Optional[dict]:
        """Today's Upstash spend as this process counts it (8.3's load test reads it)."""
        upstash = next((s for s in self._stores if isinstance(s, UpstashCache)), None)
        return {**upstash.spent(), "daily_commands": upstash.daily_commands,
                "daily_bytes": upstash.daily_bytes} if upstash else None


_cache: Optional[ResultCache] = None
_cache_lock = threading.Lock()


def cache() -> ResultCache:
    global _cache
    if _cache is None:
        with _cache_lock:
            if _cache is None:
                from api.main import deploy_environment
                _cache = ResultCache(configured_stores(deploy_environment()))
    return _cache


def set_cache(new: Optional[ResultCache]) -> None:
    """Replace the process's cache (tests); None rebuilds it from the environment."""
    global _cache
    with _cache_lock:
        _cache = new


def cached(kind: str, response: type[BaseModel]):
    """Serve a seeded run of a (sync) endpoint from the cache, by its caller.

    Goes **outside** ``simulation_slot``, so a hit never waits for the slot.
    The wrapper takes the caller as one more dependency (the same
    ``require_user`` every route already runs, so FastAPI resolves it once);
    called from Python without one, it just runs the endpoint. ``__wrapped__``
    stays the bare endpoint, which is what jobs/kinds.py ``_unwrapped`` wants:
    the job runner holds the slot and does its own lookup (jobs/runner.py).
    """
    def decorate(endpoint):
        bare = getattr(endpoint, "__wrapped__", endpoint)

        @functools.wraps(endpoint)
        def wrapper(req, cache_user: Any = None):
            owner = cache_user.subject if isinstance(cache_user, AuthUser) else None
            return cache().run(kind, owner, req, response, lambda: endpoint(req))

        wrapper.__wrapped__ = bare
        params = list(inspect.signature(bare).parameters.values())
        params.append(inspect.Parameter("cache_user", inspect.Parameter.KEYWORD_ONLY,
                                        default=Depends(require_user), annotation=AuthUser))
        wrapper.__signature__ = inspect.Signature(params)
        return wrapper
    return decorate
