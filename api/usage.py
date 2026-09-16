"""Usage counters behind the limits (PLAN.md 1.6), sized for Upstash's free
500,000 Redis commands a month.

**Counting is in memory; sharing is batched.** Every check reads and bumps a
counter in this process, so a request never waits on the network and a single
free Render instance enforces its limits exactly. Only counters for long
windows (``shared=True``, e.g. runs per day) are shared, so they survive a
restart and hold across instances. Their increments are sent together every
``FLUSH_INTERVAL_S`` -- one Redis script call for all of them -- and only when
something changed, so an idle API sends nothing and Redis cost grows with
active users, not requests. Short windows (per minute) stay local: a free
instance only restarts after 15 idle minutes, by which time they are over.

**Budget guard.** Each sync also counts, in Redis, the commands it used today
(conservatively: the script call and every command inside it). Past the daily
budget for the environment (``DAILY_COMMAND_BUDGET``) the process stops using
Redis until the next UTC day. The budgets add up to well under the free
monthly allowance whatever the traffic; ``monthly_redis_commands`` estimates
normal use (``python -m api.usage``, DEPLOY.md "Usage limits").

**Fallback.** If Upstash isn't configured, fails, or is over budget, the batch
goes to the ``usage_counters`` table instead (db/usage.py). If that fails too,
the increments stay pending for the next sync and limits keep working from
memory.

**Privacy.** Keys hold a hash of the user or address, never the id itself, so
nothing personal reaches Redis or the table (``identity_hash``).
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

import requests

from api.observability import log_event, utc_now_iso

FLUSH_INTERVAL_S = 300.0
# Upstash Redis free plan
REDIS_MONTHLY_COMMANDS = 500_000
# Commands a day each environment may send; the rest of the day uses the
# database. 31 x (8,000 + 2,000 + 1,000) = 341,000: under the free 500,000
# even if every environment hits its budget every day of a long month.
DAILY_COMMAND_BUDGET = {"production": 8_000, "staging": 2_000}
OTHER_DAILY_COMMAND_BUDGET = 1_000
# After Upstash fails, wait this long before trying it again
STORE_RETRY_S = 300.0
HTTP_TIMEOUT_S = 5.0
# How long a Redis reachability check is reused (PING is not billed, but it's
# still a network call from a public endpoint)
PING_CACHE_S = 600.0

# Adds each increment and sets the expiry on keys it created; the last key is
# today's command count. Returns the new totals, command count last.
REDIS_SCRIPT = """
local n = #KEYS - 1
local out = {}
for i = 1, n do
  local delta = tonumber(ARGV[2 * i - 1])
  local total = redis.call('INCRBY', KEYS[i], delta)
  if total == delta then redis.call('EXPIRE', KEYS[i], ARGV[2 * i]) end
  out[i] = total
end
local cost = tonumber(ARGV[2 * n + 1])
local used = redis.call('INCRBY', KEYS[n + 1], cost)
if used == cost then redis.call('EXPIRE', KEYS[n + 1], 172800) end
out[n + 1] = used
return out
"""


def identity_hash(identity: str) -> str:
    """A short, stable stand-in for a user id or address in counter keys."""
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def redis_commands_for_sync(n_keys: int) -> int:
    """Upper bound on what one sync costs: the script call, plus an increment
    and an expiry per key, plus the command counter's own two."""
    return 1 + 2 * n_keys + 2


def utc_day(now: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if now is None else now, timezone.utc).strftime("%Y-%m-%d")


class StoreError(RuntimeError):
    """A counter store could not take the batch."""


Batch = Sequence[tuple[str, int, int]]   # (key, increment, seconds to keep)


class UpstashStore:
    """Counters in Upstash Redis over its REST API (no Redis client needed)."""

    name = "upstash"

    def __init__(self, url: str, token: str, *, prefix: str, daily_budget: int,
                 session=None, clock: Callable[[], float] = time.time):
        self.url = url.rstrip("/")
        self._token = token
        self.prefix = prefix
        self.daily_budget = daily_budget
        self._session = session or requests
        self._clock = clock
        self._over_budget_day: Optional[str] = None
        self.commands_today: Optional[int] = None

    def over_budget(self) -> bool:
        return self._over_budget_day == utc_day(self._clock())

    def _call(self, command: list) -> object:
        try:
            resp = self._session.post(self.url, json=command, timeout=HTTP_TIMEOUT_S,
                                      headers={"Authorization": f"Bearer {self._token}"})
            body = resp.json()
        except (requests.RequestException, ValueError) as exc:
            raise StoreError(f"Upstash unreachable ({type(exc).__name__})") from None
        if resp.status_code != 200 or "error" in body:
            raise StoreError(f"Upstash refused the command (HTTP {resp.status_code})")
        return body.get("result")

    def add(self, batch: Batch) -> dict[str, int]:
        today = utc_day(self._clock())
        if self._over_budget_day == today:
            raise StoreError("Upstash daily command budget reached")
        keys = [self.prefix + key for key, _, _ in batch] + [f"{self.prefix}redis-commands:{today}"]
        args: list[str] = []
        for _, delta, ttl in batch:
            args += [str(int(delta)), str(int(ttl))]
        args.append(str(redis_commands_for_sync(len(batch))))
        result = self._call(["EVAL", REDIS_SCRIPT, str(len(keys)), *keys, *args])
        if not isinstance(result, list) or len(result) != len(keys):
            raise StoreError("Upstash returned an unexpected result")
        self.commands_today = int(result[-1])
        if self.commands_today >= self.daily_budget:
            self._over_budget_day = today
            log_event("redis_budget_reached", logging.WARNING,
                      commands_today=self.commands_today, budget=self.daily_budget)
        return {key: int(total) for (key, _, _), total in zip(batch, result)}

    def ping(self) -> bool:
        """PING, which Upstash doesn't bill."""
        try:
            return self._call(["PING"]) == "PONG"
        except StoreError:
            return False


class DatabaseStore:
    """Counters in the ``usage_counters`` table (the fallback)."""

    name = "database"

    def add(self, batch: Batch) -> dict[str, int]:
        from db import DatabaseUnavailable
        from db.usage import add_counts
        try:
            return add_counts(batch)
        except DatabaseUnavailable as exc:
            raise StoreError(str(exc)) from None


def configured_stores(environment: str) -> list:
    """Upstash first when its URL and token are set, then the database when set."""
    stores: list = []
    url = (os.environ.get("UPSTASH_REDIS_REST_URL") or "").strip()
    token = (os.environ.get("UPSTASH_REDIS_REST_TOKEN") or "").strip()
    if url and token:
        stores.append(UpstashStore(
            url, token, prefix=f"fse:{environment}:",
            daily_budget=DAILY_COMMAND_BUDGET.get(environment, OTHER_DAILY_COMMAND_BUDGET)))
    from db.engine import is_configured
    if is_configured():
        stores.append(DatabaseStore())
    return stores


@dataclass
class _Entry:
    expires: float          # epoch seconds when the window ends
    shared: bool
    synced: int = 0         # total in the shared store at the last sync (ours included)
    pending: int = 0        # counted here, not sent yet


@dataclass(frozen=True)
class Check:
    """One limit to test for a request."""

    key: str
    limit: int
    window_end: float
    shared: bool = False
    count: bool = True      # False: refuse at the limit but don't add to it


class UsageCounters:
    """The process's counters and their sync to the shared stores."""

    def __init__(self, stores: Optional[list] = None, *, clock: Callable[[], float] = time.time,
                 background: bool = True):
        self._stores = list(stores or [])
        self._clock = clock
        self._background = background
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._failed_until: dict[str, float] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.last_sync: Optional[dict] = None

    @property
    def stores(self) -> list:
        return list(self._stores)

    def usage(self, key: str) -> int:
        with self._lock:
            entry = self._entries.get(key)
            return entry.synced + entry.pending if entry and entry.expires > self._clock() else 0

    def consume(self, checks: Sequence[Check]) -> Optional[Check]:
        """Refuse with the first check that is at its limit; otherwise count them all.

        Nothing is counted for a refused request.
        """
        now = self._clock()
        with self._lock:
            for check in checks:
                entry = self._entries.get(check.key)
                used = entry.synced + entry.pending if entry and entry.expires > now else 0
                if used + 1 > check.limit:
                    return check
            for check in checks:
                if check.count:
                    self._add_locked(check, 1, now)
        if self._background and any(c.shared and c.count for c in checks):
            self._ensure_thread()
        return None

    def add(self, check: Check, n: int = 1) -> None:
        with self._lock:
            self._add_locked(check, n, self._clock())
        if self._background and check.shared:
            self._ensure_thread()

    def _add_locked(self, check: Check, n: int, now: float) -> None:
        entry = self._entries.get(check.key)
        if entry is None or entry.expires <= now:
            entry = self._entries[check.key] = _Entry(expires=check.window_end, shared=check.shared)
        entry.pending += n

    # -- sync ---------------------------------------------------------------
    def flush(self) -> Optional[str]:
        """Send pending shared increments; returns the store used, or None."""
        now = self._clock()
        with self._lock:
            for key in [k for k, e in self._entries.items() if e.expires <= now]:
                del self._entries[key]
            batch = [(key, e.pending, max(60, int(e.expires - now) + 60))
                     for key, e in self._entries.items() if e.shared and e.pending > 0]
        if not batch:
            return None
        for store in self._stores:
            if self._failed_until.get(store.name, 0) > now:
                continue
            if isinstance(store, UpstashStore) and store.over_budget():
                continue
            try:
                totals = store.add(batch)
            except Exception as exc:  # noqa: BLE001 - any failure means: try the next store
                self._failed_until[store.name] = now + STORE_RETRY_S
                log_event("usage_sync_failed", logging.WARNING, store=store.name,
                          error=type(exc).__name__, keys=len(batch))
                continue
            with self._lock:
                for key, delta, _ in batch:
                    entry = self._entries.get(key)
                    if entry is None:
                        continue
                    entry.pending -= delta
                    # Another store may know less than we already counted (a fallback)
                    entry.synced = max(entry.synced + delta, totals.get(key, 0))
            self._failed_until.pop(store.name, None)
            self.last_sync = {"store": store.name, "at": utc_now_iso(), "keys": len(batch)}
            return store.name
        self.last_sync = {"store": None, "at": utc_now_iso(), "keys": len(batch)}
        return None

    def _ensure_thread(self) -> None:
        if self._thread is not None or not self._stores:
            return
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="usage-sync", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(FLUSH_INTERVAL_S):
            try:
                self.flush()
            except Exception as exc:  # noqa: BLE001 - the thread must survive
                log_event("usage_sync_error", logging.ERROR, error=type(exc).__name__)

    def close(self) -> None:
        """Stop the background sync and send what's pending (at shutdown)."""
        self._stop.set()
        self.flush()


# ---------------------------------------------------------------------------
# The process's counters
# ---------------------------------------------------------------------------
_counters: Optional[UsageCounters] = None
_counters_lock = threading.Lock()
_ping_cache: dict[str, tuple[float, bool]] = {}


def counters() -> UsageCounters:
    global _counters
    if _counters is None:
        with _counters_lock:
            if _counters is None:
                from api.main import deploy_environment
                _counters = UsageCounters(configured_stores(deploy_environment()))
    return _counters


def set_counters(new: Optional[UsageCounters]) -> None:
    """Replace the process's counters (tests); None rebuilds from the environment."""
    global _counters
    with _counters_lock:
        _counters = new
    _ping_cache.clear()


def status() -> dict:
    """What ``/api/health/limits`` reports. Never touches the database."""
    c = counters()
    upstash = next((s for s in c.stores if isinstance(s, UpstashStore)), None)
    reachable = None
    if upstash is not None:
        cached = _ping_cache.get(upstash.url)
        now = time.monotonic()
        if cached is None or now - cached[0] > PING_CACHE_S:
            cached = _ping_cache[upstash.url] = (now, upstash.ping())
        reachable = cached[1]
    if upstash is not None and reachable and not upstash.over_budget():
        store = "upstash"
    elif any(isinstance(s, DatabaseStore) for s in c.stores):
        store = "database"
    else:
        store = "memory"
    return {
        "store": store,
        "upstash": {"configured": upstash is not None, "reachable": reachable,
                    "over_budget": upstash.over_budget() if upstash else False,
                    "daily_budget": upstash.daily_budget if upstash else None,
                    "commands_today": upstash.commands_today if upstash else None},
        "flush_interval_s": FLUSH_INTERVAL_S,
        "last_sync": c.last_sync,
    }


# ---------------------------------------------------------------------------
# Command budget estimate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrafficProfile:
    environment: str
    daily_active_users: int
    active_minutes_per_user: float
    busy_hours_per_day: float
    shared_counters_per_user: int = 1   # api/limits.py: runs per day


# The traffic the free phase is sized for. Every active user is assumed to run
# simulations (the only shared counter), which overstates real use.
TARGET_TRAFFIC = (
    TrafficProfile("production", daily_active_users=200, active_minutes_per_user=45,
                   busy_hours_per_day=12),
    TrafficProfile("staging", daily_active_users=20, active_minutes_per_user=45,
                   busy_hours_per_day=12),
)
# tests.yml's real round trip: a handful of commands per CI run, ~150 runs a month
CI_MONTHLY_COMMANDS = 3_000


def monthly_redis_commands(profile: TrafficProfile, *, flush_interval_s: float = FLUSH_INTERVAL_S,
                           days: int = 31) -> int:
    """Commands a month for ``profile``, counted the way the budget guard counts them.

    A user's counter is in one sync per interval they are active; a sync costs
    three commands plus two per counter in it.
    """
    syncs_per_user = math.ceil(profile.active_minutes_per_user * 60 / flush_interval_s)
    counters_per_day = profile.daily_active_users * profile.shared_counters_per_user * syncs_per_user
    syncs_per_day = min(math.ceil(profile.busy_hours_per_day * 3600 / flush_interval_s),
                        counters_per_day)
    per_day = 2 * counters_per_day + 3 * syncs_per_day
    return per_day * days


def budget_report() -> dict:
    rows = [{"environment": p.environment, "daily_active_users": p.daily_active_users,
             "monthly_commands": monthly_redis_commands(p)} for p in TARGET_TRAFFIC]
    total = sum(r["monthly_commands"] for r in rows) + CI_MONTHLY_COMMANDS
    ceiling = 31 * (sum(DAILY_COMMAND_BUDGET.values()) + OTHER_DAILY_COMMAND_BUDGET)
    return {"target": rows, "ci_monthly_commands": CI_MONTHLY_COMMANDS,
            "estimated_monthly_commands": total, "free_monthly_commands": REDIS_MONTHLY_COMMANDS,
            "estimated_fraction": total / REDIS_MONTHLY_COMMANDS,
            "budget_ceiling_monthly_commands": ceiling}


if __name__ == "__main__":
    report = budget_report()
    for row in report["target"]:
        print(f"{row['environment']:<11} {row['daily_active_users']:>4} daily users "
              f"{row['monthly_commands']:>9,} commands a month")
    print(f"{'CI':<11} {'':>16} {report['ci_monthly_commands']:>9,}")
    print(f"estimate {report['estimated_monthly_commands']:,} of {REDIS_MONTHLY_COMMANDS:,} "
          f"({report['estimated_fraction']:.0%}); hard ceiling from the daily budgets "
          f"{report['budget_ceiling_monthly_commands']:,}")
