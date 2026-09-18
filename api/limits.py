"""Usage limits and abuse protection (PLAN.md 1.6).

What is limited, and why each number:

- **Per user** (after sign-in, ``enforce_user_limits``): requests a minute,
  runs a minute and runs a day. A *run* is a request to one of ``RUN_PATHS``:
  simulations, backtests, forecasts, the ML models and EDGAR lookups.
- **Per network address** (before sign-in, ``LimitsMiddleware``): requests a
  minute, and refused sign-ins a minute, which stops token guessing. The
  address limit is set high because many users can share one address (an
  office, a proxy); signed-in users are held to their own limits instead.
- **Simulation size**: at most ``MAX_SIMULATION_PATHS`` paths. Measured on the
  API, a 15-year Monte Carlo run peaks at about 1.35 KB a path (100,000 paths:
  134 MB; the four scenarios: 160 MB), and the process itself uses about
  200 MB of Render free's 512 MB.
- **One simulation at a time** (``simulation_slot``): two runs at the cap
  would not fit in memory, and the free instance's fraction of a CPU makes
  parallel runs no faster. Others wait up to ``SLOT_WAIT_S``, then get 503.
- **Request size**: bodies over ``MAX_BODY_BYTES`` get 413.
- **Run timeout**: a simulation request that hasn't answered within
  ``RUN_TIMEOUT_S`` gets 504 (the work finishes in the background, still
  holding its slot, so a slow run can't be stacked on).

Every refusal has a sentence a person can act on in ``detail`` and says when
to try again (``Retry-After``). Health checks are never limited.

Counting lives in api/usage.py. ``FSE_LIMITS_MULTIPLIER`` raises every count
limit for the browser tests; it is ignored in production.
"""
from __future__ import annotations

import asyncio
import contextlib
import functools
import json
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, Field

from api.auth import AuthUser, require_user
from api.usage import Check, counters, identity_hash

MAX_SIMULATION_PATHS = 100_000
MAX_FORECAST_PATHS = 200_000       # the forecast overlay is lighter: 55 MB at 200,000
MAX_BODY_BYTES = 1_000_000
RUN_TIMEOUT_S = 100.0              # under the web proxy's own timeout, so this message arrives
SIMULATION_SLOTS = 1
SLOT_WAIT_S = 30.0


@dataclass(frozen=True)
class Rule:
    name: str
    what: str           # how the message names what was counted
    per: str            # "user" or "address"
    limit: int
    window_s: int
    shared: bool = False  # kept across restarts and instances (api/usage.py)


USER_REQUESTS = Rule("user_requests_per_minute", "requests", "user", 240, 60)
USER_RUNS = Rule("user_runs_per_minute", "model runs", "user", 30, 60)
USER_RUNS_DAY = Rule("user_runs_per_day", "model runs", "user", 1_000, 86_400, shared=True)
ADDRESS_REQUESTS = Rule("address_requests_per_minute", "requests", "address", 1_200, 60)
ADDRESS_REFUSED_SIGN_INS = Rule("address_refused_sign_ins_per_minute", "refused sign-ins",
                                "address", 60, 60)

# Requests that run a model or call an outside data source
RUN_PATHS = frozenset({
    "/api/montecarlo/run", "/api/montecarlo/scenarios", "/api/export/montecarlo-sample",
    "/api/backtesting/run", "/api/forecasting/run",
    "/api/ml/deal-risk", "/api/ml/surrogate", "/api/ml/macro-regime",
})
RUN_PATH_PREFIXES = ("/api/edgar/",)
# The memory-heavy runs: one at a time, with a timeout
SIMULATION_PATHS = frozenset({
    "/api/montecarlo/run", "/api/montecarlo/scenarios", "/api/export/montecarlo-sample",
    "/api/backtesting/run", "/api/forecasting/run",
})
UNLIMITED_PREFIX = "/api/health"


# Submitting a background job (PLAN.md 1.9) is a run too; polling it is not
JOBS_PATH = "/api/jobs"


def is_run_path(path: str) -> bool:
    return path in RUN_PATHS or path.startswith(RUN_PATH_PREFIXES)


def is_run_request(method: str, path: str) -> bool:
    return is_run_path(path) or (method == "POST" and path == JOBS_PATH)


def limit_multiplier() -> float:
    from api.main import deploy_environment
    if deploy_environment() == "production":
        return 1.0
    try:
        return max(1.0, float(os.environ.get("FSE_LIMITS_MULTIPLIER") or 1))
    except ValueError:
        return 1.0


def effective_limit(rule: Rule) -> int:
    return int(rule.limit * limit_multiplier())


def check_for(rule: Rule, identity: str, now: float, *, count: bool = True) -> Check:
    window = int(now // rule.window_s)
    return Check(key=f"{rule.name}:{identity_hash(identity)}:{window}", limit=effective_limit(rule),
                 window_end=(window + 1) * rule.window_s, shared=rule.shared, count=count)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
class LimitInfo(BaseModel):
    rule: str
    limit: int
    window_s: int
    retry_after_s: int


class LimitRefusal(BaseModel):
    """Body of a 429 response."""

    detail: str = Field(description="What was refused and when to try again, for people")
    limit: LimitInfo


def _period(window_s: int) -> str:
    return {60: "a minute", 3600: "an hour", 86_400: "a day"}.get(window_s, f"every {window_s} seconds")


def _wait(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    if seconds < 90 * 60:
        return f"{math.ceil(seconds / 60)} minutes"
    return f"{math.ceil(seconds / 3600)} hours"


def refusal(rule: Rule, check: Check, now: float) -> tuple[dict, dict]:
    """JSON body and headers for a 429."""
    retry = max(1, math.ceil(check.window_end - now))
    who = "your account" if rule.per == "user" else "this network address"
    message = (f"Too many {rule.what}: {who} can make {check.limit:,} {_period(rule.window_s)}. "
               f"Try again in {_wait(retry)}.")
    body = {"detail": message, "limit": {"rule": rule.name, "limit": check.limit,
                                         "window_s": rule.window_s, "retry_after_s": retry}}
    return body, {"Retry-After": str(retry)}


class LimitExceeded(Exception):
    def __init__(self, body: dict, headers: dict):
        super().__init__(body["detail"])
        self.body = body
        self.headers = headers


def _enforce(rules: list[tuple[Rule, str]], *, now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    checks = [check_for(rule, identity, now) for rule, identity in rules]
    refused = counters().consume(checks)
    if refused is not None:
        rule = rules[checks.index(refused)][0]
        raise LimitExceeded(*refusal(rule, refused, now))


def enforce_user_limits(request: Request, user: AuthUser = Depends(require_user)) -> None:
    """FastAPI dependency: the signed-in user's request and run limits."""
    rules = [(USER_REQUESTS, user.subject)]
    if is_run_request(request.method, request.url.path):
        rules += [(USER_RUNS, user.subject), (USER_RUNS_DAY, user.subject)]
    _enforce(rules)


# ---------------------------------------------------------------------------
# One simulation at a time
# ---------------------------------------------------------------------------
_slots = threading.BoundedSemaphore(SIMULATION_SLOTS)


@contextlib.contextmanager
def holding_simulation_slot(timeout: Optional[float] = None):
    """Hold the simulation slot for a block, waiting up to ``timeout`` (None:
    as long as it takes). The job runner uses it, so a background run and a
    direct request never simulate at the same time. Yields whether it got it."""
    got = _slots.acquire(timeout=timeout) if timeout is not None else _slots.acquire()
    try:
        yield got
    finally:
        if got:
            _slots.release()


def simulation_slot(endpoint):
    """Run a (sync) endpoint only while holding the simulation slot.

    Taken and released in the worker thread doing the work, so a run that
    outlives its request's timeout still holds the slot until it ends.
    """
    @functools.wraps(endpoint)
    def wrapper(*args, **kwargs):
        if not _slots.acquire(timeout=SLOT_WAIT_S):
            raise HTTPException(
                status_code=503, headers={"Retry-After": "10"},
                detail="The server is busy with other simulations. Try again in a few seconds.")
        try:
            return endpoint(*args, **kwargs)
        finally:
            _slots.release()
    return wrapper


# ---------------------------------------------------------------------------
# Middleware: address limits, request size, run timeout
# ---------------------------------------------------------------------------
async def _json(send, status: int, body: dict, headers: Optional[dict] = None) -> None:
    raw = json.dumps(body).encode()
    hdrs = [(b"content-type", b"application/json"), (b"content-length", str(len(raw)).encode())]
    hdrs += [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    await send({"type": "http.response.start", "status": status, "headers": hdrs})
    await send({"type": "http.response.body", "body": raw})


def _megabytes(n: int) -> str:
    if n < 1_000_000:
        return f"{n / 1000:.0f} KB"
    return f"{n / 1_000_000:.1f} MB".replace(".0 MB", " MB")


class LimitsMiddleware:
    """Pure ASGI middleware for the limits that apply before sign-in."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not path.startswith("/api/") or path.startswith(UNLIMITED_PREFIX):
            await self.app(scope, receive, send)
            return

        now = time.time()
        address = (scope.get("client") or ("unknown", 0))[0] or "unknown"
        rules = [(ADDRESS_REQUESTS, True), (ADDRESS_REFUSED_SIGN_INS, False)]
        checks = [check_for(rule, address, now, count=count) for rule, count in rules]
        refused = counters().consume(checks)
        if refused is not None:
            body, headers = refusal(rules[checks.index(refused)][0], refused, now)
            await _json(send, 429, body, headers)
            return

        if scope.get("method") in ("POST", "PUT", "PATCH"):
            receive = await self._read_body(scope, receive, send)
            if receive is None:
                return

        status = 0

        async def send_status(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            if path in SIMULATION_PATHS:
                await self._with_timeout(scope, receive, send_status)
            else:
                await self.app(scope, receive, send_status)
        finally:
            if status == 401:
                counters().add(checks[1])

    async def _read_body(self, scope, receive, send):
        """Read the body up to the limit; a replaying ``receive``, or None after a 413."""
        declared = dict(scope.get("headers") or []).get(b"content-length")
        try:
            size = int(declared) if declared is not None else None
        except ValueError:
            size = None
        if size is not None and size > MAX_BODY_BYTES:
            await self._too_large(send, size)
            return None
        chunks, total = [], 0
        while True:
            message = await receive()
            if message["type"] != "http.request":
                return None     # the client went away; there is no one to answer
            chunk = message.get("body", b"")
            total += len(chunk)
            if total > MAX_BODY_BYTES:
                await self._too_large(send, None)
                return None
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        return replay

    @staticmethod
    async def _too_large(send, size: Optional[int]) -> None:
        got = f" (this one is {_megabytes(size)})" if size else ""
        await _json(send, 413, {"detail": f"The request is too large: the limit is "
                                          f"{_megabytes(MAX_BODY_BYTES)}{got}."})

    async def _with_timeout(self, scope, receive, send):
        started = False
        timed_out = False

        async def guarded(message):
            nonlocal started
            if timed_out:
                return      # the late answer of a run that already got a 504
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        task = asyncio.ensure_future(self.app(scope, receive, guarded))
        done, _ = await asyncio.wait({task}, timeout=RUN_TIMEOUT_S)
        if task in done or started:
            await task
            return
        timed_out = True
        task.add_done_callback(lambda t: t.cancelled() or t.exception())
        seconds = f"{RUN_TIMEOUT_S:g}"
        await _json(send, 504, {"detail": f"The run took longer than {seconds} seconds, so the "
                                          "server stopped waiting for it. Try fewer paths or a "
                                          "shorter holding period."})
