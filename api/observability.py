"""Monitoring for the API: request IDs, structured logs, model-run timings and
error tracking (PLAN.md 1.2).

- Every request gets an ID. A well-formed ``X-Request-ID`` from the caller
  (the web app sends one with each call) is kept, anything else is replaced.
  The ID goes back in the response header, in every log line and on every
  Sentry event, so one ID links the browser error, the API error and the logs.
- Logs are one JSON object per line on stdout, timestamps in UTC. They carry
  method, route template, status and timings, never request bodies or query
  strings.
- ``model_timer`` times a model run; the request log line and the
  ``Server-Timing`` response header carry the total.
- Sentry switches on when ``SENTRY_DSN`` is set. Events never include request
  bodies, query strings, cookies, headers, local variables, IP addresses or
  user details (``scrub_event``), so no deal contents or personal data leave
  the server.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import sentry_sdk
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse

logger = logging.getLogger("fse")

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
# Render's health checks hit this every few seconds; logging them buries real traffic
QUIET_PATHS = {"/api/health"}


def deploy_environment() -> str:
    """Which copy of the API this is: production, staging or local.

    FSE_ENV wins when set. Otherwise Render's RENDER_SERVICE_NAME decides:
    a service whose name ends in -staging (fse-api-staging) is staging, any
    other Render service is production. Read per call so tests can set it.
    """
    explicit = os.environ.get("FSE_ENV", "").strip().lower()
    if explicit:
        return explicit
    service = os.environ.get("RENDER_SERVICE_NAME", "")
    if not service:
        return "local"
    return "staging" if service.endswith("-staging") else "production"


def utc_now_iso() -> str:
    """Current time as ISO 8601 UTC with a Z suffix (how every time is stored)."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def clean_request_id(value: Optional[str]) -> str:
    """The caller's request ID if well formed, otherwise a new random one."""
    if value and _REQUEST_ID_RE.match(value):
        return value
    return uuid.uuid4().hex


@dataclass
class RequestContext:
    request_id: str
    # Mutated from the endpoint's worker thread; the object itself is shared
    model_runs: list = field(default_factory=list)

    @property
    def model_ms(self) -> float:
        return sum(ms for _, ms in self.model_runs)


_current: ContextVar[Optional[RequestContext]] = ContextVar("fse_request", default=None)


def current_request_id() -> Optional[str]:
    ctx = _current.get()
    return ctx.request_id if ctx else None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc)
                          .isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        rid = getattr(record, "request_id", None) or current_request_id()
        if rid:
            entry["request_id"] = rid
        entry.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            # Type and traceback only: exception messages can quote input values
            entry["error"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            entry["traceback"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging() -> None:
    """JSON lines on stdout for the ``fse`` logger (idempotent)."""
    if any(getattr(h, "_fse_json", False) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler._fse_json = True
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_event(event: str, level: int = logging.INFO, exc_info=None, **fields) -> None:
    logger.log(level, event, exc_info=exc_info, extra={"fields": fields})


# Called with (model name, "start" | "end") around each model run. A job
# runner sets it to turn model runs into progress (jobs/runner.py).
model_run_listener: ContextVar[Optional[Callable[[str, str], None]]] = ContextVar(
    "fse_model_run_listener", default=None)


@contextmanager
def model_timer(name: str):
    """Time one model run and record it on the current request.

    Logs ``model_run`` with the model name and duration even if the run fails.
    """
    listener = model_run_listener.get()
    if listener is not None:
        listener(name, "start")
    t0 = time.perf_counter()
    ok = False
    try:
        yield
        if listener is not None:
            listener(name, "end")
        ok = True
    finally:
        ms = round((time.perf_counter() - t0) * 1000, 2)
        ctx = _current.get()
        if ctx is not None:
            ctx.model_runs.append((name, ms))
        log_event("model_run", model=name, duration_ms=ms, ok=ok)


# ---------------------------------------------------------------------------
# Sentry
# ---------------------------------------------------------------------------
_KEPT_REQUEST_KEYS = ("method",)


def scrub_event(event: dict, hint: Optional[dict] = None) -> dict:
    """Strip anything that could hold deal contents or personal data."""
    request = event.get("request")
    if isinstance(request, dict):
        kept = {k: request[k] for k in _KEPT_REQUEST_KEYS if k in request}
        if isinstance(request.get("url"), str):
            kept["url"] = request["url"].split("?", 1)[0]
        event["request"] = kept
    for key in ("user", "extra", "breadcrumbs", "server_name"):
        event.pop(key, None)
    for exc in (event.get("exception") or {}).get("values") or []:
        for frame in (exc.get("stacktrace") or {}).get("frames") or []:
            frame.pop("vars", None)
    return event


def init_sentry(dsn: Optional[str], *, environment: str, release: Optional[str],
                transport=None) -> bool:
    """Start error tracking when a DSN is given. Returns whether it is on."""
    if not dsn:
        return False
    from sentry_sdk.integrations.logging import LoggingIntegration

    sentry_sdk.init(
        dsn=dsn,
        environment=environment,
        release=release,
        send_default_pii=False,
        include_local_variables=False,
        max_request_body_size="never",
        traces_sample_rate=0,
        before_send=scrub_event,
        # Our own JSON logs aren't forwarded: errors are captured explicitly,
        # with their request ID, in RequestContextMiddleware
        integrations=[LoggingIntegration(level=None, event_level=None)],
        transport=transport,
    )
    return True


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
class RequestContextMiddleware:
    """Pure ASGI middleware: request ID, request log line, error capture."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw = dict(scope.get("headers") or []).get(b"x-request-id", b"")
        ctx = RequestContext(clean_request_id(raw.decode("latin-1")))
        token = _current.set(ctx)
        status = 500
        started = False
        t0 = time.perf_counter()

        async def send_with_id(message):
            nonlocal status, started
            if message["type"] == "http.response.start":
                started = True
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = ctx.request_id
                if ctx.model_runs:
                    headers.append("Server-Timing", f"model;dur={ctx.model_ms:.1f}")
            await send(message)

        with sentry_sdk.isolation_scope() as sentry_scope:
            sentry_scope.set_tag("request_id", ctx.request_id)
            try:
                await self.app(scope, receive, send_with_id)
            except Exception as exc:
                route = _route_template(scope)
                sentry_sdk.capture_exception(exc, tags={"request_id": ctx.request_id, "route": route})
                log_event("unhandled_error", logging.ERROR, exc_info=(type(exc), exc, exc.__traceback__),
                          method=scope.get("method"), route=route)
                if started:
                    raise
                response = JSONResponse(
                    {"detail": "Internal server error", "request_id": ctx.request_id}, status_code=500)
                await response(scope, receive, send_with_id)
            finally:
                path = scope.get("path", "")
                if not (path in QUIET_PATHS and status < 400):
                    fields = {"method": scope.get("method"), "route": _route_template(scope),
                              "status": status,
                              "duration_ms": round((time.perf_counter() - t0) * 1000, 2)}
                    if ctx.model_runs:
                        fields["model_ms"] = round(ctx.model_ms, 2)
                    log_event("request", logging.WARNING if status >= 500 else logging.INFO, **fields)
                _current.reset(token)


def _route_template(scope) -> str:
    """The matched route's path template (``/api/edgar/{ticker}``), never raw values."""
    route = scope.get("route")
    template = getattr(route, "path", None)
    if not isinstance(template, str):
        return "unmatched"
    # Routes in included routers report their path without the include prefix
    # (/deal/run for /api/deal/run). The prefix is static, so take it from the
    # raw path: every segment before the template's own.
    raw = scope.get("path", "").split("/")
    own = template.split("/")
    # both start with an empty segment: ["", "api", "deal", "run"] vs ["", "deal", "run"]
    prefix = "/".join(raw[:len(raw) - len(own) + 1]) if len(raw) > len(own) else ""
    return prefix + template
