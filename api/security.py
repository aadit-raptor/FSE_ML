"""Security headers and allowed browser origins for the API (PLAN.md 1.7).

The web app reaches the API through Vercel's proxy (web/next.config.ts), so
the browser sees these headers on every ``/api`` response of the app's own
origin, and on the Render URL itself.

- Every response gets ``SECURITY_HEADERS`` and ``Cache-Control: no-store``:
  API answers carry deal contents and must never sit in a browser or proxy
  cache.
- Every response except the interactive docs gets ``API_CSP``: a JSON API
  loads nothing, so nothing is allowed.
- ``/api/docs`` (Swagger UI) loads its script and style from jsDelivr and runs
  one inline script. Its policy allows exactly that script by hash, computed
  from the page itself, so a changed page can't run anything else.

CORS (``cors_origins``) allows only the app's own origins. The deployed web app
doesn't need CORS at all (same origin through the proxy); the list is for
direct API use from the app's domain and for local development.
"""
from __future__ import annotations

import base64
import hashlib
import logging
import os
import re
from urllib.parse import urlsplit

from starlette.datastructures import MutableHeaders

from api.observability import log_event

SECURITY_HEADERS: dict[str, str] = {
    # Two years, the value browsers' preload lists expect; ignored over plain HTTP
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
}

API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'"

DOCS_PATH = "/api/docs"
# What FastAPI's Swagger UI page loads (fastapi.openapi.docs)
DOCS_CDN = "https://cdn.jsdelivr.net"
DOCS_FAVICON = "https://fastapi.tiangolo.com"

_INLINE_SCRIPT = re.compile(rb"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)


def docs_csp(html: bytes) -> str:
    """The policy for the Swagger UI page: its CDN files, and its inline scripts by hash."""
    hashes = " ".join(
        "'sha256-" + base64.b64encode(hashlib.sha256(body).digest()).decode() + "'"
        for body in _INLINE_SCRIPT.findall(html))
    return (f"default-src 'none'; script-src {DOCS_CDN} {hashes}; style-src {DOCS_CDN}; "
            f"img-src {DOCS_FAVICON} data:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")


class SecurityHeadersMiddleware:
    """Pure ASGI middleware adding the headers above to every HTTP response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path") == DOCS_PATH:
            await self._docs(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                _apply(MutableHeaders(scope=message), API_CSP)
            await send(message)

        await self.app(scope, receive, send_with_headers)

    async def _docs(self, scope, receive, send):
        # The page is a few KB: hold it back to hash its inline script
        start = None
        body = bytearray()

        async def buffer(message):
            nonlocal start
            if message["type"] == "http.response.start":
                start = message
                return
            if message["type"] == "http.response.body":
                body.extend(message.get("body", b""))
                if message.get("more_body"):
                    return
                headers = MutableHeaders(scope=start)
                _apply(headers, docs_csp(bytes(body)))
                headers["Content-Length"] = str(len(body))
                await send(start)
                await send({"type": "http.response.body", "body": bytes(body)})
                return
            await send(message)

        await self.app(scope, receive, buffer)


def _apply(headers: MutableHeaders, csp: str) -> None:
    for name, value in SECURITY_HEADERS.items():
        headers[name] = value
    headers["Content-Security-Policy"] = csp
    headers["Cache-Control"] = "no-store"


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
PRODUCTION_WEB_ORIGIN = "https://fse-ml.vercel.app"
LOCAL_WEB_ORIGIN = "http://localhost:3000"

# Browser origins allowed per environment when FSE_CORS_ORIGINS is unset.
# Staging's web app is a Vercel preview on a changing URL and always goes
# through the proxy, so it needs none.
DEFAULT_ORIGINS: dict[str, tuple[str, ...]] = {
    "production": (PRODUCTION_WEB_ORIGIN,),
    "staging": (),
    "local": (LOCAL_WEB_ORIGIN,),
}

# The headers the web app sends; nothing else is allowed cross-origin
CORS_ALLOW_HEADERS = ("Authorization", "Content-Type", "X-Request-ID")

_LOOPBACK = {"localhost", "127.0.0.1", "::1"}


def origin_refusal(origin: str, environment: str) -> str | None:
    """Why ``origin`` can't be allowed, or None when it can.

    An origin is exactly ``scheme://host[:port]``: no wildcard, path, query or
    credentials. Deployed copies accept HTTPS only; plain HTTP is for a
    loopback host in a local run.
    """
    if "*" in origin:
        return "wildcard"
    parts = urlsplit(origin)
    if parts.scheme not in ("https", "http") or not parts.hostname:
        return "not an origin"
    if parts.path or parts.query or parts.fragment or parts.username or parts.password:
        return "not an origin"
    if origin != origin.lower() or origin.endswith("/"):
        return "not an origin"
    if parts.scheme == "http" and not (environment == "local" and parts.hostname in _LOOPBACK):
        return "plain http"
    return None


def cors_origins(environment: str) -> list[str]:
    """Browser origins allowed to call this API.

    ``FSE_CORS_ORIGINS`` (comma-separated) replaces the environment's default.
    Entries that aren't a single HTTPS origin are dropped and logged, never
    widened: a bad value locks the API down rather than opening it.
    """
    raw = os.environ.get("FSE_CORS_ORIGINS")
    if raw is None or not raw.strip():
        candidates = list(DEFAULT_ORIGINS.get(environment, ()))
    else:
        candidates = [o.strip() for o in raw.split(",") if o.strip()]
    allowed = []
    for origin in candidates:
        reason = origin_refusal(origin, environment)
        if reason:
            # The value is configuration, not user data, so it may be logged
            log_event("cors_origin_refused", logging.WARNING, origin=origin[:200], reason=reason)
        elif origin not in allowed:
            allowed.append(origin)
    return allowed
