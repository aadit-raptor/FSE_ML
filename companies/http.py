"""HTTP for the filing and economic data sources: one transport, polite pacing, no leaked keys.

Every connector fetches through ``get``. It

- waits between calls to the same host (``MIN_INTERVAL_S``), so a refresh
  stays inside each source's published rules (SEC: 10 requests a second;
  Companies House: 600 every five minutes; the others ask for restraint);
- sends a User-Agent naming the product, as the SEC requires;
- stops reading a response at ``MAX_RESPONSE_BYTES``;
- raises ``SourceError`` with the host and the status only. A request URL
  can carry a key (EDINET takes it as a query parameter), so no URL, query
  string or exception text from the HTTP library ever reaches a message.

Tests and the recorder swap the transport (``use_transport``): tests replay
recorded responses, the recorder (companies/record.py) saves real ones.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional
from urllib.parse import urlsplit

USER_AGENT = "Variater (+https://variater.com)"
TIMEOUT_S = 30.0
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
# Seconds between two calls to one host
MIN_INTERVAL_S = {
    "data.sec.gov": 0.15,
    "www.sec.gov": 0.15,
    "filings.xbrl.org": 0.5,
    "api.company-information.service.gov.uk": 0.5,
    "document-api.company-information.service.gov.uk": 0.5,
    "api.edinet-fsa.go.jp": 1.0,
    "disclosure2dl.edinet-fsa.go.jp": 1.0,
    "api.gleif.org": 1.0,
    # Economic data (economy/connectors.py); the OECD allows few calls an hour
    "www.imf.org": 1.0,
    "api.worldbank.org": 0.5,
    "stats.bis.org": 1.0,
    "sdmx.oecd.org": 5.0,
    "data-api.ecb.europa.eu": 0.5,
    "api.stlouisfed.org": 0.6,
    # Industry averages (benchmarks/): a university's web server, so gently
    "pages.stern.nyu.edu": 1.0,
}
DEFAULT_INTERVAL_S = 1.0


class SourceError(Exception):
    """A filing source failed. ``reason`` is a short code the screen can
    translate; the message never holds a URL or a key."""

    def __init__(self, source: str, reason: str, status: Optional[int] = None):
        self.source, self.reason, self.status = source, reason, status
        super().__init__(f"{source}: {reason}" + (f" (HTTP {status})" if status else ""))


class NotFound(SourceError):
    def __init__(self, source: str, reason: str = "not_found"):
        super().__init__(source, reason, 404)


class NotConfigured(SourceError):
    def __init__(self, source: str):
        super().__init__(source, "not_configured")


@dataclass(frozen=True)
class Request:
    url: str
    params: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    # (user, password) for HTTP basic auth, or None
    auth: Optional[tuple] = None
    # Parameter names holding a key: never recorded, never shown
    secret_params: tuple = ()

    @property
    def host(self) -> str:
        return urlsplit(self.url).hostname or ""

    def key(self) -> str:
        """The request without its secrets: how recordings are named."""
        public = sorted((k, v) for k, v in self.params.items() if k not in self.secret_params)
        query = "&".join(f"{k}={v}" for k, v in public)
        return self.url + (f"?{query}" if query else "")


@dataclass(frozen=True)
class Response:
    status: int
    content: bytes
    content_type: str = ""


Transport = Callable[[Request], Response]


# Where a source may send us on: Companies House hands its documents out
# from Amazon S3. A redirect goes on without the query string or credentials.
REDIRECT_HOST_SUFFIXES = (".amazonaws.com", ".company-information.service.gov.uk")
MAX_REDIRECTS = 3


def redirect_allowed(url: str) -> bool:
    parts = urlsplit(url)
    return parts.scheme == "https" and any((parts.hostname or "").endswith(s) for s in REDIRECT_HOST_SUFFIXES)


def _requests_transport(req: Request) -> Response:
    import requests

    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate", **req.headers}
    url, params, auth = req.url, dict(req.params), req.auth
    try:
        for _ in range(MAX_REDIRECTS + 1):
            with requests.get(url, params=params, headers=headers, auth=auth, timeout=TIMEOUT_S,
                              stream=True, allow_redirects=False) as resp:
                if resp.is_redirect:
                    target = resp.headers.get("Location", "")
                    if not redirect_allowed(target):
                        raise SourceError(req.host, "redirect_refused")
                    # Never carry a key on: the next host gets neither params nor auth
                    url, params, auth = target, {}, None
                    continue
                chunks, size = [], 0
                for chunk in resp.iter_content(64 * 1024):
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise SourceError(req.host, "response_too_large")
                    chunks.append(chunk)
                return Response(resp.status_code, b"".join(chunks), resp.headers.get("Content-Type", ""))
        raise SourceError(req.host, "redirect_refused")
    except requests.RequestException:
        # The exception's text holds the URL, and so possibly a key
        raise SourceError(req.host, "unreachable") from None


_transport: list = [_requests_transport]
_paced = [True]
_last_call: dict[str, float] = {}
_pace_lock = threading.Lock()


def use_transport(transport: Optional[Transport], *, paced: bool = False) -> None:
    """Replace the transport (tests replay, unpaced; the recorder calls the
    real sources, paced); None restores the real one."""
    _transport[0] = transport or _requests_transport
    _paced[0] = transport is None or paced


def _pace(host: str) -> None:
    if not _paced[0]:
        return
    interval = MIN_INTERVAL_S.get(host, DEFAULT_INTERVAL_S)
    with _pace_lock:                       # book this host's next slot ...
        now = time.monotonic()
        slot = max(now, _last_call.get(host, 0.0) + interval)
        _last_call[host] = slot
    if slot > now:                         # ... and wait for it outside the lock
        time.sleep(slot - now)


def get(source: str, url: str, *, params: Optional[Mapping[str, str]] = None,
        headers: Optional[Mapping[str, str]] = None, auth: Optional[tuple] = None,
        secret_params: tuple = (), allow_404: bool = False) -> Optional[Response]:
    """GET ``url``; the response, or None for a 404 when ``allow_404``.
    Any other failure raises ``SourceError`` naming ``source``."""
    req = Request(url, dict(params or {}), dict(headers or {}), auth, secret_params)
    _pace(req.host)
    try:
        resp = _transport[0](req)
    except SourceError as exc:
        raise SourceError(source, exc.reason, exc.status) from None
    if resp.status == 404:
        if allow_404:
            return None
        raise NotFound(source)
    if resp.status in (401, 403):
        raise SourceError(source, "refused", resp.status)
    if resp.status == 429:
        raise SourceError(source, "rate_limited", resp.status)
    if resp.status >= 400:
        raise SourceError(source, "failed", resp.status)
    return resp


def get_json(source: str, url: str, **kwargs):
    import json

    resp = get(source, url, **kwargs)
    if resp is None:
        return None
    try:
        return json.loads(resp.content)
    except ValueError:
        raise SourceError(source, "unreadable") from None
