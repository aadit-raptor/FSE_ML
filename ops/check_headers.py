"""Header scan of a deployed copy (PLAN.md 1.7): security headers, the content
security policy and CORS, on the web app and the API.

    python3 ops/check_headers.py --web https://fse-ml.vercel.app --api https://fse-api.onrender.com

- Web: ``/sign-in`` (a page anyone may open) is fetched twice. It must carry a
  strict content security policy with a fresh nonce each time, HSTS, nosniff,
  no framing, a strict referrer policy, a permissions policy and no
  ``X-Powered-By``. ``/api/health`` through the web app's proxy must carry the
  API's own headers.
- API: ``/api/health`` must carry the API headers (``default-src 'none'``,
  ``no-store``), and a cross-origin preflight from an unknown origin must not
  be allowed.

Prints every finding and exits 1 on any. Redirects are never followed, so a
Vercel bypass secret (``VERCEL_AUTOMATION_BYPASS_SECRET``, for staging
previews) only ever goes to the web URL given. Standard library only; used by
live.yml (production) and staging.yml.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Mapping, Optional

MIN_HSTS_S = 180 * 24 * 3600
STRICT_REFERRERS = {"no-referrer", "same-origin", "strict-origin", "strict-origin-when-cross-origin"}
UNKNOWN_ORIGIN = "https://header-scan.invalid"


def parse_csp(value: str) -> dict[str, list[str]]:
    directives: dict[str, list[str]] = {}
    for part in value.split(";"):
        tokens = part.split()
        if tokens:
            directives.setdefault(tokens[0].lower(), tokens[1:])
    return directives


def _get(headers: Mapping[str, str], name: str) -> Optional[str]:
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def common_problems(headers: Mapping[str, str]) -> list[str]:
    """Headers every response must have."""
    found = []
    hsts = _get(headers, "Strict-Transport-Security") or ""
    match = re.search(r"max-age=(\d+)", hsts)
    if not match or int(match.group(1)) < MIN_HSTS_S:
        found.append(f"Strict-Transport-Security missing or under 180 days ({hsts or 'absent'})")
    if (_get(headers, "X-Content-Type-Options") or "").lower() != "nosniff":
        found.append("X-Content-Type-Options is not nosniff")
    if (_get(headers, "X-Frame-Options") or "").upper() != "DENY":
        found.append("X-Frame-Options is not DENY")
    referrer = (_get(headers, "Referrer-Policy") or "").lower()
    if referrer not in STRICT_REFERRERS:
        found.append(f"Referrer-Policy is not strict ({referrer or 'absent'})")
    if not _get(headers, "Permissions-Policy"):
        found.append("Permissions-Policy is missing")
    if _get(headers, "X-Powered-By"):
        found.append("X-Powered-By reveals the framework")
    return found


def page_problems(headers: Mapping[str, str]) -> list[str]:
    """A web page: common headers plus a strict, nonce-based policy."""
    found = common_problems(headers)
    csp = _get(headers, "Content-Security-Policy")
    if not csp:
        return found + ["Content-Security-Policy is missing"]
    d = parse_csp(csp)
    scripts = d.get("script-src", d.get("default-src"))
    if scripts is None:
        found.append("CSP has no script-src or default-src")
    else:
        if not any(s.startswith("'nonce-") for s in scripts):
            found.append("CSP script-src has no nonce")
        if "'strict-dynamic'" not in scripts:
            found.append("CSP script-src lacks 'strict-dynamic'")
        for weak in ("'unsafe-eval'", "*", "https:", "http:", "data:"):
            if weak in scripts:
                found.append(f"CSP script-src allows {weak}")
        if "'unsafe-inline'" in scripts and not any(s.startswith("'nonce-") for s in scripts):
            found.append("CSP script-src allows 'unsafe-inline'")
    if d.get("object-src") != ["'none'"]:
        found.append("CSP object-src is not 'none'")
    if d.get("frame-ancestors") != ["'none'"]:
        found.append("CSP frame-ancestors is not 'none'")
    if d.get("base-uri") not in (["'self'"], ["'none'"]):
        found.append("CSP base-uri is not restricted")
    if "default-src" not in d:
        found.append("CSP has no default-src")
    return found


def nonce(headers: Mapping[str, str]) -> Optional[str]:
    match = re.search(r"'nonce-([^']+)'", _get(headers, "Content-Security-Policy") or "")
    return match.group(1) if match else None


def api_problems(headers: Mapping[str, str]) -> list[str]:
    """An API answer: common headers, nothing loadable, never cached."""
    found = common_problems(headers)
    d = parse_csp(_get(headers, "Content-Security-Policy") or "")
    if d.get("default-src") != ["'none'"]:
        found.append("API CSP is not default-src 'none'")
    if d.get("frame-ancestors") != ["'none'"]:
        found.append("API CSP frame-ancestors is not 'none'")
    if "no-store" not in (_get(headers, "Cache-Control") or ""):
        found.append("API answers may be cached (Cache-Control lacks no-store)")
    return found


def cors_problems(status: int, headers: Mapping[str, str]) -> list[str]:
    """A preflight from an origin that isn't the app's."""
    allowed = _get(headers, "Access-Control-Allow-Origin")
    if allowed in ("*", UNKNOWN_ORIGIN):
        return [f"CORS allows an unknown origin ({allowed})"]
    if _get(headers, "Access-Control-Allow-Credentials"):
        return ["CORS allows credentials for an unknown origin"]
    return []


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: ARG002
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def fetch(url: str, *, method: str = "GET", headers: Optional[dict] = None,
          attempts: int = 4, timeout: int = 120) -> tuple[int, dict[str, str]]:
    """Status and headers, waiting for a sleeping free service to wake."""
    last: Exception = RuntimeError("no attempt")
    for i in range(attempts):
        request = urllib.request.Request(url, method=method, headers={"User-Agent": "fse-header-scan", **(headers or {})})
        try:
            with _opener.open(request, timeout=timeout) as resp:
                return resp.status, dict(resp.headers.items())
        except urllib.error.HTTPError as e:
            if e.code < 500:
                return e.code, dict(e.headers.items())
            last = e
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
        time.sleep(10 * (i + 1))
    raise SystemExit(f"could not reach {url}: {last}")


def scan(web: Optional[str], api: Optional[str], bypass: Optional[str]) -> list[str]:
    found: list[str] = []
    if web:
        web = web.rstrip("/")
        extra = {"x-vercel-protection-bypass": bypass} if bypass else {}
        nonces = []
        for _ in range(2):
            status, headers = fetch(f"{web}/sign-in", headers=extra)
            if status != 200:
                found.append(f"web /sign-in answered {status}, not 200")
                break
            nonces.append(nonce(headers))
        else:
            found += [f"web page: {p}" for p in page_problems(headers)]
            if nonces[0] and nonces[0] == nonces[1]:
                found.append("web page: the CSP nonce is reused between requests")
        status, headers = fetch(f"{web}/api/health", headers=extra)
        found += [f"web /api/health: {p}" for p in api_problems(headers)]
    if api:
        api = api.rstrip("/")
        status, headers = fetch(f"{api}/api/health")
        if status != 200:
            found.append(f"API /api/health answered {status}")
        found += [f"API: {p}" for p in api_problems(headers)]
        status, headers = fetch(f"{api}/api/deal/run", method="OPTIONS", headers={
            "Origin": UNKNOWN_ORIGIN, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "authorization,content-type"})
        found += [f"API: {p}" for p in cors_problems(status, headers)]
    return found


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--web", help="the web app's URL")
    parser.add_argument("--api", help="the API's URL")
    args = parser.parse_args(argv)
    if not (args.web or args.api):
        parser.error("give --web, --api or both")
    found = scan(args.web, args.api, os.environ.get("VERCEL_AUTOMATION_BYPASS_SECRET") or None)
    for p in found:
        print(f"::error::{p}")
    print(f"header scan: {len(found)} problem(s)" + ("" if found else " -- passed"))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
