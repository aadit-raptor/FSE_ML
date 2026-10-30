"""Who someone is: their name for a team's screens, their email address for a mention (PLAN.md 7.2).

Nothing personal is stored (CLAUDE.md, accounts): a person is their Clerk user id
(``users.subject``). Names come from Clerk's Backend API when a screen asks
(``names``, kept in this process's memory for ``NAME_CACHE_S`` so a team's page
doesn't spend a call a member), and an email address only when an email is
about to be sent (``email``, never kept). Neither is ever logged.

Three directories (``get_directory``):

- ``ClerkDirectory`` when ``CLERK_SECRET_KEY`` is set: ``GET /v1/users`` with
  the ids asked for, the key in a header only;
- the development sign-in's people (``dev:<name>``, local runs and tests): the
  name is ``<name>`` and the address ``<name>@example.test``;
- otherwise no names (the screens say "A team member") and no addresses (a
  mention is kept and sent once a directory is set up, within its time).

A failing directory answers no names rather than failing the page (the screens
fall back to "A team member"); ``email`` raises ``DirectoryUnavailable``, so the
mail sender tries again later instead of deciding the person has no address.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Iterable, Mapping, Optional, Protocol

from api.observability import log_event

CLERK_API = "https://api.clerk.com/v1"
TIMEOUT_S = 10.0
NAME_CACHE_S = 600
MAX_CACHED = 5000
# Clerk's list endpoint takes up to this many ids at once
BATCH = 100
DEV_PREFIX = "dev:"
DEV_DOMAIN = "example.test"


class DirectoryUnavailable(RuntimeError):
    """The directory couldn't be asked just now (an email waits and is tried again)."""


class Directory(Protocol):
    def names(self, subjects: Iterable[str]) -> dict[str, str]: ...

    def email(self, subject: str) -> Optional[str]: ...


# A GET: (url, params, headers) -> (status, JSON body or None); tests swap it
Get = Callable[[str, list, Mapping[str, str]], tuple[int, object]]


def _requests_get(url: str, params: list, headers: Mapping[str, str]) -> tuple[int, object]:
    import requests

    answer = requests.get(url, params=params, headers=dict(headers), timeout=TIMEOUT_S, allow_redirects=False)
    try:
        return answer.status_code, answer.json()
    except ValueError:
        return answer.status_code, None


def _full_name(user: Mapping) -> Optional[str]:
    parts = [p.strip() for p in (user.get("first_name"), user.get("last_name")) if isinstance(p, str) and p.strip()]
    name = " ".join(parts) or (user.get("username") if isinstance(user.get("username"), str) else "")
    return name[:120] or None


def _primary_email(user: Mapping) -> Optional[str]:
    primary = user.get("primary_email_address_id")
    for entry in user.get("email_addresses") or []:
        if isinstance(entry, Mapping) and entry.get("id") == primary:
            address = entry.get("email_address")
            verified = (entry.get("verification") or {}).get("status") == "verified"
            return address if isinstance(address, str) and verified else None
    return None


class ClerkDirectory:
    """Clerk's Backend API, read with the instance's secret key."""

    def __init__(self, secret_key: str, get: Get = _requests_get):
        self._headers = {"Authorization": f"Bearer {secret_key}", "Accept": "application/json"}
        self._get = get
        self._lock = threading.Lock()
        self._names: dict[str, tuple[float, Optional[str]]] = {}

    def _users(self, subjects: list[str]) -> Optional[list[Mapping]]:
        try:
            status, body = self._get(f"{CLERK_API}/users", [("user_id", s) for s in subjects]
                                     + [("limit", str(BATCH))], self._headers)
        except Exception as exc:   # an outage is not the page's problem
            log_event("directory_unavailable", logging.WARNING, error=type(exc).__name__)
            return None
        if status != 200 or not isinstance(body, list):
            log_event("directory_refused", logging.WARNING, status=status)
            return None
        return [u for u in body if isinstance(u, Mapping)]

    def names(self, subjects: Iterable[str]) -> dict[str, str]:
        wanted = sorted({s for s in subjects if isinstance(s, str) and s.startswith("user_")})
        now = time.monotonic()
        found: dict[str, str] = {}
        missing = []
        with self._lock:
            for s in wanted:
                held = self._names.get(s)
                if held and now - held[0] < NAME_CACHE_S:
                    if held[1]:
                        found[s] = held[1]
                else:
                    missing.append(s)
        for start in range(0, len(missing), BATCH):
            chunk = missing[start:start + BATCH]
            users = self._users(chunk)
            if users is None:
                continue
            got = {u.get("id"): _full_name(u) for u in users}
            with self._lock:
                if len(self._names) > MAX_CACHED:
                    self._names.clear()
                for s in chunk:
                    self._names[s] = (now, got.get(s))
            found.update({s: n for s, n in got.items() if isinstance(s, str) and n and s in chunk})
        return found

    def email(self, subject: str) -> Optional[str]:
        if not subject.startswith("user_"):
            return None
        users = self._users([subject])
        if users is None:
            raise DirectoryUnavailable()
        for user in users:
            if user.get("id") == subject:
                return _primary_email(user)
        return None


class DevDirectory:
    """The development sign-in's people: ``dev:alice`` is Alice at ``alice@example.test``."""

    def names(self, subjects: Iterable[str]) -> dict[str, str]:
        return {s: s[len(DEV_PREFIX):] for s in subjects if isinstance(s, str) and s.startswith(DEV_PREFIX)}

    def email(self, subject: str) -> Optional[str]:
        if not subject.startswith(DEV_PREFIX):
            return None
        return f"{subject[len(DEV_PREFIX):].lower()}@{DEV_DOMAIN}"


class NoDirectory:
    def names(self, subjects: Iterable[str]) -> dict[str, str]:
        return {}

    def email(self, subject: str) -> Optional[str]:
        return None


_lock = threading.Lock()
_directory: Optional[Directory] = None


def configured_kind() -> str:
    """Which directory this server uses: ``clerk``, ``dev`` or ``none``."""
    if os.environ.get("CLERK_SECRET_KEY", "").strip():
        return "clerk"
    from api.auth import clerk_issuer, dev_auth_enabled
    if not clerk_issuer() and dev_auth_enabled():
        return "dev"
    return "none"


def get_directory() -> Directory:
    global _directory
    with _lock:
        if _directory is None:
            kind = configured_kind()
            _directory = (ClerkDirectory(os.environ["CLERK_SECRET_KEY"].strip()) if kind == "clerk"
                          else DevDirectory() if kind == "dev" else NoDirectory())
        return _directory


def use_directory(directory: Optional[Directory]) -> None:
    """Use this directory (tests), or None to choose again from the environment."""
    global _directory
    with _lock:
        _directory = directory
