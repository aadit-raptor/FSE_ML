"""The admin switch that shows or hides the reference library (PLAN.md 4.5).

The library is on unless one of two things turns it off:

- an **administrator** switches it off in the app (Library -> Coverage). The
  choice is stored in the database (``db/flags.py``), so it holds for every
  account in that environment and survives restarts. Administrators are the
  accounts listed in ``FSE_ADMINS`` (Clerk user ids, comma-separated, set on
  the Render service); with none listed nobody can switch it;
- the server's own ``FSE_EXAMPLE_LIBRARY=0``, which forces it off whatever
  the stored choice, for an operator who wants it gone without signing in.

Off means hidden from everyone: the Library screens, the example deals in
Backtest and the library's endpoints answer that it is off. Nothing outside
the library reads it, so every other screen and every model result is the
same either way (``tests/test_library.py``, ``web/e2e/library.spec.ts``).

The stored choice is read at most once per ``CACHE_S`` per process, so the
library's own screens cost one database round trip a minute, not one a call.
While the database can't be reached the default (on) applies.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Optional

from db import DatabaseUnavailable, flags
from db.engine import is_configured as database_configured

FLAG = "library"
DEFAULT_ON = True
CACHE_S = 60.0

_cache: dict = {}
_lock = threading.Lock()


class NotAdmin(PermissionError):
    """Only an administrator may switch the library."""


class LockedOff(RuntimeError):
    """``FSE_EXAMPLE_LIBRARY`` forces the library off on this server."""


class NoStore(RuntimeError):
    """No database to keep the choice in (a local run without one)."""


def server_allows() -> bool:
    return os.environ.get("FSE_EXAMPLE_LIBRARY", "1").strip().lower() not in {"0", "false", "no", "off"}


def admins() -> frozenset[str]:
    return frozenset(s.strip() for s in os.environ.get("FSE_ADMINS", "").split(",") if s.strip())


def is_admin(subject: Optional[str]) -> bool:
    return bool(subject) and subject in admins()


def reset_cache() -> None:
    with _lock:
        _cache.clear()


def _stored() -> Optional[flags.Flag]:
    if not database_configured():
        return None
    with _lock:
        hit = _cache.get(FLAG)
        if hit and time.monotonic() - hit[0] < CACHE_S:
            return hit[1]
    try:
        found = flags.get(FLAG)
    except DatabaseUnavailable:
        # Showing the library is the default; a database outage shouldn't hide
        # the examples, which need no database. Not cached, so it is asked again.
        return None
    with _lock:
        _cache[FLAG] = (time.monotonic(), found)
    return found


def stored_choice() -> bool:
    found = _stored()
    return found.enabled if found else DEFAULT_ON


def enabled() -> bool:
    """Whether the library is shown."""
    return server_allows() and stored_choice()


def state(subject: Optional[str]) -> dict:
    """What the screens need: shown or not, and whether this caller may switch it."""
    found = _stored() if server_allows() else None
    return {
        "enabled": server_allows() and (found.enabled if found else DEFAULT_ON),
        "locked_off": not server_allows(),
        "can_switch": is_admin(subject) and database_configured() and server_allows(),
        "updated_at": found.updated_at if found else None,
    }


def switch(subject: str, enabled_: bool) -> dict:
    """Store an administrator's choice and answer the new state."""
    if not is_admin(subject):
        raise NotAdmin(subject)
    if not server_allows():
        raise LockedOff()
    if not database_configured():
        raise NoStore()
    flags.put(subject, FLAG, enabled_, default=DEFAULT_ON)
    reset_cache()
    return state(subject)
