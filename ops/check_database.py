"""Check a deployed API's database: reachable, migrations current, storage
under the free-plan warning level (PLAN.md 1.3).

    python3 ops/check_database.py https://fse-api-staging.onrender.com

Calls ``/api/health/database`` (waiting for a sleeping free service to wake),
prints a one-line summary and exits 1 on any problem. Used by live.yml
(production, daily) and staging.yml (after each deploy). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Optional


def problems(result: dict) -> list[str]:
    """What's wrong with a /api/health/database result; empty when healthy."""
    found = []
    status = result.get("status")
    if status != "ok":
        detail = result.get("error") or status
        found.append(f"database status {status} ({detail})")
        return found
    migrations = result.get("migrations") or {}
    if migrations.get("status") != "current":
        found.append(f"migrations {migrations.get('status')}: at {migrations.get('revision')}, "
                     f"latest {migrations.get('head')}")
    storage = result.get("storage") or {}
    if storage.get("warning"):
        found.append(f"storage at {storage.get('used_fraction', 0):.0%} of the free "
                     f"{storage.get('limit_bytes', 0) // (1024 * 1024)} MB: plan cleanup or phase 12.2")
    return found


def fetch(api: str, attempts: int = 4, timeout: int = 120) -> dict:
    url = api.rstrip("/") + "/api/health/database"
    last: Exception = RuntimeError("no attempt")
    for i in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code == 503:  # the API answered: the database is down
                return json.load(e)
            last = e
        except (urllib.error.URLError, TimeoutError) as e:  # still waking
            last = e
        time.sleep(10 * (i + 1))
    raise SystemExit(f"could not reach {url}: {last}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("api")
    args = parser.parse_args(argv)
    result = fetch(args.api)
    storage = result.get("storage") or {}
    print(f"database {result.get('status')}; migrations {(result.get('migrations') or {}).get('status')}; "
          f"storage {storage.get('database_bytes', 0) / 1024 / 1024:.1f} MB "
          f"({storage.get('used_fraction', 0):.1%} of free); connect attempts {result.get('connect_attempts')}; "
          f"{result.get('latency_ms')} ms")
    found = problems(result)
    for p in found:
        print(f"::error::{p}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
