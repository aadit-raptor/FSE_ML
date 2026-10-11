"""The economic regime's answers the browser tests replay (PLAN.md 5.7).

    python -m tests.e2e_regime           # rewrites web/e2e/fixtures/regime.json

CI's browser tests never run the monthly ``regime-refresh``, so the real
endpoint answers every deal "no data". To show the tile, they replay what
``POST /api/ml/macro-regime`` gives for the fit of the recorded IMF figures
(tests/fixtures/regime/imf), judged on the day they were recorded.
tests/test_macro_regime.py fails when the file is stale.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
OUT = ROOT / "web" / "e2e" / "fixtures" / "regime.json"
FIXTURE = ROOT / "tests" / "fixtures" / "regime" / "imf"
# name: the deal's country. India (its own region), Germany (the euro
# area's) and Canada (the world standing in); the three the browser tests
# have recorded starting figures for (tests/e2e_benchmarks.py)
CASES = {"in": "IN", "de": "DE", "ca": "CA"}


def recorded_on() -> date:
    return date.fromisoformat(json.loads((FIXTURE / "index.json").read_text(encoding="utf-8"))["_recorded_on"])


def recorded_fit():
    """The stored table a refit on the recorded day would write."""
    from benchmarks.damodaran import Table
    from companies import http
    from companies.record import replay
    from ml import macro_regime as mr
    today = recorded_on()
    http.use_transport(replay(FIXTURE))
    try:
        return Table(mr.TABLE, today, mr.IMF_PAGE, mr.fit(mr.read(today), today))
    finally:
        http.use_transport(None)


def build() -> dict:
    from api.auth import AuthUser, require_user
    from api.main import app
    table, today = recorded_fit(), recorded_on()
    signed_in = require_user in app.dependency_overrides
    out = {}
    try:
        if not signed_in:
            app.dependency_overrides[require_user] = lambda: AuthUser(subject="dev:fixtures", is_dev=True)
        client = TestClient(app)
        for name, country in CASES.items():
            with mock.patch("api.routers.integrations.is_configured", lambda: True), \
                    mock.patch("db.benchmarks.table", lambda _name: table), \
                    mock.patch("api.routers.integrations._today", lambda: today):
                resp = client.post("/api/ml/macro-regime", json={"country": country})
            assert resp.status_code == 200, resp.text
            answer = resp.json()
            answer["model"]["commit"] = None        # the same file on every machine
            out[name] = answer
    finally:
        if not signed_in:
            app.dependency_overrides.pop(require_user, None)
    return out


def text() -> str:
    return json.dumps(build(), indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> None:
    OUT.write_text(text(), encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
