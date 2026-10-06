"""The economic data answers the browser tests replay (PLAN.md 4.2).

    python -m tests.e2e_economy          # rewrites web/e2e/fixtures/economy.json

CI's browser tests have no network to the data sources and no nightly
refresh, so they answer ``/api/economy/reference-rates`` from this file: what
the real endpoint gives for the recorded series in tests/fixtures/economy,
judged on the day they were recorded. tests/test_economy.py fails when the
file is stale.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from companies import http
from companies.record import replay

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "economy"
OUT = ROOT / "web" / "e2e" / "fixtures" / "economy.json"


def recorded_on() -> date:
    return date.fromisoformat(json.loads((FIXTURES / "economy_open" / "index.json").read_text())["_recorded_on"])


def build() -> dict:
    from api.auth import AuthUser, require_user
    from api.main import app
    from economy import connectors, refresh
    today = recorded_on()
    replays = [replay(FIXTURES / case) for case in ("economy_open", "economy_fred")]

    def answer(req):
        for r in replays:
            resp = r(req)
            if resp.status != 404:
                return resp
        return resp
    http.use_transport(answer)
    try:
        with mock.patch.dict("os.environ", {connectors.FRED_KEY_ENV: "recorded"}):
            found, _ = refresh._read(today)
    finally:
        http.use_transport(None)
    stored = {s.key: s for s in found}
    signed_in = require_user in app.dependency_overrides
    with (mock.patch("api.routers.economy._stored", lambda: (stored, None)),
          mock.patch("api.routers.economy._today", lambda: today)):
        try:
            if not signed_in:
                app.dependency_overrides[require_user] = lambda: AuthUser(subject="dev:fixtures", is_dev=True)
            resp = TestClient(app).get("/api/economy/reference-rates")
            assert resp.status_code == 200, resp.text
            body = resp.json()
        finally:
            if not signed_in:
                app.dependency_overrides.pop(require_user, None)
    return {"reference_rates": body}


def main() -> None:
    OUT.write_text(json.dumps(build(), indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
