"""The growth calibrator's answers the browser tests replay (PLAN.md 5.5).

    python -m tests.e2e_growth           # rewrites web/e2e/fixtures/growth.json

CI's browser tests store no economic data (the nightly ``economy-refresh``
never runs there), so the real endpoint answers every deal with
"no_growth". To show the range, they replay what ``POST /api/ml/growth``
gives with the recorded economic series (tests/fixtures/economy, judged on
the day they were recorded, as tests.e2e_benchmarks does).
tests/test_growth_calibrator.py fails when the file is stale.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
OUT = ROOT / "web" / "e2e" / "fixtures" / "growth.json"
# name: deal inputs. A US software deal (its own sector), a German machinery
# deal (Europe's industrials) and a hold the card did not test
CASES = {
    "us": {"country": "US", "industry": "software_system_application", "hold": 5},
    "de": {"country": "DE", "industry": "machinery", "currency": "EUR", "hold": 5},
    "untested": {"country": "DE", "industry": "machinery", "currency": "EUR", "hold": 9},
}


def build() -> dict:
    from api.auth import AuthUser, require_user
    from api.main import app
    from tests.e2e_benchmarks import recorded_sources
    _, series, today = recorded_sources()
    signed_in = require_user in app.dependency_overrides
    out = {}
    try:
        if not signed_in:
            app.dependency_overrides[require_user] = lambda: AuthUser(subject="dev:fixtures", is_dev=True)
        client = TestClient(app)
        for name, inputs in CASES.items():
            with mock.patch("api.routers.integrations.is_configured", lambda: True), \
                    mock.patch("db.economy.all_series", lambda: (series, None)), \
                    mock.patch("api.routers.integrations._today", lambda: today):
                resp = client.post("/api/ml/growth", json={"inputs": inputs})
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
