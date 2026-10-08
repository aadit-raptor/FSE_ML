"""The deal risk answers the browser tests replay (PLAN.md 5.2).

    python -m tests.e2e_deal_risk           # rewrites web/e2e/fixtures/deal-risk.json

CI's browser tests have no stored industry averages (the nightly
``benchmarks-refresh`` never runs there), so the real endpoint answers every
deal with "not enough data". To show the tile with figures, they replay what
``POST /api/ml/deal-risk`` gives for the recorded January 2026 averages
(ml/evaluation/data/peer_tables.json) and the repository's reference
transactions as the library. tests/test_deal_risk.py fails when the file is
stale.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).parent.parent
OUT = ROOT / "web" / "e2e" / "fixtures" / "deal-risk.json"
# name: (deal inputs, library on): a US deal where the card shows the score
# and the library lists deals like it, a German one where only the
# comparison shows, a thin Canadian group, and the US deal with the library off
CASES = {
    "us": ({"country": "US", "industry": "retail_special_lines", "currency": "USD", "ebitda": 500.0}, True),
    "us_library_off": ({"country": "US", "industry": "retail_special_lines", "currency": "USD", "ebitda": 500.0},
                       False),
    "de": ({"country": "DE", "industry": "machinery", "currency": "EUR"}, False),
    "thin": ({"country": "CA", "industry": "shipbuilding_marine", "currency": "CAD"}, False),
}


def build() -> dict:
    from api.auth import AuthUser, require_user
    from api.main import app
    from library import references
    from ml.evaluation.deal_risk import peer_tables
    tables = peer_tables()
    signed_in = require_user in app.dependency_overrides
    out = {}
    try:
        if not signed_in:
            app.dependency_overrides[require_user] = lambda: AuthUser(subject="dev:fixtures", is_dev=True)
        client = TestClient(app)
        for name, (inputs, library_on) in CASES.items():
            library = references.repository_deals() if library_on else None
            with mock.patch("api.routers.integrations._peer_data", lambda currency, lib=library: (tables, lib, 1.0)):
                resp = client.post("/api/ml/deal-risk", json={"inputs": inputs})
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
