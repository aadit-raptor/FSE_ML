"""The starting-figure answers the browser tests replay (PLAN.md 4.3).

    python -m tests.e2e_benchmarks          # rewrites web/e2e/fixtures/benchmarks.json

CI's browser tests have no network to the sources and no scheduled refresh,
so they answer ``/api/benchmarks/industries`` and ``/api/benchmarks/starting``
from this file: what the real endpoints give for the recorded workbooks
(tests/fixtures/benchmarks) and economic series (tests/fixtures/economy),
judged on the day the series were recorded. tests/test_benchmarks.py fails
when the file is stale.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from companies import http
from companies.record import replay
from tests.e2e_economy import FIXTURES as ECONOMY_FIXTURES, recorded_on

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "benchmarks" / "damodaran"
OUT = ROOT / "web" / "e2e" / "fixtures" / "benchmarks.json"
# (country, industry, currency): a German industrials deal, a thin group
# falling back to global, and the test account's own country
CASES = (("DE", "machinery", "EUR"), ("CA", "shipbuilding_marine", "CAD"), ("GB", "all", "GBP"))


def build() -> dict:
    from api.auth import AuthUser, require_user
    from api.main import app
    from benchmarks import damodaran
    from economy import connectors, refresh
    today = recorded_on()
    replays = [replay(FIXTURES), *(replay(ECONOMY_FIXTURES / c) for c in ("economy_open", "economy_fred"))]

    def answer(req):
        for r in replays:
            resp = r(req)
            if resp.status != 404:
                return resp
        return resp
    http.use_transport(answer)
    try:
        tables = {t.name: t for t in (call() for _, call in damodaran.every_table())}
        with mock.patch.dict("os.environ", {connectors.FRED_KEY_ENV: "recorded"}):
            found, _ = refresh._read(today)
    finally:
        http.use_transport(None)
    series = {s.key: s for s in found}
    signed_in = require_user in app.dependency_overrides
    with (mock.patch("api.routers.benchmarks._stored", lambda: (tables, None, series)),
          mock.patch("api.routers.benchmarks._today", lambda: today)):
        try:
            if not signed_in:
                app.dependency_overrides[require_user] = lambda: AuthUser(subject="dev:fixtures", is_dev=True)
            client = TestClient(app)
            industries = client.get("/api/benchmarks/industries")
            assert industries.status_code == 200, industries.text
            starting = {}
            for country, industry, currency in CASES:
                resp = client.get("/api/benchmarks/starting",
                                  params={"country": country, "industry": industry, "currency": currency})
                assert resp.status_code == 200, resp.text
                starting[f"{country}|{industry}|{currency}"] = resp.json()
        finally:
            if not signed_in:
                app.dependency_overrides.pop(require_user, None)
    return {"industries": industries.json(), "starting": starting}


def main() -> None:
    OUT.write_text(json.dumps(build(), indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
