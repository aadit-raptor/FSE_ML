"""The starting-figure and risk answers the browser tests replay (PLAN.md 4.3, 4.4).

    python -m tests.e2e_benchmarks          # rewrites web/e2e/fixtures/benchmarks.json

CI's browser tests have no network to the sources and no scheduled refresh,
so they answer ``/api/benchmarks/industries``, ``/api/benchmarks/starting``
and ``/api/benchmarks/risk`` from this file: what the real endpoints give for
the recorded workbooks and history (tests/fixtures/benchmarks) and economic
series (tests/fixtures/economy),
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
HISTORY = ROOT / "tests" / "fixtures" / "benchmarks" / "history"
OUT = ROOT / "web" / "e2e" / "fixtures" / "benchmarks.json"
# (country, industry, currency): a German industrials deal, a thin group
# falling back to global, and the test account's own country
CASES = (("DE", "machinery", "EUR"), ("CA", "shipbuilding_marine", "CAD"), ("GB", "all", "GBP"),
         ("GB", "machinery", "GBP"), ("IN", "machinery", "INR"))
# The risk ranges (PLAN.md 4.4): a UK and an Indian machinery deal
RISK_CASES = (("GB", "machinery", "GBP"), ("IN", "machinery", "INR"))


def recorded_sources() -> tuple[dict, dict, object]:
    """The industry tables (with history) and economic series the recorded
    fixtures give, and the day they were recorded (tests.ml_regional_deals
    reads them too)."""
    from benchmarks import damodaran, history
    from economy import connectors, refresh
    today = recorded_on()
    replays = [replay(FIXTURES), replay(HISTORY),
               *(replay(ECONOMY_FIXTURES / c) for c in ("economy_open", "economy_fred"))]

    def answer(req):
        for r in replays:
            resp = r(req)
            if resp.status != 404:
                return resp
        return resp
    http.use_transport(answer)
    try:
        tables = {t.name: t for t in (call() for _, call in damodaran.every_table())}
        tables[history.MACRO_TABLE] = history.read_macro(today)
        found_history, _ = history.fill(tables, today, batch=10_000)
        tables.update({t.name: t for t in found_history})
        with mock.patch.dict("os.environ", {connectors.FRED_KEY_ENV: "recorded"}):
            found, _ = refresh._read(today)
    finally:
        http.use_transport(None)
    return tables, {s.key: s for s in found}, today


def build() -> dict:
    from api.auth import AuthUser, require_user
    from api.main import app
    tables, series, today = recorded_sources()
    signed_in = require_user in app.dependency_overrides
    with (mock.patch("api.routers.benchmarks._stored", lambda history=False: (tables, None, series)),
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
            risk = {}
            for country, industry, currency in RISK_CASES:
                resp = client.get("/api/benchmarks/risk",
                                  params={"country": country, "industry": industry, "currency": currency})
                assert resp.status_code == 200, resp.text
                risk[f"{country}|{industry}|{currency}"] = resp.json()
        finally:
            if not signed_in:
                app.dependency_overrides.pop(require_user, None)
    return {"industries": industries.json(), "starting": starting, "risk": risk}


def main() -> None:
    OUT.write_text(json.dumps(build(), indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
