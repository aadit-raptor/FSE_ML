"""The company answers the browser tests replay (PLAN.md 4.1b).

    python -m tests.e2e_companies        # rewrites web/e2e/fixtures/companies.json

CI's browser tests have no network to the registers, so they answer
``/api/companies/*`` from this file. Every answer is what the API gives
today for the recorded filings in tests/fixtures/companies (and SAP's
company facts in tests/fixtures/edgar), run through the real endpoints;
tests/test_company_use.py fails when the file is stale.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from fastapi.testclient import TestClient

from companies import companies_house, edinet, http, sec
from companies.record import replay

ROOT = Path(__file__).parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "companies"
SAP_FACTS = ROOT / "tests" / "fixtures" / "edgar" / "sap_companyfacts.json"
OUT = ROOT / "web" / "e2e" / "fixtures" / "companies.json"

# (fixture case, query) for each search the browser tests make
SEARCHES = {"Toyota": "search_names", "MCD": "sec_mcd", "GB00BLGZ9862": "esef_tesco", "SAP": "sap",
            "00482197": "ch_cambridge_united", "Zzyzx": "search_names"}
# (fixture case, source, id) for each load
LOADS = (
    ("esef_tesco", "esef", "2138002P5RNKC5W2JZ46"),
    ("edinet_toyota", "edinet", "E02144"),
    ("sap", "sec", "1000184"),
    ("ch_cambridge_united", "companies_house", "00482197"),
)


def replay_case(case: str) -> None:
    """Answer from a recorded case; EDINET's recorded day lists are indexed
    first, as the nightly refresh would have done."""
    if case == "sap":
        replay_sap()
        return
    sec.reset_cache()
    edinet.reset_cache()
    edinet.use_index(None)
    http.use_transport(replay(FIXTURES / case))
    index = json.loads((FIXTURES / case / "index.json").read_text(encoding="utf-8"))
    if os.environ.get(edinet.KEY_ENV):
        for key in index:
            if "documents.json?" in key:
                edinet.scan_day(date.fromisoformat(key.split("date=")[1][:10]))


def replay_sap() -> None:
    """SAP SE at the SEC: its recorded company facts (20-F, IFRS, euros) and
    the two small answers naming it. Anything else is a 404."""
    sec.reset_cache()
    edinet.reset_cache()
    edinet.use_index(None)
    answers = {
        sec.TICKERS_URL: {"0": {"cik_str": 1000184, "ticker": "SAP", "title": "SAP SE"}},
        sec.SUBMISSIONS_URL.format(cik="0001000184"): {"name": "SAP SE", "tickers": ["SAP"], "fiscalYearEnd": "1231",
                                                       "addresses": {"business": {"stateOrCountry": "2M"}}},
    }
    facts_url = sec.FACTS_URL.format(cik="0001000184")
    facts = SAP_FACTS.read_bytes()

    def transport(req: http.Request) -> http.Response:
        if req.url == facts_url:
            return http.Response(200, facts, "application/json")
        if req.url in answers:
            return http.Response(200, json.dumps(answers[req.url]).encode(), "application/json")
        return http.Response(404, b"")
    http.use_transport(transport)


@contextmanager
def _keys():
    """Both keyed sources set up, as on the deployed API, and no database,
    so nothing is stored and no answer carries a time."""
    saved = {k: os.environ.get(k) for k in (companies_house.KEY_ENV, edinet.KEY_ENV, "DATABASE_URL")}
    os.environ.update({companies_house.KEY_ENV: "k", edinet.KEY_ENV: "k"})
    os.environ.pop("DATABASE_URL", None)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def build() -> dict:
    from api.auth import AuthUser, require_user
    from api.main import app
    client = TestClient(app)
    out: dict = {"search": {}, "load": {}}
    signed_in = require_user in app.dependency_overrides
    with _keys():
        try:
            if not signed_in:
                app.dependency_overrides[require_user] = lambda: AuthUser(subject="dev:fixtures", is_dev=True)
            for query, case in SEARCHES.items():
                replay_case(case)
                resp = client.get("/api/companies/search", params={"q": query})
                assert resp.status_code == 200, resp.text
                out["search"][query] = resp.json()
            for case, source, company_id in LOADS:
                replay_case(case)
                resp = client.post("/api/companies/load", json={"source": source, "id": company_id})
                assert resp.status_code == 200, resp.text
                body = resp.json()          # keyed as the screen asks: the id a search answered
                out["load"][f"{body['company']['source']}/{body['company']['id']}"] = body
        finally:
            if not signed_in:
                app.dependency_overrides.pop(require_user, None)
            http.use_transport(None)
            edinet.use_index(None)
            sec.reset_cache()
            edinet.reset_cache()
    return out


if __name__ == "__main__":
    OUT.write_text(json.dumps(build(), indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
