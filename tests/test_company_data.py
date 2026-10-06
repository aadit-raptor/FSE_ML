"""Company data stored, served and refreshed (PLAN.md 4.1): the tables, the
storage budget, the /api/companies endpoints and the scheduled refresh.

Loads replay recorded filings (tests/fixtures/companies), so the figures
are real; the database tests use a throwaway database (tests/conftest.py).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text, update

from api.limits import RUN_PATHS
from api.main import app
from companies import companies_house, edinet, http, refresh, sec, sources
from companies.edinet import IndexedReport
from companies.record import replay
from db import companies as store
from db import engine as db_engine
from db.models import Company, utc_now

FIXTURES = Path(__file__).parent / "fixtures" / "companies"
TESCO_LEI = "2138002P5RNKC5W2JZ46"
client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_sources(monkeypatch):
    monkeypatch.delenv(companies_house.KEY_ENV, raising=False)
    monkeypatch.delenv(edinet.KEY_ENV, raising=False)
    sec.reset_cache()
    edinet.reset_cache()
    edinet.use_index(None)
    http.use_transport(lambda req: pytest.fail(f"unrecorded request to {req.host}"))
    yield
    http.use_transport(None)
    edinet.use_index(None)
    sec.reset_cache()
    edinet.reset_cache()


def use(case: str) -> None:
    sec.reset_cache()
    edinet.reset_cache()
    http.use_transport(replay(FIXTURES / case))


def tesco():
    use("esef_tesco")
    return sources.fetch("esef", TESCO_LEI)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
def test_a_stored_company_reads_back_exactly(fresh_db):  # noqa: ARG001
    data = tesco()
    saved_at = store.save(data)
    back, refreshed_at = store.get("esef", TESCO_LEI)
    assert refreshed_at == saved_at
    assert back.ref == data.ref
    assert (back.currency, back.unit, back.accounting_standard, back.fiscal_year_end_month) == \
        ("GBP", "millions", "ifrs", 2)
    assert back.years == data.years                    # figures, period ends and filing links
    assert back.warnings == data.warnings
    assert store.get("esef", "0" * 20) is None


def test_storing_again_replaces_the_years(fresh_db):  # noqa: ARG001
    data = tesco()
    store.save(data)
    fewer = type(data)(data.ref, data.currency, data.accounting_standard, data.fiscal_year_end_month,
                       data.years[-1:], [])
    store.save(fewer)
    back, _ = store.get("esef", TESCO_LEI)
    assert [y.fiscal_year for y in back.years] == [2026] and back.warnings == []
    with db_engine.connect() as conn:
        assert conn.execute(text("SELECT count(*) FROM companies")).scalar_one() == 1
        assert conn.execute(text("SELECT count(*) FROM company_years")).scalar_one() == 1


def test_a_company_fits_the_budget(fresh_db):  # noqa: ARG001
    """The free 0.5 GB: a stored company (its row and three years) is
    measured, and MAX_COMPANIES of them, twice over for indexes and
    TOAST, stay inside the budget the health check reports against."""
    store.save(tesco())
    with db_engine.connect() as conn:
        row = conn.execute(text("SELECT pg_column_size(c.*) FROM companies c")).scalar_one()
        years = conn.execute(text("SELECT sum(pg_column_size(y.*)) FROM company_years y")).scalar_one()
    per_company = row + years * store.MAX_YEARS / 3
    assert per_company < 4000
    assert store.MAX_COMPANIES * per_company * 2 < store.BUDGET_BYTES
    use = store.usage()
    assert use["companies"] == 1 and 0 < use["bytes"] < store.BUDGET_BYTES and use["warning"] is False


def test_the_least_recently_used_companies_go_first(fresh_db):  # noqa: ARG001
    data = tesco()
    for i in range(4):
        ref = type(data.ref)("esef", f"{i:020d}", f"Co {i}", None, {})
        store.save(type(data)(ref, "EUR", "ifrs", 12, data.years, []))
    with db_engine.transaction() as conn:
        for i in range(4):
            conn.execute(update(Company).where(Company.source_id == f"{i:020d}")
                         .values(last_used_at=utc_now() - timedelta(days=10 - i)))
    assert store.evict(max_companies=2) == 2
    with db_engine.connect() as conn:
        left = conn.execute(text("SELECT source_id FROM companies ORDER BY source_id")).scalars().all()
        orphans = conn.execute(text("SELECT count(*) FROM company_years WHERE company_id NOT IN "
                                    "(SELECT id FROM companies)")).scalar_one()
    assert left == [f"{2:020d}", f"{3:020d}"] and orphans == 0


def test_stale_companies_come_oldest_first(fresh_db):  # noqa: ARG001
    data = tesco()
    store.save(data)
    assert store.stale(10, utc_now() - timedelta(days=1)) == []
    assert store.stale(10, utc_now() + timedelta(seconds=1)) == [("esef", TESCO_LEI)]
    store.mark_refreshed("esef", TESCO_LEI)
    assert store.get("esef", TESCO_LEI, touch=False)[1] > utc_now() - timedelta(minutes=1)


def test_edinet_index_keeps_the_newest_report_per_period(fresh_db):  # noqa: ARG001
    index = store.DatabaseReportIndex()
    first = IndexedReport("E02144", "S100AAAA", date(2024, 4, 1), date(2025, 3, 31), datetime(2025, 6, 18, 15, 0))
    amended = IndexedReport("E02144", "S100BBBB", date(2024, 4, 1), date(2025, 3, 31), datetime(2025, 9, 1, 9, 0))
    older = IndexedReport("E02144", "S100CCCC", date(2023, 4, 1), date(2024, 3, 31), datetime(2024, 6, 20, 9, 0))
    assert index.add([first, older]) == 2
    assert index.add([amended]) == 1
    assert index.add([first]) == 0                      # an older submission never replaces a newer one
    got = index.reports("E02144")
    assert [(r.doc_id, r.period_end) for r in got] == [("S100BBBB", date(2025, 3, 31)), ("S100CCCC", date(2024, 3, 31))]
    assert got[0].submitted_at == datetime(2025, 9, 1, 0, 0, tzinfo=timezone.utc)   # 09:00 in Tokyo
    assert index.scanned_through() is None
    index.set_scanned_through(date(2026, 10, 5))
    index.set_scanned_through(date(2026, 10, 6))
    assert index.scanned_through() == date(2026, 10, 6)
    assert index.prune(date(2027, 4, 1)) == 1           # three years kept
    assert [r.doc_id for r in index.reports("E02144")] == ["S100BBBB"]


def test_the_database_check_reports_company_data(fresh_db):  # noqa: ARG001
    from ops.check_database import problems
    store.save(tesco())
    result = client.get("/api/health/database").json()
    assert result["company_data"]["companies"] == 1
    assert result["company_data"]["budget_bytes"] == store.BUDGET_BYTES
    assert problems(result) == []
    over = {**result, "company_data": {**result["company_data"], "warning": True,
                                       "bytes": 60 * 1024 * 1024}}
    assert problems(over) == ["company data at 60 MB of its 64 MB budget: lower "
                              "db.companies.MAX_COMPANIES (PLAN.md 4.1)"]


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------
def test_searching_and_loading_count_as_runs_and_need_an_account(signed_out):  # noqa: ARG001
    assert {"/api/companies/search", "/api/companies/load"} <= RUN_PATHS
    assert client.get("/api/companies/sources").status_code == 401
    assert client.get("/api/companies/search", params={"q": "Tesco"}).status_code == 401


def test_the_sources_say_what_they_cover_and_whether_they_are_set_up(monkeypatch):
    body = client.get("/api/companies/sources").json()
    by_id = {s["id"]: s for s in body["sources"]}
    assert set(by_id) == {"sec", "esef", "companies_house", "edinet"}
    assert by_id["edinet"]["coverage"] == ["JP"] and by_id["edinet"]["needs_key"] is True
    assert by_id["edinet"]["configured"] is False and by_id["sec"]["configured"] is True
    assert body["fallback"] == "document_upload"
    monkeypatch.setenv(edinet.KEY_ENV, "k")
    assert {s["id"]: s for s in client.get("/api/companies/sources").json()["sources"]}["edinet"]["configured"]


def test_search_answers_every_source_at_once():
    use("search_names")
    body = client.get("/api/companies/search", params={"q": "Toyota"}).json()
    toyota = next(r for r in body["results"] if r["source"] == "edinet" and r["id"] == "E02144")
    assert toyota["name"] == "TOYOTA MOTOR CORPORATION" and toyota["country"] == "JP"
    assert toyota["loadable"] is False and toyota["stored"] is False       # no EDINET key here
    assert {"source": "companies_house", "reason": "not_configured"} in body["unavailable"]
    assert body["fallback"] == "document_upload" and body["matched"] is None
    assert client.get("/api/companies/search", params={"q": "T"}).status_code == 422


def test_loading_answers_the_figures_in_the_filings_currency():
    use("esef_tesco")
    resp = client.post("/api/companies/load", json={"source": "esef", "id": TESCO_LEI})
    assert resp.status_code == 200
    body = resp.json()
    assert body["money"] == {"currency": "GBP", "unit": "millions"}
    assert body["accounting_standard"] == "ifrs" and body["fiscal_year_end_month"] == 2
    latest = body["years"][-1]
    assert latest["fiscal_year"] == 2026 and latest["figures"]["revenue"] == 73712.0
    assert latest["figures"]["ebitda"] == 2985.0 + 1895.0
    assert latest["filing"]["url"].startswith("https://filings.xbrl.org/")
    assert body["company"]["stored"] is False and body["refreshed_at"] is None   # no database here
    assert body["warnings"] == [{"code": "missing_figure", "field": "capital_expenditures"}]


def test_a_loaded_company_is_stored_and_served_without_the_source(fresh_db):  # noqa: ARG001
    use("esef_tesco")
    loaded = client.post("/api/companies/load", json={"source": "esef", "id": TESCO_LEI}).json()
    assert loaded["company"]["stored"] is True and loaded["refreshed_at"]
    http.use_transport(lambda req: pytest.fail("the stored copy needs no source"))
    stored = client.get(f"/api/companies/esef/{TESCO_LEI}").json()
    assert stored["years"] == loaded["years"] and stored["money"] == loaded["money"]
    use("search_names")
    hit = next(r for r in client.get("/api/companies/search", params={"q": TESCO_LEI}).json()["results"])
    assert hit["stored"] is True


@pytest.mark.parametrize("path, status", [
    ("/api/companies/esef/" + "0" * 20, 404),
    ("/api/companies/esef/not-an-lei", 422),
    ("/api/companies/bloomberg/x", 422),
])
def test_reading_a_company_that_is_not_there(fresh_db, path, status):  # noqa: ARG001
    assert client.get(path).status_code == status


def test_load_refusals_say_what_to_do():
    bad = client.post("/api/companies/load", json={"source": "edinet", "id": "7203"})
    assert bad.status_code == 422
    no_key = client.post("/api/companies/load", json={"source": "edinet", "id": "E02144"})
    assert no_key.status_code == 503 and "API key" in no_key.json()["detail"]
    http.use_transport(lambda req: http.Response(404, b""))
    assert client.post("/api/companies/load", json={"source": "esef", "id": "0" * 20}).status_code == 404
    http.use_transport(lambda req: http.Response(503, b""))
    down = client.post("/api/companies/load", json={"source": "sec", "id": "63908"})
    assert down.status_code == 502 and "http" not in down.json()["detail"]


# ---------------------------------------------------------------------------
# The scheduled refresh
# ---------------------------------------------------------------------------
def test_the_refresh_is_a_scheduled_task_and_skips_without_a_database():
    from jobs.scheduled import TASKS
    assert "company-refresh" in TASKS
    assert refresh.run() == {"skipped_no_database": True}


def test_the_refresh_reloads_stale_companies(fresh_db):  # noqa: ARG001
    store.save(tesco())
    with db_engine.transaction() as conn:
        conn.execute(update(Company).values(refreshed_at=utc_now() - timedelta(days=30)))
        conn.execute(text("DELETE FROM company_years WHERE fiscal_year = 2024"))
    summary = refresh.run()
    assert summary["companies_refreshed"] == 1 and summary["refresh_failures"] == 0
    assert summary["edinet_not_configured"] is True and summary["more"] is False
    back, refreshed_at = store.get("esef", TESCO_LEI)
    assert [y.fiscal_year for y in back.years] == [2024, 2025, 2026]
    assert refreshed_at > utc_now() - timedelta(minutes=1)


def test_a_refresh_that_fails_keeps_the_figures_and_moves_on(fresh_db):  # noqa: ARG001
    store.save(tesco())
    with db_engine.transaction() as conn:
        conn.execute(update(Company).values(refreshed_at=utc_now() - timedelta(days=30)))
    http.use_transport(lambda req: http.Response(503, b""))
    summary = refresh.run()
    assert summary["companies_refreshed"] == 0 and summary["refresh_failures"] == 1
    back, refreshed_at = store.get("esef", TESCO_LEI)
    assert len(back.years) == 3 and refreshed_at > utc_now() - timedelta(minutes=1)


def _edinet_days(case: str, empty_days: bool = True):
    """Replay a case; a day list that wasn't recorded is a day with no filings."""
    recorded = replay(FIXTURES / case)

    def transport(req):
        resp = recorded(req)
        if resp.status == 404 and req.url.endswith("/documents.json") and empty_days:
            return http.Response(200, json.dumps({"metadata": {}, "results": []}).encode())
        return resp
    return transport


def test_the_refresh_reads_edinets_day_lists_into_the_index(fresh_db, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv(edinet.KEY_ENV, "test-key")
    index = json.loads((FIXTURES / "edinet_toyota" / "index.json").read_text(encoding="utf-8"))
    days = sorted(k.split("date=")[1][:10] for k in index if "documents.json" in k)
    filed = date.fromisoformat(days[-1])
    http.use_transport(_edinet_days("edinet_toyota"))
    db_index = store.DatabaseReportIndex()
    db_index.set_scanned_through(filed - timedelta(days=3))
    now = datetime.combine(filed + timedelta(days=4), datetime.min.time(), tzinfo=timezone.utc)
    summary = refresh.run(now=now)
    assert summary["edinet_days_scanned"] == 6 and summary["edinet_days_left"] == 0
    assert summary["edinet_reports_added"] >= 1 and summary["more"] is False
    assert db_index.scanned_through() == filed + timedelta(days=3)
    assert db_index.reports("E02144")[0].doc_id in {e["file"].split("_")[2] for e in index.values()
                                                    if "S100" in e["file"]}


def test_a_first_refresh_backfills_a_bounded_number_of_days(fresh_db, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv(edinet.KEY_ENV, "test-key")
    http.use_transport(lambda req: http.Response(200, b'{"results": []}'))
    summary = refresh.run(now=datetime(2026, 10, 6, 12, tzinfo=timezone.utc))
    assert summary["edinet_days_scanned"] == refresh.EDINET_DAYS_PER_RUN
    assert summary["edinet_days_left"] == refresh.EDINET_BACKFILL_DAYS - refresh.EDINET_DAYS_PER_RUN
    assert summary["more"] is True


def test_a_wrong_edinet_key_fails_the_refresh_so_it_alerts(fresh_db, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv(edinet.KEY_ENV, "wrong")
    http.use_transport(lambda req: http.Response(401, b""))
    with pytest.raises(http.SourceError):
        refresh.run()
    http.use_transport(lambda req: http.Response(503, b""))
    assert refresh.run()["edinet_error"] == "failed"


def test_the_scheduler_runs_the_task_again_while_there_is_more():
    from ops import scheduled as ops_scheduled

    answers = [{"more": True}, {"more": True}, {"more": False}, {"more": True}]

    class Api:
        calls = 0

        def wake(self):
            return {}

        def call(self, method, path):
            Api.calls += 1
            return 200, {"status": "succeeded", "summary": answers[Api.calls - 1]}

    ops_scheduled.run_task(Api(), "company-refresh", log=lambda *_: None, repeat=12)
    assert Api.calls == 3
    Api.calls = 0
    ops_scheduled.run_task(Api(), "company-refresh", log=lambda *_: None)
    assert Api.calls == 1
