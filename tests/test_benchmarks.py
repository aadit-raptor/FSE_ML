"""Sourced starting figures by region, sector and size (PLAN.md 4.3).

The industry averages replay Damodaran's real January 2026 workbooks,
recorded and trimmed by ``python -m benchmarks.record``
(tests/fixtures/benchmarks); the economic figures replay 4.2's recorded
series (tests/fixtures/economy). Every expected number below was read off
those files, and each derived figure is worked by hand from them. The
database tests use a throwaway database (tests/conftest.py).
"""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.limits import RUN_PATHS
from api.main import app
from benchmarks import catalogue, damodaran, history, record, refresh, starting
from benchmarks.countries import NAME_TO_ISO
from companies import http
from companies.record import replay
from economy import connectors
from economy import refresh as economy_refresh
from jobs import scheduled

FIXTURES = Path(__file__).parent / "fixtures"
DAMODARAN = FIXTURES / "benchmarks" / "damodaran"
HISTORY = FIXTURES / "benchmarks" / "history"         # PLAN.md 4.4: what the refresh also reads
ECONOMY = (FIXTURES / "economy" / "economy_open", FIXTURES / "economy" / "economy_fred")
# The day the economic figures were recorded: "current" is judged on it
ECONOMY_DAY = date.fromisoformat(json.loads((ECONOMY[0] / "index.json").read_text())["_recorded_on"])
client = TestClient(app)


def transport(*cases: Path, fail: dict | None = None, calls: list | None = None):
    replays = [replay(c) for c in cases]

    def answer(req: http.Request) -> http.Response:
        if calls is not None:
            calls.append(req.url)
        if fail and any(part in req.url for part in fail):
            return http.Response(next(v for k, v in fail.items() if k in req.url), b"")
        for r in replays:
            resp = r(req)
            if resp.status != 404:
                return resp
        if req.url.startswith(history.ARCHIVE_BASE):
            return http.Response(404, b"")               # not archived: not recorded either
        pytest.fail(f"unrecorded request to {req.url}")
    return answer


@pytest.fixture(autouse=True)
def recorded(monkeypatch):
    monkeypatch.setenv(connectors.FRED_KEY_ENV, "not-a-real-key")
    http.use_transport(transport(DAMODARAN, HISTORY, *ECONOMY))
    yield
    http.use_transport(None)


def tables() -> dict[str, damodaran.Table]:
    return {t.name: t for t in (call() for _, call in damodaran.every_table())}


def economy() -> dict:
    found, problems = economy_refresh._read(ECONOMY_DAY)
    assert problems == {}
    return {s.key: s for s in found}


def start(country: str, industry: str, currency: str, t=None, e=None) -> dict:
    return starting.starting_assumptions(country, industry, currency, tables() if t is None else t,
                                         economy() if e is None else e, ECONOMY_DAY)


def figure(answer: dict, name: str) -> dict:
    return next(f for f in answer["figures"] if f["field"] == name)


# ---------------------------------------------------------------------------
# Reading the workbooks
# ---------------------------------------------------------------------------
def test_every_table_reads_with_its_date_and_thousands_of_companies():
    t = tables()
    assert set(t) == {name for name, _ in damodaran.every_table()} and len(t) == 41
    assert {x.published for x in t.values()} == {date(2026, 1, 5)}
    world = t["margins.global"].rows
    # The all-industry row (without financials) is the 43,000-company sample
    assert (world["all"]["name"], world["all"]["firms"], world["all"]["gross_margin"]) == (
        "Total Market (without financials)", 43056, 0.295871)


def test_european_machinery_reads_as_printed():
    t = tables()
    assert t["margins.europe"].rows["machinery"] == {
        "name": "Machinery", "firms": 210, "gross_margin": 0.446829, "operating_margin": 0.117119,
        "ebitda_margin": 0.129671}
    assert t["multiples.europe"].rows["machinery"]["ev_ebitda"] == 14.980532
    assert t["capex.europe"].rows["machinery"]["capex_to_da"] == 0.791256
    assert t["working_capital.europe"].rows["machinery"] == {
        "name": "Machinery", "firms": 210, "receivables_sales": 0.184073, "inventory_sales": 0.177707,
        "payables_sales": 0.101753, "noncash_wc_sales": 0.17367}
    assert t["debt.europe"].rows["machinery"] == {
        "name": "Machinery", "firms": 210, "debt_ebitda": 1.871635, "interest_coverage": 10.645658}


def test_financial_industries_and_totals_are_left_out():
    rows = tables()["margins.us"].rows
    names = {r["name"] for r in rows.values()}
    assert not names & catalogue.FINANCIAL_INDUSTRIES
    assert not names & catalogue.TOTAL_ROWS
    assert len(rows) == 84 and "all" in rows


def test_industry_ids_are_stable_slugs():
    assert damodaran.industry_id("Oil/Gas (Integrated)") == "oil_gas_integrated"
    assert damodaran.industry_id("Rubber& Tires") == "rubber_tires"
    assert damodaran.industry_id(catalogue.ALL_INDUSTRIES) == "all"


def test_a_countrys_first_tax_row_wins_over_the_stale_repeat_at_the_end():
    rows = tables()["country_tax"].rows
    assert rows["GB"] == {"rate": 0.25}          # the repeat at the end still says 19%
    assert rows["TR"] == {"rate": 0.25}          # ... and 22%
    assert rows["DE"] == {"rate": 0.2993} and rows["US"] == {"rate": 0.2563}
    assert rows["KR"] == {"rate": 0.264}         # "Republic of Korea", not the North
    assert len(rows) > 200


def test_every_country_the_tax_workbook_names_is_placed():
    import xlrd
    path = DAMODARAN / json.loads((DAMODARAN / "index.json").read_text())[
        damodaran.workbook_url("countrytaxrates")]["file"]
    sheet = xlrd.open_workbook(str(path)).sheet_by_index(0)
    names = {str(sheet.cell_value(r, 0)).strip() for r in range(2, sheet.nrows)} - {""}
    assert names - set(NAME_TO_ISO) == set()


def test_a_workbook_of_another_shape_is_unreadable():
    with pytest.raises(damodaran.Unreadable):
        damodaran.parse_industries(b"not a workbook", catalogue.DATASETS["margins"])
    other = (DAMODARAN / json.loads((DAMODARAN / "index.json").read_text())[
        damodaran.workbook_url("vebitda", catalogue.REGIONS["us"])]["file"]).read_bytes()
    with pytest.raises(damodaran.Unreadable, match="grossmargin"):
        damodaran.parse_industries(other, catalogue.DATASETS["margins"])


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------
def test_each_country_looks_in_its_own_groups_closest_first():
    assert catalogue.chain("DE") == ("europe", "global")
    assert catalogue.chain("GB") == ("europe", "global")
    assert catalogue.chain("US") == ("us", "global")
    assert catalogue.chain("IN") == ("india", "emerging", "global")
    assert catalogue.chain("CA") == ("aus_nz_canada", "global")
    assert catalogue.chain("BR") == ("emerging", "global")
    assert catalogue.region_of("IN") == "emerging" and catalogue.region_of("JP") == "global"


def test_a_retired_country_code_is_read_as_the_current_one(fresh_db, monkeypatch):  # noqa: ARG001
    """The account screen once offered "DD" (East Germany) beside "DE", both named Germany."""
    store_everything()
    monkeypatch.setattr("api.routers.benchmarks._today", lambda: ECONOMY_DAY)
    get = lambda c: client.get("/api/benchmarks/starting",  # noqa: E731
                               params={"country": c, "industry": "machinery", "currency": "EUR"}).json()
    assert get("DD")["country"] == "DE" and get("DD")["inputs"] == get("DE")["inputs"]
    assert catalogue.canonical("UK") == "GB" and catalogue.canonical("FR") == "FR"


# ---------------------------------------------------------------------------
# Starting figures
# ---------------------------------------------------------------------------
def test_a_german_industrials_deal_gets_sourced_figures_with_sample_sizes():
    """PLAN.md 4.3 "done when": each figure worked by hand from the
    recorded European machinery row (210 companies) and Germany's data."""
    a = start("DE", "machinery", "EUR")
    gm, op, ebitda = 0.446829, 0.117119, 0.129671
    da = ebitda - op
    g = (1.008 * 1.027 - 1)                      # IMF 2026: real 0.8%, inflation 2.7%
    expected = {
        "growth": round(g * 100, 2),             # 3.52
        "tax": 29.93,                            # Tax Foundation, Germany
        "gross_margin": 44.68,
        "opex": round((gm - op) * 100, 2),       # 32.97
        "da": round(da * 100, 2),                # 1.26
        "entry_mult": 14.98, "exit_mult": 14.98,
        "capex": round(da * 0.791256 * 100, 2),  # 0.99
        "ar_days": round(0.184073 * 365, 1),
        "inv_days": round(0.177707 * 365 / (1 - gm), 1),
        "ap_days": round(0.101753 * 365 / (1 - gm), 1),
        "nwc": round(0.17367 * g / (1 + g) * 100, 2),
        "debt_pct": round(1.871635 / 14.980532 * 100, 2),
        "senior_pct": 100.0,
        "base_rate": round(2.6350455 + 0.40, 2),  # EURIBOR + AAA spread (coverage 10.6x)
    }
    assert a["inputs"] == expected
    assert a["missing"] == []
    for name in ("gross_margin", "entry_mult", "capex", "ar_days", "debt_pct"):
        f = figure(a, name)
        assert (f["source"], f["area"], f["level"], f["sample"], f["sample_kind"], f["as_of"]) == (
            "damodaran", "europe", "region", 210, "companies", "2026-01-05")
        assert f["url"].startswith(catalogue.DAMODARAN_BASE) and f["skipped"] == []
    assert (figure(a, "growth")["source"], figure(a, "growth")["level"]) == ("imf", "country")
    assert (figure(a, "tax")["source"], figure(a, "tax")["area"]) == ("tax_foundation", "DE")
    rate = figure(a, "base_rate")["detail"]
    assert (rate["benchmark"], rate["rating"], rate["spread"]) == ("EURIBOR", "AAA", 0.40)
    assert figure(a, "senior_pct")["source"] == "choice"
    assert set(a["notes"]) == {"size_not_split", "exit_equals_entry", "listed_company_leverage"}


def test_a_thin_group_falls_back_to_global_and_says_why():
    """Shipbuilding in Australia, NZ and Canada: 8 companies, under 20."""
    a = start("CA", "shipbuilding_marine", "CAD")
    gm = figure(a, "gross_margin")
    assert (gm["area"], gm["level"], gm["sample"]) == ("global", "global", 359)
    assert gm["skipped"] == [{"area": "aus_nz_canada", "reason": "thin", "sample": 8}]
    assert a["inputs"]["gross_margin"] == 26.53          # global 0.265294, not the region's 0.269181


def test_an_unusable_figure_falls_back_too():
    t = tables()
    europe = t["margins.europe"]
    broken = dict(europe.rows)
    broken["machinery"] = {**broken["machinery"], "ebitda_margin": -0.02}
    t["margins.europe"] = replace(europe, rows=broken)
    gm = figure(start("DE", "machinery", "EUR", t=t), "gross_margin")
    assert gm["area"] == "global"
    assert gm["skipped"] == [{"area": "europe", "reason": "unusable", "sample": 210}]


def test_a_country_without_its_own_figures_takes_its_regions_median():
    a = start("AT", "machinery", "EUR")
    growth = figure(a, "growth")
    assert (growth["area"], growth["level"], growth["sample"], growth["sample_kind"]) == (
        "europe", "region", 10, "economies")
    assert growth["skipped"] == [{"area": "AT", "reason": "missing", "sample": None}]
    # The median of the ten European economies' nominal growth, each the IMF's real growth and inflation
    e = economy()
    europe = ("GB", "DE", "FR", "IT", "ES", "NL", "CH", "SE", "NO", "PL")
    nominal = sorted(((1 + e[f"gdp_growth.{c}.imf"].observations[-6][1] / 100)
                      * (1 + e[f"inflation.{c}.imf"].observations[-6][1] / 100) - 1) * 100 for c in europe)
    assert [e[f"gdp_growth.{c}.imf"].observations[-6][0] for c in europe] == ["2026"] * 10
    assert growth["value"] == round((nominal[4] + nominal[5]) / 2, 2)
    assert figure(a, "tax")["area"] == "AT"           # Austria has its own tax rate
    nowhere = start("AQ", "all", "USD")               # no tax rate, no economic data
    tax = figure(nowhere, "tax")
    assert (tax["area"], tax["sample_kind"], tax["detail"]) == ("emerging", "countries", {"statistic": "median"})


def test_a_currency_without_a_benchmark_takes_the_countrys_policy_rate():
    a = start("BR", "all", "BRL")
    rate = figure(a, "base_rate")
    assert rate["dataset"] == "policy_rate" and rate["detail"]["benchmark_source"] == "bis"


def test_listed_company_leverage_is_the_industrys_debt_over_its_multiple():
    a = start("US", "software_system_application", "USD")
    d = figure(a, "debt_pct")["detail"]
    assert a["inputs"]["debt_pct"] == round(d["debt_ebitda"] / d["ev_ebitda"] * 100, 2)
    assert figure(a, "debt_pct")["area"] == "us" and figure(a, "debt_pct")["level"] == "country"


def test_without_stored_data_every_figure_is_missing_and_the_deal_keeps_its_own():
    a = starting.starting_assumptions("DE", "machinery", "EUR", {}, {}, ECONOMY_DAY)
    assert a["inputs"] == {} and a["figures"] == []
    assert {m["field"] for m in a["missing"]} == {
        "growth", "tax", "gross_margin", "opex", "da", "capex", "ar_days", "inv_days", "ap_days", "nwc",
        "entry_mult", "exit_mult", "debt_pct", "base_rate"}


def test_the_starting_figures_run_through_the_deal_model():
    """Verify by output: a deal started from them runs, and its EBITDA
    margin is the industry's."""
    a = start("DE", "machinery", "EUR")
    r = client.post("/api/deal/run", json={"inputs": {**a["inputs"], "currency": "EUR"}, "settings": {}})
    assert r.status_code == 200, r.text
    returns = r.json()["returns"]
    default = client.post("/api/deal/run", json={"inputs": {}, "settings": {}}).json()["returns"]
    margin = a["inputs"]["gross_margin"] - a["inputs"]["opex"] + a["inputs"]["da"]
    assert margin == pytest.approx(12.97, abs=0.01)                       # Europe machinery EBITDA / sales
    assert returns["exit_multiple"] == 14.98
    assert 0 < returns["irr"] < 1 and returns["irr"] != pytest.approx(default["irr"], abs=0.001)


def test_industries_list_every_one_with_its_global_sample():
    found = starting.industries(tables())
    assert found[0] == {"id": "all", "name": "Total Market (without financials)", "firms": 43056}
    assert {"id": "machinery", "name": "Machinery", "firms": 1553} in found
    assert len(found) == 84


# ---------------------------------------------------------------------------
# The refresh
# ---------------------------------------------------------------------------
def at(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 3, tzinfo=timezone.utc)


def test_a_refresh_stores_every_table(fresh_db):  # noqa: ARG001
    from db import benchmarks as store
    summary = refresh.run(at(ECONOMY_DAY))
    assert (summary["tables_saved"], summary["tables_stored"], summary["industries"]) == (41, 41, 84)
    assert summary["published"] == "2026-01-05" and summary["tables_missing"] == ""
    stored, _ = store.all_tables()
    assert stored["debt.europe"].rows["machinery"]["debt_ebitda"] == 1.871635


def test_a_refresh_reads_again_only_once_a_week_has_passed(fresh_db):  # noqa: ARG001
    refresh.run(at(ECONOMY_DAY))
    calls: list = []
    http.use_transport(transport(DAMODARAN, HISTORY, *ECONOMY, calls=calls))
    current = lambda: [c for c in calls if "/pc/datasets/" in c]  # noqa: E731
    assert refresh.run(at(ECONOMY_DAY + timedelta(days=3)))["read"] is False
    assert current() == []
    assert refresh.run(at(ECONOMY_DAY + timedelta(days=8)))["read"] is True
    assert len(current()) == 41


def test_a_failing_workbook_keeps_what_was_stored(fresh_db):  # noqa: ARG001
    from db import benchmarks as store
    refresh.run(at(ECONOMY_DAY))
    http.use_transport(transport(DAMODARAN, HISTORY, *ECONOMY, fail={"datasets/marginEurope": 503}))
    summary = refresh.run(at(ECONOMY_DAY), force=True)
    assert summary["source_problems"] == "margins.europe:failed" and summary["tables_saved"] == 40
    assert store.all_tables()[0]["margins.europe"].rows["machinery"]["firms"] == 210


def test_a_workbook_that_lost_most_of_its_rows_does_not_replace_the_stored_one(fresh_db):  # noqa: ARG001
    from db import benchmarks as store
    refresh.run(at(ECONOMY_DAY))
    real = damodaran.parse_industries

    def few(content, dataset):
        published, rows = real(content, dataset)
        return published, dict(list(rows.items())[:10]) if dataset.id == "debt" else rows
    damodaran.parse_industries = few
    try:
        summary = refresh.run(at(ECONOMY_DAY), force=True)
    finally:
        damodaran.parse_industries = real
    assert "debt.europe:shrunk" in summary["source_problems"] and summary["tables_saved"] == 33
    assert len(store.all_tables()[0]["debt.europe"].rows) == 84


def test_a_date_cell_that_is_no_date_is_left_blank():
    import xlwt, io
    book = xlwt.Workbook()
    sheet = book.add_sheet("Industry Averages")
    for c, v in enumerate(["Date updated:", 1e300]):
        sheet.write(0, c, v)
    for c, v in enumerate(["Industry Name", "Number of firms", "EV/EBITDA"]):
        sheet.write(1, c, v)
    for c, v in enumerate([catalogue.ALL_INDUSTRIES, 100.0, 9.5]):
        sheet.write(2, c, v)
    buf = io.BytesIO()
    book.save(buf)
    published, rows = damodaran.parse_industries(buf.getvalue(), catalogue.DATASETS["multiples"])
    assert published is None and rows["all"]["ev_ebitda"] == 9.5


def test_a_first_refresh_missing_a_table_fails_so_it_alerts(fresh_db):  # noqa: ARG001
    http.use_transport(transport(DAMODARAN, HISTORY, *ECONOMY, fail={"datasets/vebitdaJapan": 404}))
    with pytest.raises(refresh.Incomplete, match="multiples.japan"):
        refresh.run(at(ECONOMY_DAY))


def test_an_unreadable_workbook_is_named_and_the_rest_still_count(fresh_db):  # noqa: ARG001
    base = transport(DAMODARAN, HISTORY, *ECONOMY)

    def odd(req):
        return http.Response(200, b"<html>moved</html>") if "datasets/wcdataIndia" in req.url else base(req)
    refresh.run(at(ECONOMY_DAY))
    http.use_transport(odd)
    summary = refresh.run(at(ECONOMY_DAY), force=True)
    assert summary["source_problems"] == "working_capital.india:unreadable"


def test_a_runs_summary_is_what_the_task_endpoint_answers(fresh_db):  # noqa: ARG001
    from api.routers.scheduled import TaskRun
    summary = refresh.run(at(ECONOMY_DAY))
    run = TaskRun(task="benchmarks-refresh", status="succeeded", summary=summary, workflow="scheduled.yml",
                  github_run_id=1)
    assert run.summary == summary


def test_the_refresh_is_a_scheduled_task_and_skips_without_a_database():
    assert "benchmarks-refresh" in scheduled.TASKS
    assert refresh.run(at(ECONOMY_DAY)) == {"skipped_no_database": True}


def test_the_workflows_refresh_on_both_environments_and_on_each_staging_deploy():
    workflows = Path(__file__).parent.parent / ".github" / "workflows"
    nightly = (workflows / "scheduled.yml").read_text(encoding="utf-8")
    staging = (workflows / "staging.yml").read_text(encoding="utf-8")
    assert "task benchmarks-refresh" in nightly and "environment: staging" in nightly
    assert "task benchmarks-refresh" in staging and ".tables_stored == 41" in staging
    assert "benchmarks" in staging.split("API_PATHS:")[1].splitlines()[0]


def test_the_database_check_reports_the_averages(fresh_db):  # noqa: ARG001
    from db import benchmarks as store
    from ops.check_database import problems
    refresh.run(at(ECONOMY_DAY))
    result = client.get("/api/health/database").json()
    assert result["benchmark_data"]["tables"] == 41
    assert 0 < result["benchmark_data"]["bytes"] < store.BUDGET_BYTES
    assert problems(result) == []
    over = {**result, "benchmark_data": {**result["benchmark_data"], "warning": True, "bytes": 8000 * 1024}}
    assert problems(over) == ["industry averages at 8000 KB of their 8192 KB budget: store fewer figures "
                              "(benchmarks/catalogue.py DATASETS, PLAN.md 4.3)"]


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------
def store_everything():
    from db import economy as economy_store
    refresh.run(at(ECONOMY_DAY))
    economy_store.save_series(list(economy().values()))


def test_the_api_answers_the_starting_figures(fresh_db, monkeypatch):  # noqa: ARG001
    store_everything()
    monkeypatch.setattr("api.routers.benchmarks._today", lambda: ECONOMY_DAY)
    r = client.get("/api/benchmarks/starting", params={"country": "DE", "industry": "machinery", "currency": "EUR"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["inputs"] == start("DE", "machinery", "EUR")["inputs"]
    assert body["refreshed_at"] and body["source"]["id"] == "damodaran" and body["chain"] == ["europe", "global"]
    industries = client.get("/api/benchmarks/industries").json()
    assert industries["published"] == "2026-01-05" and len(industries["industries"]) == 84


def test_the_api_refuses_an_unknown_industry_or_a_malformed_code(fresh_db):  # noqa: ARG001
    store_everything()
    get = lambda **p: client.get("/api/benchmarks/starting", params=p).status_code  # noqa: E731
    assert get(country="DE", industry="banks_regional", currency="EUR") == 404
    assert get(country="de", industry="machinery", currency="EUR") == 422
    assert get(country="DE", industry="machinery", currency="euro") == 422


def test_without_a_database_the_api_answers_nothing_found():
    body = client.get("/api/benchmarks/starting", params={"country": "DE", "currency": "EUR"}).json()
    assert body["inputs"] == {} and body["industry"] == "all"
    assert client.get("/api/benchmarks/industries").json()["industries"] == []


def test_the_endpoints_read_storage_only_so_none_is_a_run(signed_out):  # noqa: ARG001
    assert not any(p.startswith("/api/benchmarks") for p in RUN_PATHS)
    assert client.get("/api/benchmarks/industries").status_code == 401


def test_the_browser_tests_replay_what_the_api_answers_today():
    """web/e2e/fixtures/benchmarks.json is the endpoints' answers for the
    recorded data; regenerate it when an answer changes."""
    from tests.e2e_benchmarks import OUT, build
    assert json.loads(OUT.read_text(encoding="utf-8")) == build(), \
        "web/e2e/fixtures/benchmarks.json is stale: run python -m tests.e2e_benchmarks"


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------
def test_recording_the_fixture_again_from_itself_gives_the_same_bytes(tmp_path):
    out = record.record(tmp_path, real=transport(DAMODARAN))
    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in DAMODARAN.iterdir())
    for p in out.iterdir():
        if p.name != "index.json":
            assert p.read_bytes() == (DAMODARAN / p.name).read_bytes(), p.name
