"""Economic data by country, reference rates and exchange rates (PLAN.md 4.2).

Every source replays recorded responses (tests/fixtures/economy: the keyless
sources recorded with ``python -m economy.record economy_open``, FRED by the
``record-economy`` workflow with the repository's key), so the figures are
real ones; each expected number below was read off the recorded file. The
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
from companies import http
from companies.record import replay
from core.debt import REFERENCE_RATES
from db import economy as store
from economy import connectors, record, refresh, views
from economy.catalogue import COUNTRIES, CURRENCY_BENCHMARKS, REFERENCE_SOURCES, SOURCES, series_key
from economy.model import FxDay, Series
from jobs import scheduled

FIXTURES = Path(__file__).parent / "fixtures" / "economy"
OPEN, FRED = FIXTURES / "economy_open", FIXTURES / "economy_fred"
RECORDED_ON = date.fromisoformat(json.loads((OPEN / "index.json").read_text())["_recorded_on"])
FAKE_KEY = "not-a-real-fred-key"
FX_ASKED_FROM = (RECORDED_ON - timedelta(days=store.KEEP_FX_DAYS)).isoformat()
client = TestClient(app)


def transport(*cases: Path, fail: dict | None = None):
    """Replay several recorded cases; ``fail`` maps a host to a status it
    answers. The exchange rates answer the recorded days whatever day a
    refresh starts from (that depends on what is already stored)."""
    replays = [replay(c) for c in cases]

    def answer(req: http.Request) -> http.Response:
        if fail and req.host in fail:
            return http.Response(fail[req.host], b"")
        if req.url.endswith("/EXR/D..EUR.SP00.A"):
            req = http.Request(req.url, {**req.params, "startPeriod": FX_ASKED_FROM}, req.headers)
        for r in replays:
            resp = r(req)
            if resp.status != 404:
                return resp
        pytest.fail(f"unrecorded request to {req.host}")
    return answer


@pytest.fixture(autouse=True)
def recorded(monkeypatch):
    monkeypatch.setenv(connectors.FRED_KEY_ENV, FAKE_KEY)
    http.use_transport(transport(OPEN, FRED))
    yield
    http.use_transport(None)


def everything(today: date = RECORDED_ON) -> dict[str, Series]:
    found, problems = refresh._read(today)
    assert problems == {}
    return {s.key: s for s in found}


# ---------------------------------------------------------------------------
# What each source says (read off the recorded files)
# ---------------------------------------------------------------------------
def test_each_source_reads_its_recorded_figures():
    s = everything()
    assert dict(s["gdp_growth.GB.worldbank"].observations)["2025"] == pytest.approx(1.38844207858726)
    assert s["policy_rate.GB.bis"].latest == ("2026-09-28", 3.75)
    assert s["policy_rate.XM.bis"].latest == ("2026-09-29", 2.5)
    assert s["bond_yield_10y.DE.oecd"].latest == ("2026-09", 3.469524)
    assert s["short_rate_3m.AU.oecd"].latest == ("2026-08", 4.51)
    assert s["reference_rate.ESTR.ecb"].latest == ("2026-10-05", 2.438)
    assert s["reference_rate.EURIBOR.ecb"].latest == ("2026-09", 2.6350455)
    assert s["gdp_growth.GB.imf"].source_series == "NGDP_RPCH"
    # The WEO keeps six years back and five ahead of the year it is read in
    years = [p for p, _ in s["gdp_growth.GB.imf"].observations]
    assert years[0] == str(RECORDED_ON.year - 6) and years[-1] == str(RECORDED_ON.year + 5)


def test_fred_gives_sofr_sonia_and_the_treasury_yield():
    s = everything()
    for key in ("reference_rate.SOFR.fred", "reference_rate.SONIA.fred", "bond_yield_10y.US.fred"):
        period, value = s[key].latest
        assert s[key].frequency == "D" and 0 < value < 20
        assert (RECORDED_ON - date.fromisoformat(period)).days < 10


def test_every_economy_and_every_figure_names_its_source_and_link():
    for series in everything().values():
        assert series.source in SOURCES and series.url.startswith("https://")
        assert series.source_series and series.observations


def test_the_ecb_rates_are_units_per_euro_by_day():
    days = connectors.ecb_fx(RECORDED_ON - timedelta(days=store.KEEP_FX_DAYS))
    assert len(days) == record.FX_DAYS_KEPT
    last = days[-1]
    assert last.day == date(2026, 10, 6)
    assert (last.rates["EUR"], last.rates["USD"], last.rates["GBP"]) == (1.0, 1.1269, 0.8488)
    assert len(last.rates) == 30


def test_fred_needs_its_key_and_never_names_it(monkeypatch):
    seen = []
    http.use_transport(lambda req: seen.append(req) or transport(OPEN, FRED)(req))
    connectors.fred()
    assert seen and all(FAKE_KEY not in r.key() for r in seen)
    assert all(r.params["api_key"] == FAKE_KEY for r in seen)
    monkeypatch.delenv(connectors.FRED_KEY_ENV)
    with pytest.raises(http.NotConfigured):
        connectors.fred()


def test_a_series_keeps_only_its_newest_observations():
    daily = Series("policy_rate", "GB", "bis", "x", "D", "https://x",
                   tuple((f"2026-01-{d:02d}", d) for d in range(31, 0, -1)))
    assert len(daily.observations) == 24
    assert daily.observations[0] == ("2026-01-08", 8.0) and daily.latest == ("2026-01-31", 31.0)


# ---------------------------------------------------------------------------
# What is current
# ---------------------------------------------------------------------------
def test_at_least_twenty_economies_have_current_sourced_data():
    s = everything()
    current = views.current_economies(s, RECORDED_ON)
    assert len(current) >= 20
    # Singapore steers its exchange rate, not a policy rate: no BIS series
    assert set(COUNTRIES) - set(current) == {"SG"}
    gb = views.country_figures(s, "GB", RECORDED_ON)
    assert gb["gdp_growth"].basis == "projection" and gb["gdp_growth"].period == str(RECORDED_ON.year)
    assert gb["gdp_growth"].value == dict(s["gdp_growth.GB.imf"].observations)[str(RECORDED_ON.year)]
    assert gb["gdp_growth_actual"] == views.Figure(
        1.38844207858726, "2025", "worldbank", "NY.GDP.MKTP.KD.ZG",
        "https://data.worldbank.org/indicator/NY.GDP.MKTP.KD.ZG?locations=GB", "actual", True)
    assert gb["policy_rate"].value == 3.75 and gb["policy_rate"].source == "bis"


def test_a_euro_member_takes_the_ecbs_policy_rate():
    fig = views.country_figures(everything(), "DE", RECORDED_ON)["policy_rate"]
    assert (fig.value, fig.basis, fig.source_series) == (2.5, "euro_area", "WS_CBPOL/D.XM")


def test_a_sovereign_spread_is_the_yield_over_germanys_in_the_same_month():
    s = everything()
    it = views.country_figures(s, "IT", RECORDED_ON)["sovereign_spread"]
    # Italy's latest month is August: 3.986 - Germany's August 3.185714
    assert (it.period, it.basis) == ("2026-08", "computed")
    assert it.value == pytest.approx(3.986 - 3.185714)
    assert "sovereign_spread" not in views.country_figures(s, "DE", RECORDED_ON)
    assert "sovereign_spread" not in views.country_figures(s, "GB", RECORDED_ON)


def test_a_spread_waits_for_the_month_germany_has_published():
    s = everything()
    italy = s["bond_yield_10y.IT.oecd"]
    s[italy.key] = replace(italy, observations=(*italy.observations, ("2026-10", 9.0)))
    it = views.country_figures(s, "IT", RECORDED_ON)["sovereign_spread"]
    # Italy now has October, Germany only up to September: the spread stays
    # on the latest month both have (August, as Italy has no September)
    assert (it.period, it.value) == ("2026-08", pytest.approx(3.986 - 3.185714))


def test_the_us_yield_is_freds_daily_one_before_the_oecds_monthly_average():
    s = everything()
    fig = views.country_figures(s, "US", RECORDED_ON)["bond_yield_10y"]
    assert fig.source == "fred" and fig.value == s["bond_yield_10y.US.fred"].latest[1]
    del s["bond_yield_10y.US.fred"]
    fig = views.country_figures(s, "US", RECORDED_ON)["bond_yield_10y"]
    assert (fig.source, fig.period, fig.value) == ("oecd", "2026-09", 4.99)


def test_old_figures_are_still_shown_but_not_current():
    later = RECORDED_ON + timedelta(days=200)
    s = everything()
    gb = views.country_figures(s, "GB", later)
    assert gb["policy_rate"].value == 3.75 and not gb["policy_rate"].current
    assert views.current_economies(s, later) == []
    # A daily rate counts for 45 days, a policy rate for 120
    assert views.is_current(s["reference_rate.ESTR.ecb"], "2026-10-05", date(2026, 11, 19))
    assert not views.is_current(s["reference_rate.ESTR.ecb"], "2026-10-05", date(2026, 11, 20))
    assert views.is_current(s["policy_rate.GB.bis"], "2026-09-28", date(2027, 1, 26))
    assert not views.is_current(s["policy_rate.GB.bis"], "2026-09-28", date(2027, 1, 27))


# ---------------------------------------------------------------------------
# Reference rates: what a new floating facility starts from
# ---------------------------------------------------------------------------
def test_every_benchmark_but_custom_has_a_source_and_every_currency_benchmark_exists():
    assert set(REFERENCE_SOURCES) == set(REFERENCE_RATES) - {"custom"}
    assert set(CURRENCY_BENCHMARKS.values()) <= set(REFERENCE_SOURCES)


def test_sonia_is_the_bank_of_englands_published_rate():
    s = everything()
    sonia = views.reference_rates(s, RECORDED_ON)["SONIA"]
    assert (sonia.source, sonia.source_series, sonia.basis) == ("fred", "IUDSOIA", "benchmark")
    assert (sonia.period, sonia.value) == s["reference_rate.SONIA.fred"].latest
    assert sonia.url == "https://fred.stlouisfed.org/series/IUDSOIA"


def test_without_fred_sonia_falls_back_to_bank_rate_and_says_so():
    s = {k: v for k, v in everything().items() if v.source != "fred"}
    rates = views.reference_rates(s, RECORDED_ON)
    assert (rates["SONIA"].value, rates["SONIA"].basis, rates["SONIA"].source) == (3.75, "policy_rate", "bis")
    assert (rates["SOFR"].value, rates["SOFR"].basis) == (3.875, "policy_rate")


def test_benchmarks_without_a_free_source_say_what_stands_in():
    rates = views.reference_rates(everything(), RECORDED_ON)
    assert (rates["ESTR"].value, rates["ESTR"].basis) == (2.438, "benchmark")
    assert (rates["EURIBOR"].value, rates["EURIBOR"].basis) == (2.6350455, "benchmark")
    assert (rates["TONA"].value, rates["TONA"].basis) == (1.25, "policy_rate")
    assert (rates["SARON"].value, rates["SARON"].basis) == (0.0, "policy_rate")
    assert (rates["BBSY"].value, rates["BBSY"].basis) == (4.51, "interbank_3m")
    assert (rates["MIBOR"].value, rates["MIBOR"].basis) == (5.25, "policy_rate")


def test_a_stale_benchmark_is_left_out_rather_than_defaulted_to():
    later = RECORDED_ON + timedelta(days=200)
    assert views.reference_rates(everything(), later) == {}
    # A stale first candidate gives way to a current second one
    s = everything()
    old = s["reference_rate.SONIA.fred"]
    s["reference_rate.SONIA.fred"] = replace(old, observations=(("2026-01-02", 4.7),))
    assert views.reference_rates(s, RECORDED_ON)["SONIA"].basis == "policy_rate"


# ---------------------------------------------------------------------------
# Exchange rates
# ---------------------------------------------------------------------------
def test_rates_rebase_to_any_currency_the_ecb_quotes():
    day = FxDay(date(2026, 10, 6), {"EUR": 1.0, "USD": 1.1269, "GBP": 0.8488})
    per_dollar = views.rebase(day, "USD")
    assert per_dollar["USD"] == 1.0
    assert per_dollar["EUR"] == pytest.approx(1 / 1.1269)
    assert per_dollar["GBP"] == pytest.approx(0.8488 / 1.1269)
    assert views.rebase(day, "XYZ") is None


# ---------------------------------------------------------------------------
# Storage and the nightly refresh
# ---------------------------------------------------------------------------
def at(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 6, tzinfo=timezone.utc)


def test_a_refresh_stores_every_series_and_the_exchange_rates(fresh_db):  # noqa: ARG001
    summary = refresh.run(at(RECORDED_ON))
    stored, refreshed_at = store.all_series()
    assert summary["series_saved"] == len(stored) == len(everything())
    assert stored == everything()                                     # read back exactly
    assert refreshed_at is not None
    assert summary["economies_current"] == 23 and summary["economies_not_current"] == ["SG"]
    assert summary["reference_rates"] == sorted(REFERENCE_SOURCES)
    assert (summary["fx_date"], summary["fx_currencies"], summary["fx_days_saved"]) == ("2026-10-06", 30, 10)
    assert summary["source_problems"] == {}
    assert summary["economy_bytes"] < store.BUDGET_BYTES / 10 and not summary["economy_budget_warning"]


def test_exchange_rates_are_found_on_a_day_or_the_last_publication_before(fresh_db):  # noqa: ARG001
    refresh.run(at(RECORDED_ON))
    assert store.fx_on().day == date(2026, 10, 6)
    sunday = store.fx_on(date(2026, 10, 4))
    assert sunday.day == date(2026, 10, 2)                           # the Friday before
    assert store.fx_on(date(2020, 1, 1)) is None


def test_a_second_refresh_reads_from_a_week_before_the_last_stored_day(fresh_db):  # noqa: ARG001
    refresh.run(at(RECORDED_ON))
    asked = []
    inner = transport(OPEN, FRED)

    def watch(req):
        if req.url.endswith("/EXR/D..EUR.SP00.A"):
            asked.append(req.params["startPeriod"])
            return http.Response(200, b"KEY,FREQ,CURRENCY,TIME_PERIOD,OBS_VALUE\n")
        return inner(req)
    http.use_transport(watch)
    refresh.run(at(RECORDED_ON))
    assert asked == ["2026-09-29"]


def test_old_exchange_rates_are_pruned(fresh_db):  # noqa: ARG001
    refresh.run(at(RECORDED_ON))
    oldest_kept = RECORDED_ON - timedelta(days=store.KEEP_FX_DAYS)
    store.save_fx([FxDay(oldest_kept - timedelta(days=1), {"EUR": 1.0}), FxDay(oldest_kept, {"EUR": 1.0})])
    assert store.prune_fx(RECORDED_ON) == 1
    assert store.usage()["fx_days"] == 11 and store.fx_on(oldest_kept).day == oldest_kept


def test_a_failing_source_keeps_what_was_stored(fresh_db):  # noqa: ARG001
    refresh.run(at(RECORDED_ON))
    before, _ = store.all_series()
    http.use_transport(transport(OPEN, FRED, fail={"sdmx.oecd.org": 503}))
    summary = refresh.run(at(RECORDED_ON))
    assert summary["source_problems"] == {"oecd": "failed"}
    after, _ = store.all_series()
    assert after["bond_yield_10y.DE.oecd"] == before["bond_yield_10y.DE.oecd"]


def test_a_refused_fred_key_fails_the_run_so_it_alerts_after_storing_the_rest(fresh_db):  # noqa: ARG001
    http.use_transport(transport(OPEN, FRED, fail={"api.stlouisfed.org": 403}))
    with pytest.raises(refresh.KeyRefused):
        refresh.run(at(RECORDED_ON))
    stored, _ = store.all_series()
    assert "policy_rate.GB.bis" in stored and "reference_rate.ESTR.ecb" in stored
    assert not any(s.source == "fred" for s in stored.values())
    assert store.fx_on().day == date(2026, 10, 6)


def test_a_source_answering_an_unknown_shape_is_unreadable_and_the_rest_still_count(fresh_db):  # noqa: ARG001
    inner = transport(OPEN, FRED)

    def odd(req):
        if req.host == "www.imf.org":
            return http.Response(200, b'{"values": ["a list, not a map"]}', "application/json")
        return inner(req)
    http.use_transport(odd)
    with pytest.raises(refresh.NotCurrent):        # no growth or inflation projections: nothing current
        refresh.run(at(RECORDED_ON))
    stored, _ = store.all_series()
    assert "policy_rate.GB.bis" in stored and not any(s.source == "imf" for s in stored.values())
    found, problems = refresh._read(RECORDED_ON)
    assert problems == {"imf": "unreadable"} and found


def test_without_a_fred_key_the_run_still_succeeds_and_says_so(fresh_db, monkeypatch):  # noqa: ARG001
    monkeypatch.delenv(connectors.FRED_KEY_ENV)
    summary = refresh.run(at(RECORDED_ON))
    assert summary["source_problems"] == {"fred": "not_configured"}
    assert summary["economies_current"] == 23


def test_too_few_current_economies_fail_the_run(fresh_db):  # noqa: ARG001
    with pytest.raises(refresh.NotCurrent, match="0 economies current"):
        refresh.run(at(RECORDED_ON + timedelta(days=400)))


def test_the_refresh_is_a_scheduled_task_and_skips_without_a_database():
    assert "economy-refresh" in scheduled.TASKS
    assert refresh.run(at(RECORDED_ON)) == {"skipped_no_database": True}


def test_the_workflows_refresh_nightly_on_both_environments_and_on_each_staging_deploy():
    workflows = Path(__file__).parent.parent / ".github" / "workflows"
    nightly = (workflows / "scheduled.yml").read_text(encoding="utf-8")
    staging = (workflows / "staging.yml").read_text(encoding="utf-8")
    assert "task economy-refresh" in nightly and "environment: staging" in nightly
    assert "task economy-refresh" in staging and '.summary.economies_current >= 20' in staging


def test_the_database_check_reports_economic_data(fresh_db):  # noqa: ARG001
    from ops.check_database import problems
    refresh.run(at(RECORDED_ON))
    result = client.get("/api/health/database").json()
    assert result["economic_data"]["series"] == len(everything())
    assert result["economic_data"]["fx_days"] == 10
    assert result["economic_data"]["budget_bytes"] == store.BUDGET_BYTES
    assert problems(result) == []
    over = {**result, "economic_data": {**result["economic_data"], "warning": True, "bytes": 7 * 1024 * 1024}}
    assert problems(over) == ["economic data at 7168 KB of its 8192 KB budget: lower db.economy.KEEP_FX_DAYS "
                              "or economy.model.KEEP_OBSERVATIONS (PLAN.md 4.2)"]


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------
def test_reference_rates_answer_sonias_current_level(fresh_db):  # noqa: ARG001
    refresh.run(at(RECORDED_ON))
    r = client.get("/api/economy/reference-rates")
    assert r.status_code == 200
    body = r.json()
    sonia = everything()["reference_rate.SONIA.fred"].latest
    assert (body["rates"]["SONIA"]["period"], body["rates"]["SONIA"]["value"]) == sonia
    assert body["rates"]["SONIA"]["basis"] == "benchmark"
    assert body["currency_benchmarks"]["GBP"] == "SONIA"
    assert body["refreshed_at"]


def test_countries_answer_every_economy_with_its_figures(fresh_db, monkeypatch):  # noqa: ARG001
    refresh.run(at(RECORDED_ON))
    monkeypatch.setattr("api.routers.economy._today", lambda: RECORDED_ON)
    body = client.get("/api/economy/countries").json()
    areas = {a["area"]: a for a in body["areas"]}
    assert set(areas) == {*COUNTRIES, "XM"}
    assert sum(a["current"] for a in areas.values()) == 23
    assert areas["IT"]["figures"]["sovereign_spread"]["value"] == pytest.approx(0.800286)
    assert areas["JP"]["currency"] == "JPY"
    assert {s["id"] for s in body["sources"]} == set(SOURCES)
    assert [s["configured"] for s in body["sources"] if s["id"] == "fred"] == [True]


def test_exchange_rates_answer_against_any_base(fresh_db):  # noqa: ARG001
    refresh.run(at(RECORDED_ON))
    body = client.get("/api/economy/exchange-rates", params={"base": "USD"}).json()
    assert (body["base"], body["published_on"]) == ("USD", "2026-10-06")
    assert body["rates"]["GBP"] == pytest.approx(0.8488 / 1.1269)
    assert client.get("/api/economy/exchange-rates", params={"base": "ZZZ"}).status_code == 404
    assert client.get("/api/economy/exchange-rates", params={"base": "usd"}).status_code == 422
    friday = client.get("/api/economy/exchange-rates", params={"on": "2026-10-04"}).json()
    assert friday["published_on"] == "2026-10-02" and friday["rates"]["EUR"] == 1.0


def test_without_a_database_the_endpoints_answer_empty():
    assert client.get("/api/economy/reference-rates").json()["rates"] == {}
    areas = client.get("/api/economy/countries").json()["areas"]
    assert all(a["figures"] == {} and not a["current"] for a in areas)
    assert client.get("/api/economy/exchange-rates").status_code == 404


def test_the_endpoints_read_storage_only_so_none_is_a_run(signed_out):  # noqa: ARG001
    assert not any(p.startswith("/api/economy") for p in RUN_PATHS)
    assert client.get("/api/economy/reference-rates").status_code == 401


def test_the_browser_tests_replay_what_the_api_answers_today():
    """web/e2e/fixtures/economy.json is the endpoint's answer for the recorded
    series; regenerate it when the answer changes."""
    from tests.e2e_economy import OUT, build
    assert json.loads(OUT.read_text(encoding="utf-8")) == build(),         "web/e2e/fixtures/economy.json is stale: run python -m tests.e2e_economy"


# ---------------------------------------------------------------------------
# The recorder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["economy_open", "economy_fred"])
def test_recording_a_fixture_again_from_itself_gives_the_same_bytes(case, tmp_path):
    out = record.record(record.CASES[case], tmp_path, RECORDED_ON, real=transport(OPEN, FRED))
    original = FIXTURES / case
    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in original.iterdir())
    for p in out.iterdir():
        if p.name != "index.json":
            assert p.read_bytes() == (original / p.name).read_bytes(), p.name


def test_the_recorder_refuses_to_write_a_key(tmp_path, monkeypatch):
    leak = transport(OPEN, FRED)

    def leaky(req):                         # a source echoing the key back in its answer
        resp = leak(req)
        return http.Response(resp.status, resp.content.replace(b'"lin"', f'"{FAKE_KEY}"'.encode()),
                             resp.content_type)
    with pytest.raises(SystemExit, match="key appeared"):
        record.record(record.CASES["economy_fred"], tmp_path, RECORDED_ON, real=leaky)


def test_series_keys_fit_their_column():
    longest = max((series_key(i, a, s) for i in ("sovereign_spread", "reference_rate")
                   for a in ("EURIBOR", *COUNTRIES) for s in SOURCES), key=len)
    assert len(longest) <= 48
