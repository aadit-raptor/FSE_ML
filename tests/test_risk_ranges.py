"""Risk ranges, correlations and scenarios by region (PLAN.md 4.4).

The industry history replays Damodaran's real archived editions, and the
economic history the IMF's and the BIS's real answers, recorded and trimmed
by ``python -m benchmarks.record --history`` (tests/fixtures/benchmarks/history);
the current editions and today's rates replay 4.3's and 4.2's fixtures.
Expected figures are worked from the recorded files directly (xlrd, csv,
json), not through the code under test.
"""
from __future__ import annotations

import csv
import io
import json
import statistics
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from benchmarks import damodaran, history, record, refresh, risk
from benchmarks.catalogue import REGIONS
from companies import http
from companies.record import replay
from core.config import DEFAULTS, build_corr_matrix, is_valid_corr, resolve_config
from economy import connectors
from economy import refresh as economy_refresh

FIXTURES = Path(__file__).parent / "fixtures"
HISTORY = FIXTURES / "benchmarks" / "history"
DAMODARAN = FIXTURES / "benchmarks" / "damodaran"
ECONOMY = (FIXTURES / "economy" / "economy_open", FIXTURES / "economy" / "economy_fred")
ECONOMY_DAY = date.fromisoformat(json.loads((ECONOMY[0] / "index.json").read_text())["_recorded_on"])
INDEX = json.loads((HISTORY / "index.json").read_text(encoding="utf-8"))
client = TestClient(app)


def transport(*cases: Path, fail: dict | None = None):
    replays = [replay(c) for c in cases]

    def answer(req: http.Request) -> http.Response:
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
    http.use_transport(transport(HISTORY, DAMODARAN, *ECONOMY))
    yield
    http.use_transport(None)


_TABLES: dict = {}


def tables() -> dict:
    """Everything a full refresh would store: the current editions, every
    archived one, and the economic history."""
    if not _TABLES:
        stored = {name: call() for name, call in damodaran.every_table()}
        stored[history.MACRO_TABLE] = history.read_macro(ECONOMY_DAY)
        while True:
            found, summary = history.fill(stored, ECONOMY_DAY, batch=500)
            stored.update({t.name: t for t in found})
            if summary["left"] == 0:
                break
        _TABLES.update(stored)
    return dict(_TABLES)


def economy() -> dict:
    found, problems = economy_refresh._read(ECONOMY_DAY)
    assert problems == {}
    return {s.key: s for s in found}


def answer(country: str, industry: str = "machinery", currency: str = "GBP", t=None) -> dict:
    return risk.risk_assumptions(country, industry, currency, tables() if t is None else t, economy(), ECONOMY_DAY)


def figure(a: dict, name: str) -> dict:
    return next(f for f in a["figures"] if f["field"] == name)


# ---------------------------------------------------------------------------
# Independent readings of the recorded files
# ---------------------------------------------------------------------------
def recorded_workbook(dataset: str, group: str, year: int):
    import xlrd
    url = history.archive_url(history.HISTORY_DATASETS[dataset], REGIONS[group], year)
    entry = INDEX.get(url)
    try:
        return None if entry is None else xlrd.open_workbook(str(HISTORY / entry["file"]))
    except Exception:              # an edition recorded as the reader can't open it
        return None


def column(dataset: str, group: str, industry_name: str, header: str) -> dict[int, tuple[float, float]]:
    """{year: (firms, value)} read straight off the archived workbooks."""
    out = {}
    for year in history.archive_years(ECONOMY_DAY):
        book = recorded_workbook(dataset, group, year)
        if book is None:
            continue
        sheet = book.sheet_by_index(0)
        heads = ["".join(str(h).split()).lower() for h in sheet.row_values(1)]
        if header not in heads:
            continue
        for r in range(2, sheet.nrows):
            row = sheet.row_values(r)
            if str(row[0]).strip() == industry_name and isinstance(row[heads.index(header)], float):
                out[year] = (row[1], row[heads.index(header)])
    return out


def imf_years(code: str, iso3: str) -> dict[int, float]:
    entry = next(v for k, v in INDEX.items() if k.endswith(f"/{code}"))
    values = json.loads((HISTORY / entry["file"]).read_text())["values"][code][iso3]
    return {int(y): v for y, v in values.items() if history.WINDOW_START <= int(y) < ECONOMY_DAY.year}


def bis_months(area: str) -> dict[str, list[float]]:
    entry = next(v for k, v in INDEX.items() if "WS_CBPOL/M." in k)
    rows = csv.DictReader(io.StringIO((HISTORY / entry["file"]).read_text(encoding="utf-8-sig")))
    out: dict[str, list[float]] = {}
    for r in rows:
        if r["REF_AREA"] == area and r["OBS_VALUE"]:
            out.setdefault(r["TIME_PERIOD"][:4], []).append(float(r["OBS_VALUE"]))
    return out


# ---------------------------------------------------------------------------
# Reading the archive
# ---------------------------------------------------------------------------
def test_an_older_edition_reads_the_columns_it_has():
    status, rows = history.read_archive("multiples", "europe", 2011)
    assert status == "ok" and set(rows["machinery"]) == {"name", "firms", "ev_ebitda"}
    status, rows = history.read_archive("margins", "europe", 2016)
    assert status == "ok" and "ebitda_margin" in rows["machinery"] and "gross_margin" not in rows["machinery"]
    status, rows = history.read_archive("margins", "europe", 2024)
    assert {"gross_margin", "ebitda_margin"} <= set(rows["machinery"])


def test_an_edition_with_none_of_the_columns_or_none_at_all_is_said_so():
    # Developed Europe's 2011 margins print neither gross margin nor EBITDA/sales
    assert history.read_archive("margins", "europe", 2011) == ("unreadable", {})
    assert history.read_archive("margins", "china", 2023) == ("missing", {})


def test_an_archived_file_is_the_edition_for_the_year_before_it_was_published():
    """marginEurope16.xls is the January 2017 edition: it describes 2016;
    the current edition (January 2026) describes 2025."""
    assert history.archive_url(history.HISTORY_DATASETS["margins"], REGIONS["europe"], 2016).endswith(
        "/archives/marginEurope16.xls")
    assert list(history.archive_years(ECONOMY_DAY))[-1] == ECONOMY_DAY.year - 2
    years = history.industry_years(tables(), "europe", "machinery")
    assert years[ECONOMY_DAY.year - 1]["ev_ebitda"] == 14.980532        # the current edition (4.3)


def test_the_history_holds_each_industrys_figures_year_by_year_as_printed():
    years = history.industry_years(tables(), "europe", "machinery")
    printed = column("multiples", "europe", "Machinery", "ev/ebitda")
    assert printed and all(years[y]["ev_ebitda"] == round(v, 6) for y, (_, v) in printed.items())
    gross = column("margins", "europe", "Machinery", "grossmargin")
    assert min(gross) == 2017 and all(years[y]["gross_margin"] == round(v, 6) for y, (_, v) in gross.items())


def test_a_year_keeps_the_smaller_of_its_two_company_counts():
    merged = history.merge({}, 2020, {"x": {"name": "X", "firms": 30, "ev_ebitda": 9.0}})
    history.merge(merged, 2020, {"x": {"name": "X", "firms": 25, "gross_margin": 0.3}})
    assert merged["x"]["years"]["2020"] == {"firms": 25, "ev_ebitda": 9.0, "gross_margin": 0.3}


def test_a_file_read_once_is_not_read_again_except_a_missing_newest_one():
    newest = list(history.archive_years(ECONOMY_DAY))[-1]
    read = {history.read_key(d, g, y): {"status": "ok"} for d in history.HISTORY_DATASETS for g in REGIONS
            for y in history.archive_years(ECONOMY_DAY)}
    assert history.to_read(read, ECONOMY_DAY) == []
    read[history.read_key("margins", "china", newest - 1)] = {"status": "missing"}
    read[history.read_key("margins", "china", newest)] = {"status": "missing"}
    assert history.to_read(read, ECONOMY_DAY) == [("margins", "china", newest)]
    assert history.to_read(read, ECONOMY_DAY, retry=False) == []


def test_a_fill_reads_a_batch_and_says_what_is_left():
    stored = {name: call() for name, call in damodaran.every_table()}
    found, summary = history.fill(stored, ECONOMY_DAY, batch=5)
    everything = len(history.to_read({}, ECONOMY_DAY))
    assert summary["left"] == everything - 5
    log = next(t for t in found if t.name == history.READ_TABLE).rows
    assert len(log) == 5 and all(v["status"] in ("ok", "missing", "unreadable") for v in log.values())


def test_a_failed_call_is_asked_again_next_time():
    stored = {name: call() for name, call in damodaran.every_table()}
    http.use_transport(transport(HISTORY, fail={"vebitdaEurope11": 503}))
    found, summary = history.fill(stored, ECONOMY_DAY, batch=20)
    key = history.read_key("multiples", "europe", 2011)
    assert summary["problems"] == {key: "failed"}
    assert key not in next(t for t in found if t.name == history.READ_TABLE).rows


def test_the_economic_history_is_the_imf_by_year_and_the_bis_averaged_over_each_full_year():
    macro = history.read_macro(ECONOMY_DAY).rows
    assert history.macro_series({history.MACRO_TABLE: history.read_macro(ECONOMY_DAY)}, "GB", "real_growth") == \
        imf_years("NGDP_RPCH", "GBR")
    months = bis_months("GB")
    full = {y: v for y, v in months.items() if len(v) == 12 and int(y) < ECONOMY_DAY.year}
    assert macro["GB"]["policy_rate"] == {y: round(sum(v) / 12, 4) for y, v in full.items()}
    assert "policy_rate" not in macro["DE"] and "policy_rate" in macro["XM"]      # a euro member takes XM's
    assert min(map(int, macro["GB"]["real_growth"])) == history.WINDOW_START


def test_a_year_of_policy_rates_counts_only_when_all_twelve_months_are_published():
    rows = ["FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE"]
    rows += [f"M,GB,2010-{m:02d},1.0" for m in range(1, 13)] + [f"M,GB,2011-{m:02d},2.0" for m in range(1, 7)]
    http.use_transport(lambda req: http.Response(200, "\n".join(rows).encode(), "text/csv"))
    assert history._bis(ECONOMY_DAY) == {"GB": {"policy_rate": {"2010": 1.0}}}


# ---------------------------------------------------------------------------
# Ranges
# ---------------------------------------------------------------------------
def test_uk_and_india_deals_get_different_sourced_ranges():
    """PLAN.md 4.4 "done when": the same industry in the UK and in India."""
    uk, india = answer("GB", currency="GBP"), answer("IN", currency="INR")
    for key in ("mc_growth_std", "mc_exit_std", "mc_rate_std", "mc_gm_std"):
        assert uk["settings"][key] != india["settings"][key], key
    assert figure(uk, "mc_exit_std")["area"] == "europe" and figure(india, "mc_exit_std")["area"] in ("india", "emerging")
    assert figure(uk, "mc_growth_std")["area"] == "GB" and figure(india, "mc_growth_std")["area"] == "IN"


def test_the_exit_and_margin_spreads_are_the_industrys_own_year_to_year_spread():
    a = answer("GB")
    ev = {y: v for y, (n, v) in column("multiples", "europe", "Machinery", "ev/ebitda").items() if n >= 20}
    ev[ECONOMY_DAY.year - 1] = 14.980532
    assert a["settings"]["mc_exit_std"] == round(statistics.stdev(round(v, 6) for v in ev.values()), 2)
    f = figure(a, "mc_exit_std")
    assert f["detail"]["years"] == len(ev) and f["sample"] == 210 and f["sample_kind"] == "companies"
    gm = {y: v for y, (n, v) in column("margins", "europe", "Machinery", "grossmargin").items() if n >= 20}
    gm[ECONOMY_DAY.year - 1] = 0.446829
    assert a["settings"]["mc_gm_std"] == round(statistics.stdev(round(v, 6) for v in gm.values()) * 100, 2)


def test_growth_and_rate_spreads_are_the_countrys_own_year_to_year_spread():
    a = answer("GB")
    real, inflation = imf_years("NGDP_RPCH", "GBR"), imf_years("PCPIPCH", "GBR")
    nominal = [((1 + real[y] / 100) * (1 + inflation[y] / 100) - 1) * 100 for y in real if y in inflation]
    assert a["settings"]["mc_growth_std"] == round(statistics.stdev(nominal), 2)
    # 1999 is read too, so the window's first change is 2000's
    rates = {int(y): round(sum(v) / 12, 4) for y, v in bis_months("GB").items()
             if len(v) == 12 and int(y) < ECONOMY_DAY.year}
    changes = [rates[y] - rates[y - 1] for y in rates if y - 1 in rates and y >= history.WINDOW_START]
    assert a["settings"]["mc_rate_std"] == round(statistics.stdev(changes), 2)
    # A euro member's rate is the euro area's
    assert figure(answer("DE", currency="EUR"), "mc_rate_std")["detail"]["policy_area"] == "XM"


def test_the_means_are_the_deals_own_sourced_starting_figures():
    from benchmarks import starting
    a = answer("GB")
    s = starting.starting_assumptions("GB", "machinery", "GBP", tables(), economy(), ECONOMY_DAY)["inputs"]
    assert (a["settings"]["mc_growth_mean"], a["settings"]["mc_exit_mean"], a["settings"]["mc_rate_mean"],
            a["settings"]["mc_gm_mean"]) == (s["growth"], s["exit_mult"], s["base_rate"], s["gross_margin"])


def test_a_short_history_hands_over_to_a_wider_group_and_says_why():
    t = tables()
    # Only three years of Indian machinery left: India hands over to emerging markets
    t[history.table_name("india")] = damodaran.Table(history.table_name("india"), None, "", {
        "machinery": {"name": "Machinery", "years": {str(y): f for y, f in list(
            t[history.table_name("india")].rows["machinery"]["years"].items())[:3]}}})
    t.pop("margins.india"), t.pop("multiples.india")
    f = figure(answer("IN", currency="INR", t=t), "mc_exit_std")
    assert f["area"] == "emerging"
    assert f["skipped"] == [{"area": "india", "reason": "short", "sample": 3}]


def test_a_country_outside_the_catalogue_takes_its_regions_median_spread():
    a = answer("PT", currency="EUR")                              # Portugal: not one of the 24 economies
    f = figure(a, "mc_growth_std")
    assert (f["area"], f["level"], f["detail"]["statistic"]) == ("europe", "region", "median")
    assert f["skipped"] == [{"area": "PT", "reason": "missing", "sample": None}]


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("group,country", [("us", "US"), ("japan", "JP"), ("china", "CN"), ("india", "IN"),
                                           ("europe", "GB"), ("aus_nz_canada", "CA"), ("emerging", "BR"),
                                           ("global", None)])
def test_every_regions_correlation_matrix_is_valid(group, country):
    """PLAN.md 4.4 "done when": every group's matrix passes the simulation's
    own check, and the Settings a deal there gets build that same matrix."""
    rows, growth, rates, _ = risk.panel(tables(), group, ECONOMY_DAY)
    assert rows
    matrix, _ = risk.valid_matrix(risk.raw_correlations(rows, growth, rates)[0])
    assert is_valid_corr(matrix) and np.linalg.eigvalsh(matrix).min() > 0
    if country:
        a = answer(country, currency="USD")
        assert a["correlation"]["group"] == group
        m = build_corr_matrix(resolve_config({k: v for k, v in a["settings"].items() if k.startswith("corr_")}))
        assert is_valid_corr(m) and np.allclose(m, matrix)


def test_a_correlation_is_the_rank_correlation_of_the_panel_turned_normal():
    t = tables()
    rows, growth, rates, _ = risk.panel(t, "europe", ECONOMY_DAY)
    pairs = [(r[1], r[3]) for r in rows if r[1] is not None and r[3] is not None]
    from scipy.stats import spearmanr
    rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
    raw, counts = risk.raw_correlations(rows, growth, rates)
    assert raw[1, 3] == pytest.approx(2 * np.sin(np.pi * rho / 6)) and counts[1, 3] == len(pairs)


def test_an_invalid_matrix_is_shrunk_toward_none_until_it_is_valid():
    bad = np.array([[1, .9, -.9, 0, 0], [.9, 1, .9, 0, 0], [-.9, .9, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1.]])
    assert not is_valid_corr(bad)
    fixed, s = risk.valid_matrix(bad)
    # The smallest shrink that leaves an eigenvalue of at least the floor (rounding moves it by at most 0.02)
    assert 0 < s < 1 and is_valid_corr(fixed)
    assert np.linalg.eigvalsh(fixed).min() >= risk.EIGEN_FLOOR - 0.02
    less = s - risk.SHRINK_STEP
    assert np.linalg.eigvalsh((1 - less) * bad + less * np.eye(5)).min() < risk.EIGEN_FLOOR
    same, none = risk.valid_matrix(np.eye(5))
    assert none == 0 and np.array_equal(same, np.eye(5))


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------
def test_each_scenario_lists_its_historical_periods():
    """PLAN.md 4.4 "done when": the years each preset is built from, from the
    country's own history, joined into periods."""
    a = answer("GB")
    real, inflation = imf_years("NGDP_RPCH", "GBR"), imf_years("PCPIPCH", "GBR")
    n = round(len(real) * risk.SHARE)
    weakest = sorted(sorted(real, key=lambda y: (real[y], -y))[:n])
    hottest = sorted(sorted(inflation, key=lambda y: (inflation[y], y), reverse=True)[:n])
    assert a["scenarios"]["recession"]["years"] == weakest and {2009, 2020} <= set(weakest)
    assert a["scenarios"]["stagflation"]["years"] == hottest and 2022 in hottest
    for sc in a["scenarios"].values():
        assert sc["periods"] and sum(p["end"] - p["start"] + 1 for p in sc["periods"]) == len(sc["years"])
        assert sc["area"] == "GB" and sc["of_years"] == len(real)
    assert risk.periods([2008, 2009, 2020]) == [{"start": 2008, "end": 2009}, {"start": 2020, "end": 2020}]


def test_a_scenario_moves_each_mean_by_what_happened_in_its_years():
    a = answer("GB")
    years = a["scenarios"]["recession"]["years"]
    real, inflation = imf_years("NGDP_RPCH", "GBR"), imf_years("PCPIPCH", "GBR")
    nominal = {y: ((1 + real[y] / 100) * (1 + inflation[y] / 100) - 1) * 100 for y in real}
    gap = statistics.fmean(nominal[y] for y in years) - statistics.fmean(nominal.values())
    assert a["settings"]["rec_growth_adj"] == round(gap, 2)
    assert a["settings"]["rec_growth_floor"] == round(min(nominal[y] for y in years), 2)
    rates = {int(y): round(sum(v) / 12, 4) for y, v in bis_months("GB").items() if len(v) == 12}
    change = statistics.fmean(rates[y] - rates[y - 1] for y in years if y in rates and y - 1 in rates)
    r = a["settings"]["mc_rate_mean"]
    assert a["settings"]["rec_rate_mult"] == round(max(r + change, 0) / r, 3)
    ev = {y: f["ev_ebitda"] for y, f in history.industry_years(tables(), "europe", "machinery").items()
          if f.get("firms", 0) >= 20 and "ev_ebitda" in f}
    inside = [ev[y] for y in years if y in ev]
    assert a["settings"]["rec_exit_mult"] == round(statistics.fmean(inside) / statistics.fmean(ev.values()), 3)


def test_bull_growth_is_a_multiplier_on_the_sourced_mean():
    a = answer("GB")
    f = figure(a, "bull_growth_mult")
    g = a["settings"]["mc_growth_mean"]
    # the change is shown to two decimals; the multiplier was worked on the unrounded one
    assert f["value"] == pytest.approx((g + f["detail"]["change"]) / g, abs=2e-3) and f["detail"]["base"] == g
    assert f["value"] > 1 and f["detail"]["change"] > 0


def test_the_sourced_settings_change_the_simulation_and_it_accepts_them():
    """Verify by output: the Monte Carlo answer moves with the sourced Settings."""
    a = answer("GB")
    body = {"mc": {"n": 2000, "ebitda": 100, "entry_mult": 10, "hold": 5, "hurdle": 20,
                   "growth_mean": 5, "growth_std": 3, "exit_mean": 10, "exit_std": 1.5,
                   "rate_mean": 6.5, "rate_std": 1.5, "gm_mean": 40, "gm_std": 3},
            "deal": {}, "seed": 42}
    sourced = {**body, "settings": a["settings"]}
    r0 = client.post("/api/montecarlo/scenarios", json=body)
    r1 = client.post("/api/montecarlo/scenarios", json=sourced)
    assert r0.status_code == r1.status_code == 200, r1.text
    assert r0.json()["scenarios"]["recession"]["mean_irr"] != r1.json()["scenarios"]["recession"]["mean_irr"]
    assert r0.json()["scenarios"]["base"]["mean_irr"] != r1.json()["scenarios"]["base"]["mean_irr"]


def test_every_setting_produced_is_a_known_setting():
    a = answer("GB")
    assert set(a["settings"]) == set(risk.SETTINGS) and set(risk.SETTINGS) <= set(DEFAULTS)


def test_without_stored_data_every_setting_is_missing():
    a = risk.risk_assumptions("GB", "machinery", "GBP", {}, {}, ECONOMY_DAY)
    assert a["settings"] == {} and {m["field"] for m in a["missing"]} == set(risk.SETTINGS)


# ---------------------------------------------------------------------------
# Refresh, storage and the API
# ---------------------------------------------------------------------------
def at(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, 3, tzinfo=timezone.utc)


def fill_store():
    summaries = [refresh.run(at(ECONOMY_DAY))]
    while summaries[-1]["more"]:
        summaries.append(refresh.run(at(ECONOMY_DAY)))
    return summaries


def test_a_refresh_fills_the_history_a_batch_a_call_until_nothing_is_left(fresh_db):  # noqa: ARG001
    summaries = fill_store()
    first, last = summaries[0], summaries[-1]
    assert first["history_files_read"] == 0 and first["more"] is True and first["macro_economies"] == 25
    assert all(s["history_files_read"] <= history.BATCH for s in summaries)
    assert (last["history_left"], last["history_groups"], last["more"]) == (0, 8, False)
    assert last["tables_stored"] == 41
    # A later run reads nothing from the archive
    again = refresh.run(at(ECONOMY_DAY))
    assert (again["history_files_read"], again["more"]) == (0, False)


def test_the_history_stays_inside_its_budget(fresh_db):  # noqa: ARG001
    from db import benchmarks as store
    fill_store()
    use = store.usage()
    assert 0 < use["bytes"] < store.BUDGET_BYTES * 0.5 and use["warning"] is False


def test_the_api_answers_the_sourced_risk_figures(fresh_db, monkeypatch):  # noqa: ARG001
    from db import economy as economy_store
    fill_store()
    economy_store.save_series(list(economy().values()))
    monkeypatch.setattr("api.routers.benchmarks._today", lambda: ECONOMY_DAY)
    r = client.get("/api/benchmarks/risk", params={"country": "GB", "industry": "machinery", "currency": "GBP"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["settings"] == answer("GB")["settings"] and body["industry_name"] == "Machinery"
    assert body["scenarios"]["recession"]["periods"]
    get = lambda **p: client.get("/api/benchmarks/risk", params=p).status_code  # noqa: E731
    assert get(country="GB", industry="banks_regional", currency="GBP") == 404
    assert get(country="gb", industry="machinery", currency="GBP") == 422


def test_without_a_database_the_api_answers_every_setting_missing():
    body = client.get("/api/benchmarks/risk", params={"country": "GB", "currency": "GBP"}).json()
    assert body["settings"] == {} and len(body["missing"]) == len(risk.SETTINGS)


def test_the_workflows_finish_the_backfill_and_staging_checks_it():
    workflows = Path(__file__).parent.parent / ".github" / "workflows"
    nightly = (workflows / "scheduled.yml").read_text(encoding="utf-8")
    staging = (workflows / "staging.yml").read_text(encoding="utf-8")
    assert "task benchmarks-refresh --repeat" in nightly and "task benchmarks-refresh --repeat" in staging
    assert ".history_left == 0" in staging and ".history_groups == 8" in staging


def test_recording_the_history_again_from_itself_gives_the_same_bytes(tmp_path):
    out = record.record_history(tmp_path, ECONOMY_DAY, real=transport(HISTORY))
    assert sorted(p.name for p in out.iterdir()) == sorted(p.name for p in HISTORY.iterdir())
    for p in out.iterdir():
        if p.name != "index.json":
            assert p.read_bytes() == (HISTORY / p.name).read_bytes(), p.name
