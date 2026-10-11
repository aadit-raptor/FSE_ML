"""Economic regime by region (PLAN.md 5.7).

"Done when: each region shows a dated regime; an India deal shows India's
regime."

The figures are the IMF's own, recorded (tests/fixtures/regime/imf and
ml/evaluation/data/regime_history.json, both from python -m
tests.ml_regime_history); each expected number below was read off them.
The database tests use a throwaway database (tests/conftest.py).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from api.limits import RUN_PATHS
from api.main import app
from companies import http
from companies.record import replay
from jobs.scheduled import TASKS
from ml import macro_regime as mr
from ml.evaluation import regime as card_eval
from tests import e2e_regime

ROOT = Path(__file__).parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "regime" / "imf"
RECORDED_ON = e2e_regime.recorded_on()
NOW = datetime(RECORDED_ON.year, RECORDED_ON.month, RECORDED_ON.day, 5, tzinfo=timezone.utc)
CARD = json.loads(mr.CARD.read_text(encoding="utf-8"))
client = TestClient(app)


@pytest.fixture
def recorded():
    http.use_transport(replay(FIXTURE))
    yield
    http.use_transport(None)


@pytest.fixture
def offline():
    """Any request fails the test: nothing should be read."""
    def refuse(req: http.Request) -> http.Response:
        pytest.fail(f"unexpected request to {req.host}")
    http.use_transport(refuse)
    yield
    http.use_transport(None)


def flat(years: range, growth: float = 2.0, inflation: float = 2.0) -> dict[int, tuple]:
    return {y: (growth, inflation) for y in years}


# ---------------------------------------------------------------------------
# The rule, by hand
# ---------------------------------------------------------------------------
def test_a_year_is_measured_against_the_median_of_the_five_before_it():
    series = {1: (1.0, 2.0), 2: (2.0, 2.0), 3: (3.0, 2.0), 4: (10.0, 8.0), 5: (-4.0, 2.0), 6: (2.5, 5.0)}
    # Medians, not means (2.4 and 3.2): one crash and its rebound move nothing
    assert mr.norm(series, 6) == (2.0, 2.0)
    assert mr.gap(series, 6) == (0.5, 3.0)
    # A projection made at the time stands in for the year's own figure
    assert mr.gap(series, 6, (0.0, 2.0)) == (-2.0, 0.0)
    # A year missing among the five leaves no norm
    assert mr.norm(series, 8) is None and mr.norm({k: v for k, v in series.items() if k != 3}, 6) is None


def test_a_cut_off_is_the_edge_of_a_fifth():
    values = [-3, -2, -1, 0, 0, 1, 1, 2, 3, 5]
    assert mr.cutoff(values, top=False) == -2 and mr.cutoff(values, top=True) == 3
    assert mr.cutoff([4.0, 1.0, 2.0], top=True) == 4.0          # never fewer than one year


@pytest.mark.parametrize("gaps,regime", [
    ((-2.0, 5.0), "recession"),         # weak growth first, whatever inflation does; the edge counts
    ((-1.9, 3.0), "stagflation"),
    ((3.0, 3.0), "stagflation"),        # high inflation before strong growth
    ((3.0, 2.9), "bull"),
    ((2.9, 2.9), "base"),
    ((0.0, 0.0), "base"),
])
def test_the_rule_names_weak_growth_then_high_inflation_then_strong_growth(gaps, regime):
    assert mr.name(gaps, {"weak_growth": -2.0, "high_inflation": 3.0, "strong_growth": 3.0}) == regime


def test_a_call_by_hand():
    # Twenty flat years, then: 2020 growth 3 below, 2021 3 above, 2022 inflation 4 above
    series = {**flat(range(1995, 2020)), 2020: (-1.0, 2.0), 2021: (5.0, 2.0), 2022: (2.0, 6.0), 2023: (2.0, 2.0)}
    # Up to 2020 every gap was zero, so both growth cut-offs are zero and a
    # year at its norm sits on the weak edge
    assert mr.call(series, 2020)["regime"] == "recession"
    found = mr.call(series, 2023)
    # 2023's norm: the medians of 2018-2022, (2, 2, -1, 5, 2) and (2, 2, 2, 2, 6)
    assert (found["growth_norm"], found["inflation_norm"]) == (2.0, 2.0)
    assert (found["growth_gap"], found["inflation_gap"]) == (0.0, 0.0)
    # 23 earlier gaps (2000-2022): a fifth is five years (23 x 0.2 = 4.6).
    # Growth gaps sorted: -3, 0 ... 0, 3, so both edges are 0; inflation's: 4, 0 ...
    assert found["history_years"] == 23
    assert (found["weak_growth"], found["strong_growth"], found["high_inflation"]) == (0.0, 0.0, 0.0)
    assert found["regime"] == "recession" and (found["growth"], found["inflation"]) == (2.0, 2.0)


def test_a_call_reads_nothing_from_the_year_itself_or_later():
    data = card_eval.load()
    series = card_eval.series(data, "us")
    before = mr.call(series, 2009, (-2.8, -0.9))
    changed = {y: ((99.0, 99.0) if y >= 2009 else v) for y, v in series.items()}
    assert mr.call(changed, 2009, (-2.8, -0.9)) == before


def test_a_short_history_gives_no_regime():
    # Ten earlier gaps need fifteen earlier years (five of them have no norm)
    assert mr.call(flat(range(2000, 2016)), 2015) is not None
    assert mr.call(flat(range(2001, 2016)), 2015) is None
    assert mr.thresholds(flat(range(2001, 2016)), 2015) is None


def test_the_year_named_is_last_years_until_the_imfs_april_figures_are_in():
    assert mr.regime_year(date(2026, 4, 30)) == 2025
    assert mr.regime_year(date(2026, 5, 1)) == 2026
    assert mr.regime_year(date(2027, 1, 2)) == 2026


@pytest.mark.parametrize("country,region,own", [
    ("IN", "india", True), ("US", "us", True), ("GB", "uk", True), ("JP", "japan", True), ("CN", "china", True),
    ("DE", "eurozone", True), ("PT", "eurozone", True), ("in", "india", True),
    ("BR", "global", False), ("CH", "global", False), ("", "global", False), (None, "global", False),
])
def test_a_country_takes_its_own_region_the_euro_areas_or_the_world(country, region, own):
    assert mr.region_of(country) == (region, own)


# ---------------------------------------------------------------------------
# The recorded figures
# ---------------------------------------------------------------------------
def test_the_imfs_figures_are_read_for_the_seven_regions(recorded):
    series = mr.read(RECORDED_ON)
    assert set(series) == set(mr.REGIONS) == {"us", "eurozone", "uk", "japan", "china", "india", "global"}
    assert series["india"][2026] == (6.5, 4.7) and series["india"][2020] == (-5.8, 6.2)
    assert series["global"][2026][0] == 3.1 and series["eurozone"][2026][0] == 1.1
    # The IMF's euro area starts in 1992; nothing after this year is kept
    assert min(series["eurozone"]) == 1992 and min(series["us"]) == mr.FIRST_YEAR
    assert max(max(s) for s in series.values()) == RECORDED_ON.year


def test_every_region_has_a_dated_regime(recorded):
    rows = mr.fit(mr.read(RECORDED_ON), RECORDED_ON)
    for region in mr.REGIONS:
        found = rows[region]["fit"]
        assert found["regime"] in mr.REGIMES, region
        assert (found["year"], found["basis"]) == (RECORDED_ON.year, "projection"), region
        assert len(found["recent"]) == mr.RECENT
        # 2020 was a recession everywhere, on final figures
        assert {r["year"]: r["regime"] for r in found["recent"]}[2020] == "recession", region
        # What is stored stops at the regime's year
        assert max(map(int, rows[region]["series"])) == RECORDED_ON.year


def test_indias_regime_by_hand(recorded):
    india = mr.fit(mr.read(RECORDED_ON), RECORDED_ON)["india"]["fit"]
    # 2021-2025: growth 9.7, 7.6, 7.2, 7.1, 7.6 (median 7.6); inflation 5.5, 6.6, 5.4, 4.6, 2.1 (median 5.4)
    assert (india["growth"], india["inflation"]) == (6.5, 4.7)
    assert (india["growth_norm"], india["inflation_norm"]) == (7.6, 5.4)
    assert (india["growth_gap"], india["inflation_gap"]) == (-1.1, -0.7)
    assert india["weak_growth"] < india["growth_gap"] < india["strong_growth"]
    assert india["inflation_gap"] < india["high_inflation"] and india["regime"] == "base"
    assert india["first_year"] == 1980 and india["history_years"] == 41


def test_before_may_the_regime_is_last_years_estimate(recorded):
    early = date(RECORDED_ON.year + 1, 2, 1)
    found = mr.fit(mr.read(early), early)["india"]["fit"]
    assert (found["year"], found["basis"]) == (RECORDED_ON.year, "estimate")


# ---------------------------------------------------------------------------
# The monthly refit
# ---------------------------------------------------------------------------
def test_the_refit_stores_seven_dated_regimes_and_reads_once_a_month(fresh_db, recorded):  # noqa: ARG001
    from db import benchmarks as store
    first = mr.refresh(NOW)
    assert first["read"] is True and (first["regions"], first["dated"]) == (7, 7)
    assert first["year"] == RECORDED_ON.year and first["read_on"] == RECORDED_ON.isoformat()
    assert "india:base" in first["regimes"].split(",") and first["source_problem"] == ""
    # Flat, as a scheduled run's summary must be
    assert all(isinstance(v, (int, float, bool, str)) or v is None for v in first.values())
    stored = store.table(mr.TABLE)
    assert stored.published == RECORDED_ON and set(stored.rows) == set(mr.REGIONS)
    # Not an industry-average table: the benchmark refresh never sees it
    assert mr.TABLE not in store.all_tables(history=True)[0]
    assert store.usage()["tables"] == 0 and store.usage()["history_tables"] == 0

    # Later the same month: answered from storage, nothing read
    http.use_transport(lambda req: pytest.fail("read again in the same month"))
    again = mr.refresh(NOW.replace(day=28))
    assert again["read"] is False and again["dated"] == 7 and again["read_on"] == RECORDED_ON.isoformat()

    # The first run of the next month refits
    http.use_transport(replay(FIXTURE))
    next_month = datetime(RECORDED_ON.year, RECORDED_ON.month + 1, 1, 5, tzinfo=timezone.utc)
    later = mr.refresh(next_month)
    assert later["read"] is True and later["read_on"] == next_month.date().isoformat()


def test_a_failed_source_keeps_the_stored_fit_and_is_tried_again(fresh_db, recorded):  # noqa: ARG001
    mr.refresh(NOW)
    http.use_transport(lambda req: http.Response(503, b""))
    next_month = datetime(RECORDED_ON.year, RECORDED_ON.month + 1, 1, 5, tzinfo=timezone.utc)
    kept = mr.refresh(next_month)
    assert kept["read"] is False and kept["source_problem"] == "failed"
    assert kept["dated"] == 7 and kept["read_on"] == RECORDED_ON.isoformat()
    # Still due: the next night reads
    http.use_transport(replay(FIXTURE))
    assert mr.refresh(next_month + timedelta(days=1))["read"] is True


def test_with_nothing_stored_a_failed_source_fails_the_run(fresh_db):  # noqa: ARG001
    http.use_transport(lambda req: http.Response(503, b""))
    try:
        with pytest.raises(mr.Incomplete):
            mr.refresh(NOW)
        http.use_transport(lambda req: http.Response(200, b'{"values": []}', "application/json"))
        with pytest.raises(mr.Incomplete):
            mr.refresh(NOW)
    finally:
        http.use_transport(None)


def test_figures_that_leave_a_region_without_a_regime_never_replace_a_whole_fit(fresh_db, recorded):  # noqa: ARG001
    from db import benchmarks as store
    with mock.patch.object(mr, "MIN_YEARS", 40), pytest.raises(mr.Incomplete, match="6 of 7"):
        mr.refresh(NOW)                                         # the euro area has 29 earlier gaps
    assert store.table(mr.TABLE) is None
    mr.refresh(NOW)
    next_month = datetime(RECORDED_ON.year, RECORDED_ON.month + 1, 1, 5, tzinfo=timezone.utc)
    with mock.patch.object(mr, "MIN_YEARS", 40), pytest.raises(mr.Incomplete, match="unchanged"):
        mr.refresh(next_month)
    kept = store.table(mr.TABLE)
    assert kept.published == RECORDED_ON and all(kept.rows[r]["fit"] for r in mr.REGIONS)
    # Still due: the next night reads again
    assert mr.refresh(next_month + timedelta(days=1))["read"] is True


def test_an_estimate_of_the_year_just_ended_is_gated_by_the_autumn_set(fresh_db, recorded):  # noqa: ARG001
    early = datetime(RECORDED_ON.year + 1, 2, 1, 5, tzinfo=timezone.utc)
    mr.refresh(early)
    body = ask("IN", early.date())
    assert body["regime"]["basis"] == "estimate" and body["card"]["set"] == "autumn"
    assert body["card"]["model"] == CARD["evaluation"]["sets"]["autumn"]["by_region"]["india"]["model"]["accuracy"]
    real = mr.card_result

    def card(region, basis="projection"):
        found = real(region, basis)
        return {**found, "verdict": "does_not_beat_baseline"} if basis == "estimate" else found
    with mock.patch.object(mr, "card_result", card):
        assert ask("IN", early.date())["hidden"] == "does_not_beat_baseline"


def test_the_scheduled_task_runs_the_refit(fresh_db, recorded):  # noqa: ARG001
    with mock.patch.object(mr, "datetime") as clock:
        clock.now.return_value = NOW
        summary = TASKS["regime-refresh"].run()
    assert summary["dated"] == 7 and summary["read"] is True


def test_without_a_database_the_refit_is_skipped(offline):  # noqa: ARG001
    assert mr.refresh(NOW) == {"skipped_no_database": True}


def test_the_nightly_economic_data_task_refits_and_fails_with_it(fresh_db, recorded):  # noqa: ARG001
    """The workflows schedule ``economy-refresh`` on both environments and
    after each staging deploy; the refit rides on it."""
    from economy import refresh as economy_refresh
    workflows = ROOT / ".github" / "workflows"
    for name in ("scheduled.yml", "staging.yml"):
        assert "task economy-refresh --api" in (workflows / name).read_text(encoding="utf-8")
    with mock.patch.object(economy_refresh, "run", lambda: {"economies_current": 24}), \
            mock.patch.object(mr, "datetime") as clock:
        clock.now.return_value = NOW
        summary = TASKS["economy-refresh"].run()
        assert summary["economies_current"] == 24
        assert (summary["regime_dated"], summary["regime_year"], summary["regime_read"]) == (7, 2026, True)
        assert all(isinstance(v, (int, float, bool, str)) or v is None for v in summary.values())
        # A region left without a regime fails the scheduled run, so it alerts
        clock.now.return_value = datetime(RECORDED_ON.year, RECORDED_ON.month + 1, 1, 5, tzinfo=timezone.utc)
        with mock.patch.object(mr, "MIN_YEARS", 40), pytest.raises(mr.Incomplete):
            TASKS["economy-refresh"].run()

    # The data refresh failing does not stop the refit, and still fails the run
    def broken():
        raise RuntimeError("too few economies")
    with mock.patch.object(economy_refresh, "run", broken), mock.patch.object(mr, "refresh") as refit, \
            pytest.raises(RuntimeError, match="too few"):
        TASKS["economy-refresh"].run()
    assert refit.called


# ---------------------------------------------------------------------------
# What a deal is shown
# ---------------------------------------------------------------------------
def ask(country: str, today: date = RECORDED_ON) -> dict:
    with mock.patch("api.routers.integrations._today", lambda: today):
        resp = client.post("/api/ml/macro-regime", json={"country": country})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_an_india_deal_shows_indias_regime(fresh_db, recorded):  # noqa: ARG001
    mr.refresh(NOW)
    body = ask("IN")
    assert (body["status"], body["region"], body["own_region"], body["shown"]) == ("ok", "india", True, True)
    india = body["regime"]
    assert (india["regime"], india["year"], india["growth"], india["inflation"]) == ("base", 2026, 6.5, 4.7)
    assert body["read_on"] == RECORDED_ON.isoformat()
    assert (body["card"]["region"], body["card"]["set"]) == ("india", "spring")
    # Not the euro area's figures, nor the world's
    assert ask("DE")["regime"]["growth"] == 1.1 and ask("DE")["region"] == "eurozone"
    world = ask("BR")
    assert (world["region"], world["own_region"], world["regime"]["growth"]) == ("global", False, 3.1)
    assert body["model"]["settings_fingerprint"] is None


def test_every_region_is_listed_with_its_regime_and_year(fresh_db, recorded):  # noqa: ARG001
    mr.refresh(NOW)
    regions = ask("IN")["regions"]
    assert [r["region"] for r in regions] == list(mr.REGIONS)
    for r in regions:
        assert r["shown"] and r["regime"] in mr.REGIMES and r["year"] == 2026 and r["basis"] == "projection"


def test_a_regime_is_shown_only_where_its_card_beats_the_baseline(fresh_db, recorded):  # noqa: ARG001
    mr.refresh(NOW)
    real = mr.card_result

    def card(region, basis="projection"):
        found = real(region, basis)
        return {**found, "verdict": "does_not_beat_baseline"} if region == "india" else found
    with mock.patch.object(mr, "card_result", card):
        body = ask("IN")
        other = ask("US")
    assert (body["shown"], body["hidden"], body["regime"]) == (False, "does_not_beat_baseline", None)
    listed = {r["region"]: r for r in other["regions"]}
    assert listed["india"]["regime"] is None and listed["india"]["hidden"] == "does_not_beat_baseline"
    assert listed["india"]["year"] == 2026 and other["shown"] and listed["us"]["regime"] is not None


def test_an_old_fit_is_not_shown(fresh_db, recorded):  # noqa: ARG001
    mr.refresh(NOW)
    assert ask("IN", RECORDED_ON + timedelta(days=mr.MAX_AGE_DAYS))["shown"] is True
    stale = ask("IN", RECORDED_ON + timedelta(days=mr.MAX_AGE_DAYS + 1))
    assert (stale["shown"], stale["hidden"], stale["regime"]) == (False, "stale", None)
    assert all(r["hidden"] == "stale" and r["regime"] is None for r in stale["regions"])


def test_nothing_stored_or_no_database_says_there_is_no_data(fresh_db):  # noqa: ARG001
    body = ask("IN")
    assert (body["status"], body["shown"], body["regime"], body["regions"]) == ("no_data", False, None, [])
    assert body["region"] == "india" and body["read_on"] is None


def test_without_a_database_the_endpoint_says_there_is_no_data():
    body = ask("IN")
    assert body["status"] == "no_data" and body["region"] == "india"


def test_the_endpoint_counts_as_a_run_and_refuses_a_long_country():
    assert "/api/ml/macro-regime" in RUN_PATHS
    assert client.post("/api/ml/macro-regime", json={"country": "India"}).status_code == 422
    assert client.post("/api/ml/macro-regime", json={"nation": "IN"}).status_code == 422


def test_the_browser_tests_replay_what_the_endpoint_answers():
    committed = e2e_regime.OUT.read_text(encoding="utf-8")
    assert committed == e2e_regime.text(), "run python -m tests.e2e_regime"


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------
def test_the_card_tests_each_year_on_the_imfs_report_of_that_year():
    data = card_eval.load()
    rows = card_eval.scored(data, "S")
    assert min(r[0].year for r in rows) == card_eval.FIRST_TEST
    by_id = {r[0].id: r for r in rows}
    # 2020, India: in April the IMF projected 1.9% growth (a recession on
    # India's record) and the year ended at -5.8%; last year's was the baseline
    case, called, last = by_id["india.2020"]
    assert case.features == {"growth": 1.9, "inflation": 3.3} and case.truth == "recession"
    assert called == "recession" and last == mr.call(card_eval.series(data, "india"), 2019, (4.2, 4.5))["regime"]
    # 2009, the United States: the April report already had the recession
    assert by_id["us.2009"][1] == by_id["us.2009"][0].truth == "recession"
    # 2021, the United States: April's 6.4% growth and 2.3% inflation read as a
    # bull year; inflation ended at 4.7% and made it stagflation. "The same
    # as last year" said recession
    case, called, last = by_id["us.2021"]
    assert case.features == {"growth": 6.4, "inflation": 2.3}
    assert (case.truth, called, last) == ("stagflation", "bull", "recession")


def test_the_regime_beats_last_years_in_every_region_in_both_reports():
    assert CARD["evaluation"]["headline_set"] == mr.GATE_SET == "spring"
    for name in ("spring", "autumn"):
        group = CARD["evaluation"]["sets"][name]
        assert set(group["by_region"]) == set(mr.REGIONS)
        for region, found in group["by_region"].items():
            assert found["verdict"] == "beats_baseline", (name, region)
            assert found["model"]["accuracy"] > found["baseline"]["accuracy"]
        assert group["overall"]["model"]["accuracy"] > 0.7
    for region in mr.REGIONS:
        assert mr.card_result(region)["verdict"] == "beats_baseline"
        assert mr.card_result(region)["model"] == CARD["evaluation"]["sets"]["spring"]["by_region"][region][
            "model"]["accuracy"]


def test_the_cards_data_and_the_recorded_figures_are_one_reading(recorded):
    data = card_eval.load()
    assert data["written_on"] == RECORDED_ON.isoformat()
    series = mr.read(RECORDED_ON)
    for region in mr.REGIONS:
        assert card_eval.series(data, region) == {y: v for y, v in series[region].items() if y < RECORDED_ON.year}


# ---------------------------------------------------------------------------
# What it replaces
# ---------------------------------------------------------------------------
def test_the_us_only_fred_models_are_gone():
    ml = ROOT / "ml"
    assert not (ml / "correlation_updater.py").exists()
    assert not list(ml.glob("hmm_*.pkl")) and not (ml / "macro_data.parquet").exists()
    requirements = (ROOT / "requirements-ml.txt").read_text(encoding="utf-8")
    assert "hmmlearn" not in requirements and "fredapi" not in requirements
    source = (ml / "macro_regime.py").read_text(encoding="utf-8")
    assert "import numpy" not in source and "import pandas" not in source   # runs in the core image
