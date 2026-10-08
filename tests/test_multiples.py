"""The multiple predictor by region (PLAN.md 5.4).

"Done when: ranges contain actual multiples for most held-out companies per
region (card); it works with the library off."

The industry multiples are Damodaran's real archive and January 2026
edition, every industry, as kept in ml/evaluation/data/multiple_history.json
(checked below against the recorded fixtures for the industries those keep).
"""
from __future__ import annotations

import json
import math
from datetime import date
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from api.limits import RUN_PATHS
from api.main import app
from benchmarks.damodaran import Table
from library import references
from ml import multiple_predictor as mp
from ml.evaluation import multiples as card_eval
from ml.evaluation.card import paths

TABLES = card_eval.tables()
TODAY = date(2026, 10, 8)
client = TestClient(app)


def predict(country: str, industry: str = "machinery", hold: int = 5, **kw) -> dict:
    return mp.predict(country, industry, hold, TABLES, TODAY, **kw)


# ---------------------------------------------------------------------------
# The method, by hand
# ---------------------------------------------------------------------------
def tiny(rows: dict[str, dict[int, float]], firms: int = 30) -> dict[str, Table]:
    """A group "us" whose industries have these multiples by year."""
    history = {k: {"name": k, "years": {str(y): {"firms": firms, "ev_ebitda": v} for y, v in s.items()}}
               for k, s in rows.items()}
    return {"history.us": Table("history.us", None, "u", history)}


def test_a_range_by_hand():
    """Four industries, two years: the market median sits between the middle
    two; the line through (gap, move) and its misses give the range."""
    rows = {"a": {2020: 5.0, 2021: 6.0}, "b": {2020: 8.0, 2021: 8.4}, "c": {2020: 10.0, 2021: 9.0},
            "d": {2020: 20.0, 2021: 16.0}}
    t = tiny(rows)
    g = mp.group_series(t, "us")
    assert mp.market_median(g, 2020) == 9.0
    points = sorted(mp.moves(g, 1))
    assert points == pytest.approx(sorted((math.log(s[2020] / 9.0), math.log(s[2021] / s[2020]))
                                          for s in rows.values()))
    with mock.patch.object(mp, "MIN_PAIRS", 4):
        f = mp.fit(points)
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        mx, my = sum(xs) / 4, sum(ys) / 4
        b = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x in xs)
        assert f.b == pytest.approx(b) and f.b < 0                  # the dear industries cheapened
        assert f.a == pytest.approx(my - b * mx)
        misses = sorted(y - f.a - f.b * x for x, y in points)
        # Four misses: the 10th percentile is 0.3 of the way from the first to
        # the second, the 50th halfway between the middle two
        low = misses[0] + (misses[1] - misses[0]) * 0.3
        middle = (misses[1] + misses[2]) / 2
        centre = f.a + f.b * math.log(1.2)
        found = mp.band(12.0, math.log(1.2), f)
        assert found["low"] == pytest.approx(12.0 * math.exp(centre + low))
        assert found["median"] == pytest.approx(12.0 * math.exp(centre + middle))
        assert found["low"] < found["median"] < found["high"] and found["pairs"] == 4
    # Under MIN_PAIRS moves there is no range
    assert mp.fit(points) is None and mp.band(12.0, 0.0, None) is None


def test_years_with_too_few_companies_or_no_usable_multiple_are_not_read():
    t = tiny({"a": {2020: 5.0, 2021: 6.0, 2022: -3.0, 2023: 150.0}})
    t["history.us"].rows["a"]["years"]["2021"]["firms"] = 19
    assert mp.series(t, "us", "a") == {2020: 5.0}


def test_a_german_machinery_deal_by_hand():
    """Developed Europe's 210 machinery companies at 14.98x in 2025, against a
    market median of 12.03x (methodology §13's worked figures)."""
    found = predict("DE")
    assert (found["status"], found["group"], found["firms"], found["latest_year"], found["latest"]) == (
        "ok", "europe", 210, 2025, 14.98)
    g = mp.group_series(TABLES, "europe")
    assert mp.market_median(g, 2025) == pytest.approx(12.0276, abs=1e-4)
    line = mp.fit(mp.moves(g, 1))
    gap = math.log(14.98 / mp.market_median(g, 2025))
    misses = list(line.residuals)
    expected = [round(14.98 * math.exp(line.a + line.b * gap + mp.quantile(misses, q)), 2) for q in mp.QUANTILES]
    entry = found["entry"]
    assert (entry["horizon"], entry["year"]) == (1, 2026)
    assert [entry["range"][k] for k in ("low", "median", "high")] == expected == [10.92, 15.43, 20.5]
    assert entry["range"]["pairs"] == 801
    exit_ = found["exit"]
    assert (exit_["horizon"], exit_["year"]) == (6, 2031)
    assert [exit_["range"][k] for k in ("low", "median", "high")] == [10.2, 16.12, 24.17]


def test_expensive_industries_cheapen_and_cheap_ones_catch_up_in_every_region():
    for group in card_eval.GROUP_REGION:
        line = mp.fit(mp.moves(mp.group_series(TABLES, group), 1))
        assert line.b < 0, group


def test_a_longer_hold_widens_the_exit_range():
    widths = [(r := predict("US", hold=h)["exit"]["range"])["high"] - r["low"] for h in (3, 5, 7)]
    assert widths == sorted(widths) and widths[0] < widths[-1]


def test_an_older_stored_edition_moves_the_horizons_on():
    late = mp.predict("US", "machinery", 5, TABLES, date(2028, 3, 1))
    assert (late["entry"]["horizon"], late["exit"]["horizon"]) == (3, 8)


# ---------------------------------------------------------------------------
# Each region has its own peers; a thin one says "not enough data"
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("country, group, region", [
    ("US", "us", "us"), ("JP", "japan", "other_developed"), ("CN", "china", "emerging"),
    ("IN", "india", "emerging"), ("DE", "europe", "europe"), ("AU", "aus_nz_canada", "other_developed"),
    ("BR", "emerging", "emerging"),
])
def test_each_country_reads_its_own_regions_companies(country, group, region):
    found = predict(country)
    assert (found["status"], found["group"], found["region"]) == ("ok", group, region)
    assert found["latest"] == round(TABLES[f"multiples.{group}"].rows["machinery"]["ev_ebitda"], 2)
    assert [r["group"] for r in found["regions"] if r["used"]] == [group]


def test_the_regions_give_different_ranges():
    found = {c: predict(c)["entry"]["range"]["median"] for c in ("US", "JP", "CN", "IN", "DE", "AU", "BR")}
    assert len(set(found.values())) == 7


def test_a_thin_region_says_not_enough_data_and_never_borrows_the_global_group():
    """Australia, New Zealand and Canada have too few shipbuilders; the
    global group has plenty but is never used."""
    rows = {r["group"]: r for r in predict("CA", "shipbuilding_marine")["regions"]}
    assert rows["aus_nz_canada"]["firms"] < mp.MIN_FIRMS and rows["global"]["enough"]
    found = predict("CA", "shipbuilding_marine")
    assert (found["status"], found["reason"], found["entry"], found["exit"]) == ("not_enough_data", "no_peers", None, None)
    assert found["skipped"] == [{"area": "aus_nz_canada", "reason": "thin", "sample": rows["aus_nz_canada"]["firms"]}]


def test_without_a_country_or_stored_averages_there_is_no_range():
    assert predict("")["reason"] == "no_country"
    empty = mp.predict("DE", "machinery", 5, {}, TODAY)
    assert (empty["status"], empty["reason"], empty["regions"]) == ("not_enough_data", "no_peers", [])


def test_a_retired_country_code_reads_its_current_country():
    assert predict("UK") == {**predict("GB"), "country": "GB"}


def test_comparables_are_the_regions_industries_in_the_same_sector():
    found = predict("DE")
    sector = found["sector"]
    assert {s["industry"] for s in sector} >= {"machinery", "electrical_equipment", "building_materials"}
    assert [s["industry"] for s in sector if s["this"]] == ["machinery"]
    assert [s["multiple"] for s in sector] == sorted(s["multiple"] for s in sector)
    assert all(s["firms"] >= mp.MIN_FIRMS for s in sector)
    assert predict("DE", "")["sector"] == []                  # the whole market has no sector


# ---------------------------------------------------------------------------
# The card: most multiples inside the range in every region; shown only where it beats
# ---------------------------------------------------------------------------
def committed_card() -> dict:
    return json.loads(paths("multiples")[0].read_text(encoding="utf-8"))


@pytest.mark.parametrize("set_name", ["entry", "exit"])
def test_the_card_says_most_held_out_multiples_fall_inside_the_range_in_every_region(set_name):
    s = committed_card()["evaluation"]["sets"][set_name]
    for region, group in s["by_region"].items():
        assert group["cases"] >= 100, region
        assert group["model"]["inside"] > 0.5, region
        assert group["verdict"] == "beats_baseline", region


def test_the_walk_forward_never_reads_a_later_year():
    """Changing 2025's multiples changes no prediction of an earlier year."""
    def by_case(t):
        return {r[0].id: r[1] for r in card_eval.scored((1,), t) if r[0].year < 2025}
    later = {k: Table(v.name, v.published, v.url, json.loads(json.dumps(v.rows))) for k, v in TABLES.items()}
    for row in later["multiples.us"].rows.values():
        if "ev_ebitda" in row:
            row["ev_ebitda"] *= 3
    assert by_case(later) == by_case(TABLES)


def test_a_card_that_does_not_beat_the_baseline_hides_that_range():
    card = committed_card()
    card["evaluation"]["sets"]["exit"]["by_region"]["us"]["verdict"] = "does_not_beat_baseline"
    found = predict("US", card=card)
    assert found["entry"]["shown"] is True and found["entry"]["range"] is not None
    assert found["exit"]["shown"] is False and found["exit"]["range"] is None
    assert found["exit"]["card"]["verdict"] == "does_not_beat_baseline"
    assert found["latest"] is not None                         # the published figure still shows


def test_the_card_is_evaluated_on_the_recorded_multiples():
    """multiple_history.json against the fixtures recorded from the same
    archive and edition, for the industries those keep."""
    from companies import http
    from companies.record import replay
    from tests import ml_multiple_history
    from tests.e2e_benchmarks import FIXTURES, HISTORY
    replays = [replay(FIXTURES), replay(HISTORY)]

    def answer(req):
        for r in replays:
            resp = r(req)
            if resp.status != 404:
                return resp
        return resp
    recorded_on = date.fromisoformat(json.loads((HISTORY / "index.json").read_text(encoding="utf-8"))["_recorded_on"])
    http.use_transport(answer)
    try:
        fixture = ml_multiple_history.build(recorded_on)
    finally:
        http.use_transport(None)
    stored = json.loads(ml_multiple_history.OUT.read_text(encoding="utf-8"))
    assert fixture["read"] == stored["read"]
    for name, table in fixture["tables"].items():
        kept = table["rows"]
        assert kept, name
        assert stored["tables"][name]["published"] == table["published"]
        for industry, row in kept.items():
            if name.startswith("history."):
                assert stored["tables"][name]["rows"][industry]["years"] == row["years"], (name, industry)
            else:
                assert stored["tables"][name]["rows"][industry] == row, (name, industry)


# ---------------------------------------------------------------------------
# The library: deals like it when on; everything else the same when off
# ---------------------------------------------------------------------------
def test_with_the_library_off_the_ranges_are_the_same_and_no_deal_is_listed():
    on = predict("US", "retail_special_lines", library=references.repository_deals(), ev_usd_m=5_000.0)
    off = predict("US", "retail_special_lines", library=None, ev_usd_m=5_000.0)
    assert off["deals"] == {"enabled": False, "region": None, "sector": None, "size": None, "deals": []}
    assert on["deals"]["enabled"] is True
    assert {d["key"] for d in on["deals"]["deals"]} == {"toys-r-us-2005", "dollar-general-2007", "dominos-1998",
                                                        "gymboree-2010"}
    assert all(d["entry_multiple"] for d in on["deals"]["deals"])
    assert {k: v for k, v in on.items() if k != "deals"} == {k: v for k, v in off.items() if k != "deals"}


# ---------------------------------------------------------------------------
# The endpoint
# ---------------------------------------------------------------------------
def post(inputs: dict, library=None, usd_per=1.0):
    with mock.patch("api.routers.integrations._peer_data",
                    lambda currency, history=False: (TABLES, library, usd_per)):
        resp = client.post("/api/ml/multiples", json={"inputs": inputs})
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_the_endpoint_answers_the_deals_country_industry_and_hold():
    body = post({"country": "DE", "industry": "machinery", "hold": 7})
    assert (body["status"], body["group"], body["hold"], body["region"]) == ("ok", "europe", 7, "europe")
    assert body["exit"]["horizon"] == body["entry"]["horizon"] + 7
    assert body["entry"]["range"]["median"] > 0 and body["model"]["engine_version"]
    other = post({"country": "US", "industry": "machinery", "hold": 7})
    assert other["entry"]["range"] != body["entry"]["range"]


def test_the_endpoint_works_with_the_library_off_and_lists_deals_with_it_on():
    inputs = {"country": "US", "industry": "retail_special_lines", "ebitda": 500.0, "entry_mult": 10.0}
    off, on = post(inputs), post(inputs, library=references.repository_deals())
    assert off["status"] == "ok" and off["deals"]["enabled"] is False and off["deals"]["deals"] == []
    assert on["deals"]["enabled"] is True and on["deals"]["size"] == "1bn_10bn" and len(on["deals"]["deals"]) == 4
    assert off["entry"] == on["entry"] and off["exit"] == on["exit"]


def test_the_endpoint_reads_the_stored_history():
    seen = {}

    def peer_data(currency, history=False):
        seen["history"] = history
        return TABLES, None, 1.0
    with mock.patch("api.routers.integrations._peer_data", peer_data):
        client.post("/api/ml/multiples", json={"inputs": {"country": "US"}})
    assert seen == {"history": True}


def test_a_server_without_a_database_says_not_enough_data():
    with mock.patch("api.routers.integrations.is_configured", lambda: False):
        resp = client.post("/api/ml/multiples", json={"inputs": {"country": "DE", "industry": "machinery"}})
    body = resp.json()
    assert resp.status_code == 200 and body["status"] == "not_enough_data" and body["reason"] == "no_peers"


def test_the_endpoint_counts_as_a_model_run():
    assert "/api/ml/multiples" in RUN_PATHS


def test_the_browser_tests_recorded_answers_are_current():
    from tests import e2e_multiples
    assert e2e_multiples.OUT.read_text(encoding="utf-8") == e2e_multiples.text(), (
        "web/e2e/fixtures/multiples.json is stale: run python -m tests.e2e_multiples")
