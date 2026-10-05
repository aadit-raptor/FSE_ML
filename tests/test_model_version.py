"""Model version on every result (PLAN.md 3.1).

Every result carries the engine version, the commit, a fingerprint of the
Settings it ran with and the vintage of the published data; saved deals and
versions store that stamp with the deal's IRR and MOIC; reopening a deal whose
results have changed since it was saved says so, and why; exports show it.
The engine version is tied to the numbers: reference deals' results are
pinned per version, so a change that moves one fails until the version is
raised in ``core/model_version.py`` and ``MODEL_CHANGELOG.md``.
"""
import io
import json
import re
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from api.auth import AuthUser, require_user
from api.main import app
from core import model_version as mv
from core.config import DEFAULTS, resolve_config
from db import engine as db_engine

client = TestClient(app)
ROOT = Path(__file__).resolve().parents[1]
PINS = Path(__file__).with_name("model_version_pins.json")

PROFILE = {"country": "DE", "preferred_currency": "EUR", "locale": "de-DE",
           "time_zone": "Europe/Berlin"}
INPUTS = {"ebitda": 240.0, "entry_mult": 9.0, "exit_mult": 10.5, "hold": 6, "growth": 7.5,
          "debt_pct": 55.0, "currency": "EUR", "unit": "thousands"}
SETTINGS = {"tx_fee_pct": 3.0}
STAMP_KEYS = {"engine_version", "commit", "settings_fingerprint", "data_vintage",
              "data_fingerprint", "data_sets"}


def ok(resp, code=200):
    assert resp.status_code == code, resp.text
    return resp.json()


def backtest_body() -> dict:
    deal = ok(client.get("/api/backtesting/deals"))[0]
    hold = int(deal["entry"]["holding_period"])
    return {"entry": deal["entry"], "actual": {k: v[:hold] for k, v in deal["actual"].items()},
            "actual_exit": deal["actual_exit"], "n": 1000, "settings": SETTINGS}


def forecast_body() -> dict:
    defaults = ok(client.get("/api/forecasting/defaults"))
    grid = {k: [float(v)] * 3 for k, v in defaults["seeded_assumptions"].items()}
    return {"history": defaults["history"], "assumptions": grid}


PLAN_ACTUAL = {"plan": {**INPUTS, "unit": "millions"}, "n": 1000,
               "actuals": {"currency": "EUR", "unit": "millions",
                           "years": [{"ebitda": 250.0}, {"ebitda": 270.0}]}}

# Every endpoint that answers a model result, and a body for it (None: built
# in the test). Each one reads Settings except the forecast.
RESULTS = [
    ("/api/deal/run", {"inputs": INPUTS, "settings": SETTINGS}, True),
    ("/api/deal/sources-and-uses", {"settings": SETTINGS}, True),
    ("/api/montecarlo/run", {"mc": {"n": 1000}, "seed": 3, "settings": SETTINGS}, True),
    ("/api/montecarlo/scenarios", {"mc": {"n": 1000}, "seed": 3, "settings": SETTINGS}, True),
    ("/api/backtesting/run", None, True),
    ("/api/backtesting/plan-vs-actual", {**PLAN_ACTUAL, "settings": SETTINGS}, True),
    ("/api/forecasting/run", None, False),
]


# ---------------------------------------------------------------------------
# Every result carries the stamp
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("path, body, reads_settings", RESULTS)
def test_every_result_carries_the_model_stamp(monkeypatch, path, body, reads_settings):
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abc123def456")
    if body is None:
        body = backtest_body() if "backtesting" in path else forecast_body()
    answer = ok(client.post(path, json=body))
    model = answer["model"]
    assert set(model) == STAMP_KEYS
    assert model["engine_version"] == mv.ENGINE_VERSION
    assert model["commit"] == "abc123def456"
    assert model["data_vintage"] == max(model["data_sets"].values())
    assert model["data_sets"]["damodaran_ratings_2026"] == "2026-01"
    assert model["data_sets"]["sp_default_study_2024"] == "2025-03-27"
    expected = mv.settings_fingerprint(resolve_config(SETTINGS)) if reads_settings else None
    assert model["settings_fingerprint"] == expected


def test_health_names_the_engine_version():
    assert ok(client.get("/api/health"))["engine_version"] == mv.ENGINE_VERSION


def test_the_settings_fingerprint_follows_the_settings_a_run_used():
    def fingerprint(settings):
        return ok(client.post("/api/deal/run", json={"inputs": INPUTS, "settings": settings}))[
            "model"]["settings_fingerprint"]

    plain, again = fingerprint({}), fingerprint({})
    assert plain == again == mv.settings_fingerprint(DEFAULTS)
    assert fingerprint({"tx_fee_pct": 3.0}) != plain
    # Stating a default is the same Settings as leaving it out
    assert fingerprint({"tx_fee_pct": DEFAULTS["tx_fee_pct"]}) == plain
    assert len(plain) == mv.FINGERPRINT_DIGITS and re.fullmatch("[0-9a-f]+", plain)


def test_whole_numbers_and_their_floats_share_a_fingerprint():
    cfg = resolve_config({})
    whole = {k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in cfg.items()}
    assert mv.settings_fingerprint(whole) == mv.settings_fingerprint(cfg)


def test_the_data_fingerprint_changes_with_any_data_set(monkeypatch):
    before = mv.data_fingerprint()
    newer = {**mv.data_sets(), "damodaran_ratings_2026": "2027-01"}
    monkeypatch.setattr(mv, "data_sets", lambda: newer)
    assert mv.data_fingerprint() != before
    assert mv.data_vintage() == "2027-01"


def test_a_background_job_carries_the_same_stamp(job_queue):
    from jobs.runner import Runner

    body = {"mc": {"n": 1000}, "seed": 3, "settings": SETTINGS}
    direct = ok(client.post("/api/montecarlo/run", json=body))
    job = ok(client.post("/api/jobs", json={"kind": "montecarlo.run", "input": body}), 202)
    Runner(lambda: job_queue).run_next()
    done = ok(client.get(f"/api/jobs/{job['id']}"))
    assert done["result"]["model"] == direct["model"]


# ---------------------------------------------------------------------------
# The version is tied to the numbers
# ---------------------------------------------------------------------------
def test_the_changelog_describes_this_engine_version():
    changelog = (ROOT / "MODEL_CHANGELOG.md").read_text(encoding="utf-8")
    versions = re.findall(r"^## (\d+\.\d+\.\d+)\b", changelog, flags=re.M)
    assert versions, "MODEL_CHANGELOG.md has no '## x.y.z' entries"
    assert versions[0] == mv.ENGINE_VERSION, (
        f"MODEL_CHANGELOG.md's newest entry is {versions[0]}, the engine says {mv.ENGINE_VERSION}")
    assert len(versions) == len(set(versions))


# Reference deals, each exercising a part of the model: the defaults, written-
# out tranches with PIK and a revolver, tax rules, IFRS 16 leases, a deal in
# thousands with Settings changed
REFERENCE_DEALS = {
    "defaults": ({}, {}),
    "tranches": ({"tranches": [
        {"name": "TLB", "kind": "institutional_term_loan", "amount": 350.0, "floating": True,
         "reference_path": [4.0, 3.5], "floor": 1.0, "margin": 4.0, "amort_pct": 1.0, "sweep": True},
        {"name": "PIK", "kind": "pik_notes", "amount": 120.0, "fixed_rate": 12.0, "pik_share": 100.0,
         "maturity_years": 7},
        {"name": "RCF", "kind": "revolver", "amount": 80.0, "drawn_pct": 25.0, "fixed_rate": 6.0,
         "commitment_fee_pct": 0.5, "sweep": True, "allow_redraw": True}]}, {}),
    "tax_rules": ({"tax_interest_limit": "ebitda_share", "tax_interest_limit_pct": 30.0,
                   "tax_loss_carryforward": True, "tax_minimum_pct": 15.0}, {}),
    "leases": ({"accounting_standard": "ifrs", "lease_cost": 12.0, "lease_liability": 80.0}, {}),
    "thousands": ({**INPUTS}, SETTINGS),
}
PINNED = ("irr", "moic", "net_exit_equity", "entry_equity")


def reference_results() -> dict:
    out = {}
    for name, (inputs, settings) in REFERENCE_DEALS.items():
        run = ok(client.post("/api/deal/run", json={"inputs": inputs, "settings": settings}))
        out[name] = {k: run["returns"][k] for k in PINNED}
    # The simulation, on both its paths: the two-bucket one and the tranche one
    for name, deal in (("montecarlo", {}), ("montecarlo_tranches", REFERENCE_DEALS["tranches"][0])):
        mc = ok(client.post("/api/montecarlo/run", json={"mc": {"n": 2000}, "seed": 7, "deal": deal}))
        out[name] = {k: mc["summary"][k] for k in ("mean_irr", "median_irr", "p5_irr", "p95_irr")}
    return out


def test_results_are_those_recorded_for_this_engine_version():
    """A change that moves any of these numbers is a new engine version:
    raise ``ENGINE_VERSION``, add its entry to MODEL_CHANGELOG.md (what moved
    and why), and record its results here with
    ``python -m tests.test_model_version`` -- never edit an older version's."""
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    assert mv.ENGINE_VERSION in pins, (
        f"no results recorded for engine {mv.ENGINE_VERSION}: run python -m tests.test_model_version")
    recorded = pins[mv.ENGINE_VERSION]
    now = reference_results()
    assert set(now) == set(recorded)
    for deal, figures in recorded.items():
        for key, value in figures.items():
            assert now[deal][key] == pytest.approx(value, rel=1e-9, abs=1e-12), (
                f"{deal} {key} moved from {value} to {now[deal][key]} under engine "
                f"{mv.ENGINE_VERSION}: a model change needs a new engine version")


# ---------------------------------------------------------------------------
# Exports show the version and the data vintage
# ---------------------------------------------------------------------------
def about_sheet(resp) -> dict:
    assert resp.status_code == 200, resp.text
    about = pd.read_excel(io.BytesIO(resp.content), sheet_name="About")
    return dict(zip(about["Item"], about["Value"]))


def test_a_workbook_shows_the_stamp_of_the_result_it_holds():
    stamp = ok(client.post("/api/deal/run", json={"inputs": INPUTS, "settings": SETTINGS}))["model"]
    sent = {**stamp, "engine_version": "0.9.0", "commit": "feedface0000"}  # an older result
    about = about_sheet(client.post("/api/export/workbook", json={
        "filename": "deal.xlsx", "model": sent,
        "sheets": [{"name": "P&L", "columns": ["", "Y1"], "rows": [["Revenue", 1.0]]}]}))
    assert about["Model engine version"] == "0.9.0"
    assert about["Model commit"] == "feedface0000"
    assert about["Settings fingerprint"] == stamp["settings_fingerprint"]
    assert about["Data vintage"] == stamp["data_vintage"]
    assert about["Data: damodaran_ratings_2026"] == "2026-01"


@pytest.mark.parametrize("bad", [
    {"data_sets": {"x": "=1+1"}}, {"data_sets": {"=cmd": "2026-01"}},
    {"engine_version": "=1+1"}, {"commit": "=A1"}, {"data_vintage": "=A1"},
])
def test_a_workbook_refuses_a_stamp_that_could_become_a_formula(bad):
    stamp = ok(client.post("/api/deal/run", json={"inputs": INPUTS}))["model"]
    resp = client.post("/api/export/workbook", json={
        "model": {**stamp, **bad}, "sheets": [{"name": "a", "columns": ["x"], "rows": [[1]]}]})
    assert resp.status_code == 422


def test_a_workbook_without_a_stamp_shows_this_apis_own():
    about = about_sheet(client.post("/api/export/workbook", json={
        "sheets": [{"name": "P&L", "columns": ["", "Y1"], "rows": [["Revenue", 1.0]]}]}))
    assert about["Model engine version"] == mv.ENGINE_VERSION
    assert about["Data vintage"] == mv.data_vintage()


def test_the_simulation_sample_shows_its_stamp():
    about = about_sheet(client.post("/api/export/montecarlo-sample",
                                    json={"mc": {"n": 1000}, "seed": 3, "settings": SETTINGS}))
    assert about["Model engine version"] == mv.ENGINE_VERSION
    assert about["Settings fingerprint"] == mv.settings_fingerprint(resolve_config(SETTINGS))
    assert about["Data vintage"] == mv.data_vintage()


# ---------------------------------------------------------------------------
# Saved deals store the stamp; reopening says when results changed
# ---------------------------------------------------------------------------
@pytest.fixture
def account(fresh_db):  # noqa: ARG001
    app.dependency_overrides[require_user] = lambda: AuthUser(subject="user_vera")
    ok(client.post("/api/account", json=PROFILE))
    yield


def deal_irr(inputs=INPUTS, settings=SETTINGS):
    run = ok(client.post("/api/deal/run", json={"inputs": inputs, "settings": settings}))
    return run["returns"]["irr"], run["returns"]["moic"]


def stored(table: str, deal_id: str):
    where = "id" if table == "deals" else "deal_id"
    with db_engine.connect() as conn:
        return conn.execute(text(f"SELECT model FROM {table} WHERE {where} = :id ORDER BY 1"),
                            {"id": deal_id}).scalars().all()


def age(deal_id: str, **changes):
    """Pretend the working copy was saved by an older model."""
    with db_engine.transaction() as conn:
        model = conn.execute(text("SELECT model FROM deals WHERE id = :id"), {"id": deal_id}).scalar()
        conn.execute(text("UPDATE deals SET model = CAST(:m AS jsonb) WHERE id = :id"),
                     {"id": deal_id, "m": json.dumps({**model, **changes}) if changes else None})


def test_a_saved_deal_and_its_versions_store_the_stamp_and_results(account):  # noqa: ARG001
    deal = ok(client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS, "settings": SETTINGS}), 201)
    irr, moic = deal_irr()
    (model,) = stored("deals", deal["id"])
    assert model["engine_version"] == mv.ENGINE_VERSION
    assert model["irr"] == pytest.approx(irr, rel=1e-12) and model["moic"] == pytest.approx(moic, rel=1e-12)
    assert model["settings_fingerprint"] == mv.settings_fingerprint(resolve_config(SETTINGS))
    assert model["data_fingerprint"] == mv.data_fingerprint()
    assert "data_sets" not in model  # the fingerprint stands for them, keeping versions small
    # A version keeps the stamp without the content fingerprint, which only
    # the working copy needs (its content is what an older release may change)
    plain = {k: v for k, v in model.items() if k != "content"}
    assert model["content"] and stored("deal_versions", deal["id"]) == [plain]
    assert deal["model"] == model
    version = ok(client.get(f"/api/deals/{deal['id']}/versions/1"))
    assert version["model"] == {**plain, "content": None}

    # An edit restamps the working copy with the new results
    edited = {**INPUTS, "exit_mult": 12.0}
    ok(client.put(f"/api/deals/{deal['id']}/draft", json={"inputs": edited, "settings": SETTINGS}))
    (after,) = stored("deals", deal["id"])
    assert after["irr"] == pytest.approx(deal_irr(edited)[0], rel=1e-12) and after["irr"] != model["irr"]


def test_reopening_an_unchanged_deal_shows_no_notice(account):  # noqa: ARG001
    deal = ok(client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS, "settings": SETTINGS}), 201)
    check = ok(client.get(f"/api/deals/{deal['id']}"))["model_check"]
    assert check["status"] == "unchanged" and check["causes"] == []


def test_reopening_a_deal_saved_by_an_older_model_says_its_results_changed(account):  # noqa: ARG001
    deal = ok(client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS, "settings": SETTINGS}), 201)
    irr, moic = deal_irr()
    age(deal["id"], engine_version="0.9.0", irr=irr - 0.012, moic=moic - 0.1)
    check = ok(client.get(f"/api/deals/{deal['id']}"))["model_check"]
    assert check["status"] == "changed"
    assert check["causes"] == ["engine_version"]
    assert check["saved"]["engine_version"] == "0.9.0" and check["now"]["engine_version"] == mv.ENGINE_VERSION
    assert check["saved"]["irr"] == pytest.approx(irr - 0.012)
    assert check["now"]["irr"] == pytest.approx(irr, rel=1e-12)
    assert check["now"]["moic"] == pytest.approx(moic, rel=1e-12)


def test_a_new_version_that_leaves_this_deal_alone_is_not_a_change(account):  # noqa: ARG001
    deal = ok(client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS, "settings": SETTINGS}), 201)
    age(deal["id"], engine_version="0.9.0")
    check = ok(client.get(f"/api/deals/{deal['id']}"))["model_check"]
    assert check["status"] == "unchanged" and check["causes"] == ["engine_version"]


def test_a_changed_default_is_named_as_the_cause(account, monkeypatch):  # noqa: ARG001
    """An untouched deal stores no overrides, so before 3.1 a changed default
    moved its results silently; now the fingerprint names it."""
    deal = ok(client.post("/api/deals", json={"name": "Plain", "inputs": INPUTS, "settings": {}}), 201)
    monkeypatch.setitem(DEFAULTS, "tx_fee_pct", DEFAULTS["tx_fee_pct"] + 1.5)
    check = ok(client.get(f"/api/deals/{deal['id']}"))["model_check"]
    assert check["status"] == "changed" and check["causes"] == ["settings"]
    assert check["now"]["irr"] < check["saved"]["irr"]


def test_a_deal_saved_before_stamps_were_recorded_is_unknown(account):  # noqa: ARG001
    deal = ok(client.post("/api/deals", json={"name": "Old", "inputs": INPUTS, "settings": SETTINGS}), 201)
    age(deal["id"])  # model = NULL, as for every deal saved before 3.1
    check = ok(client.get(f"/api/deals/{deal['id']}"))["model_check"]
    assert check["status"] == "unknown" and check["saved"] is None
    assert check["now"]["irr"] == pytest.approx(deal_irr()[0], rel=1e-12)


def test_a_stamp_left_behind_by_an_older_release_is_unknown(account):  # noqa: ARG001
    """An API from before 3.1 (a deploy rolling out, a rollback) overwrites
    the content and leaves the stamp: it then describes other content."""
    deal = ok(client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS, "settings": SETTINGS}), 201)
    irr, _ = deal_irr()
    # As if an older release had saved an edit: content changed, stamp kept
    with db_engine.transaction() as conn:
        conn.execute(text("UPDATE deals SET inputs = jsonb_set(inputs, '{exit_mult}', '12.0') "
                          "WHERE id = :id"), {"id": deal["id"]})
    check = ok(client.get(f"/api/deals/{deal['id']}"))["model_check"]
    assert check["status"] == "unknown"
    assert check["saved"]["irr"] == pytest.approx(irr, rel=1e-12) and check["now"]["irr"] != check["saved"]["irr"]


def test_restoring_a_version_brings_back_its_stamp(account):  # noqa: ARG001
    deal = ok(client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS, "settings": SETTINGS}), 201)
    (first,) = stored("deals", deal["id"])
    ok(client.put(f"/api/deals/{deal['id']}/draft",
                  json={"inputs": {**INPUTS, "exit_mult": 12.0}, "settings": SETTINGS}))
    restored = ok(client.post(f"/api/deals/{deal['id']}/versions/1/restore"))
    assert restored["model"] == first
    assert restored["model_check"]["status"] == "unchanged"
    # The edits kept before restoring carry the stamp they were saved with
    versions = {v["number"]: v for v in (ok(client.get(f"/api/deals/{deal['id']}/versions/{n}"))
                                         for n in (2, 3))}
    assert versions[2]["model"]["irr"] != first["irr"] and versions[3]["model"]["irr"] == first["irr"]


def test_a_deal_the_sponsor_could_not_fund_stores_no_results(account):  # noqa: ARG001
    unfundable = {**INPUTS, "tranches": [{"name": "Too much", "kind": "unitranche", "amount": 1e9}]}
    deal = ok(client.post("/api/deals", json={"name": "Over", "inputs": unfundable, "settings": {}}), 201)
    assert deal["model"]["irr"] is None and deal["model"]["moic"] is None


def test_a_version_with_its_stamp_is_still_a_few_hundred_bytes(account, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv("RENDER_GIT_COMMIT", "0123456789abcdef0123456789abcdef01234567")  # as deployed
    deal = ok(client.post("/api/deals", json={"name": "Alpine", "inputs": INPUTS, "settings": SETTINGS}), 201)
    with db_engine.connect() as conn:
        row_bytes = conn.execute(text(
            "SELECT pg_column_size(v.*) FROM deal_versions v WHERE deal_id = :id"), {"id": deal["id"]}
        ).scalar()
    assert row_bytes < 900, row_bytes


if __name__ == "__main__":
    # Record the reference results for a new engine version (see the test above)
    from api.auth import AuthUser as _User

    app.dependency_overrides[require_user] = lambda: _User(subject="dev:pins")
    pins = json.loads(PINS.read_text(encoding="utf-8")) if PINS.exists() else {}
    if mv.ENGINE_VERSION in pins:
        raise SystemExit(f"engine {mv.ENGINE_VERSION} is already recorded: raise the version first")
    pins[mv.ENGINE_VERSION] = reference_results()
    PINS.write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    print(f"recorded engine {mv.ENGINE_VERSION} in {PINS.name}")
