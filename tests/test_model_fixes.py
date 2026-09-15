"""Deliberate fixes to open model findings (see CLAUDE.md, "Model findings").

Each test pins the corrected behaviour with an independent check, so the
Streamlit golden snapshot stays the record of the old behaviour and every
departure from it is explained here.
"""
import threading

import numpy as np
import pytest

from simulation.vectorized_simulation import SimulationParams, run_vectorized_simulation_full


# ---------------------------------------------------------------------------
# Finding 9: a fixed seed must be reproducible under concurrent simulations
# ---------------------------------------------------------------------------
def _irr(seed, n=20000):
    return run_vectorized_simulation_full(SimulationParams(n=n), seed=seed).df["IRR"].values


def test_seeded_simulations_are_reproducible_when_run_concurrently():
    seeds = [11, 22, 33, 44, 55, 66, 77, 88]
    expected = {s: _irr(s) for s in seeds}
    results, errors = {}, []
    barrier = threading.Barrier(len(seeds) * 3)

    def worker(seed, k):
        try:
            barrier.wait()
            results[(seed, k)] = _irr(seed)
        except Exception as e:  # pragma: no cover - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(s, k)) for s in seeds for k in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for (seed, _), irr in results.items():
        np.testing.assert_array_equal(irr, expected[seed])


def test_local_generator_keeps_the_legacy_seeded_stream():
    # Golden Monte Carlo numbers depend on np.random.seed(seed) followed by
    # standard_normal; the local RandomState must reproduce that stream.
    np.random.seed(42)
    legacy = np.random.standard_normal((5, 1000))
    local = np.random.RandomState(42).standard_normal((5, 1000))
    np.testing.assert_array_equal(legacy, local)


# ---------------------------------------------------------------------------
# Finding 4: forecast target labels were inverted
# ---------------------------------------------------------------------------
def test_forecast_targets_above_plan_are_bull_and_below_are_bear():
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    d = client.get("/api/forecasting/defaults").json()
    body = client.post("/api/forecasting/run", json={
        "history": d["history"],
        "assumptions": {k: [v] * d["n_fwd"] for k, v in d["seeded_assumptions"].items()},
        "simulate": True, "n_sim": 5000,
    }).json()
    plan = body["simulation"]["ebitda_final"]["deterministic"]
    targets = body["simulation"]["target_probabilities"]
    assert [t["scenario"] for t in targets] == ["Bear", "Bear", "Base", "Bull", "Bull"]
    for t in targets:
        if t["scenario"] == "Bull":
            assert t["target"] > plan and t["probability"] < 0.5
        if t["scenario"] == "Bear":
            assert t["target"] < plan and t["probability"] > 0.5


# ---------------------------------------------------------------------------
# Finding 2: settings the model ignored
# ---------------------------------------------------------------------------
def _deal(settings=None, **inputs):
    from fastapi.testclient import TestClient

    from api.main import app

    r = TestClient(app).post("/api/deal/run", json={"inputs": inputs, "settings": settings or {}})
    assert r.status_code == 200, r.text
    return r.json()


def test_senior_amortisation_setting_drives_mandatory_repayments():
    base = _deal()
    senior0 = base["tranches"]["Senior Term Loan"][0]["beginning_balance"]
    assert base["tranches"]["Senior Term Loan"][0]["mandatory_repayment"] == pytest.approx(senior0 * 0.05, abs=0.01)
    faster = _deal({"def_senior_amort": 12.0})
    assert faster["tranches"]["Senior Term Loan"][0]["mandatory_repayment"] == pytest.approx(senior0 * 0.12, abs=0.01)
    assert faster["returns"]["irr"] != base["returns"]["irr"]


def test_clip_irr_setting_reaches_the_simulation():
    from core.config import resolve_config
    from core.deal import DealInputs
    from core.montecarlo import MCInputs, build_sim_params

    on = build_sim_params(MCInputs(), DealInputs(), resolve_config({}))
    off = build_sim_params(MCInputs(), DealInputs(), resolve_config({"mc_clip_irr": False}))
    assert on.clip_irr is True and off.clip_irr is False
    # An extreme upside deal: clipping caps IRR at 500%
    extreme = dict(n=4000, holding_period=1, debt_pct=0.95, exit_mean=40.0, exit_std=5.0, growth_mean=0.4)
    clipped = run_vectorized_simulation_full(SimulationParams(**extreme, clip_irr=True), seed=1).df["IRR"]
    raw = run_vectorized_simulation_full(SimulationParams(**extreme, clip_irr=False), seed=1).df["IRR"]
    assert raw.max() > 5.0 and clipped.max() <= 5.0


# ---------------------------------------------------------------------------
# Finding 7: every sensitivity cell must be a real run for that hold and exit
# Finding 2 (part 2): the grid's ranges come from settings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("inputs", [{}, {"exit_mult": 12.0, "growth": 8.0, "mincash": 15.0}])
def test_every_sensitivity_cell_equals_a_full_run(inputs):
    body = _deal(**inputs)
    s = body["exit_sensitivity"]
    assert s["exit_multiples"] == [6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]   # sens_em_* defaults
    assert s["holding_periods"] == [3, 4, 5, 6, 7]                          # sens_hp_* defaults
    for i, em in enumerate(s["exit_multiples"]):
        for j, hp in enumerate(s["holding_periods"]):
            full = _deal(**{**inputs, "exit_mult": em, "hold": hp})
            assert s["table"][i][j] == pytest.approx(full["returns"]["irr"], abs=6e-5), (em, hp)


def test_sensitivity_ranges_follow_settings():
    s = _deal({"sens_em_min": 8.0, "sens_em_max": 14.0, "sens_em_steps": 4,
               "sens_hp_min": 4, "sens_hp_max": 6})["exit_sensitivity"]
    assert s["exit_multiples"] == [8.0, 10.0, 12.0, 14.0]
    assert s["holding_periods"] == [4, 5, 6]
    assert len(s["table"]) == 4 and all(len(r) == 3 for r in s["table"])


# ---------------------------------------------------------------------------
# Finding 1: sponsor equity funds the minimum cash
# ---------------------------------------------------------------------------
def test_minimum_cash_is_funded_by_equity_and_the_bridge_closes():
    base, with_cash = _deal(), _deal(mincash=20.0)
    assert with_cash["returns"]["entry_equity"] == pytest.approx(base["returns"]["entry_equity"] + 20.0)
    assert with_cash["equity_bridge"]["residual"] == pytest.approx(0.0, abs=0.011)
    # More equity for the same business can't raise the return
    assert with_cash["returns"]["irr"] < base["returns"]["irr"]


def test_sources_and_uses_include_minimum_cash_and_match_the_engine():
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    su = client.post("/api/deal/sources-and-uses", json={"mincash": 20.0, "senior_x": 4.2, "mezz_x": 1.8}).json()
    assert su["cash_to_balance_sheet"] == 20.0 and su["balanced"]
    run = _deal(mincash=20.0)
    assert su["sponsor_equity"] == pytest.approx(run["returns"]["entry_equity"], abs=1e-6)


# ---------------------------------------------------------------------------
# Finding 8: the interest circularity converges
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("inputs", [{}, {"exit_mult": 12.0}, {"debt_pct": 90.0, "base_rate": 12.0}, {"hold": 7, "mincash": 15.0}])
def test_interest_converges_and_the_income_statement_matches_the_debt_schedule(inputs):
    body = _deal(**inputs)
    assert body["interest_converged"]
    pl = body["operating_model"]["interest_expense"]
    schedule = body["debt_schedule"]["total_interest_expense"]
    # The engine rounds each year's interest to cents
    assert max(abs(a - b) for a, b in zip(pl, schedule)) <= 0.011


# ---------------------------------------------------------------------------
# Finding 3: the Monte Carlo heatmap is the deal model, not a fee-free shortcut
# ---------------------------------------------------------------------------
def test_heatmap_cells_equal_deal_model_runs_with_fees():
    from fastapi.testclient import TestClient

    from api.main import app

    client = TestClient(app)
    body = client.post("/api/montecarlo/run", json={"mc": {"n": 2000}, "seed": 1}).json()
    h = body["heatmap"]
    p = body["params"]
    for i, j in [(0, 0), (3, 4), (6, 7)]:
        g, em = h["growth"][j], h["exit_multiple"][i]
        deal = _deal(ebitda=p["entry_ebitda"], entry_mult=p["entry_multiple"],
                     exit_mult=em, hold=p["holding_period"], growth=g * 100,
                     gross_margin=p["gross_margin_mean"] * 100, base_rate=p["interest_mean"] * 100)
        assert h["irr"][i][j] == pytest.approx(deal["returns"]["irr"], abs=1e-6), (g, em)
    # Fees are charged: doubling transaction fees lowers every cell
    dearer = client.post("/api/montecarlo/run", json={"mc": {"n": 2000}, "seed": 1,
                                                      "settings": {"tx_fee_pct": 4.6}}).json()["heatmap"]["irr"]
    assert all(a < b for ra, rb in zip(dearer, h["irr"]) for a, b in zip(ra, rb))
