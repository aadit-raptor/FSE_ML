"""core/ reproduces the Streamlit app's outputs.

tests/golden/golden.json was captured from the Streamlit app before the model
logic moved into core/ (see tests/golden/generate_golden.py). These tests feed
the recorded inputs straight into core/ and require the same results, to within
floating-point noise (see golden_compare.py).
"""
import dataclasses
import json
import os

import numpy as np
import pytest

from tests.golden_compare import assert_close, assert_min_cash_funded

from core.backtesting import PRELOADED_DEALS, backtest_summary, predicted_ebitda, run_prediction_sim
from core.config import DEFAULTS, resolve_config
from core.deal import DealInputs, run_deal
from core.forecasting import (
    ASSUMPTION_KEYS, assumptions_from_grid, default_history, ltm_from_history,
    run_3_statement_model, run_forecast_simulation, seed_assumptions,
)
from core.montecarlo import MCInputs, apply_scenario, build_sim_params

GOLDEN = json.load(open(os.path.join(os.path.dirname(__file__), "golden", "golden.json"),
                        encoding="utf-8"))


def plain(x):
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def cfg_from(recorded):
    return resolve_config({k: v for k, v in recorded.items() if k in DEFAULTS})


def deal_from(inputs):
    fields = {f.name for f in dataclasses.fields(DealInputs)}
    return DealInputs(**{k[2:]: v for k, v in inputs.items() if k[2:] in fields})


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(GOLDEN["deal"]))
def test_deal_matches_streamlit(case):
    g = GOLDEN["deal"][case]
    d = deal_from(g["inputs"])
    r = run_deal(d, cfg_from(g["cfg"]))
    # exit_sensitivity is left out on purpose: the Streamlit grid reused one
    # hold's exit values for every column (finding 7). The corrected grid is
    # checked cell by cell in test_model_fixes.py.
    parts = ["operating_model", "cash_flow", "debt_schedule", "interest_converged"]
    if d.mincash:
        # Sponsor equity now funds the minimum cash (finding 1)
        assert_min_cash_funded(plain(r.returns), plain(r.equity_bridge), g, d.mincash)
    else:
        parts += ["returns", "equity_bridge"]
    for part in parts:
        assert_close(plain(getattr(r, part)), g[part])


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(GOLDEN["montecarlo"]))
def test_simulation_params_match_streamlit(case):
    g = GOLDEN["montecarlo"][case]
    i = g["inputs"]
    mc = MCInputs(n=i["mc_n"], ebitda=i["mc_fi_eb"], entry_mult=i["mc_fi_em"],
                  hold=i["mc_fi_hold"], hurdle=i["mc_hurdle"],
                  growth_mean=i["mc_growth_mean"], growth_std=i["mc_growth_std"],
                  exit_mean=i["mc_exit_mean"], exit_std=i["mc_exit_std"],
                  rate_mean=i["mc_rate_mean"], rate_std=i["mc_rate_std"],
                  gm_mean=i["mc_gm_mean"], gm_std=i["mc_gm_std"])
    cfg = cfg_from(g["cfg"])
    params = build_sim_params(mc, deal_from(i), cfg)
    if g["preset"]:
        params = apply_scenario(g["preset"].lower(), params, cfg)
    assert_close(plain(params), g["sim_params"])


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", sorted(GOLDEN["forecasting"]))
def test_forecast_matches_streamlit(case):
    g = GOLDEN["forecasting"][case]
    state = g["state"]
    history = default_history(3)
    for key, value in state.items():
        if key.startswith("hist_"):
            field, year = key[len("hist_"):].rsplit("_", 1)
            history[field][int(year)] = value
    ltm = ltm_from_history(history)
    assert_close(plain(ltm), g["ltm"])

    n_fwd = len(g["fwd"])
    # The page seeds its assumption grid when it first renders -- from the
    # default historicals -- and edits to the historicals made afterwards do
    # not re-seed it (only an EDGAR fetch does). The capture typed its custom
    # historicals in after that first render.
    seeds = seed_assumptions(ltm_from_history(default_history(3)))
    grid = {k: [float(seeds[k])] * n_fwd for k in ASSUMPTION_KEYS}
    for key, value in state.items():
        if key.startswith("fwd_"):
            field, year = key[len("fwd_"):].rsplit("_", 1)
            grid[field][int(year)] = value
    assumptions = assumptions_from_grid(grid, n_fwd)
    assert_close(plain(assumptions), g["assumptions"])

    assert_close(plain(run_3_statement_model(ltm, assumptions)), g["fwd"])

    paths = run_forecast_simulation(ltm, assumptions, n=30000)
    for key in ("revenue", "ebitda"):
        a = paths[key]
        got = {q: np.percentile(a, int(q[1:]), axis=0).tolist() for q in ("p5", "p50", "p95")}
        got["mean"] = a.mean(axis=0).tolist()
        assert_close(got, g["sim"][key])
    assert_close(float(paths["g_draws"].mean()), g["sim"]["g_draws_mean"])
    assert_close(float(paths["m_draws"].mean()), g["sim"]["m_draws_mean"])


# ---------------------------------------------------------------------------
@pytest.mark.parametrize("deal", sorted(GOLDEN["backtesting"]["helpers"]))
def test_backtest_prediction_matches_streamlit(deal):
    g = GOLDEN["backtesting"]["helpers"][deal]
    cfg = resolve_config()
    assert_close(plain(predicted_ebitda(g["entry"])), g["predicted_ebitda"])
    irr = run_prediction_sim(g["entry"], cfg)
    got = {"mean": float(irr.mean()), "p5": float(np.percentile(irr, 5)),
           "p50": float(np.percentile(irr, 50)), "p95": float(np.percentile(irr, 95)),
           "n": int(irr.size)}
    assert_close(got, g["prediction_irr"])


def test_backtest_summary_matches_rendered_page():
    rendered = GOLDEN["backtesting"]["rendered_default"]
    d = PRELOADED_DEALS[rendered["deal"]]
    hold = d["entry"]["holding_period"]
    actual = {k: [float(v) for v in d["actual"][k][:hold]] for k in d["actual"]}
    bt = backtest_summary(d["entry"], actual, d["actual_exit"], resolve_config())
    m = rendered["metrics"]
    assert f"{bt['predicted_irr_mean']:.1f}%" == m["Predicted IRR (mean)"]
    assert f"{bt['predicted_moic']:.2f}x" == m["Predicted MOIC (base)"]
    assert f"${bt['predicted_ebitda'][-1]:,.0f}M" == m["Predicted exit EBITDA"]
    summary = {row["Metric"]: row for row in rendered["tables"][1]}
    assert f"${bt['predicted_equity_entry']:,.0f}M" == summary["Entry equity ($M)"]["Predicted"]
    assert f"${bt['predicted_exit_equity']:,.0f}M" == summary["Exit equity ($M)"]["Predicted"]


# ---------------------------------------------------------------------------
def test_default_history_matches_streamlit_inputs():
    widgets = GOLDEN["forecasting_inputs"]["history_widgets"]
    history = default_history(3)
    got = {f"hist_{k}_{j}": v for k, vals in history.items() for j, v in enumerate(vals)}
    assert_close(got, widgets)


def test_historical_metrics_match_rendered_table():
    from core.forecasting import historical_metrics
    table = GOLDEN["forecasting_inputs"]["metrics_table"]
    rows = historical_metrics(default_history(3), 3)

    def pct(v):
        return "—" if v is None else f"{v:.1%}"

    for row, m in zip(table, rows):
        rev_col = next(c for c in row if c.startswith("Revenue ("))
        assert row[rev_col] == f"${m['revenue']:,.1f}"
        assert row["Revenue growth"] == pct(m["revenue_growth"])
        assert row["Gross margin"] == pct(m["gross_margin"])
        assert row["R&D %"] == pct(m["rd_pct"])
        assert row["SG&A %"] == pct(m["sga_pct"])
        assert row["EBITDA margin"] == pct(m["ebitda_margin"])
        assert row["Adj EBITDA margin"] == pct(m["adj_ebitda_margin"])


@pytest.mark.parametrize("case", sorted(GOLDEN["deal"]))
def test_entry_costs_match_engine(case):
    # The inputs page shows sponsor equity using entry_costs(); the engine
    # funds the same fees inside run_lbo(). They must agree.
    from core.deal import entry_costs
    g = GOLDEN["deal"][case]
    d, cfg = deal_from(g["inputs"]), cfg_from(g["cfg"])
    ev = d.ebitda * d.entry_mult
    debt = ev * d.debt_pct / 100
    fees_in_engine = g["returns"]["entry_equity"] - (ev - debt)
    assert entry_costs(ev, debt, cfg) == pytest.approx(fees_in_engine, abs=1e-6)


@pytest.mark.parametrize("case", ["defaults", "wsp_mode"])
def test_sources_and_uses_equity_matches_engine(case):
    # These cases run the page's default financing: 3.4x senior + 0.8x mezz.
    from core.deal import capital_structure_from_multiples, sources_and_uses
    g = GOLDEN["deal"][case]
    d, cfg = deal_from(g["inputs"]), cfg_from(g["cfg"])
    debt_pct, senior_pct = capital_structure_from_multiples(d.ebitda, d.entry_mult, 3.4, 0.8)
    assert (debt_pct, senior_pct) == (d.debt_pct, d.senior_pct)
    su = sources_and_uses(d.ebitda, d.entry_mult, 3.4, 0.8, cfg)
    assert su["balanced"]
    assert su["sponsor_equity"] == pytest.approx(g["returns"]["entry_equity"], abs=1e-6)
