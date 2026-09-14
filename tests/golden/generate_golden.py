"""Snapshot what the Streamlit app computes, to prove the core/ extraction
changes no numbers.

Run against the Streamlit app as it stood before core/ existed to create
golden.json; tests/test_core_parity.py then checks core/ against it. Running
it again after the extraction and diffing the output also shows the Streamlit
app itself still computes the same values.

Everything is captured through the app's own session state and widgets
(headless AppTest), so no model code is duplicated here. Monte Carlo draws are
unseeded in the app, so for simulations the snapshot records the exact
parameters sent to the engine rather than the random outcomes.

Delete this directory when the Streamlit app is retired.

    python tests/golden/generate_golden.py [output.json]
"""
import dataclasses
import json
import logging
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
logging.disable(logging.WARNING)

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(ROOT, "app.py")


def plain(x):
    """Dataclasses / numpy / pandas -> JSON-safe, full float precision."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return {f.name: plain(getattr(x, f.name)) for f in dataclasses.fields(x)}
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return plain(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if hasattr(x, "to_dict") and hasattr(x, "columns"):   # DataFrame
        return None                                         # not snapshotted
    return x


def new_app():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    return at


def goto(at, mode, page=1):
    at.session_state["mode"] = mode
    at.session_state["page"] = page
    at.run()


def click(at, label):
    next(b for b in at.button if label in str(b.label)).click()
    at.run()


def assert_clean(at, where):
    if at.exception:
        raise RuntimeError(f"{where}: {[e.value for e in at.exception]}")


# ---------------------------------------------------------------------------
# Deal wizard
# ---------------------------------------------------------------------------
DEAL_CASES = {
    "defaults": {"state": {}, "cfg": {}},
    "wsp_mode": {
        "state": {"d_wsp_mode": True, "d_ar_days": 38.0, "d_inv_days": 52.0,
                  "d_ap_days": 41.0, "d_growth": 7.5},
        "cfg": {},
    },
    "custom_deal_and_fees": {
        "state": {"d_ebitda": 250.0, "d_entry_mult": 12.0, "d_exit_mult": 10.0,
                  "d_hold": 7, "d_growth": 8.0, "d_gross_margin": 55.0,
                  "d_opex": 25.0, "d_tax": 21.0, "d_da": 3.0, "d_debt_pct": 55.0,
                  "d_senior_pct": 65.0, "d_base_rate": 5.25, "d_mezz_spread": 5.0,
                  "d_capex": 3.5, "d_nwc": 2.0, "d_mincash": 15.0},
        "cfg": {"tx_fee_pct": 3.0, "fin_fee_pct": 2.0, "other_uses": 10.0},
    },
}


def capture_deal(name, case):
    at = new_app()
    for k, v in case["cfg"].items():
        at.session_state[f"cfg_{k}"] = v
    for k, v in case["state"].items():
        at.session_state[k] = v
    goto(at, "deal", 2)
    click(at, "Run deal model")
    assert_clean(at, f"deal/{name}")
    r = at.session_state["lbo_result"]
    inputs = {k: at.session_state[k] for k in at.session_state.filtered_state
              if isinstance(k, str) and k.startswith("d_")}
    cfg = {k[4:]: at.session_state[k] for k in at.session_state.filtered_state
           if isinstance(k, str) and k.startswith("cfg_")}
    return {
        "inputs": plain(inputs),
        "cfg": plain(cfg),
        "returns": plain(r.returns),
        "equity_bridge": plain(r.equity_bridge),
        "exit_sensitivity": plain(r.exit_sensitivity),
        "operating_model": plain(r.operating_model),
        "cash_flow": plain(r.cash_flow),
        "debt_schedule": plain(r.debt_schedule),
        "interest_converged": plain(r.interest_converged),
    }


# ---------------------------------------------------------------------------
# Monte Carlo — parameters sent to the engine
# ---------------------------------------------------------------------------
MC_CASES = {
    "defaults": {"state": {}, "cfg": {}, "preset": None},
    "recession_preset": {"state": {}, "cfg": {}, "preset": "RECESSION"},
    "bull_custom_cfg": {
        "state": {"mc_growth_mean": 9.0, "mc_exit_std": 2.25, "mc_rate_mean": 7.1,
                  "d_debt_pct": 48.0, "d_opex": 21.0},
        "cfg": {"bull_growth_mult": 1.8, "bull_rate_mult": 0.7, "corr_g_em": 0.45,
                "corr_ir_sh": -0.35, "mc_n_passes": 3, "tx_fee_pct": 1.5},
        "preset": "BULL",
    },
    "stagflation_preset": {"state": {"mc_growth_mean": 1.0}, "cfg": {},
                           "preset": "STAGFLATION"},
}


def capture_mc(name, case):
    at = new_app()
    for k, v in case["cfg"].items():
        at.session_state[f"cfg_{k}"] = v
    for k, v in case["state"].items():
        at.session_state[k] = v
    at.session_state["mc_n"] = 2000          # keep the capture fast
    goto(at, "mc")
    if case["preset"]:
        click(at, case["preset"])
    click(at, "RUN SIMULATION")
    assert_clean(at, f"mc/{name}")
    sim = at.session_state["mc_result"]
    inputs = {k: at.session_state[k] for k in at.session_state.filtered_state
              if isinstance(k, str) and (k.startswith("d_") or k.startswith("mc_"))
              and k not in ("mc_result",)}
    cfg = {k[4:]: at.session_state[k] for k in at.session_state.filtered_state
           if isinstance(k, str) and k.startswith("cfg_")}
    return {"inputs": plain(inputs), "cfg": plain(cfg), "preset": case["preset"],
            "sim_params": plain(sim.params)}


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
FORECAST_CASES = {
    "defaults_with_overlay": {"state": {}},
    "custom_historicals_and_assumptions": {
        "state": {"hist_h_rev_2": 410.0, "hist_h_cogs_2": -205.0, "hist_h_sga_2": -40.0,
                  "hist_h_cash_2": 90.0, "hist_h_ltd_2": 150.0, "hist_h_re_2": 60.0,
                  "hist_h_ncl_2": 25.0, "hist_h_lta_2": 30.0,
                  "fwd_rev_g_0": 12.0, "fwd_rev_g_1": 9.0, "fwd_min_cash_2": 120.0,
                  "fwd_divs_3": 20.0, "fwd_ltd_chg_1": -30.0},
    },
}


def summarize_paths(paths):
    if paths is None:
        return None
    out = {}
    for key in ("revenue", "ebitda"):
        a = paths[key]
        out[key] = {q: np.percentile(a, int(q[1:]), axis=0).tolist()
                    for q in ("p5", "p50", "p95")}
        out[key]["mean"] = a.mean(axis=0).tolist()
    out["g_draws_mean"] = float(paths["g_draws"].mean())
    out["m_draws_mean"] = float(paths["m_draws"].mean())
    return out


def capture_forecast(name, case):
    at = new_app()
    goto(at, "forecast")
    for k, v in case["state"].items():
        at.session_state[k] = v
    at.run()
    cb = next(c for c in at.checkbox if "Monte Carlo" in str(c.label))
    cb.check()
    at.run()
    click(at, "Run 3-statement model")
    assert_clean(at, f"forecast/{name}")
    res = at.session_state["fc2_result"]
    return {"state": case["state"], "ltm": plain(res["ltm"]),
            "assumptions": plain(res["assumptions"]), "fwd": plain(res["fwd"]),
            "sim": summarize_paths(res["sim_paths"])}


# ---------------------------------------------------------------------------
# Backtesting — rendered results for every preloaded deal
# ---------------------------------------------------------------------------
def capture_backtests():
    """Rendered results for the default deal, and the page's calculation
    helpers for every preloaded deal.

    Only the default deal (Burger King) is captured through the page: in the
    Streamlit app, switching the deal selector updates the description and
    years but its keyed input widgets keep the first deal's numbers, so the
    page never actually runs the other deals' inputs.
    """
    from pages.backtesting import (PRELOADED_DEALS, _predicted_ebitda,
                                   _run_prediction_sim)
    at = new_app()
    goto(at, "backtest")
    click(at, "RUN BACKTEST")
    assert_clean(at, "backtest/default")
    rendered = {
        "deal": at.selectbox[0].value,
        "metrics": {str(m.label): str(m.value) for m in at.metric},
        "tables": [df.value.reset_index().astype(str).to_dict(orient="records")
                   for df in at.dataframe],
    }
    helpers = {}
    for deal, d in PRELOADED_DEALS.items():
        if deal.startswith("Custom"):
            continue
        irr = _run_prediction_sim(d["entry"])      # seeded in the app (seed=42)
        helpers[deal] = {
            "entry": d["entry"],
            "predicted_ebitda": plain(_predicted_ebitda(d["entry"])),
            "prediction_irr": {"mean": float(irr.mean()),
                               "p5": float(np.percentile(irr, 5)),
                               "p50": float(np.percentile(irr, 50)),
                               "p95": float(np.percentile(irr, 95)),
                               "n": int(irr.size)},
        }
    return {"rendered_default": rendered, "helpers": helpers}


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "golden.json")
    golden = {
        "deal": {n: capture_deal(n, c) for n, c in DEAL_CASES.items()},
        "montecarlo": {n: capture_mc(n, c) for n, c in MC_CASES.items()},
        "forecasting": {n: capture_forecast(n, c) for n, c in FORECAST_CASES.items()},
        "backtesting": capture_backtests(),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(golden, f, indent=1, sort_keys=True)
    print(f"wrote {out_path}")
