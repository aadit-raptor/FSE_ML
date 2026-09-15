"""HTTP API: results match core/ (and so the Streamlit snapshot), inputs are
validated, and optional features fail cleanly when unavailable."""
import json
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from tests.golden_compare import assert_close

client = TestClient(app)
GOLDEN = json.load(open(os.path.join(os.path.dirname(__file__), "golden", "golden.json"),
                        encoding="utf-8"))


def ok(resp):
    assert resp.status_code == 200, resp.text
    return resp.json()


# ---------------------------------------------------------------------------
# Meta and settings
# ---------------------------------------------------------------------------
def test_health_and_openapi():
    assert ok(client.get("/api/health"))["status"] == "ok"
    schema = ok(client.get("/api/openapi.json"))
    assert "/api/deal/run" in schema["paths"] and "/api/montecarlo/run" in schema["paths"]


def test_settings_defaults_and_validation():
    body = ok(client.get("/api/settings/defaults"))
    assert body["defaults"]["tx_fee_pct"] == 2.3 and body["correlation_valid"]
    bad = ok(client.post("/api/settings/validate",
                         json={"settings": {"corr_g_em": 0.99, "corr_g_ir": 0.99, "corr_em_ir": -0.99}}))
    assert bad["correlation_valid"] is False


def test_unknown_setting_is_rejected():
    r = client.post("/api/deal/run", json={"settings": {"not_a_setting": 1}})
    assert r.status_code == 422 and "not_a_setting" in r.text


# ---------------------------------------------------------------------------
# Deal
# ---------------------------------------------------------------------------
def _golden_deal_request(case):
    g = GOLDEN["deal"][case]
    fields = {"ebitda", "entry_mult", "exit_mult", "hold", "growth", "gross_margin", "opex",
              "tax", "da", "debt_pct", "senior_pct", "base_rate", "mezz_spread", "capex", "nwc",
              "mincash", "wsp_mode", "ar_days", "inv_days", "ap_days"}
    inputs = {k[2:]: v for k, v in g["inputs"].items() if k[2:] in fields}
    settings = {k: v for k, v in g["cfg"].items() if k in ("tx_fee_pct", "fin_fee_pct", "other_uses")}
    return {"inputs": inputs, "settings": settings}, g


@pytest.mark.parametrize("case", sorted(GOLDEN["deal"]))
def test_deal_run_matches_streamlit_snapshot(case):
    req, g = _golden_deal_request(case)
    body = ok(client.post("/api/deal/run", json=req))
    assert_close(body["returns"], g["returns"])
    assert_close(body["equity_bridge"], {k: g["equity_bridge"][k] for k in body["equity_bridge"]})
    # exit_sensitivity deliberately differs from the snapshot (finding 7, fixed);
    # see test_model_fixes.py
    assert_close(body["operating_model"], g["operating_model"])
    assert_close(body["cash_flow"], g["cash_flow"])
    tranches = {k: v for k, v in g["debt_schedule"].items() if k != "schedule"}
    assert_close(body["debt_schedule"], tranches)
    assert_close(body["tranches"], g["debt_schedule"]["schedule"])
    steps = body["bridge_steps"]
    assert steps[0]["is_total"] and steps[-1]["is_total"]
    # The waterfall steps close to the equity gain except for the bridge's
    # residual. Known engine behaviour, kept as is: the debt model opens with
    # cash equal to the minimum cash balance, but sponsor equity does not fund
    # it, so the residual equals the minimum cash.
    gain = g["equity_bridge"]["exit_equity"] - g["equity_bridge"]["entry_equity"]
    assert sum(s["value"] for s in steps[1:-1]) + body["equity_bridge"]["residual"] == pytest.approx(gain, abs=0.02)
    assert body["equity_bridge"]["residual"] == pytest.approx(req["inputs"]["mincash"], abs=0.02)


def test_sources_and_uses():
    body = ok(client.post("/api/deal/sources-and-uses", json={}))
    assert body["balanced"] and body["sponsor_equity"] == pytest.approx(613.92)
    assert body["debt_pct"] == pytest.approx(42.0) and body["senior_pct"] == pytest.approx(340 / 4.2)


@pytest.mark.parametrize("patch", [{"hold": 0}, {"hold": 16}, {"ebitda": 0}, {"debt_pct": 100},
                                   {"gross_margin": -1}, {"unknown_field": 1}])
def test_deal_input_validation(patch):
    assert client.post("/api/deal/run", json={"inputs": patch}).status_code == 422


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------
def test_montecarlo_run_is_reproducible_and_chart_ready():
    req = {"mc": {"n": 20000}, "seed": 7, "histogram_bins": 50, "scatter_points": 500}
    a = ok(client.post("/api/montecarlo/run", json=req))
    b = ok(client.post("/api/montecarlo/run", json=req))
    for key in ("summary", "irr_histogram", "irr_cdf", "drivers", "heatmap"):
        assert a[key] == b[key], key
    assert len(a["irr_histogram"]["edges"]) == 51 and len(a["irr_histogram"]["density"]) == 50
    assert len(a["irr_cdf"]["percentiles"]) == 101
    assert all(len(v) == 500 for v in a["scatter"].values())
    assert len(a["heatmap"]["irr"]) == 7 and len(a["heatmap"]["irr"][0]) == 8
    assert a["correlations"]["labels"][0] == "IRR"


def test_montecarlo_matches_core_with_same_seed():
    from core.config import resolve_config
    from core.deal import DealInputs
    from core.montecarlo import MCInputs, build_sim_params, risk_summary
    from simulation.vectorized_simulation import run_vectorized_simulation_full
    body = ok(client.post("/api/montecarlo/run", json={"mc": {"n": 10000}, "seed": 3}))
    sim = run_vectorized_simulation_full(
        build_sim_params(MCInputs(n=10000), DealInputs(), resolve_config()), seed=3)
    expected = risk_summary(sim, 20.0)
    assert body["summary"]["mean_irr"] == float(expected["mean_irr"])
    assert body["summary"]["p_above_hurdle"] == float(expected["p_above_hurdle"])


@pytest.mark.parametrize("case", ["recession_preset", "stagflation_preset"])
def test_montecarlo_scenario_params_match_streamlit(case):
    g = GOLDEN["montecarlo"][case]
    i = g["inputs"]
    deal_keys = {"opex", "da", "tax", "capex", "nwc", "debt_pct", "senior_pct", "mezz_spread"}
    req = {
        "mc": {"n": i["mc_n"], "ebitda": i["mc_fi_eb"], "entry_mult": i["mc_fi_em"],
               "hold": i["mc_fi_hold"], "hurdle": i["mc_hurdle"],
               "growth_mean": i["mc_growth_mean"], "growth_std": i["mc_growth_std"],
               "exit_mean": i["mc_exit_mean"], "exit_std": i["mc_exit_std"],
               "rate_mean": i["mc_rate_mean"], "rate_std": i["mc_rate_std"],
               "gm_mean": i["mc_gm_mean"], "gm_std": i["mc_gm_std"]},
        "deal": {k[2:]: v for k, v in i.items() if k[2:] in deal_keys},
        "scenario": g["preset"].lower(), "seed": 1,
    }
    body = ok(client.post("/api/montecarlo/run", json=req))
    assert_close(body["params"], g["sim_params"])


def test_montecarlo_rejects_invalid_correlations_and_oversized_runs():
    bad_corr = {"settings": {"corr_g_em": 0.99, "corr_g_ir": 0.99, "corr_em_ir": -0.99}}
    r = client.post("/api/montecarlo/run", json=bad_corr)
    assert r.status_code == 422 and "semi-definite" in r.text
    assert client.post("/api/montecarlo/run", json={"mc": {"n": 5_000_000}}).status_code == 422


def test_scenarios():
    body = ok(client.post("/api/montecarlo/scenarios", json={"mc": {"n": 5000}, "seed": 2}))
    assert set(body["scenarios"]) == {"recession", "base", "bull", "stagflation"}
    rec, bull = body["scenarios"]["recession"], body["scenarios"]["bull"]
    assert rec["mean_irr"] < bull["mean_irr"]
    assert rec["irr_box"]["p5"] <= rec["irr_box"]["p50"] <= rec["irr_box"]["p95"]


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
def test_forecasting_defaults_seed_and_run_match_streamlit():
    g = GOLDEN["forecasting"]["defaults_with_overlay"]
    defaults = ok(client.get("/api/forecasting/defaults"))
    assert_close({f"hist_{k}_{j}": v for k, vals in defaults["history"].items()
                  for j, v in enumerate(vals)}, GOLDEN["forecasting_inputs"]["history_widgets"])

    seed = ok(client.post("/api/forecasting/seed", json={"history": defaults["history"]}))
    assert_close(seed["ltm"], g["ltm"])

    n_fwd = len(g["fwd"])
    grid = {k: [float(v)] * n_fwd for k, v in seed["seeded_assumptions"].items()}
    body = ok(client.post("/api/forecasting/run",
                          json={"history": defaults["history"], "assumptions": grid,
                                "simulate": True, "n_sim": 30000}))
    assert_close(body["years"], g["fwd"])
    assert body["balanced"] and body["opening_balance_gap"] == 0.0
    sim = body["simulation"]
    assert_close(sim["revenue_bands"]["p50"], g["sim"]["revenue"]["p50"])
    assert len(sim["target_probabilities"]) == 5


def test_forecasting_reports_an_unbalanced_opening_balance_sheet():
    defaults = ok(client.get("/api/forecasting/defaults"))
    history = dict(defaults["history"], h_re=[0.0, 0.0, 0.0])
    grid = {k: [float(v)] * 5 for k, v in defaults["seeded_assumptions"].items()}
    body = ok(client.post("/api/forecasting/run", json={"history": history, "assumptions": grid}))
    assert body["balanced"] is False and body["opening_balance_gap"] == pytest.approx(127.6)
    assert all(abs(g) < 1e-6 for g in body["forecast_balance_gaps"])


@pytest.mark.parametrize("patch", [
    {"history": {"h_nope": [1.0]}},
    {"history": {"h_rev": [1.0, 2.0], "h_cogs": [1.0]}},
    {"drop_assumption": "gm"},
    {"assumptions_extra": {"bogus": [1.0] * 5}},
])
def test_forecasting_validation(patch):
    defaults = ok(client.get("/api/forecasting/defaults"))
    grid = {k: [float(v)] * 5 for k, v in defaults["seeded_assumptions"].items()}
    history = defaults["history"]
    if "history" in patch:
        history = patch["history"]
    if "drop_assumption" in patch:
        grid.pop(patch["drop_assumption"])
    if "assumptions_extra" in patch:
        grid.update(patch["assumptions_extra"])
    r = client.post("/api/forecasting/run", json={"history": history, "assumptions": grid})
    assert r.status_code == 422, r.text


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
def test_backtesting_matches_rendered_streamlit_page():
    deals = {d["name"]: d for d in ok(client.get("/api/backtesting/deals"))}
    rendered = GOLDEN["backtesting"]["rendered_default"]
    d = deals[rendered["deal"]]
    hold = int(d["entry"]["holding_period"])
    body = ok(client.post("/api/backtesting/run", json={
        "entry": d["entry"],
        "actual": {k: v[:hold] for k, v in d["actual"].items()},
        "actual_exit": d["actual_exit"],
    }))
    m = rendered["metrics"]
    assert f"{body['predicted_irr_mean']:.1f}%" == m["Predicted IRR (mean)"]
    assert f"{body['predicted_moic']:.2f}x" == m["Predicted MOIC (base)"]
    assert f"${body['predicted_ebitda'][-1]:,.0f}M" == m["Predicted exit EBITDA"]
    assert len(body["years"]) == hold and len(body["irr_histogram"]["density"]) == 80


@pytest.mark.parametrize("deal", sorted(GOLDEN["backtesting"]["helpers"]))
def test_backtesting_each_deal_uses_its_own_inputs(deal):
    # The Streamlit page only ever ran Burger King's inputs; the API must not.
    deals = {d["name"]: d for d in ok(client.get("/api/backtesting/deals"))}
    d, g = deals[deal], GOLDEN["backtesting"]["helpers"][deal]
    hold = int(d["entry"]["holding_period"])
    body = ok(client.post("/api/backtesting/run", json={
        "entry": d["entry"], "actual": {k: v[:hold] for k, v in d["actual"].items()},
        "actual_exit": d["actual_exit"]}))
    assert_close(body["predicted_ebitda"], g["predicted_ebitda"])
    assert body["predicted_irr_mean"] == pytest.approx(g["prediction_irr"]["mean"] * 100, abs=1e-9)


def test_backtesting_requires_actuals_for_every_year():
    deals = ok(client.get("/api/backtesting/deals"))
    d = next(x for x in deals if x["name"].startswith("Burger"))
    short = {k: v[:3] for k, v in d["actual"].items()}
    r = client.post("/api/backtesting/run",
                    json={"entry": d["entry"], "actual": short, "actual_exit": d["actual_exit"]})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Optional features
# ---------------------------------------------------------------------------
def test_capabilities():
    body = ok(client.get("/api/capabilities"))
    assert set(body) == {"anomaly_detector", "surrogate", "macro_regime_installed",
                         "macro_regime_trained", "edgar"}


def test_deal_risk_when_available():
    caps = ok(client.get("/api/capabilities"))
    r = client.post("/api/ml/deal-risk", json={})
    if not caps["anomaly_detector"]:
        assert r.status_code == 503
        return
    body = ok(r)
    assert body["inputs"]["rate"] == pytest.approx((3.4 * 6.5 + 0.8 * 10.5) / 4.2)
    assert 1 <= body["risk_score"] <= 10 and len(body["nearest_deals"]) > 0


def test_surrogate_when_available():
    caps = ok(client.get("/api/capabilities"))
    r = client.post("/api/ml/surrogate", json={"deal": {"opex": 27.0}})
    if not caps["surrogate"]:
        assert r.status_code == 503
        return
    body = ok(r)
    assert "irr_p50" in body["prediction"]
    assert any(t["term"] == "opex / revenue" and t["value"] == pytest.approx(0.27)
               for t in body["term_differences"])


def test_surrogate_tail_threshold():
    from core.surrogate import tail_unreliable
    assert tail_unreliable(0.02) and not tail_unreliable(0.0199)


def test_macro_regime_unavailable_is_a_clean_503():
    caps = ok(client.get("/api/capabilities"))
    if caps["macro_regime_trained"]:
        pytest.skip("regime model is trained here; this checks the unavailable path")
    r = client.post("/api/ml/macro-regime")
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]


def test_edgar_maps_filings_to_forecast_history(monkeypatch):
    import ml.edgar_extractor as ee

    def fake_fetch(ticker, n_years=3):
        return ee.ExtractedFinancials(ticker=ticker, company_name="TEST CORP",
                                      years=[2023, 2024, 2025], data={}, warnings=["w"])

    def fake_state(extracted):
        return {"hist_h_rev_0": 10.0, "hist_h_rev_1": 11.0, "hist_h_rev_2": 12.0,
                "hist_h_cash_2": 5.0, "hist_h_cash_0": 3.0, "hist_h_cash_1": 4.0}

    monkeypatch.setattr(ee, "fetch_financials", fake_fetch)
    monkeypatch.setattr(ee, "financials_to_session_state", fake_state)
    body = ok(client.get("/api/edgar/msft"))
    assert body["ticker"] == "MSFT" and body["company_name"] == "TEST CORP"
    assert body["history"] == {"h_rev": [10.0, 11.0, 12.0], "h_cash": [3.0, 4.0, 5.0]}


def test_edgar_unknown_ticker_is_404(monkeypatch):
    import ml.edgar_extractor as ee

    def fake_fetch(ticker, n_years=3):
        raise ValueError(f"Ticker '{ticker}' not found in SEC EDGAR.")

    monkeypatch.setattr(ee, "fetch_financials", fake_fetch)
    assert client.get("/api/edgar/ZZZZZZ").status_code == 404
