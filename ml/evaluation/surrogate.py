"""The live sliders' evaluation (core/live.py; PLAN.md 5.1, 5.8).

Question: for a deal a user in each region would actually enter, how close
is the Live step's estimate to the simulation it stands in for?

Since PLAN.md 5.8 the estimate is the simulation itself on a small, fixed
set of paths (``core.live``), not a trained network, so there is nothing to
train and no file to hash; the id stays ``surrogate`` so that
ops/model_gate.py compares it with the network's card it replaced.

- **Cases**: ``varied_deals`` (the headline) -- for every economy the app
  covers and ``INDUSTRIES`` across sectors, a whole deal as the API takes
  it: what "Use sourced figures" sets (PLAN.md 4.3, 4.4), the country's tax
  preset, and its own entry multiple, hold, fees, leases and one of five
  debt structures. ``all_regional_deals`` are the same economies and
  industries as the eleven inputs the retired network took, on its one
  fixed deal (``NETWORK_DEAL``), and ``in_training_range`` those inside the
  ranges it was trained on: both kept so the gate compares like with like.
  All written from the recorded sources by ``python -m
  tests.ml_regional_deals``.
- **Truth**: the simulation itself on ``TRUTH_PATHS`` paths.
- **Model**: ``core.live``'s paths and seed (``live_paths``, ``LIVE_SEED``);
  the varied deals go through ``core.live.live_estimate`` itself.
- **Baseline**: the same simulation on ``BASELINE_PATHS`` paths, the cruder
  estimate a live answer could also be.
- The cases are simulated and undated, so there is no time split: they are
  the newest sourced figures.
"""
from __future__ import annotations

import json
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from core.config import DEFAULTS, resolve_config
from core.deal import DealInputs
from core.live import LIVE_PATHS, LIVE_SEED, MIN_LIVE_PATHS, live_estimate, live_params, live_paths
from core.montecarlo import MCInputs
from ml.evaluation.harness import Case, Metric, evaluate_set, mean_abs, metric_specs
from simulation.vectorized_simulation import SimulationParams, run_vectorized_simulation_full

MODEL_ID = "surrogate"
REGIONAL_DEALS = Path(__file__).parent / "data" / "regional_deals.json"
TRUTH_PATHS = 100_000
BASELINE_PATHS = 200
# The network's training setting, kept for its one fixed deal
INTEREST_PASSES = 2
# Damodaran industry ids across sectors: the ones whose history the recorded
# fixtures keep (benchmarks/record.py HISTORY_INDUSTRIES), which the sourced
# spreads need
INDUSTRIES = ("machinery", "chemical_basic", "steel", "food_processing", "auto_parts", "building_materials",
              "electrical_equipment", "retail_general", "telecom_services", "utility_general")
HEADLINE_SET = "varied_deals"
PERCENTILES = {"p5": 5, "p50": 50, "p95": 95}
# PLAN.md 5.8's "done when": the median IRR within this of the full
# simulation, overall and in every region (tests/test_model_cards.py)
MEDIAN_ERROR_LIMIT_PP = 0.5

# The one deal the retired network was trained on, and the eleven inputs it
# took with the ranges they were sampled from
NETWORK_DEAL = dict(
    entry_ebitda=100.0, entry_multiple=10.0, holding_period=5, opex_pct=0.18, tax_rate=0.25,
    senior_pct=0.70, mezz_spread=0.04, interest_std=0.015,
    transaction_fees_pct=DEFAULTS["tx_fee_pct"] / 100, financing_fees_pct=DEFAULTS["fin_fee_pct"] / 100,
    other_uses=DEFAULTS["other_uses"],
)
X_COLS = ['growth_mean', 'growth_std', 'exit_mean', 'exit_std', 'interest_mean',
          'gross_margin_mean', 'gross_margin_std', 'da_pct', 'capex_pct', 'nwc_pct', 'debt_pct']
L_BOUNDS = [0.00, 0.005, 5.0, 0.3, 0.015, 0.15, 0.01, 0.02, 0.01, 0.005, 0.25]
U_BOUNDS = [0.18, 0.10,  20.0, 4.0, 0.14,  0.80, 0.08, 0.09, 0.10, 0.05,  0.90]


def in_range(features: dict) -> bool:
    """Whether every input lies inside the ranges the retired network was trained on."""
    return all(lo <= features[k] <= hi for k, lo, hi in zip(X_COLS, L_BOUNDS, U_BOUNDS))


def features_from(risk_inputs: dict, starting_inputs: dict, debt_pct: float) -> Optional[dict]:
    """The network's eleven inputs from the sourced Monte Carlo Settings and
    starting figures (per cent, as the API answers them), or None when one
    is missing. ``debt_pct`` is the deal's debt / value, per cent."""
    pairs = {"growth_mean": risk_inputs.get("mc_growth_mean"), "growth_std": risk_inputs.get("mc_growth_std"),
             "interest_mean": risk_inputs.get("mc_rate_mean"),
             "gross_margin_mean": risk_inputs.get("mc_gm_mean"), "gross_margin_std": risk_inputs.get("mc_gm_std"),
             "da_pct": starting_inputs.get("da"), "capex_pct": starting_inputs.get("capex"),
             "nwc_pct": starting_inputs.get("nwc"), "debt_pct": debt_pct}
    if any(v is None for v in pairs.values()) or None in (risk_inputs.get("mc_exit_mean"),
                                                          risk_inputs.get("mc_exit_std")):
        return None
    out = {k: round(v / 100, 6) for k, v in pairs.items()}
    out["exit_mean"] = risk_inputs["mc_exit_mean"]
    out["exit_std"] = risk_inputs["mc_exit_std"]
    return {k: out[k] for k in X_COLS}


def load_cases() -> list[dict]:
    return json.loads(REGIONAL_DEALS.read_text(encoding="utf-8"))["deals"]


def load_varied() -> list[dict]:
    return json.loads(REGIONAL_DEALS.read_text(encoding="utf-8"))["varied"]


def _seed(deal_id: str, salt: int) -> int:
    return (zlib.crc32(deal_id.encode()) + salt) % (2 ** 31)


def _answer(sim) -> dict:
    out = {k: float(np.percentile(sim.irr, q)) for k, q in PERCENTILES.items()}
    out["wipeout"] = float(sim.wipeout_rate)
    return out


def simulate(features: dict, paths: int, seed: int) -> dict:
    """The simulation's IRR percentiles and wipeout rate for these inputs on
    the retired network's fixed deal."""
    params = SimulationParams(n=paths, **NETWORK_DEAL, **features, n_interest_passes=INTEREST_PASSES)
    return _answer(run_vectorized_simulation_full(params, seed=seed))


def live_on_network_deal(features: dict) -> dict:
    """What the Live step answers for these inputs on the network's fixed
    deal: the simulation on ``core.live``'s paths and seed."""
    return simulate(features, live_paths(NETWORK_DEAL["holding_period"], 0, INTEREST_PASSES), LIVE_SEED)


def request_of(deal: dict) -> tuple[MCInputs, DealInputs, dict]:
    """A varied deal as the inputs the API builds from its request."""
    return MCInputs(**deal["mc"]), DealInputs(**deal["deal"]), resolve_config(deal["settings"])


def simulate_deal(deal: dict, paths: int, seed: int) -> dict:
    """The simulation's answer for a varied deal on ``paths`` paths: the
    Live step's own parameters with only the size and the seed changed."""
    params = replace(live_params(*request_of(deal)), n=paths)
    return _answer(run_vectorized_simulation_full(params, seed=seed))


def live_on_deal(deal: dict) -> dict:
    """What the Live step answers for a varied deal (``core.live.live_estimate``)."""
    p = live_estimate(*request_of(deal))["prediction"]
    return {"p5": p["irr_p5"], "p50": p["irr_p50"], "p95": p["irr_p95"], "wipeout": p["p_wipeout"]}


def _mae_pp(key: str):
    def compute(cases: Sequence[Case], preds: Sequence[dict]) -> Optional[float]:
        errors = [(p[key] - c.truth[key]) * 100 for c, p in zip(cases, preds)]
        return mean_abs(errors)
    return compute


METRICS = (
    Metric("median_irr_error_pp", "lower", _mae_pp("p50"), 0.02,
           "Mean absolute error of the median IRR, percentage points"),
    Metric("p5_irr_error_pp", "lower", _mae_pp("p5"), 0.02,
           "Mean absolute error of the 5th-percentile IRR, percentage points"),
    Metric("p95_irr_error_pp", "lower", _mae_pp("p95"), 0.02,
           "Mean absolute error of the 95th-percentile IRR, percentage points"),
    Metric("wipeout_error_pp", "lower", _mae_pp("wipeout"), 0.02,
           "Mean absolute error of the share of paths that lose all equity, percentage points"),
)
HEADLINE = "median_irr_error_pp"


def _rows(deals: Sequence[dict], truth, model, baseline) -> list[tuple]:
    return [(Case(id=d["id"], region=d["region"], year=None, features=d.get("features") or {}, truth=truth(d)),
             model(d), baseline(d)) for d in deals]


def evaluate() -> dict:
    """The three sets for the card."""
    rows = _rows(load_cases(),
                 lambda d: simulate(d["features"], TRUTH_PATHS, _seed(d["id"], 0)),
                 lambda d: live_on_network_deal(d["features"]),
                 lambda d: simulate(d["features"], BASELINE_PATHS, _seed(d["id"], 1)))
    inside = [r for r in rows if in_range(r[0].features)]
    varied = _rows(load_varied(),
                   lambda d: simulate_deal(d, TRUTH_PATHS, _seed(d["id"], 0)),
                   live_on_deal,
                   lambda d: simulate_deal(d, BASELINE_PATHS, _seed(d["id"], 1)))
    meta = json.loads(REGIONAL_DEALS.read_text(encoding="utf-8"))
    return {
        "headline_set": HEADLINE_SET,
        "headline_metric": HEADLINE,
        "metrics": metric_specs(METRICS),
        "split": {"kind": "none", "why": "the cases are simulated deals, which have no dates: the sourced "
                  f"figures recorded on {meta['recorded_on']}"},
        "baseline": {"id": f"simulation_{BASELINE_PATHS}",
                     "description": f"The simulation itself on {BASELINE_PATHS:,} paths, the cruder estimate a "
                                    f"live answer could be; truth is the simulation on {TRUTH_PATHS:,} paths"},
        "sets": {
            "varied_deals": {
                "description": "Every regional deal with its own entry multiple, hold, fees, tax rules, leases "
                               "and one of five debt structures, as the Live step would be sent it",
                **evaluate_set(varied, METRICS, HEADLINE)},
            "all_regional_deals": {
                "description": "Every regional deal's eleven inputs on the retired network's one fixed deal",
                **evaluate_set(rows, METRICS, HEADLINE)},
            "in_training_range": {
                "description": "The regional deals whose every input lies inside the ranges the retired "
                               "network was trained on",
                **evaluate_set(inside, METRICS, HEADLINE)},
        },
    }


CARD = {
    "id": MODEL_ID,
    "name": "Live sliders (the simulation on a small, fixed set of paths)",
    "module": "core/live.py",
    "used_by": "Monte Carlo -> Live (POST /api/montecarlo/live)",
    "estimates": "The Monte Carlo simulation's IRR percentiles, mean, spread and the shares of paths above "
                 "the hurdle, losing money and losing everything, for any deal, in a few milliseconds",
    "training": {
        "data": f"None: nothing is trained. Each estimate reruns the simulation on {LIVE_PATHS:,} seeded "
                f"paths (fewer for a long hold with many facilities, never under {MIN_LIVE_PATHS:,}). It "
                "replaced a network trained on 100,000 simulations of one fixed deal (PLAN.md 5.8)",
        "command": None,
        "deterministic": True,
    },
    "limitations": [
        "The estimate carries sampling error: the Live step shows a 95% interval for the median, about half "
        "a percentage point wide for an ordinary deal and wider on fewer paths.",
        "Close to losing everything the 5th percentile falls steeply, so it moves more between samples "
        "than the median does (the p5 column).",
        f"Truth is itself a simulation of {TRUTH_PATHS:,} paths: a median error below about 0.04 points is "
        "within its noise.",
        "Whatever the simulation leaves out the estimate leaves out too (CLAUDE.md findings 12 and 13: "
        "the percentages path drops surplus cash once debt is repaid, and neither path reads minimum cash).",
    ],
}
