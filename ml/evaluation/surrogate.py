"""The live sliders' surrogate evaluation (ml/surrogate/; PLAN.md 5.1).

Question: for a deal a user in each region would actually enter, how close
is the network's IRR distribution to the simulation it stands in for?

- **Cases**: ``REGIONAL_DEALS`` -- for every economy the app covers and
  ``INDUSTRIES`` across sectors, the Monte Carlo inputs a user gets from
  "Use sourced figures" (PLAN.md 4.3, 4.4: the country's growth, the
  industry's exit multiple, margins, D&A, capex, working capital and their
  spreads; the currency's benchmark rate plus the industry's spread), with
  the deal's default debt share. Written from the recorded sources by
  ``python -m tests.ml_regional_deals``; the surrogate's fixed training
  terms (entry multiple, hold, fees ...) apply, as on the Live screen.
- **Truth**: the simulation itself on ``TRUTH_PATHS`` paths.
- **Baseline**: the same simulation on ``BASELINE_PATHS`` paths, which is
  fast enough to run live: the network must beat it to be worth having.
- **Sets**: ``all_regional_deals`` (the headline: what a user in each
  region gets) and ``in_training_range`` (every input inside the ranges the
  network was trained on, ``L_BOUNDS``/``U_BOUNDS``). The training data are
  simulated and undated, so there is no time split: the cases are the newest
  sourced figures.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ml.evaluation.harness import Case, Metric, evaluate_set, mean_abs, metric_specs
from ml.surrogate.generate_data import L_BOUNDS, TRAINING_FIXED, U_BOUNDS, X_COLS
from simulation.vectorized_simulation import SimulationParams, run_vectorized_simulation_full

MODEL_ID = "surrogate"
REGIONAL_DEALS = Path(__file__).parent / "data" / "regional_deals.json"
TRUTH_PATHS = 10_000
BASELINE_PATHS = 200
# The training data's own setting
INTEREST_PASSES = 2
# Damodaran industry ids across sectors: the ones whose history the recorded
# fixtures keep (benchmarks/record.py HISTORY_INDUSTRIES), which the sourced
# spreads need
INDUSTRIES = ("machinery", "chemical_basic", "steel", "food_processing", "auto_parts", "building_materials",
              "electrical_equipment", "retail_general", "telecom_services", "utility_general")
HEADLINE_SET = "all_regional_deals"
PERCENTILES = {"p5": 5, "p50": 50, "p95": 95}


def in_range(features: dict) -> bool:
    """Whether every input lies inside the ranges the network was trained on."""
    return all(lo <= features[k] <= hi for k, lo, hi in zip(X_COLS, L_BOUNDS, U_BOUNDS))


def features_from(risk_inputs: dict, starting_inputs: dict, debt_pct: float) -> Optional[dict]:
    """The eleven surrogate inputs from the sourced Monte Carlo Settings and
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


def _seed(deal_id: str, salt: int) -> int:
    return (zlib.crc32(deal_id.encode()) + salt) % (2 ** 31)


def simulate(features: dict, paths: int, seed: int) -> dict:
    """The simulation's IRR percentiles and wipeout rate for these inputs on
    the surrogate's fixed training deal."""
    params = SimulationParams(n=paths, **TRAINING_FIXED, **features, n_interest_passes=INTEREST_PASSES)
    sim = run_vectorized_simulation_full(params, seed=seed)
    out = {k: float(np.percentile(sim.irr, q)) for k, q in PERCENTILES.items()}
    out["wipeout"] = float(sim.wipeout_rate)
    return out


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


def evaluate(base: Optional[str] = None) -> dict:
    """Both sets for the card, with the network in ``base`` (default: the
    committed one)."""
    from ml.surrogate.predict import BASE, SurrogatePredictor
    predictor = SurrogatePredictor(base or BASE)
    rows, inside = [], []
    for deal in load_cases():
        f = deal["features"]
        truth = simulate(f, TRUTH_PATHS, _seed(deal["id"], 0))
        case = Case(id=deal["id"], region=deal["region"], year=None, features=f, truth=truth)
        p = predictor.predict(**f)
        model = {"p5": p.irr_p5, "p50": p.irr_p50, "p95": p.irr_p95, "wipeout": p.p_wipeout}
        row = (case, model, simulate(f, BASELINE_PATHS, _seed(deal["id"], 1)))
        rows.append(row)
        if in_range(f):
            inside.append(row)
    meta = json.loads(REGIONAL_DEALS.read_text(encoding="utf-8"))
    return {
        "headline_set": HEADLINE_SET,
        "headline_metric": HEADLINE,
        "metrics": metric_specs(METRICS),
        "split": {"kind": "none", "why": "trained on simulated deals, which have no dates; the cases are "
                  f"the sourced figures recorded on {meta['recorded_on']}"},
        "baseline": {"id": f"simulation_{BASELINE_PATHS}",
                     "description": f"The simulation itself on {BASELINE_PATHS:,} paths, fast enough to run "
                                    "live; truth is the simulation on "
                                    f"{TRUTH_PATHS:,} paths"},
        "sets": {
            "all_regional_deals": {
                "description": "Every regional deal, as a user there would enter it",
                **evaluate_set(rows, METRICS, HEADLINE)},
            "in_training_range": {
                "description": "The regional deals whose every input lies inside the network's training ranges",
                **evaluate_set(inside, METRICS, HEADLINE)},
        },
    }


CARD = {
    "id": MODEL_ID,
    "name": "Live sliders (surrogate network)",
    "module": "ml/surrogate/predict.py",
    "used_by": "Monte Carlo -> Live (POST /api/ml/surrogate)",
    "estimates": "The Monte Carlo simulation's IRR percentiles, mean, spread, chance of beating 20% and "
                 "wipeout rate, from eleven inputs, in under a millisecond",
    "training": {
        "data": "100,000 simulations of one fixed deal (entry multiple 10x, five-year hold, the default fees "
                "and debt split), Latin hypercube over the eleven inputs; 2,000 paths each",
        "command": "python -m ml.evaluation train surrogate",
        "deterministic": False,
    },
    "limitations": [
        "Every term outside the eleven inputs is the fixed training deal's; the Live screen lists the "
        "deal's differences. PLAN.md 5.8 retrains across deal terms.",
        "Sourced regional inputs often fall outside the training ranges (D&A, working capital and capex "
        "shares below them, exit multiples and their spreads above them); the network extrapolates there, "
        "and the in-training-range set shows how it does inside them.",
        "Truth is itself a simulation of 10,000 paths, so a median error below about 0.1 points is "
        "within its noise.",
    ],
}
