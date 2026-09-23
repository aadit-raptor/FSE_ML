"""3-statement forecasting endpoints."""
from fastapi import APIRouter, HTTPException

from api.limits import simulation_slot
from api.schemas import (
    ForecastDefaultsResponse, Money, ForecastRunRequest, ForecastRunResponse, HistoryRequest, SeedResponse,
)
from api.observability import model_timer
from api.serialize import to_json
from core.money import in_unit
from core.forecasting import (
    ASSUMPTION_KEYS, HISTORICAL_FIELDS, assumptions_from_grid, default_history,
    historical_metrics, ltm_from_history, opening_bs_gap, revenue_cagr,
    run_3_statement_model, run_forecast_simulation, seed_assumptions, simulation_summary,
)

router = APIRouter(prefix="/forecasting", tags=["forecasting"])

N_HIST = 3
N_FWD = 5
BALANCE_TOLERANCE = 0.5     # millions (in_unit converts); the Streamlit page warned above this


def _checked_history(history):
    known = {k for k, _, _ in HISTORICAL_FIELDS}
    unknown = set(history) - known
    if unknown:
        raise HTTPException(422, f"unknown history fields: {sorted(unknown)}")
    lengths = {len(v) for v in history.values()}
    if len(lengths) > 1 or not lengths or min(lengths) < 1 or max(lengths) > 10:
        raise HTTPException(422, "every history field needs the same number of years (1-10)")
    n = lengths.pop()
    # Fields left out fall back to their defaults for that many years
    return {**default_history(n), **history}, n


@router.get("/defaults", response_model=ForecastDefaultsResponse)
def get_defaults():
    """Default historical inputs and the assumptions seeded from them."""
    history = default_history(N_HIST)
    return {
        "n_hist": N_HIST, "n_fwd": N_FWD,
        "fields": [{"key": k, "default_latest": d, "step": s} for k, d, s in HISTORICAL_FIELDS],
        "history": history,
        "assumption_keys": ASSUMPTION_KEYS,
        "seeded_assumptions": to_json(seed_assumptions(ltm_from_history(history))),
        "money": Money(),
    }


@router.post("/seed", response_model=SeedResponse)
def post_seed(req: HistoryRequest):
    """LTM year, historical ratios and seeded assumptions for given historicals."""
    history, n = _checked_history(req.history)
    ltm = ltm_from_history(history)
    return {
        "ltm": to_json(ltm),
        "historical_metrics": to_json(historical_metrics(history, n)),
        "seeded_assumptions": to_json(seed_assumptions(ltm, req.money.unit)),
        "money": req.money,
    }


@router.post("/run", response_model=ForecastRunResponse)
@simulation_slot
def post_run(req: ForecastRunRequest):
    """Run the 3-statement model, optionally with the simulation overlay."""
    history, _ = _checked_history(req.history)
    unknown = set(req.assumptions) - set(ASSUMPTION_KEYS)
    missing = set(ASSUMPTION_KEYS) - set(req.assumptions)
    if unknown or missing:
        raise HTTPException(422, f"assumptions: unknown {sorted(unknown)}, missing {sorted(missing)}")
    lengths = {len(v) for v in req.assumptions.values()}
    if len(lengths) != 1 or not 1 <= min(lengths) <= 10:
        raise HTTPException(422, "every assumption needs the same number of forecast years (1-10)")
    n_fwd = lengths.pop()

    ltm = ltm_from_history(history)
    assumptions = assumptions_from_grid(req.assumptions, n_fwd)
    with model_timer("forecast.run"):
        fwd = run_3_statement_model(ltm, assumptions)
    gap0 = opening_bs_gap(ltm)
    model_gaps = [y.balance_check - gap0 for y in fwd]
    tolerance = in_unit(BALANCE_TOLERANCE, req.money.unit)

    simulation = None
    if req.simulate:
        with model_timer("forecast.simulation"):
            paths = run_forecast_simulation(ltm, assumptions, n=req.n_sim)
        simulation = {"n": req.n_sim, **to_json(simulation_summary(fwd, paths))}
        simulation.pop("ebitda_final_median", None)

    return {
        "ltm": to_json(ltm),
        "years": to_json(fwd),
        "revenue_cagr": to_json(revenue_cagr(ltm, fwd)) if ltm.revenue > 0 else None,
        "opening_balance_gap": gap0,
        "forecast_balance_gaps": model_gaps,
        "balanced": abs(gap0) <= tolerance and all(abs(g) <= tolerance for g in model_gaps),
        "simulation": simulation,
        "money": req.money,
    }
