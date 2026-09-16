"""Backtesting endpoints."""
import numpy as np
from fastapi import APIRouter, HTTPException

from api.deps import resolve_settings
from api.limits import simulation_slot
from api.schemas import BacktestRequest, BacktestResponse, PreloadedDeal
from api.observability import model_timer
from api.serialize import histogram, to_json
from core.backtesting import PRELOADED_DEALS, backtest_summary

router = APIRouter(prefix="/backtesting", tags=["backtesting"])


@router.get("/deals", response_model=list[PreloadedDeal])
def get_deals():
    """Historical deals with their entry assumptions and actual results."""
    return [{"name": name, **{k: v for k, v in deal.items() if k in PreloadedDeal.model_fields}}
            for name, deal in PRELOADED_DEALS.items()]


@router.post("/run", response_model=BacktestResponse)
@simulation_slot
def post_run(req: BacktestRequest):
    """Predict a deal from its entry assumptions and compare with what happened."""
    cfg = resolve_settings(req.settings)
    entry = req.entry.model_dump()
    hold = entry["holding_period"]
    actual = req.actual.model_dump()
    short = [k for k, v in actual.items() if len(v) < hold]
    if short:
        raise HTTPException(422, f"actual results need {hold} years for: {short}")
    actual = {k: v[:hold] for k, v in actual.items()}

    with model_timer("backtest.run"):
        bt = backtest_summary(entry, actual, req.actual_exit.model_dump(), cfg, n=req.n)
    dist = bt.pop("predicted_irr_distribution")
    pred = bt["predicted_ebitda"]
    return {
        **to_json(bt),
        "predicted_irr_p5": float(np.percentile(dist, 5)) * 100,
        "predicted_irr_p95": float(np.percentile(dist, 95)) * 100,
        "irr_histogram": histogram(dist * 100, req.histogram_bins),
        "years": [{
            "year_index": i + 1,
            "predicted_ebitda": pred[i],
            "actual_ebitda": actual["ebitda"][i],
            "ebitda_variance": actual["ebitda"][i] - pred[i],
            "actual_revenue": actual["revenue"][i],
            "actual_fcf": actual["fcf"][i],
            "actual_total_debt": actual["total_debt"][i],
        } for i in range(hold)],
    }
