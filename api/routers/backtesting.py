"""Backtesting endpoints.

``/plan-vs-actual`` is the backtest (PLAN.md 2.7): any deal as the plan
against what happened. ``/examples`` is the optional example library.
``/deals`` and ``/run`` are the original backtest of the four example deals,
kept as the Streamlit parity record (tests/golden) and for API callers.
"""
import numpy as np
from fastapi import APIRouter, HTTPException

from api.deps import resolve_settings
from api.limits import simulation_slot
from api.schemas import (
    BacktestRequest, BacktestResponse, ExampleLibrary, PlanActualRequest, PlanActualResponse, PreloadedDeal,
)
from api.observability import model_timer
from api.serialize import histogram, to_json
from core.backtesting import (
    BACKTEST_MONEY_KEYS, PRELOADED_DEALS, PRELOADED_MONEY, backtest_in_millions, backtest_summary,
)
from core.deal import DealInputs, in_millions
from core.examples import examples, library_enabled
from core.money import in_unit, rescale
from core.plan_actual import (
    PLAN_ACTUAL_MONEY_KEYS, ActualExit, Actuals, PlanActualMismatch, actuals_in_millions, compare,
)

router = APIRouter(prefix="/backtesting", tags=["backtesting"])


@router.get("/deals", response_model=list[PreloadedDeal])
def get_deals():
    """Historical deals with their entry assumptions and actual results."""
    if not library_enabled():
        return []
    money = {"currency": PRELOADED_MONEY.currency, "unit": PRELOADED_MONEY.unit}
    return [{"name": name, **{k: v for k, v in deal.items() if k in PreloadedDeal.model_fields}, "money": money}
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

    # Money runs in millions, as the deal model does, and comes back in the deal's unit
    entry, actual, actual_exit, cfg = backtest_in_millions(
        entry, actual, req.actual_exit.model_dump(), cfg, req.money.unit)
    with model_timer("backtest.run"):
        bt = backtest_summary(entry, actual, actual_exit, cfg, n=req.n)
    dist = bt.pop("predicted_irr_distribution")
    pred = bt["predicted_ebitda"]
    answer = {
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
    return {**rescale(answer, in_unit(1.0, req.money.unit), BACKTEST_MONEY_KEYS), "money": req.money}


@router.get("/examples", response_model=ExampleLibrary)
def get_examples():
    """The example library: each example deal as a plan and its actuals.
    Empty, and ``enabled`` false, when the library is switched off."""
    return {"enabled": library_enabled(), "examples": examples()}


@router.post("/plan-vs-actual", response_model=PlanActualResponse)
@simulation_slot
def post_plan_vs_actual(req: PlanActualRequest):
    """Compare a deal's plan with its actual results and exit.

    The plan runs through the deal model and is simulated around its own
    assumptions; money comes back in the plan's currency and unit, whatever
    unit the actuals were entered in.
    """
    plan, acts = req.plan, req.actuals
    if acts.currency != plan.currency:
        raise HTTPException(
            422, f"The actuals are in {acts.currency} and the deal in {plan.currency}: "
                 f"enter the actuals in the deal's currency.")
    cfg = resolve_settings(req.settings, check_correlations=True)
    deal, cfg = in_millions(DealInputs(**plan.model_dump()), cfg)
    actuals = actuals_in_millions(Actuals(
        years=tuple(y.model_dump() for y in acts.years),
        exit=ActualExit(**acts.exit.model_dump()) if acts.exit else None,
    ), acts.unit)
    try:
        with model_timer("backtest.plan_vs_actual"):
            answer = compare(deal, cfg, actuals, n=req.n)
    except PlanActualMismatch as exc:
        # The message names no figure, but like every refusal it is answered, not logged
        raise HTTPException(422, str(exc)) from None
    paths = answer.pop("irr_paths")
    answer = {**to_json(answer), "irr_histogram": histogram(paths, req.histogram_bins)}
    return {**rescale(answer, in_unit(1.0, plan.unit), PLAN_ACTUAL_MONEY_KEYS), "money": plan.money()}
