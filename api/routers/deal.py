"""Deal wizard endpoints."""
from fastapi import APIRouter

from api.deps import resolve_settings
from api.observability import model_timer
from api.schemas import (
    DealRunRequest, DealRunResponse, SourcesUsesRequest, SourcesUsesResponse,
)
from api.serialize import to_json
from core.deal import (
    DealInputs, bridge_steps, capital_structure_from_multiples, run_deal, sources_and_uses,
)

router = APIRouter(prefix="/deal", tags=["deal"])


@router.post("/sources-and-uses", response_model=SourcesUsesResponse)
def post_sources_and_uses(req: SourcesUsesRequest):
    """Sources & uses for a deal financed with senior and mezz debt multiples."""
    cfg = resolve_settings(req.settings)
    su = sources_and_uses(req.ebitda, req.entry_mult, req.senior_x, req.mezz_x, cfg, mincash=req.mincash)
    debt_pct, senior_pct = capital_structure_from_multiples(
        req.ebitda, req.entry_mult, req.senior_x, req.mezz_x)
    return {**to_json(su), "debt_pct": debt_pct, "senior_pct": senior_pct}


@router.post("/run", response_model=DealRunResponse)
def post_run(req: DealRunRequest):
    """Run the full LBO model: operating model, cash flow, debt, returns."""
    cfg = resolve_settings(req.settings)
    with model_timer("deal.run"):
        result = run_deal(DealInputs(**req.inputs.model_dump()), cfg)
    br = result.equity_bridge
    debt = to_json(result.debt_schedule)
    tranches = debt.pop("schedule")
    return {
        "returns": to_json(result.returns),
        "operating_model": to_json(result.operating_model),
        "cash_flow": to_json(result.cash_flow),
        "debt_schedule": debt,
        "tranches": tranches,
        "equity_bridge": to_json(br),
        "bridge_steps": [
            {"key": axis.replace("\n", " "), "label": label, "value": to_json(value),
             "is_total": is_total, "pct_of_gain": to_json(br[pct_key]) if pct_key else None}
            for axis, label, value, is_total, pct_key in bridge_steps(br)
        ],
        "exit_sensitivity": to_json(result.exit_sensitivity),
        "interest_converged": result.interest_converged,
    }
