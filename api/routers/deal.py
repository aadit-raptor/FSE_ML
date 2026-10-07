"""Deal wizard endpoints."""
from fastapi import APIRouter

from api.deps import resolve_settings
from api.observability import model_timer
from api.schemas import (
    DealRunRequest, DealRunResponse, SourcesUsesRequest, SourcesUsesResponse, TaxPresetsResponse,
)
from api.serialize import to_json
from core.deal import (
    DEAL_MONEY_KEYS, SOURCES_USES_MONEY_KEYS, DealInputs, bridge_steps,
    capital_structure_from_multiples, capital_structure_summary, in_millions, lease_summary,
    leases_of, run_deal, sources_and_uses, sources_and_uses_for,
)
from core.model_version import stamp
from core.money import in_unit, rescale, to_millions
from core.risk_warnings import credit_view, risk_warnings
from core.tax import PRESETS as TAX_PRESETS

router = APIRouter(prefix="/deal", tags=["deal"])


@router.post("/sources-and-uses", response_model=SourcesUsesResponse)
def post_sources_and_uses(req: SourcesUsesRequest):
    """Sources & uses for a deal financed with senior and mezz debt multiples.

    Money in and out is in ``money``'s unit; the sums run in millions like the
    deal model's.
    """
    cfg = resolve_settings(req.settings)
    model = stamp(cfg)
    unit = req.money.unit
    cfg = {**cfg, "other_uses": to_millions(cfg["other_uses"], unit)}
    # Leases (PLAN.md 2.6): valued and taken over the way the deal model does
    leases = dict(accounting_standard=req.accounting_standard, lease_view=req.lease_view,
                  lease_cost=req.lease_cost, lease_liability=req.lease_liability)
    if req.tranches:
        # Sized facility by facility: each one is its own source, and their
        # arrangement fees are their own use of funds (PLAN.md 2.4)
        deal, cfg = in_millions(
            DealInputs(ebitda=req.ebitda, entry_mult=req.entry_mult, mincash=req.mincash,
                       unit=unit, tranches=[t.model_dump() for t in req.tranches], **leases),
            resolve_settings(req.settings))
        su = sources_and_uses_for(deal, cfg)
        debt_pct, senior_pct = None, None
    else:
        deal, _ = in_millions(DealInputs(ebitda=req.ebitda, unit=unit, **leases), cfg)
        su = sources_and_uses(to_millions(req.ebitda, unit), req.entry_mult, req.senior_x, req.mezz_x, cfg,
                              mincash=to_millions(req.mincash, unit),
                              leases=leases_of(deal) if (req.lease_cost or req.lease_liability) else None)
        debt_pct, senior_pct = capital_structure_from_multiples(
            req.ebitda, req.entry_mult, req.senior_x, req.mezz_x)
    su = rescale(to_json(su), in_unit(1.0, unit), SOURCES_USES_MONEY_KEYS)
    return {**su, "debt_pct": debt_pct, "senior_pct": senior_pct, "money": req.money, "model": model}


@router.get("/tax-presets", response_model=TaxPresetsResponse)
def get_tax_presets():
    """Country presets for a deal's tax rules, each with its source and the
    date it was checked. A starting point, not advice: every field stays
    editable once applied."""
    return {"presets": [to_json(p) for p in TAX_PRESETS]}


@router.post("/run", response_model=DealRunResponse)
def post_run(req: DealRunRequest):
    """Run the full LBO model: operating model, cash flow, debt, returns.

    Money in and out is in the deal's currency and unit. The engine runs in
    millions (core/money.py), so a deal in thousands answers exactly as the
    same deal in millions, times a thousand.
    """
    cfg = resolve_settings(req.settings)
    model = stamp(cfg)
    deal, cfg = in_millions(DealInputs(**req.inputs.model_dump()), cfg)
    with model_timer("deal.run"):
        result = run_deal(deal, cfg)
    br = result.equity_bridge
    debt = to_json(result.debt_schedule)
    tranches = debt.pop("schedule")
    answer = {
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
        "capital_structure": capital_structure_summary(deal),
        "tax": to_json(result.tax),
        "leases": to_json(lease_summary(deal, result)),
        "risk_warnings": risk_warnings(deal, result),
        "credit": credit_view(result),
    }
    return {**rescale(answer, in_unit(1.0, req.inputs.unit), DEAL_MONEY_KEYS), "money": req.inputs.money(),
            "model": model}
