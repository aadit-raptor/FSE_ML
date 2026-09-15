"""Deal wizard: capital structure, sources & uses, and the LBO run.

Moved from app.py (run_deal, _entry_costs, _bridge_steps and the calculations
inside page_deal_inputs). Inputs use the same units as the wizard: percentages
as numbers like 60.0, money in $M.
"""
from dataclasses import dataclass
from typing import Mapping

from lbo_engine.model import LBOParams, run_lbo


@dataclass
class DealInputs:
    """Deal wizard inputs. Defaults match the wizard's first load."""
    ebitda: float = 100.0         # LTM EBITDA ($M)
    entry_mult: float = 10.0
    exit_mult: float = 11.0
    hold: int = 5                 # years
    growth: float = 5.0           # revenue growth %
    gross_margin: float = 40.0    # %
    opex: float = 18.0            # % of revenue
    tax: float = 25.0             # %
    da: float = 4.0               # % of revenue
    debt_pct: float = 60.0        # total debt / EV %
    senior_pct: float = 70.0      # senior / total debt %
    base_rate: float = 6.5        # senior rate %
    mezz_spread: float = 4.0      # %
    capex: float = 4.0            # % of revenue
    nwc: float = 1.0              # change in NWC % of revenue
    mincash: float = 0.0          # $M
    # WSP-style working capital (replaces flat nwc when wsp_mode is on)
    wsp_mode: bool = False
    ar_days: float = 45.0
    inv_days: float = 30.0
    ap_days: float = 60.0


def capital_structure_from_multiples(ebitda, entry_mult, senior_x, mezz_x):
    """Debt/EV % and senior/total-debt % from senior and mezz debt multiples.

    Returns None for either value the wizard leaves unchanged (no EV).
    """
    entry_ev = ebitda * entry_mult
    total_debt_abs = (senior_x + mezz_x) * ebitda
    if entry_ev > 0:
        debt_pct = min((total_debt_abs / entry_ev) * 100, 99.0)
        senior_pct = (senior_x / (senior_x + mezz_x)) * 100 \
            if (senior_x + mezz_x) > 0 else 70.0
        return debt_pct, senior_pct
    return None, None


def entry_costs(entry_ev, total_debt, cfg: Mapping):
    """Fees and other uses funded by sponsor equity at close ($M)."""
    return (entry_ev * cfg['tx_fee_pct'] / 100
            + total_debt * cfg['fin_fee_pct'] / 100
            + cfg['other_uses'])


def sources_and_uses(ebitda, entry_mult, senior_x, mezz_x, cfg: Mapping) -> dict:
    """Sources & uses of funds for a deal financed with senior + mezz multiples."""
    entry_ev       = ebitda * entry_mult
    total_debt_abs = (senior_x + mezz_x) * ebitda
    tx_fees        = entry_ev * cfg['tx_fee_pct'] / 100
    fin_fees       = total_debt_abs * cfg['fin_fee_pct'] / 100
    total_uses     = entry_ev + tx_fees + fin_fees + cfg['other_uses']
    # Equity is the plug that balances Sources against Uses, fees included
    sponsor_eq     = max(total_uses - total_debt_abs, 0)
    total_sources  = total_debt_abs + sponsor_eq
    check          = total_sources - total_uses
    return {
        "senior_debt": senior_x * ebitda,
        "mezz_debt": mezz_x * ebitda,
        "sponsor_equity": sponsor_eq,
        "total_sources": total_sources,
        "equity_purchase_price": entry_ev,
        "transaction_fees": tx_fees,
        "financing_fees": fin_fees,
        "other_uses": cfg['other_uses'],
        "total_uses": total_uses,
        "check": check,
        "balanced": abs(check) < 1,
    }


def risk_model_inputs(d: DealInputs, senior_x, mezz_x) -> dict:
    """Inputs for the anomaly detector's deal risk score.

    The detector expects the all-in debt rate, so senior and mezz rates are
    blended by amount rather than passing the senior rate.
    """
    total_debt_abs = (senior_x + mezz_x) * d.ebitda
    total_x = senior_x + mezz_x
    blended_rate = ((senior_x * d.base_rate
                     + mezz_x * (d.base_rate + d.mezz_spread)) / total_x
                    if total_x > 0 else d.base_rate)
    return dict(
        entry_mult=d.entry_mult,
        leverage=total_debt_abs / max(d.ebitda, 1e-9),
        growth_pct=d.growth,
        ebitda_margin=d.gross_margin - d.opex + d.da,   # already in %
        rate=blended_rate,
    )


def effective_nwc_pct(d: DealInputs) -> float:
    """Change in NWC as a share of revenue, as the engine reads it.

    WSP mode: the engine has no days-based working-capital input, and it
    reads nwc_pct as the *change* in NWC as a share of that year's revenue
    (cashflow_model: delta_nwc[t] = revenue[t] * nwc_pct[t]).

    wc_from_days() with cogs = revenue * (1 - gross_margin) collapses to
        NWC(t) = revenue(t) * k,
        k = [ar_days + (1 - gm) * (inv_days - ap_days)] / 365
    so the implied change is
        dNWC(t) = k * revenue(t) * g / (1 + g).
    That makes the conversion below exact under the model's own assumptions
    (constant days, constant gross margin, constant growth), not an estimate.
    """
    if d.wsp_mode:
        g  = d.growth / 100
        gm = d.gross_margin / 100
        k  = (d.ar_days + (1 - gm) * (d.inv_days - d.ap_days)) / 365
        return k * g / (1 + g) if (1 + g) != 0 else 0.0
    return d.nwc / 100


def build_lbo_params(d: DealInputs, cfg: Mapping) -> LBOParams:
    return LBOParams(
        entry_ebitda=d.ebitda, entry_multiple=d.entry_mult,
        exit_multiple=d.exit_mult, holding_period=int(d.hold),
        debt_pct=d.debt_pct/100, senior_pct=d.senior_pct/100,
        mezz_spread=d.mezz_spread/100, interest_rate=d.base_rate/100,
        revenue_growth=d.growth/100, gross_margin=d.gross_margin/100,
        opex_pct=d.opex/100, da_pct=d.da/100, tax_rate=d.tax/100,
        capex_pct=d.capex/100, nwc_pct=effective_nwc_pct(d),
        transaction_fees_pct=cfg['tx_fee_pct']/100,
        financing_fees_pct=cfg['fin_fee_pct']/100,
        other_uses=cfg['other_uses'],
        senior_amort_pct=cfg['def_senior_amort']/100,
        minimum_cash=d.mincash, n_iterations=3,
    )


def run_deal(d: DealInputs, cfg: Mapping):
    """Run the full LBO model for the deal wizard inputs."""
    return run_lbo(build_lbo_params(d, cfg))


def bridge_steps(br):
    """(axis label, table label, value, is_total, pct_key) per bridge step.

    Fees at entry are shown only when non-zero, so a fee-free deal keeps the
    original five-step waterfall.
    """
    steps = [("Entry", "Entry equity", br["entry_equity"], True, None)]
    if abs(br["entry_costs"]) > 0.005:
        steps.append(("Fees", "Fees at entry", br["entry_costs"], False,
                      "entry_costs_pct"))
    steps += [
        ("EBITDA\ngrowth", "EBITDA growth", br["ebitda_growth"], False,
         "ebitda_growth_pct"),
        ("Multiple", "Multiple expansion", br["multiple_expansion"], False,
         "multiple_expansion_pct"),
        ("Deleverage", "Deleveraging", br["deleveraging"], False,
         "deleveraging_pct"),
        ("Exit", "Exit equity", br["exit_equity"], True, None),
    ]
    return steps
