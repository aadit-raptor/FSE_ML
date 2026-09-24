"""Deal wizard: capital structure, sources & uses, and the LBO run.

Moved from app.py (run_deal, _entry_costs, _bridge_steps and the calculations
inside page_deal_inputs). Inputs use the same units as the wizard: percentages
as numbers like 60.0, money in the deal's currency and unit (core/money.py;
US dollar millions unless the deal says otherwise).
"""
from dataclasses import dataclass, replace
from typing import Mapping, Optional

from core.money import DEFAULT_CURRENCY, DEFAULT_UNIT, to_millions
from lbo_engine.model import LBOParams, run_lbo


@dataclass
class DealInputs:
    """Deal wizard inputs. Defaults match the wizard's first load."""
    ebitda: float = 100.0         # LTM EBITDA (money)
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
    mincash: float = 0.0          # money
    # WSP-style working capital (replaces flat nwc when wsp_mode is on)
    wsp_mode: bool = False
    ar_days: float = 45.0
    inv_days: float = 30.0
    ap_days: float = 60.0
    # What the money figures are counted in. The currency is carried, never
    # calculated with; the engine always runs in millions (in_millions).
    currency: str = DEFAULT_CURRENCY
    unit: str = DEFAULT_UNIT
    # Fiscal year labels (PLAN.md 2.3a), carried like the currency: the model
    # counts years from 1 whatever they say
    fiscal_year_end_month: int = 12
    first_fiscal_year: Optional[int] = None


# Money inputs, in the deal's unit
MONEY_INPUTS = ("ebitda", "mincash")
# Settings that are money amounts (the rest are rates, multiples and counts)
MONEY_SETTINGS = ("other_uses", "def_ebitda", "def_mincash")


def in_millions(d: DealInputs, cfg: Mapping) -> tuple[DealInputs, dict]:
    """The deal and its settings with money in millions, the unit the engine
    rounds in and is pinned to (tests/golden). A deal in thousands or billions
    then gives exactly the millions answer, scaled (core.money.rescale)."""
    moved = {k: to_millions(getattr(d, k), d.unit) for k in MONEY_INPUTS}
    cfg = {**cfg, **{k: to_millions(cfg[k], d.unit) for k in MONEY_SETTINGS if k in cfg}}
    return replace(d, **moved, unit="millions"), cfg


# Every money figure in a deal-model answer (api/routers/deal.py builds it);
# the rest are rates, shares, multiples and years
DEAL_MONEY_KEYS = frozenset({
    # returns
    "entry_equity", "exit_ebitda", "net_debt_at_exit", "exit_ev", "gross_exit_equity",
    "mgmt_dilution", "net_exit_equity", "cash_flow_stream", "ebitda_growth_contribution",
    "multiple_expansion_contribution", "deleveraging_contribution",
    # operating model and cash flow
    "revenue", "cogs", "gross_profit", "opex", "ebit", "da", "ebitda", "interest_expense",
    "interest_income", "ebt", "taxes", "net_income", "capex", "delta_nwc", "mandatory_repay",
    "levered_fcf", "cumulative_fcf",
    # debt schedule, totals and per tranche
    "total_beginning_debt", "total_mandatory_repayment", "total_cash_sweep", "total_ending_debt",
    "total_interest_expense", "cash_balance", "available_for_sweep", "beginning_balance",
    "mandatory_repayment", "cash_sweep", "ending_balance",
    # equity bridge
    "entry_costs", "ebitda_growth", "multiple_expansion", "deleveraging", "exit_equity",
    "total_gain", "residual", "value",
})
# Sources and uses: every number but the implied percentages is money
SOURCES_USES_MONEY_KEYS = frozenset({
    "senior_debt", "mezz_debt", "sponsor_equity", "total_sources", "equity_purchase_price",
    "transaction_fees", "financing_fees", "other_uses", "cash_to_balance_sheet", "total_uses", "check",
})


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
    """Fees and other uses funded by sponsor equity at close (money)."""
    return (entry_ev * cfg['tx_fee_pct'] / 100
            + total_debt * cfg['fin_fee_pct'] / 100
            + cfg['other_uses'])


def sources_and_uses(ebitda, entry_mult, senior_x, mezz_x, cfg: Mapping, mincash: float = 0.0) -> dict:
    """Sources & uses of funds for a deal financed with senior + mezz multiples.

    Minimum cash left on the balance sheet at close is a use of funds, as in
    the engine (finding 1).
    """
    entry_ev       = ebitda * entry_mult
    total_debt_abs = (senior_x + mezz_x) * ebitda
    tx_fees        = entry_ev * cfg['tx_fee_pct'] / 100
    fin_fees       = total_debt_abs * cfg['fin_fee_pct'] / 100
    total_uses     = entry_ev + tx_fees + fin_fees + cfg['other_uses'] + mincash
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
        "cash_to_balance_sheet": mincash,
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


MAX_INTEREST_PASSES = 50
INTEREST_TOLERANCE = 0.001   # millions (the engine always runs in millions)


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
        # Iterate the interest circularity until it settles to a thousandth of
        # a million. Three fixed passes left P&L interest up to 0.6M off the debt schedule
        # on the default deal (finding 8).
        minimum_cash=d.mincash, n_iterations=MAX_INTEREST_PASSES,
        interest_tolerance=INTEREST_TOLERANCE,
        sensitivity_exit_multiples=sensitivity_exit_multiples(cfg),
        sensitivity_holding_periods=sensitivity_holding_periods(cfg),
    )


def sensitivity_exit_multiples(cfg: Mapping) -> list:
    """Exit multiple rows for the sensitivity grid, from settings (sens_em_*)."""
    lo, hi, steps = float(cfg['sens_em_min']), float(cfg['sens_em_max']), max(int(cfg['sens_em_steps']), 2)
    return [round(lo + (hi - lo) * i / (steps - 1), 2) for i in range(steps)]


def sensitivity_holding_periods(cfg: Mapping) -> list:
    """Holding period columns for the sensitivity grid, from settings (sens_hp_*)."""
    lo, hi = int(cfg['sens_hp_min']), int(cfg['sens_hp_max'])
    return list(range(max(lo, 1), max(hi, lo) + 1))


def run_deal(d: DealInputs, cfg: Mapping):
    """Run the full LBO model for the deal wizard inputs."""
    return run_lbo(build_lbo_params(d, cfg))


def bridge_steps(br):
    """(axis label, table label, value, is_total, pct_key) per bridge step, from
    a bridge in millions.

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
