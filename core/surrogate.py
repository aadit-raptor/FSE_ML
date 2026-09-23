"""Live-slider surrogate: model features and reliability checks.

Moved from app.py (render_surrogate_live). Importing this module does not load
torch; callers load the model themselves (ml.surrogate.predict).
"""
from typing import Mapping

from core.deal import DealInputs
from core.montecarlo import MCInputs

# Near the wipeout cliff the true 5th percentile falls steeply toward the -100%
# floor and the network smooths over it. Measured on 400 random deals:
# 5th-percentile error was 0.2pp median below 1% wipeout, but up to 23pp around
# 5%. The surrogate's own wipeout prediction is accurate, and flagging at >= 2%
# caught every error above 3pp while flagging 12% of deals.
TAIL_UNRELIABLE_WIPEOUT = 0.02


def training_terms(mc: MCInputs, deal: DealInputs, cfg: Mapping, fixed: Mapping):
    """Every term held fixed while the surrogate was trained, beside this deal's value.

    Each item: term, value, model_value, unit ("multiple", "years", "percent",
    "money") and decimals -- rates as fractions, so 0.25 is 25%. "money" is in
    the deal's currency and unit; the training deal's is zero, so it compares
    in any currency.
    """
    terms = [
        ("entry multiple", mc.entry_mult, fixed["entry_multiple"], "multiple", 1),
        ("holding period", mc.hold, fixed["holding_period"], "years", 0),
        ("opex / revenue", deal.opex / 100, fixed["opex_pct"], "percent", 1),
        ("tax rate", deal.tax / 100, fixed["tax_rate"], "percent", 1),
        ("senior / total debt", deal.senior_pct / 100, fixed["senior_pct"], "percent", 0),
        ("mezz spread", deal.mezz_spread / 100, fixed["mezz_spread"], "percent", 2),
        ("interest rate std dev", mc.rate_std / 100, fixed["interest_std"], "percent", 2),
        ("transaction fees", cfg["tx_fee_pct"] / 100, fixed["transaction_fees_pct"], "percent", 1),
        ("financing fees", cfg["fin_fee_pct"] / 100, fixed["financing_fees_pct"], "percent", 1),
        ("other uses", cfg["other_uses"], fixed["other_uses"], "money", 0),
    ]
    return [{"term": name, "value": yours, "model_value": model, "unit": unit,
             "decimals": decimals}
            for name, yours, model, unit, decimals in terms]


def training_term_differences(mc: MCInputs, deal: DealInputs, cfg: Mapping, fixed: Mapping):
    """Deal terms that differ from the fixed deal the surrogate was trained on."""
    return [t for t in training_terms(mc, deal, cfg, fixed)
            if abs(float(t["value"]) - float(t["model_value"])) > 1e-6]


def surrogate_features(mc: MCInputs, deal: DealInputs, *, growth_mean, exit_mean,
                       interest_mean, gross_margin_mean, debt_pct, exit_std):
    """Keyword arguments for SurrogatePredictor.predict from the live sliders.

    Slider values are in the page's units (percent, x); the rest of the
    features come from the Monte Carlo and deal inputs.
    """
    return dict(
        growth_mean=growth_mean / 100, growth_std=mc.growth_std / 100,
        exit_mean=exit_mean, exit_std=exit_std,
        interest_mean=interest_mean / 100,
        gross_margin_mean=gross_margin_mean / 100, gross_margin_std=mc.gm_std / 100,
        da_pct=deal.da / 100, capex_pct=deal.capex / 100,
        nwc_pct=deal.nwc / 100, debt_pct=debt_pct / 100,
    )


def tail_unreliable(p_wipeout) -> bool:
    """Whether the surrogate's 5th percentile should be withheld."""
    return p_wipeout >= TAIL_UNRELIABLE_WIPEOUT
