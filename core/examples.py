"""The optional example library (PLAN.md 2.7, grown into 4.5).

The four inception-era deals the backtest used to be limited to, written out
as a plan and its actuals, so they run through plan vs actual exactly like a
user's own saved deal. They are examples, not evidence: unsourced, US, 2006
to 2013, and the screen says so (``web/src/lib/provenance.ts``).

They are part of the reference library, which an administrator can hide
(``library/switch.py``); the endpoints that serve them check the switch, and
every screen and endpoint keeps working without them.
"""
from __future__ import annotations

from core.backtesting import PRELOADED_DEALS, PRELOADED_MONEY

# A backtest example's entry assumption -> the deal input it is
ENTRY_TO_DEAL = {
    "entry_ebitda": "ebitda", "entry_multiple": "entry_mult", "exit_multiple": "exit_mult",
    "holding_period": "hold", "debt_pct": "debt_pct", "senior_pct": "senior_pct",
    "base_rate": "base_rate", "mezz_spread": "mezz_spread", "revenue_growth": "growth",
    "gross_margin": "gross_margin", "opex_pct": "opex", "da_pct": "da", "tax_rate": "tax",
    "capex_pct": "capex", "nwc_pct": "nwc",
}

# Entries that are templates rather than deals
_NOT_EXAMPLES = {"Custom deal (enter manually)"}


def _plan(deal: dict) -> dict:
    plan = {ENTRY_TO_DEAL[k]: v for k, v in deal["entry"].items()}
    plan["hold"] = int(plan["hold"])
    plan.update(currency=PRELOADED_MONEY.currency, unit=PRELOADED_MONEY.unit)
    first = deal["actual_years"][0] if deal["actual_years"] else None
    if first and first >= 1900:
        plan["first_fiscal_year"] = int(first)
    return plan


def _actuals(deal: dict) -> dict:
    lines = deal["actual"]
    n = len(deal["actual_years"])
    ex = deal["actual_exit"]
    return {
        "currency": PRELOADED_MONEY.currency, "unit": PRELOADED_MONEY.unit,
        "years": [{line: float(values[i]) for line, values in lines.items()} for i in range(n)],
        "exit": {k: float(ex[k]) for k in ("exit_ev", "net_debt_at_exit", "sponsor_equity_entry", "moic", "irr")},
    }


def examples() -> list[dict]:
    """Each example as a plan (deal inputs) and its actuals."""
    return [{
        "name": name,
        "description": deal["description"],
        "sector": deal["sector"],
        "geography": deal["geography"],
        "outcome": deal["outcome"],
        "plan": _plan(deal),
        "actuals": _actuals(deal),
    } for name, deal in PRELOADED_DEALS.items() if name not in _NOT_EXAMPLES]
