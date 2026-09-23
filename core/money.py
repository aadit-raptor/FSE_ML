"""Currency and money units (PLAN.md 2.2).

The model is currency-neutral: every money figure in a deal, a forecast or a
backtest is a plain number in that deal's own currency and unit. A deal in
euros and thousands takes EBITDA as 12,500 (EUR 12.5M) and answers in euro
thousands; the currency code never enters a calculation.

The deal engine (lbo_engine/) rounds money to two decimals and is pinned to
the golden snapshot in millions, so deals, simulations and backtests run in
millions whatever unit they arrive in: inputs go in through ``to_millions``
and the answer comes back through ``rescale``. A deal in thousands therefore
gives exactly the millions answer times a thousand. The forecast model has no
rounding and runs in the company's own unit; its fixed amounts (a balance
check) are written in millions and converted with ``in_unit``.
"""
import math
import re
from dataclasses import dataclass

DEFAULT_CURRENCY = "USD"
DEFAULT_UNIT = "millions"

# How many currency units one money unit is
UNIT_SCALE = {"thousands": 1e3, "millions": 1e6, "billions": 1e9}
MONEY_UNITS = tuple(UNIT_SCALE)

CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


@dataclass(frozen=True)
class Money:
    """What a set of money figures is counted in: an ISO 4217 code and a unit."""
    currency: str = DEFAULT_CURRENCY
    unit: str = DEFAULT_UNIT

    def __post_init__(self):
        if not isinstance(self.currency, str) or not CURRENCY_RE.match(self.currency):
            raise ValueError("currency must be a three-letter ISO 4217 code, such as EUR or INR")
        if self.unit not in UNIT_SCALE:
            raise ValueError(f"unit must be one of {', '.join(MONEY_UNITS)}")


def to_millions(amount: float, unit: str) -> float:
    """``amount`` in ``unit``, in millions: 2,500 thousands is 2.5."""
    return amount * (UNIT_SCALE[unit] / 1e6)


def in_unit(amount_in_millions: float, unit: str) -> float:
    """An amount written in millions, in ``unit``: 0.001 (a thousand) is 1.0 in thousands."""
    return amount_in_millions * (1e6 / UNIT_SCALE[unit])


def round_money(amount: float, decimals_in_millions: int, unit: str) -> float:
    """Round as ``round(amount, decimals)`` would in millions, at the same
    precision in ``unit``: to 0.1M is to the nearest 100 in thousands."""
    return round(amount, decimals_in_millions + round(math.log10(UNIT_SCALE[unit] / 1e6)))


def rescale(value, factor: float, money_keys: frozenset, key: str = ""):
    """``value`` (JSON-like) with the numbers under ``money_keys`` multiplied by
    ``factor``; everything else (rates, multiples, years, text) as it was.

    Used to hand the engine's millions back in the deal's unit. The keys are
    listed per answer (core.deal.DEAL_MONEY_KEYS and so on) rather than
    guessed from names: "exit_multiple" is a multiple in the returns but money
    in a backtest's attribution.
    """
    if key in money_keys:
        return _scale_all(value, factor)
    if isinstance(value, dict):
        return {k: rescale(v, factor, money_keys, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rescale(v, factor, money_keys, key) for v in value]
    return value


def _scale_all(value, factor: float):
    if isinstance(value, dict):
        return {k: _scale_all(v, factor) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scale_all(v, factor) for v in value]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return value * factor
