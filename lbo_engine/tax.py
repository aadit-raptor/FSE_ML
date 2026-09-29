"""
tax.py
------
Tax rules a deal can switch on (PLAN.md 2.5): an interest limit, losses
carried forward, a minimum tax.

Before these, tax was one line in ``operating_model.complete_income_statement``:
a flat rate on positive profit before tax, with a loss simply lost. That stays
exactly what happens when no rule is on (``TaxRules.active`` is False): the
caller does not come here at all, so every deal from before 2.5 -- and the
golden snapshot -- is untouched.

**One function for the deal model and the simulation.** ``tax_schedule`` uses
only ``np.minimum``/``np.maximum`` and arithmetic, so it runs on plain numbers
(one deal) and on arrays of paths (``simulation/vectorized_simulation.py``)
alike. The two can never disagree about what a rule means.

Units: decimals for rates and shares, millions for money (the engine's
unit; ``core/tax.py`` converts a deal's own inputs). The country presets and
their sources live in ``core/tax.py``, never here.

Each year, in this order:

1. **Interest limit.** Net interest (expense less income) is claimed together
   with whatever an earlier year could not deduct. The cap is a share of the
   year's EBITDA, never below an allowance that is always deductible
   (``ebitda_share``), or a fixed amount (``fixed``). What the cap refuses is
   carried forward, without expiry. Net interest *income* is income: it is
   taxed, not capped.
2. **Losses.** Profit for tax is EBIT less the deductible interest. A loss is
   carried forward; a profit absorbs losses carried in, up to an allowance in
   full plus a share of the profit above it.
3. **Tax.** The rate on what is left, raised to the minimum tax on book
   profit (profit before tax, as the P&L shows it) when that is higher.
"""

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

INTEREST_LIMITS = ("none", "ebitda_share", "fixed")


@dataclass(frozen=True)
class TaxRules:
    """What a deal's tax system does beyond a flat rate. Every default is off."""

    interest_limit: str = "none"          # "none" | "ebitda_share" | "fixed"
    interest_limit_share: float = 0.30    # of EBITDA, under "ebitda_share"
    interest_limit_amount: float = 0.0    # M: the fixed cap, or the allowance under the share
    loss_carryforward: bool = False
    loss_limit_share: float = 1.0         # of the profit above the allowance
    loss_limit_amount: float = 0.0        # M: offset in full each year
    minimum_tax_rate: float = 0.0         # of book profit

    def __post_init__(self):
        if self.interest_limit not in INTEREST_LIMITS:
            raise ValueError(f"interest_limit must be one of {', '.join(INTEREST_LIMITS)}, "
                             f"not {self.interest_limit!r}")

    @property
    def active(self) -> bool:
        """Whether any rule changes anything. When not, callers keep the flat
        rate on positive profit -- the exact expression from before 2.5."""
        return (self.interest_limit != "none" or self.loss_carryforward
                or self.minimum_tax_rate > 0)


@dataclass
class TaxSchedule:
    """The tax computation year by year (M). Lists of numbers for a deal,
    lists of arrays of paths in the simulation."""

    years: List[int] = field(default_factory=list)
    taxable_income: list = field(default_factory=list)
    interest_deductible: list = field(default_factory=list)
    interest_carried: list = field(default_factory=list)   # disallowed, at the year's end
    losses_used: list = field(default_factory=list)
    losses_carried: list = field(default_factory=list)     # at the year's end
    regular_tax: list = field(default_factory=list)
    minimum_tax_topup: list = field(default_factory=list)
    taxes: list = field(default_factory=list)


def _plain(x):
    """A number for a number, the array for an array."""
    return float(x) if np.ndim(x) == 0 else x


def tax_schedule(
    ebitda: Sequence,
    ebit: Sequence,
    net_interest: Sequence,
    tax_rate: Sequence[float],
    rules: TaxRules,
) -> TaxSchedule:
    """Tax for each year under ``rules``; see the module docstring for the
    order. ``net_interest`` is interest expense less interest income, so
    profit before tax is ``ebit - net_interest``."""
    out = TaxSchedule()
    interest_carried = 0.0
    losses = 0.0
    for t in range(len(ebit)):
        e, b, i = ebitda[t], ebit[t], net_interest[t]

        # 1. Interest limit
        if rules.interest_limit == "none":
            deductible = i
        else:
            if rules.interest_limit == "ebitda_share":
                cap = np.maximum(np.asarray(e) * rules.interest_limit_share, rules.interest_limit_amount)
            else:
                cap = rules.interest_limit_amount
            cap = np.maximum(cap, 0.0)
            claim = np.maximum(i, 0.0) + interest_carried
            allowed = np.minimum(claim, cap)
            interest_carried = claim - allowed
            deductible = allowed + np.minimum(i, 0.0)

        # 2. Losses
        profit = b - deductible
        if rules.loss_carryforward:
            positive = np.maximum(profit, 0.0)
            allowance = rules.loss_limit_amount
            limit = (np.minimum(positive, allowance)
                     + rules.loss_limit_share * np.maximum(positive - allowance, 0.0))
            used = np.minimum(losses, limit)
            losses = losses - used + np.maximum(-profit, 0.0)
            taxable = positive - used
        else:
            used = 0.0
            taxable = np.maximum(profit, 0.0)

        # 3. Tax, and the minimum tax on book profit
        regular = taxable * tax_rate[t]
        minimum = np.maximum(b - i, 0.0) * rules.minimum_tax_rate
        topup = np.maximum(minimum - regular, 0.0)

        out.years.append(t + 1)
        out.taxable_income.append(_plain(taxable))
        out.interest_deductible.append(_plain(deductible))
        out.interest_carried.append(_plain(interest_carried + 0.0 * i))
        out.losses_used.append(_plain(used + 0.0 * b))
        out.losses_carried.append(_plain(losses + 0.0 * b))
        out.regular_tax.append(_plain(regular))
        out.minimum_tax_topup.append(_plain(topup))
        out.taxes.append(_plain(regular + topup))
    return out
