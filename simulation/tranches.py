"""
simulation/tranches.py
----------------------
The simulation's debt schedule for a deal financed tranche by tranche
(PLAN.md 2.4b), vectorised over paths.

It is ``lbo_engine/debt_model.py``'s schedule on arrays of shape (N,): the
same order of steps, the same mandatory repayments, sweep waterfall, PIK
accrual, commitment fee and revolver draw. Two things differ, both on
purpose:

* **No rounding.** The engine rounds to hundredths of a million for its
  tables; the two-bucket simulation never rounded, and neither does this.
* **The rate moves.** A floating facility's rate for a year is
  ``max(reference + shock, floor) + margin``: the path's rate shock moves the
  reference, the floor bites on the moved reference, then the margin is
  added -- the order a loan agreement writes it. A fixed facility's rate
  never moves, whatever the draw.

The simulation's operating model and cash flow are the two-bucket path's
(``vectorized_simulation._run_tranche_core``); only the debt differs.

Market conventions stay in ``core/debt.py``, which builds these from a deal's
``TrancheSpec`` list (``simulation_tranches``): this module sees decimals and
money in millions, never a reference-rate name.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class SimulationTranche:
    """One facility, as the simulation needs it. Decimals and millions.

    ``rates`` is the rate a year: the **reference** rate for a floating
    facility (``floor`` and ``margin`` are applied to it path by path), the
    all-in coupon for a fixed one. A path shorter than the hold repeats its
    last year, as ``core.debt.TrancheSpec.rate_in_year`` does.

    The amortisation fields mean what they mean on ``lbo_engine``'s
    ``Tranche``, and ``core.debt`` fills both from the same code.
    """

    name: str
    amount: float                          # drawn at close (M)
    rates: tuple = (0.0,)
    floating: bool = False
    floor: float = 0.0
    margin: float = 0.0
    maturity_years: int = 7
    amort_type: str = "bullet"             # "bullet" | "amortizing" | "custom"
    amort_pct: float = 0.0                 # of the original principal a year
    amort_schedule: tuple = ()             # M a year, for "custom"
    sweep: bool = False
    sweep_share: float = 1.0               # of the cash available
    sweep_priority: int = 99
    pik_share: float = 0.0                 # of the coupon, accrued to principal
    commitment: Optional[float] = None     # facility limit; None = nothing undrawn
    commitment_fee_pct: float = 0.0        # a year, on the undrawn commitment
    allow_redraw: bool = False             # draws to cover a cash shortfall
    upfront_fee: float = 0.0               # M, paid at close by the sponsor


def rate_in_year(t: SimulationTranche, shock: np.ndarray, y: int):
    """The facility's rate in year ``y`` (0-indexed) on every path: an (N,)
    array for a floating facility, a plain number for a fixed one.

    Computed one year at a time on purpose. Holding every facility's rate for
    every path and year at once is twelve (N, hold) arrays at the API's caps
    (12 facilities, 100,000 paths, a 15-year hold): about 190 MB more than
    the two-bucket path, on a free server with 512 MB.
    """
    base = t.rates[min(y, len(t.rates) - 1)]
    if not t.floating:
        return base
    return np.maximum(base + shock, t.floor) + t.margin


def tranche_rates(t: SimulationTranche, shock: np.ndarray, n_years: int) -> np.ndarray:
    """The facility's rate on every path in every year, shape (N, n_years):
    ``rate_in_year`` stacked, for tests and inspection."""
    return np.column_stack([np.broadcast_to(rate_in_year(t, shock, y), shock.shape)
                            for y in range(n_years)]).astype(float)


def _scheduled(t: SimulationTranche, year: int, owed: np.ndarray) -> np.ndarray:
    """What the agreement says is due in ``year`` (1-indexed), before the cap
    at what is owed -- ``Tranche.mandatory_repayment`` on arrays."""
    if t.amort_type == "amortizing":
        return np.full_like(owed, t.amount * t.amort_pct)
    if t.amort_type == "custom":
        due = t.amort_schedule[year - 1] if year <= len(t.amort_schedule) else 0.0
        return np.full_like(owed, due)
    return owed if year == t.maturity_years else np.zeros_like(owed)


@dataclass
class ScheduleResult:
    interest: np.ndarray        # (N, n_years): coupon plus commitment fees, the P&L line
    non_cash: np.ndarray        # (N, n_years): the part of it that accrued (PIK)
    ending_debt: np.ndarray     # (N,): every facility's balance after the last year
    ending_cash: np.ndarray     # (N,): the balance sheet's cash after the last year
    # (N, n_years): every facility's balance at the start of each year, when
    # asked for (``track_debt``, the distress predictor, PLAN.md 5.3)
    beginning_debt: Optional[np.ndarray] = None


def run_tranche_schedule(
    tranches: Sequence[SimulationTranche],
    shock: np.ndarray,
    fcf: np.ndarray,
    minimum_cash: float,
    track_debt: bool = False,
) -> ScheduleResult:
    """The year-by-year schedule on every path at once.

    ``fcf`` is levered free cash flow before any principal, shape
    (N, n_years); ``shock`` is each path's move in reference rates, shape
    (N,). Each step matches ``lbo_engine.debt_model.run_debt_model`` in
    standard mode.
    """
    N, n_years = fcf.shape
    balances = [np.full(N, t.amount, dtype=float) for t in tranches]
    cash = np.full(N, minimum_cash, dtype=float)
    interest = np.zeros((N, n_years))
    non_cash = np.zeros((N, n_years))
    beginning_debt = np.zeros((N, n_years)) if track_debt else None
    sweep_order = sorted((i for i, t in enumerate(tranches) if t.sweep),
                         key=lambda i: tranches[i].sweep_priority)

    for y in range(n_years):
        year = y + 1
        beginning = balances
        if beginning_debt is not None and tranches:
            beginning_debt[:, y] = sum(beginning)
        # The coupon splits into what accrues and what is paid; the fee on the
        # undrawn commitment is a cash cost beside it
        pik, mandatory = [], []
        for i, t in enumerate(tranches):
            coupon = beginning[i] * rate_in_year(t, shock, y)
            fee = (np.maximum(t.commitment - beginning[i], 0.0) * t.commitment_fee_pct
                   if t.commitment is not None else 0.0)
            pik.append(coupon * t.pik_share)
            interest[:, y] += coupon + fee
            non_cash[:, y] += pik[i]
            # What is owed at maturity includes what the year accrued
            owed = beginning[i] + pik[i]
            mandatory.append(np.minimum(_scheduled(t, year, owed), owed))
        total_mandatory = sum(mandatory) if tranches else np.zeros(N)

        cash_in_hand = cash + fcf[:, y]
        available = np.maximum(cash_in_hand - total_mandatory - minimum_cash, 0.0)

        # The waterfall: each facility takes at most its share of the cash
        # available, capped by what is left and by its own balance
        remaining = available
        swept = [np.zeros(N) for _ in tranches]
        for i in sweep_order:
            take = np.minimum(np.minimum(available * tranches[i].sweep_share, remaining),
                              beginning[i] - mandatory[i])
            swept[i] = np.maximum(take, 0.0)
            remaining = remaining - swept[i]

        ending = [np.maximum(beginning[i] - mandatory[i] - swept[i] + pik[i], 0.0)
                  for i in range(len(tranches))]

        # A committed facility that may redraw funds a shortfall, in list order
        shortfall = minimum_cash - (cash_in_hand - total_mandatory)
        for i, t in enumerate(tranches):
            if not (t.allow_redraw and t.commitment is not None):
                continue
            draw = np.minimum(np.maximum(shortfall, 0.0), np.maximum(t.commitment - ending[i], 0.0))
            ending[i] = ending[i] + draw
            shortfall = shortfall - draw

        # Unswept cash stays on the balance sheet, as in the deal model. A
        # shortfall nobody funds is still restored to the minimum (finding 11).
        cash = minimum_cash + remaining
        balances = ending

    return ScheduleResult(
        interest=interest,
        non_cash=non_cash,
        ending_debt=sum(balances) if tranches else np.zeros(N),
        ending_cash=cash,
        beginning_debt=beginning_debt,
    )
