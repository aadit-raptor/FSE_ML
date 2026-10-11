"""Live sliders: the simulation itself on a small, fixed set of paths (PLAN.md 5.8).

The Live step used to ask a neural network trained on one fixed deal. The
simulation is fast enough to answer for itself: a few thousand paths take a
few milliseconds, and they are the deal's own paths -- its facilities, tax
rules, leases, fees and Settings, exactly as Run Monte Carlo treats them. So
there is no trained range and nothing to warn about; what is left is sampling
error, which the answer states (``median_interval``).

- **The same draws every time** (``LIVE_SEED``): moving a slider reruns the
  same paths with the new assumption, so the estimate moves smoothly with the
  slider instead of jumping with the sample.
- **A bounded amount of work** (``live_paths``): ``LIVE_PATHS`` for an
  ordinary deal, fewer for a long hold with many facilities and interest
  passes, never under ``MIN_LIVE_PATHS``. A live estimate holds no simulation
  slot, so its size is its own limit.

No model logic lives here: the deal is changed at its inputs (``with_sliders``)
and then takes the ordinary road, ``core.montecarlo.build_sim_params`` and
``simulation.vectorized_simulation.run_vectorized_simulation_full``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Mapping, Optional

import numpy as np

from core.accounting import lease_terms, valuation_ebitda
from core.deal import DealInputs
from core.debt import shift_references, total_debt
from core.montecarlo import MCInputs, build_sim_params, mc_in_millions, probability_of_loss
from simulation.vectorized_simulation import run_vectorized_simulation_full

LIVE_PATHS = 8_000
MIN_LIVE_PATHS = 2_000
# Hold x facilities x interest passes up to which a deal gets LIVE_PATHS (the
# default deal is 5 x 2 x 2 = 20); beyond it the paths shrink in proportion
WORK_BUDGET = 200
# Any fixed number: the same draws for every estimate
LIVE_SEED = 20261011
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
# Two-sided 95%: the normal quantile behind the median's interval
Z_95 = 1.959964
# A debt share this close to the deal's own leaves its facilities untouched
SAME_SHARE = 1e-9


@dataclass(frozen=True)
class LiveSliders:
    """The six assumptions the Live step moves, in the page's units (per cent, x)."""
    growth_mean: float
    exit_mean: float
    interest_mean: float
    gross_margin_mean: float
    debt_pct: float
    exit_std: float


def live_paths(hold: int, facilities: int, passes: int) -> int:
    """How many paths a live estimate runs: ``LIVE_PATHS``, fewer once the
    deal's size (years x facilities x interest passes; a deal sized by
    percentages has two) passes ``WORK_BUDGET``, never under
    ``MIN_LIVE_PATHS``."""
    work = max(int(hold), 1) * max(int(facilities), 2) * max(int(passes), 1)
    return int(min(LIVE_PATHS, max(MIN_LIVE_PATHS, LIVE_PATHS * WORK_BUDGET // work)))


def debt_share(mc: MCInputs, deal: DealInputs) -> float:
    """Total debt drawn at close as a share of entry value (%), whichever way
    the deal sizes it: the web's ``debtShareOfEv``."""
    if not deal.tranches:
        return deal.debt_pct
    leases = lease_terms(mc.ebitda, deal.accounting_standard, deal.lease_view,
                         deal.lease_cost, deal.lease_liability)
    entry_ev = valuation_ebitda(leases) * mc.entry_mult
    return total_debt(deal.tranches) / entry_ev * 100 if entry_ev > 0 else 0.0


def sliders_of(mc: MCInputs, deal: DealInputs) -> LiveSliders:
    """The deal's own values: where the sliders start."""
    return LiveSliders(growth_mean=mc.growth_mean, exit_mean=mc.exit_mean, interest_mean=mc.rate_mean,
                       gross_margin_mean=mc.gm_mean, debt_pct=debt_share(mc, deal), exit_std=mc.exit_std)


def _with_debt_share(mc: MCInputs, deal: DealInputs, share: float) -> DealInputs:
    """The deal with its debt at ``share`` of entry value (%). A deal sized by
    percentages takes it as its own input; a listed structure keeps its shape
    and every facility (and its repayment schedule) is scaled alike. A
    structure with nothing drawn has nothing to scale."""
    if not deal.tranches:
        return replace(deal, debt_pct=share)
    own = debt_share(mc, deal)
    if own <= 0 or abs(share - own) <= SAME_SHARE:
        return deal
    factor = share / own
    return replace(deal, tranches=tuple(
        replace(t, amount=t.amount * factor, amort_schedule=tuple(a * factor for a in t.amort_schedule))
        for t in deal.tranches))


def with_sliders(mc: MCInputs, deal: DealInputs, sliders: LiveSliders) -> tuple[MCInputs, DealInputs, float]:
    """The Monte Carlo inputs and the deal with the sliders applied, and how
    far the rate slider sits from the rail's rate mean (percentage points):
    a listed structure's floating references move by that much, as a scenario
    moves them (``core.montecarlo.apply_scenario``)."""
    moved = replace(mc, growth_mean=sliders.growth_mean, exit_mean=sliders.exit_mean, exit_std=sliders.exit_std,
                    rate_mean=sliders.interest_mean, gm_mean=sliders.gross_margin_mean)
    return moved, _with_debt_share(mc, deal, sliders.debt_pct), sliders.interest_mean - mc.rate_mean


def live_params(mc: MCInputs, deal: DealInputs, cfg: Mapping, sliders: Optional[LiveSliders] = None):
    """The simulation's parameters for a live estimate: the deal's own
    (money in its own unit here; converted to millions as every run is), with
    the sliders applied and the path count ``live_paths`` gives."""
    rate_shift = 0.0
    if sliders is not None:
        mc, deal, rate_shift = with_sliders(mc, deal, sliders)
    mc, deal, cfg = mc_in_millions(mc, deal, cfg)
    paths = live_paths(mc.hold, len(deal.tranches), cfg["mc_n_passes"])
    params = build_sim_params(replace(mc, n=paths), deal, cfg)
    if params.tranches and rate_shift:
        params.tranches = shift_references(params.tranches, rate_shift / 100)
    return params


def median_interval(irr: np.ndarray) -> tuple[float, float]:
    """A 95% interval for the median from the paths themselves: the order
    statistics ``Z_95 * sqrt(n) / 2`` ranks either side of the middle (the
    sign-test interval, which assumes nothing about the distribution)."""
    ordered = np.sort(irr[np.isfinite(irr)])
    n = len(ordered)
    if n == 0:
        return float("nan"), float("nan")
    reach = Z_95 * math.sqrt(n) / 2
    low = max(int(math.floor(n / 2 - reach)), 0)
    high = min(int(math.ceil(n / 2 + reach)), n - 1)
    return float(ordered[low]), float(ordered[high])


def summarize(sim, hurdle_pct: float) -> dict:
    """What the Live step shows of a run: IRR percentiles, mean and spread,
    the shares of paths above the hurdle, losing money and losing everything,
    and the median's interval. A path with no IRR is left out of the
    percentiles, as the Monte Carlo summary leaves it out."""
    irr = np.asarray(sim.irr, dtype=float)
    prediction = {f"irr_p{q}": float(np.nanpercentile(irr, q)) for q in PERCENTILES}
    prediction.update(irr_mean=float(np.nanmean(irr)), irr_std=float(np.nanstd(irr)),
                      p_above_hurdle=float((irr > hurdle_pct / 100).mean()),
                      p_loss=float(probability_of_loss(sim.moic)), p_wipeout=float(sim.wipeout_rate))
    return {"paths": int(len(irr)), "hurdle": hurdle_pct / 100, "prediction": prediction,
            "median_interval": list(median_interval(irr))}


def live_estimate(mc: MCInputs, deal: DealInputs, cfg: Mapping, sliders: Optional[LiveSliders] = None) -> dict:
    """The Live step's answer for a deal: ``summarize`` of the simulation on
    ``live_paths`` seeded paths, with the sliders applied (None: the deal's
    own assumptions)."""
    sim = run_vectorized_simulation_full(live_params(mc, deal, cfg, sliders), seed=LIVE_SEED)
    return summarize(sim, mc.hurdle)
