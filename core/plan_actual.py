"""Plan vs actual (PLAN.md 2.7): any deal's plan against what happened.

The plan is a deal -- a saved deal's inputs and Settings -- run through the
same deal model and simulation as the deal and Monte Carlo screens. The
actuals are what the business reported, year by year, and optionally how the
investment ended. Nothing here is a new model: it runs the existing ones and
lines their answers up against the figures the user typed in.

- **Years.** Up to the plan's hold; a deal still held compares the years it
  has. Any figure may be missing (``None``), and its variance is then missing.
- **Exit.** Optional. An exit after fewer years than planned is compared
  with the plan rerun for that hold (the same deal sold earlier), not with
  the plan's own exit.
- **Attribution.** An exact split of actual minus planned exit equity into
  exit EBITDA, exit multiple and net debt (finding 5's split, applied to the
  plan). With leases the plan values EBITDA plus its lease add-back, so the
  actual EBITDA gets the same add-back: like is compared with like.
- **Where the actual IRR landed.** The plan simulated around its own
  assumptions -- its growth, exit multiple, rate and gross margin as the
  means, Settings' Monte Carlo spreads around them.

Money comes in and goes out in millions (``core.money``); the router converts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence

import numpy as np

from core.deal import DealInputs, build_lbo_params
from core.montecarlo import MCInputs, build_sim_params
from core.money import to_millions
from lbo_engine.model import run_lbo
from simulation.vectorized_simulation import run_vectorized_simulation_full

# The figures a year of actuals may carry, in the order the screens show them
ACTUAL_LINES = ("revenue", "ebitda", "net_income", "fcf", "total_debt")

# Money in a plan-vs-actual answer (api/routers/backtesting.py builds it). Not
# derivable from names: "exit_multiple" is a multiple in the returns blocks
# but money under attribution.
PLAN_ACTUAL_MONEY_KEYS = frozenset(
    {f"{side}_{line}" for side in ("plan", "actual", "variance") for line in ACTUAL_LINES}
    | {"entry_equity", "exit_ebitda", "exit_ev", "net_debt_at_exit", "exit_equity",
       "lease_addback", "attribution"}
)


class PlanActualMismatch(ValueError):
    """Actuals the plan can't be compared with. The message is for the user
    and names no figure from the deal, but it is still returned, never logged."""


@dataclass(frozen=True)
class ActualExit:
    exit_ev: float
    net_debt_at_exit: float
    sponsor_equity_entry: float
    # What the sponsor actually made, when known (dividends, fees and timing
    # make it differ from equity in and out); else computed from those two
    moic: Optional[float] = None
    irr: Optional[float] = None          # % as a number, like every API input


@dataclass(frozen=True)
class Actuals:
    """What happened: one mapping per year (``ACTUAL_LINES`` to a figure or
    None), oldest first from the plan's year 1, and the exit if there was one."""
    years: tuple
    exit: Optional[ActualExit] = None


def actuals_in_millions(actuals: Actuals, unit: str) -> Actuals:
    """The actuals with money in millions, the unit the engine runs in."""
    def m(v):
        return None if v is None else to_millions(v, unit)
    years = tuple({line: m(year.get(line)) for line in ACTUAL_LINES} for year in actuals.years)
    ex = actuals.exit
    if ex is not None:
        ex = replace(ex, exit_ev=m(ex.exit_ev), net_debt_at_exit=m(ex.net_debt_at_exit),
                     sponsor_equity_entry=m(ex.sponsor_equity_entry))
    return Actuals(years=years, exit=ex)


def _run(deal: DealInputs, cfg: Mapping):
    """The deal model without its sensitivity grid, which nothing here shows."""
    return run_lbo(replace(build_lbo_params(deal, cfg), compute_sensitivity=False))


def plan_irr_paths(deal: DealInputs, cfg: Mapping, hold: int, n: int, seed: int = 42) -> np.ndarray:
    """The plan simulated around its own assumptions: IRR per path (fractions)."""
    mc = MCInputs(
        n=n, ebitda=deal.ebitda, entry_mult=deal.entry_mult, hold=hold,
        growth_mean=deal.growth, growth_std=float(cfg["mc_growth_std"]),
        exit_mean=deal.exit_mult, exit_std=float(cfg["mc_exit_std"]),
        rate_mean=deal.base_rate, rate_std=float(cfg["mc_rate_std"]),
        gm_mean=deal.gross_margin, gm_std=float(cfg["mc_gm_std"]),
    )
    params = build_sim_params(mc, replace(deal, hold=hold), cfg)
    return np.asarray(run_vectorized_simulation_full(params, seed=seed).irr, dtype=float)


def _plan_years(run, n_years: int) -> list[dict]:
    om, cf, debt = run.operating_model, run.cash_flow, run.debt_schedule
    return [{
        "plan_revenue": om.revenue[i],
        "plan_ebitda": om.ebitda[i],
        "plan_net_income": om.net_income[i],
        # Free cash flow after interest and tax, before debt repayment, as a
        # reported FCF is: the engine's levered FCF, whose scheduled
        # repayments the debt model takes afterwards
        "plan_fcf": cf.levered_fcf[i],
        "plan_total_debt": debt.total_ending_debt[i],
    } for i in range(n_years)]


def _variance(actual: Optional[float], plan: float) -> Optional[float]:
    return None if actual is None else actual - plan


def _irr_from_moic(moic: Optional[float], years: int) -> Optional[float]:
    if moic is None or moic <= 0:
        return None if moic is None else -1.0
    return moic ** (1.0 / years) - 1.0


def _returns(run) -> dict:
    r = run.returns
    return {
        "irr": r.irr, "moic": r.moic, "entry_equity": r.entry_equity,
        "exit_ebitda": r.exit_ebitda, "exit_multiple": r.exit_multiple, "exit_ev": r.exit_ev,
        "net_debt_at_exit": r.net_debt_at_exit,
        # Before any management dilution, as the actual side is
        "exit_equity": r.exit_ev - r.net_debt_at_exit,
    }


def attribution(plan: Mapping, actual: Mapping) -> dict:
    """Actual minus planned exit equity, split exactly in three:

        exit EBITDA   = (actual EBITDA - plan EBITDA) x plan multiple
        exit multiple = (actual EV - plan EV) - the EBITDA part
                      = (actual multiple - plan multiple) x actual EBITDA
        net debt      = plan net debt - actual net debt

    The multiple part is written as the remainder of the EV gap so the three
    add up to the cent even after the engine rounds its EV.
    """
    ebitda = (actual["exit_ebitda"] - plan["exit_ebitda"]) * plan["exit_multiple"]
    return {
        "exit_ebitda": ebitda,
        "exit_multiple": (actual["exit_ev"] - plan["exit_ev"]) - ebitda,
        "net_debt": plan["net_debt_at_exit"] - actual["net_debt_at_exit"],
    }


def _margins(revenue: Sequence, ebitda: Sequence) -> list:
    return [None if r is None or e is None or r <= 0 else e / r for r, e in zip(revenue, ebitda)]


def compare(deal: DealInputs, cfg: Mapping, actuals: Actuals, n: int, seed: int = 42) -> dict:
    """Plan against actuals for a deal and Settings, money in millions.

    Returns the comparison plus ``irr_paths``, the simulated IRRs the caller
    turns into a histogram.
    """
    hold = int(deal.hold)
    n_years = len(actuals.years)
    if not 1 <= n_years <= hold:
        raise PlanActualMismatch(
            f"The plan runs {hold} years: enter actual results for 1 to {hold} years, "
            f"or lengthen the deal's holding period.")
    ex = actuals.exit
    exit_year = n_years if ex is not None else hold
    last_ebitda = actuals.years[-1].get("ebitda")
    if ex is not None and last_ebitda is None:
        raise PlanActualMismatch(
            "The exit can only be split into its parts with the exit year's EBITDA: "
            "enter EBITDA for the last year.")

    plan_run = _run(deal, cfg)
    plan_exit = plan_run if exit_year == hold else _run(replace(deal, hold=exit_year), cfg)
    paths = plan_irr_paths(deal, cfg, exit_year, n, seed)

    years = []
    for i, (plan_year, actual) in enumerate(zip(_plan_years(plan_run, n_years), actuals.years)):
        row = {"year_index": i + 1, **plan_year}
        for line in ACTUAL_LINES:
            row[f"actual_{line}"] = actual.get(line)
            row[f"variance_{line}"] = _variance(actual.get(line), plan_year[f"plan_{line}"])
        years.append(row)

    plan = {**_returns(plan_exit),
            "irr_mean": float(np.mean(paths)),
            "irr_p5": float(np.percentile(paths, 5)),
            "irr_p95": float(np.percentile(paths, 95))}
    # What a lease adds to the EBITDA a multiple is applied to (PLAN.md 2.6):
    # the plan's valuation EBITDA less its operating EBITDA at exit
    lease_addback = plan_exit.returns.exit_ebitda - plan_exit.operating_model.exit_ebitda

    actual_block, split = None, None
    if ex is not None:
        exit_ebitda = last_ebitda + lease_addback
        exit_equity = ex.exit_ev - ex.net_debt_at_exit
        moic = ex.moic if ex.moic is not None else (
            exit_equity / ex.sponsor_equity_entry if ex.sponsor_equity_entry > 0 else None)
        irr = ex.irr / 100 if ex.irr is not None else _irr_from_moic(moic, exit_year)
        actual_block = {
            "irr": irr, "moic": moic, "irr_given": ex.irr is not None, "moic_given": ex.moic is not None,
            "entry_equity": ex.sponsor_equity_entry, "exit_ebitda": exit_ebitda,
            "exit_multiple": ex.exit_ev / exit_ebitda if exit_ebitda > 0 else None,
            "exit_ev": ex.exit_ev, "net_debt_at_exit": ex.net_debt_at_exit, "exit_equity": exit_equity,
            # Where the actual IRR landed among the plan's simulated paths (%)
            "percentile": None if irr is None else float(np.mean(paths < irr)) * 100,
        }
        split = attribution(plan, actual_block)

    revenue = [y["plan_revenue"] for y in years]
    return {
        "hold": hold,
        "years_compared": n_years,
        "exit_year": exit_year if ex is not None else None,
        "years": years,
        "plan": plan,
        "actual": actual_block,
        "attribution": split,
        "lease_addback": lease_addback,
        "plan_ebitda_margin": _margins(revenue, [y["plan_ebitda"] for y in years]),
        "actual_ebitda_margin": _margins([y["actual_revenue"] for y in years],
                                         [y["actual_ebitda"] for y in years]),
        "irr_paths": paths,
    }
