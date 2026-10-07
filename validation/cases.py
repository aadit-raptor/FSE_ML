"""Validation cases: what the model predicted, what happened, and whether the
outcome is newer than the prediction's data (PLAN.md 4.6).

**Reference transactions** test the deal summary's default risk. The
prediction is what the app answers for the deal entered with only its
filed headline figures -- EBITDA, the entry multiple (value paid over
EBITDA) and the debt's share of the value -- every other input and Setting
at the app's defaults, held for the default hold: year-one coverage, its
rating and S&P's cumulative default rate over the hold
(``core.risk_warnings.credit_view``). What happened: distress (a missed
payment, bankruptcy or restructuring) within the hold of closing. An exit
before then without distress counts as none; a deal still held counts once
the hold has passed, and not before.

**Users' deals** (opted in, with an exit) test the plan's simulated IRR
range and its probability of loss (IRR below zero, MOIC below 1). The plan
is the newest version saved **before any actual figure was first entered**,
run through plan vs actual (``core.plan_actual.compare``) with
``PATHS`` simulated paths; what happened is the actual IRR.

**Newer data.** A case is out of time only when its outcome became known
after everything its prediction was made from: for a reference transaction,
after the newest year in the published tables the default risk reads
(``fit_until``); for a user's deal, when its plan version predates the first
actuals. Anything else is in-sample and reported apart, never as the
headline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Callable, Optional

import numpy as np
from pydantic import ValidationError

from core.config import resolve_config
from core.deal import DealInputs, build_lbo_params, in_millions
from core.money import to_millions
from core.plan_actual import ActualExit, Actuals, PlanActualMismatch, actuals_in_millions, compare
from core.risk_sources import SOURCES
from core.risk_warnings import CREDIT_SOURCES, credit_view
from lbo_engine.model import run_lbo
from library import references
from validation import tags

# Simulated paths per user's deal: enough for its percentile to 0.05 points,
# and a few milliseconds each on the free server
PATHS = 2_000
SEED = 42


@dataclass(frozen=True)
class Case:
    """One prediction and its outcome, holding nothing that names the deal.

    ``origin`` is ``library`` or ``contributed``. A probability check reads
    ``predicted`` (a probability, 0-1) and ``happened``; the IRR range check
    reads ``percentile`` (where the actual IRR fell among the plan's paths,
    0-100) and ``irr_error_pp`` (actual minus planned IRR, percentage points).
    """
    origin: str
    check: str
    groups: dict
    out_of_time: bool
    predicted: Optional[float] = None
    happened: Optional[bool] = None
    percentile: Optional[float] = None
    irr_error_pp: Optional[float] = None


def fit_until() -> int:
    """The newest year of data the default risk is read from: the last year
    of S&P's study sample and the year before Damodaran's January table,
    which is built from the year before it."""
    years = []
    for sid in CREDIT_SOURCES:
        src = SOURCES[sid]
        if src.get("sample"):
            years.append(int(src["sample"]["last_year"]))
        elif src.get("published"):
            years.append(int(str(src["published"])[:4]) - 1)
    return max(years)


def predicted_default(deal: dict) -> dict:
    """The app's coverage and default risk for a reference transaction
    entered with its filed headline figures only (``credit_view``)."""
    value = references.value(deal, "transaction_value")
    ebitda = references.value(deal, "ebitda")
    debt = references.value(deal, "debt")
    inputs = DealInputs(ebitda=ebitda, entry_mult=value / ebitda, debt_pct=debt / value * 100)
    run = run_lbo(replace(build_lbo_params(inputs, resolve_config()), compute_sensitivity=False))
    return credit_view(run)


def library_case(deal: dict, *, today: date) -> Optional[Case]:
    """The default check for one approved reference transaction, or None
    while its hold has not yet passed (or the app reads no default risk)."""
    credit = predicted_default(deal)
    if credit["default_pct"] is None:
        return None
    hold = credit["years"]
    closed = references.closed_year(deal)
    outcome = deal["outcome"]
    horizon_end = closed + hold
    if outcome["kind"] == "distress" and outcome["year"] <= horizon_end:
        happened, known_in = True, outcome["year"]
    elif outcome["kind"] == "held" and today.year < horizon_end:
        return None
    else:
        # Exited (known when it exited, at the latest the hold's end), distress
        # after the hold, or held past it
        known_in = min(outcome["year"], horizon_end) if outcome["kind"] == "success" else horizon_end
        happened = False
    return Case(origin="library", check="default", groups=tags.reference_tags(deal),
                out_of_time=known_in > fit_until(), predicted=credit["default_pct"] / 100,
                happened=happened)


@dataclass(frozen=True)
class Version:
    saved_at: datetime
    inputs: dict
    settings: dict


@dataclass(frozen=True)
class Contribution:
    """A deal whose owner agreed to count it, as stored: the working copy,
    its versions, its actuals and when actuals were first saved."""
    inputs: dict
    settings: dict
    created_at: datetime
    actuals: Optional[dict]
    actuals_first_saved_at: Optional[datetime]
    versions: tuple = field(default_factory=tuple)


class Skipped(Exception):
    """A contribution that can't be checked; the reason is a short code."""


def plan_for(c: Contribution) -> tuple[dict, dict, datetime, bool]:
    """The plan to test: the newest version saved before the first actuals
    (out of time), else the working copy (in-sample)."""
    first = c.actuals_first_saved_at
    before = [v for v in c.versions if first is not None and v.saved_at < first]
    if before:
        v = max(before, key=lambda v: v.saved_at)
        return v.inputs, v.settings, v.saved_at, True
    return c.inputs, c.settings, c.created_at, False


def _read(c: Contribution):
    from api.schemas import DealActuals, DealInputsIn

    inputs, settings, saved_at, out_of_time = plan_for(c)
    try:
        plan = DealInputsIn.model_validate(inputs)
        acts = DealActuals.model_validate(c.actuals or {})
        cfg = resolve_config(settings)
    except (ValidationError, KeyError, ValueError):
        raise Skipped("unreadable") from None
    if acts.exit is None:
        raise Skipped("no_exit")
    if acts.currency != plan.currency:
        raise Skipped("currency")
    return plan, acts, cfg, saved_at, out_of_time


def contributed_cases(c: Contribution, *, usd_per: Callable[[str, date], Optional[float]]) -> list[Case]:
    """The IRR range and loss checks for one opted-in deal. Raises
    ``Skipped`` when it can't be checked (no exit yet, actuals in another
    currency, a plan the model refuses)."""
    plan, acts, cfg, saved_at, out_of_time = _read(c)
    deal, cfg = in_millions(DealInputs(**plan.model_dump()), cfg)
    actuals = actuals_in_millions(Actuals(
        years=tuple(y.model_dump() for y in acts.years), exit=ActualExit(**acts.exit.model_dump()),
    ), acts.unit)
    try:
        answer = compare(deal, cfg, actuals, n=PATHS, seed=SEED)
    except (PlanActualMismatch, ValueError, np.linalg.LinAlgError):
        raise Skipped("mismatch") from None
    actual_irr = answer["actual"]["irr"]
    if actual_irr is None:
        raise Skipped("no_return")
    groups = tags.deal_tags(
        country=plan.country, industry=plan.industry,
        ev=to_millions(plan.ebitda * plan.entry_mult, plan.unit), currency=plan.currency,
        year=saved_at.year, usd_per=lambda ccy: usd_per(ccy, saved_at.date()))
    paths = np.asarray(answer["irr_paths"], dtype=float)
    common = {"origin": "contributed", "groups": groups, "out_of_time": out_of_time}
    return [
        Case(check="irr_range", percentile=answer["actual"]["percentile"],
             irr_error_pp=(actual_irr - answer["plan"]["irr"]) * 100, **common),
        Case(check="loss", predicted=float(np.mean(paths < 0.0)), happened=actual_irr < 0.0, **common),
    ]
