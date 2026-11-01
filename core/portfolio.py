"""Portfolio tracking (PLAN.md 7.4): owned deals' quarterly results against their plan, in any currency.

Nothing here is a new model. The plan is the deal model's answer for the saved deal (as in plan vs actual,
PLAN.md 2.7, whose lines and definitions a quarter reuses: ``core.plan_actual.ACTUAL_LINES``); a quarter is
compared with it by plain arithmetic, and the reforecast is the same deal model and simulation run again with
one input changed.

- **A quarter of the plan.** The plan is yearly, so a quarter's revenue, EBITDA, net income and free cash flow
  are a quarter of its year's (no seasonality: the screen says so), and its debt is the straight line between the
  plan's debt at the end of the year before (at close, in year 1) and at the end of its year.
- **Drift.** The latest year's results so far against the plan for the same quarters, and the latest debt against
  the plan's debt at that quarter. Revenue or EBITDA below the plan, or debt above it, by more than the holding's
  tolerance (``DEFAULT_TOLERANCE`` %) is an alert. Alerts are ids and figures; the words are the web's.
- **Reforecast.** The last four quarters' EBITDA (``ltm``) against the plan's for the same quarters gives the
  growth the deal has shown since close: with ``t`` years since close, ``1 + g' = (1 + g) x (actual / plan) **
  (1 / t)``. The deal model and the simulation then run the plan with that growth and everything else as
  planned -- the entry price, the debt, the margins, the exit multiple and the hold. At a year end the
  reforecast's EBITDA for that year is the actual year's.
- **Currency.** Each holding stays in its own currency and unit; the dashboard converts at one day's ECB
  reference rates (units per euro), crossed through the euro, and says the rates it used. A currency the ECB
  doesn't quote is left out of the totals and named.

Money is in millions inside (``core.money``); the API converts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Optional, Sequence

import numpy as np

from core.deal import DealInputs
from core.money import UNIT_SCALE, to_millions
from core.plan_actual import ACTUAL_LINES, _returns, _run, plan_irr_paths

# A quarter's figures are plan vs actual's lines (PLAN.md 2.7); the first four are flows, debt is a balance
LINES = ACTUAL_LINES
FLOWS = ("revenue", "ebitda", "net_income", "fcf")
# Below the plan by more than this share (%) is an alert, unless the holding says otherwise
DEFAULT_TOLERANCE = 5.0
# Simulated paths of a reforecast
REFORECAST_PATHS = 20_000

# Money in the answers (api/routers/portfolio.py hands them back in the deal's unit)
DRIFT_MONEY_KEYS = frozenset({"actual", "plan"})
REFORECAST_MONEY_KEYS = frozenset({
    "ltm_ebitda_actual", "ltm_ebitda_plan", "entry_equity", "exit_ebitda", "exit_ev", "net_debt_at_exit",
    "exit_equity", "plan_ebitda", "reforecast_ebitda", "actual_ebitda"})
# A holding's money on the dashboard, each totalled in the reporting currency
BOARD_MONEY_KEYS = ("entry_ev", "entry_equity", "plan_exit_equity", "ltm_revenue", "ltm_ebitda", "latest_debt",
                    "reforecast_exit_equity")


class QuartersRefused(ValueError):
    """Quarters that can't be used as asked. ``code`` is what the screen words; no figure is in it."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class Quarter:
    year: int                                   # the plan's year, from 1
    quarter: int                                # 1 to 4
    figures: Mapping[str, Optional[float]]

    @property
    def index(self) -> int:
        """Quarters since close: year 1's first is 1."""
        return (self.year - 1) * 4 + self.quarter


@dataclass(frozen=True)
class PlanPath:
    """The plan's years (index 0 is year 1) and its debt at close."""
    revenue: Sequence[float]
    ebitda: Sequence[float]
    net_income: Sequence[float]
    fcf: Sequence[float]
    debt: Sequence[float]
    debt_at_close: float


def quarters_from(rows: Sequence[Mapping]) -> tuple[Quarter, ...]:
    """Quarters, oldest first; the same quarter twice is refused."""
    seen = set()
    out = []
    for row in rows:
        key = (int(row["year"]), int(row["quarter"]))
        if key in seen:
            raise QuartersRefused("duplicate_quarter")
        seen.add(key)
        out.append(Quarter(key[0], key[1], {line: row.get(line) for line in LINES}))
    return tuple(sorted(out, key=lambda q: q.index))


def check_hold(quarters: Sequence[Quarter], hold: int) -> None:
    if any(q.year > hold for q in quarters):
        raise QuartersRefused("beyond_hold")


def quarters_in_millions(quarters: Sequence[Quarter], unit: str) -> tuple[Quarter, ...]:
    def m(v):
        return None if v is None else to_millions(v, unit)
    return tuple(replace(q, figures={line: m(v) for line, v in q.figures.items()}) for q in quarters)


def plan_path(run) -> PlanPath:
    om, cf, debt = run.operating_model, run.cash_flow, run.debt_schedule
    return PlanPath(revenue=list(om.revenue), ebitda=list(om.ebitda), net_income=list(om.net_income),
                    fcf=list(cf.levered_fcf), debt=list(debt.total_ending_debt),
                    debt_at_close=debt.total_beginning_debt[0])


def plan_quarter(plan: PlanPath, line: str, year: int, quarter: int) -> float:
    """The plan's figure for one quarter: a quarter of a flow, the straight line for debt."""
    if line == "total_debt":
        start = plan.debt_at_close if year == 1 else plan.debt[year - 2]
        return start + (plan.debt[year - 1] - start) * quarter / 4
    return getattr(plan, line)[year - 1] / 4


def _gap(actual: float, planned: float) -> dict:
    return {"actual": actual, "plan": planned,
            "gap_pct": None if planned == 0 else (actual - planned) / abs(planned) * 100}


def _summed(quarters: Sequence[Quarter], plan: PlanPath, line: str) -> Optional[dict]:
    values = [q.figures.get(line) for q in quarters]
    if not quarters or any(v is None for v in values):
        return None
    return _gap(sum(values), sum(plan_quarter(plan, line, q.year, q.quarter) for q in quarters))


def year_to_date(quarters: Sequence[Quarter], plan: PlanPath, line: str) -> Optional[dict]:
    """The latest year's quarters so far, all of them there and given, against the plan's."""
    if not quarters:
        return None
    last = quarters[-1]
    held = [q for q in quarters if q.year == last.year]
    if [q.quarter for q in held] != list(range(1, last.quarter + 1)):
        return None
    return _summed(held, plan, line)


def ltm(quarters: Sequence[Quarter], plan: PlanPath, line: str) -> Optional[dict]:
    """The last four quarters, one after another and all given, against the plan's."""
    window = list(quarters[-4:])
    if len(window) < 4 or window[-1].index - window[0].index != 3:
        return None
    return _summed(window, plan, line)


def full_years(quarters: Sequence[Quarter]) -> list[dict]:
    """Complete years from year 1, as plan vs actual's yearly actuals: flows summed (missing unless every quarter
    gives them), debt the fourth quarter's."""
    years = []
    for year in range(1, max((q.year for q in quarters), default=0) + 1):
        held = [q for q in quarters if q.year == year]
        if len(held) != 4:
            break
        row = {}
        for line in FLOWS:
            values = [q.figures.get(line) for q in held]
            row[line] = None if any(v is None for v in values) else sum(values)
        row["total_debt"] = held[-1].figures.get("total_debt")
        years.append(row)
    return years


def drift(plan: PlanPath, quarters: Sequence[Quarter], tolerance: float) -> Optional[dict]:
    """Where the latest results stand against the plan, and the alerts they raise."""
    if not quarters:
        return None
    last = quarters[-1]
    share = tolerance / 100
    ytd = {line: year_to_date(quarters, plan, line) for line in ("revenue", "ebitda")}
    debt = last.figures.get("total_debt")
    debt_gap = None if debt is None else _gap(debt, plan_quarter(plan, "total_debt", last.year, last.quarter))
    alerts = []
    for line, gap in ytd.items():
        if gap is not None and gap["plan"] > 0 and gap["actual"] < gap["plan"] * (1 - share):
            alerts.append({"id": f"{line}_below_plan", **gap})
    if debt_gap is not None and debt_gap["actual"] > 0 and debt_gap["actual"] > debt_gap["plan"] + abs(debt_gap["plan"]) * share:
        alerts.append({"id": "debt_above_plan", **debt_gap})
    return {"year": last.year, "quarter": last.quarter, "tolerance": tolerance, "ytd": ytd, "debt": debt_gap,
            "alerts": alerts}


def reforecast_growth(plan_growth: float, actual: float, planned: float, years: float) -> float:
    """The growth (%) that takes the plan's EBITDA through the actual one ``years`` after close."""
    return ((1 + plan_growth / 100) * (actual / planned) ** (1 / years) - 1) * 100


def reforecast(deal: DealInputs, cfg: Mapping, quarters: Sequence[Quarter], n: int = REFORECAST_PATHS,
               seed: int = 42) -> dict:
    """The plan run again with the growth its last four quarters show, and simulated around it. Money in
    millions; IRRs as fractions."""
    plan_run = _run(deal, cfg)
    plan = plan_path(plan_run)
    window = ltm(quarters, plan, "ebitda")
    if window is None:
        raise QuartersRefused("no_ltm")
    if window["actual"] <= 0 or window["plan"] <= 0:
        raise QuartersRefused("ebitda_not_positive")
    last = quarters[-1]
    growth = reforecast_growth(deal.growth, window["actual"], window["plan"], last.index / 4)
    again = replace(deal, growth=growth)
    run = _run(again, cfg)
    paths = plan_irr_paths(again, cfg, int(deal.hold), n, seed)
    planned, forecast = _returns(plan_run), _returns(run)
    actual_years = full_years(quarters)
    years = [{"year": i + 1, "plan_ebitda": plan.ebitda[i], "reforecast_ebitda": run.operating_model.ebitda[i],
              "actual_ebitda": actual_years[i]["ebitda"] if i < len(actual_years) else None}
             for i in range(int(deal.hold))]
    return {
        "based_on": {"year": last.year, "quarter": last.quarter},
        "plan_growth": deal.growth, "reforecast_growth": growth,
        "ltm_ebitda_actual": window["actual"], "ltm_ebitda_plan": window["plan"],
        "plan": planned, "reforecast": forecast,
        "simulated": {"paths": int(len(paths)), "p5": float(np.percentile(paths, 5)),
                      "p50": float(np.percentile(paths, 50)), "p95": float(np.percentile(paths, 95)),
                      # the share of paths that miss the plan's own IRR
                      "below_plan": float(np.mean(paths < planned["irr"]))},
        "years": years,
    }


# ------------------------------------------------------------------------------------------------ currency
def _per_euro(rates: Mapping[str, float], currency: str) -> Optional[float]:
    return 1.0 if currency == "EUR" else rates.get(currency) or None


def stated_rate(currency: str, reporting: str, rates: Mapping[str, float]) -> Optional[float]:
    """Units of ``currency`` per one unit of ``reporting``, through the ECB's rates per euro."""
    if currency == reporting:
        return 1.0
    held, to = _per_euro(rates, currency), _per_euro(rates, reporting)
    return None if held is None or to is None else held / to


def fx_factor(currency: str, unit: str, reporting: str, reporting_unit: str,
              rates: Mapping[str, float]) -> Optional[float]:
    """What an amount in ``currency`` and ``unit`` is multiplied by to be in ``reporting`` and ``reporting_unit``;
    None when the ECB quotes either currency not at all."""
    rate = stated_rate(currency, reporting, rates)
    return None if rate is None else UNIT_SCALE[unit] / UNIT_SCALE[reporting_unit] / rate


def board(holdings: Sequence[Mapping], reporting: str, reporting_unit: str, rates: Mapping[str, float]) -> dict:
    """Each holding's local money converted to the reporting currency, and the totals. A holding whose currency
    has no rate keeps its local figures, is left out of every total and named in ``missing_rates``."""
    out, missing, stated = [], set(), {}
    totals = {key: {"value": None, "holdings": 0} for key in BOARD_MONEY_KEYS}
    for holding in holdings:
        factor = fx_factor(holding["currency"], holding["unit"], reporting, reporting_unit, rates)
        converted = None
        if factor is None:
            missing.add(holding["currency"])
        else:
            if holding["currency"] != reporting:
                stated[holding["currency"]] = stated_rate(holding["currency"], reporting, rates)
            converted = {k: None if v is None else v * factor for k, v in holding["local"].items()}
            for key, value in converted.items():
                if key in totals and value is not None:
                    totals[key] = {"value": (totals[key]["value"] or 0.0) + value,
                                   "holdings": totals[key]["holdings"] + 1}
        out.append({**holding, "converted": converted})
    return {"holdings": out, "totals": totals, "rates": dict(sorted(stated.items())),
            "missing_rates": sorted(missing)}
