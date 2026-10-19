"""Rolling a saved sector model forward on new results (PLAN.md R5).

A saved model knows how far its past reaches (``as_of``: the last full
year's end and how many months of the year in progress it holds). When the
analyst has accepted newer results for the company, `roll` gives the model
for the new period: **the structure, the choices and every assumption are
kept**; the past is the new results; and **the estimate made before the
results is set beside them, line by line**, as it stood when the roll was
made. The caller stores that comparison with the version kept from before
the roll, so later edits never change it.

Two kinds of roll:

- **A later quarter of the same year** (three months to date become six):
  the year to date is the new results, the rest of the year keeps its
  assumptions, and its days shrink so the year keeps its length. The model
  holds no quarters, so the estimate for the months just reported is the
  rest of the year's estimate shared out by days (``basis`` says so); the
  actual is the new year to date less the old.
- **A new year** (the year in progress has ended): last year is the year
  just reported, every assumption moves back one year (next year's become
  this year's rest; the last forecast year is repeated and marked), and the
  estimate is the old current year against the year as reported, exactly,
  with the balance sheet lines at the year end.

Figures the results don't carry (a hospital's beds, a segment's revenue) are
never guessed: a count or a balance is carried from what the model held and
marked amber with why; a flow over the period is left blank to type, and
listed. Only the rows the results fill (research/actuals.py ``FLOWS`` and
``STOCKS``) are compared.
"""
from __future__ import annotations

import calendar
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date

from research.actuals import FLOWS, QUARTER_MISSING, STOCKS, Fill
from research.definition import ASSUME, FIGURE, Company, Model, Pick, cells, label, rule_for, words_for
from research.evaluate import evaluate
from research.pieces import CY, LY, ROY, YTD, added_up

# Why a roll is refused (words: web/messages/en.json models.roll_*)
NO_AS_OF = "no_as_of"                   # the model was not started from accepted results
NOTHING_NEWER = "nothing_newer"
OLDER_RESULTS = "older_results"
TOO_FAR = "too_far"                     # more than a year on: start the model again from the results
NO_RESULTS = "no_results"               # the results needed for the new past are not all accepted

WITHIN_YEAR, NEW_YEAR = "quarter", "year"
SHARED_BY_DAYS, AS_REPORTED = "share_of_rest_by_days", "year_as_reported"


class RollRefused(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class Rolled:
    company: Company
    as_of: dict
    kind: str
    comparison: dict
    carried: list = field(default_factory=list)     # (row, period): kept from before and marked amber
    to_type: list = field(default_factory=list)     # (row, period): left blank for the analyst


def _day(text) -> date:
    return date.fromisoformat(str(text))


def _year_on(day: date) -> date:
    """The same month end a year later (29 February has none)."""
    return date(day.year + 1, day.month, calendar.monthrange(day.year + 1, day.month)[1])


def _near(a: date, b: date) -> bool:
    return abs((a - b).days) <= 7


def kind_of(old: dict | None, new: dict | None) -> str:
    """Which roll takes the model from `old` to `new` (both ``as_of``), or a refusal's code."""
    if not old:
        raise RollRefused(NO_AS_OF)
    if not new:
        raise RollRefused(NO_RESULTS)
    old_ly, new_ly = _day(old["ly_end"]), _day(new["ly_end"])
    if _near(old_ly, new_ly):
        if new["ytd_months"] > old["ytd_months"]:
            return WITHIN_YEAR
        raise RollRefused(NOTHING_NEWER if new["ytd_months"] == old["ytd_months"] else OLDER_RESULTS)
    if new_ly < old_ly:
        raise RollRefused(OLDER_RESULTS)
    if _near(_year_on(old_ly), new_ly):
        return NEW_YEAR
    raise RollRefused(TOO_FAR)


def newer(old: dict | None, new: dict | None) -> bool:
    """Whether accepted results reach further than the saved model: what the screen offers a roll for."""
    try:
        kind_of(old, new)
    except RollRefused:
        return False
    return True


def _is_flow(model: Model, row, company: Company) -> bool:
    """A row added up over a period (revenue, days): the current year is the part reported plus the rest."""
    rule = rule_for(row, model.period(CY))
    if isinstance(rule, Pick):
        rule = rule.options[company.choices[rule.choice]]
    return rule == added_up(row.key)


def _forecast_names(model: Model) -> list:
    return [p.name for p in model.periods if p.kind == "forecast"]


def in_model_unit(fill: Fill, company: Company) -> Fill:
    """The results' money in the model's unit (`unit_rupees`): a company that moved from lakh to crore stays in
    the unit its model is typed in."""
    ours, theirs = company.constants.get("unit_rupees"), fill.unit_rupees
    if not ours or not theirs or ours == theirs:
        return fill
    factor = theirs / ours
    money = set(FLOWS) | set(STOCKS)
    values = {row: {p: (round(v * factor, 6) if row in money else v) for p, v in by.items()}
              for row, by in fill.values.items()}
    return Fill(values=values, sources=fill.sources, labels=fill.labels, unit=fill.unit, currency=fill.currency,
                unit_rupees=ours, notes=list(fill.notes), as_of=fill.as_of)


def roll(model: Model, company: Company, as_of: dict | None, fill: Fill) -> Rolled:
    """The model rolled to the results in `fill` (research/actuals.py, from the same scope's accepted periods)."""
    kind = kind_of(as_of, fill.as_of)
    fill = in_model_unit(fill, company)
    if kind == WITHIN_YEAR and (QUARTER_MISSING in fill.notes or "rev" not in fill.values
                                or "ytd" not in fill.values["rev"]):
        raise RollRefused(NO_RESULTS)
    if kind == NEW_YEAR and ("rev" not in fill.values or "ly" not in fill.values["rev"]):
        raise RollRefused(NO_RESULTS)
    plan = cells(model, company, allow_missing=True)
    before = evaluate(model, company, partial=True, plan=plan).values
    words = words_for(model, company)
    old_values = deepcopy(company.values)
    values = deepcopy(company.values)
    marks = deepcopy(company.marks)
    carried, to_type = [], []

    def put(row: str, period: str, value) -> None:
        if value is None:
            values.get(row, {}).pop(period, None)
        else:
            values.setdefault(row, {})[period] = value
        marks.get(row, {}).pop(period, None)

    def carry(row: str, period: str, value, why: str) -> None:
        put(row, period, value)
        if value is not None:
            marks.setdefault(row, {})[period] = {"level": "amber", "why": why}
            carried.append((row, period))

    new_ytd = fill.labels.get("ytd") or "the year to date"
    new_ly = fill.labels.get("ly") or "last year"
    if kind == NEW_YEAR:
        forecast = _forecast_names(model)
        # assumptions move back a year: next year's become this year's (its rest and its whole-year figures)
        source = {ROY: forecast[0], CY: forecast[0], **{f: n for f, n in zip(forecast, forecast[1:])}}
        for (row, period), cell in plan.items():
            if cell.typed is not ASSUME:
                continue
            if period in source:
                from_cell = plan.get((row, source[period]))
                was = old_values.get(row, {}).get(source[period]) if from_cell and from_cell.typed is ASSUME else None
                put(row, period, was)
            elif period == forecast[-1]:
                carry(row, period, old_values.get(row, {}).get(period),
                      "Repeated from the year before when the model rolled forward: type this year's.")
        # the past: the year just reported, and the year to date if any of the new year is reported
        for (row, period), cell in plan.items():
            if cell.typed is not FIGURE:
                continue
            if period == LY:
                if row in fill.values and LY in fill.values[row]:
                    put(row, LY, fill.values[row][LY])
                else:
                    carry(row, LY, before.get(row, {}).get(CY),
                          "The model's estimate for %s when it rolled forward: type the year as reported." % new_ly)
        _past_to_date(model, company, plan, fill, values, put, carry, to_type, start_of_year=True)
        year_days = (_year_on(_day(fill.as_of["ly_end"])) - _day(fill.as_of["ly_end"])).days
        to_date = (values.get("days") or {}).get(YTD) or 0
        if ("days", ROY) in plan:
            put("days", ROY, year_days - to_date)
        comparison = _compare_year(model, company, words, before, fill)
    else:
        _past_to_date(model, company, plan, fill, values, put, carry, to_type, start_of_year=False)
        for row in fill.values:                         # last year as accepted (a filing may restate it)
            if LY in fill.values[row] and plan.get((row, LY)) is not None and plan[(row, LY)].typed is FIGURE:
                put(row, LY, fill.values[row][LY])
        old_days = old_values.get("days", {})
        if ("days", ROY) in plan and old_days.get(ROY) is not None:
            year_days = (old_days.get(YTD) or 0) + old_days[ROY]
            put("days", ROY, year_days - (values.get("days", {}).get(YTD) or 0))
        comparison = _compare_quarter(model, company, words, before, old_values, fill, as_of)
    labels = dict(company.period_labels)
    if kind == NEW_YEAR:                                # the years' own labels move back with their assumptions
        forecast = _forecast_names(model)
        for later, earlier in zip([CY] + forecast, forecast):
            if earlier in company.period_labels:
                labels[later] = company.period_labels[earlier]
            else:
                labels.pop(later, None)
        labels.pop(forecast[-1], None)
        labels.pop(ROY, None)
        if CY in labels:
            labels[ROY] = "Rest of %s (assumed)" % labels[CY]
    for period in (LY, YTD):
        if period in fill.labels:
            labels[period] = fill.labels[period]
        elif period == YTD and not fill.as_of["ytd_months"]:
            labels.pop(YTD, None)
    rolled = Company(name=company.name, unit_name=company.unit_name, sector=company.sector, basis=company.basis,
                     words=dict(company.words), period_labels=labels, constants=dict(company.constants),
                     choices=dict(company.choices), values={r: v for r, v in values.items() if v},
                     off=tuple(company.off), marks={r: m for r, m in marks.items() if m})
    cells(model, rolled, allow_missing=True)            # what the roll made must still be a company of this model
    return Rolled(rolled, dict(fill.as_of), kind, comparison, carried, to_type)


def _past_to_date(model, company, plan, fill, values, put, carry, to_type, start_of_year: bool) -> None:
    """The year to date from the new results. A figure they don't carry: a count or balance is carried (amber),
    from last year's end at the start of a year; a flow over the period is left to type."""
    months = fill.as_of["ytd_months"]
    rows = {r.key: r for r in model.data_rows()}
    for (row, period), cell in plan.items():
        if period != YTD or cell.typed is not FIGURE:
            continue
        if row in fill.values and YTD in fill.values[row]:
            put(row, YTD, fill.values[row][YTD])
        elif row == "days":
            put(row, YTD, 0 if not months else None)
        elif _is_flow(model, rows[row], company):
            put(row, YTD, None)
            if months:
                to_type.append((row, YTD))
        elif start_of_year and not months:
            put(row, YTD, values.get(row, {}).get(LY))      # nothing of the year yet: it stands where it ended
        elif start_of_year:
            carry(row, YTD, values.get(row, {}).get(LY),
                  "Carried from the year end when the model rolled forward: type it for %s." % fill.labels.get(YTD, YTD))
        else:
            carry(row, YTD, values.get(row, {}).get(YTD),
                  "Carried from the quarter before when the model rolled forward: type it for %s." % fill.labels.get(YTD, YTD))


def _line(model: Model, words: dict, row: str, what: str, estimate, actual) -> dict | None:
    if estimate is None or actual is None:
        return None
    text = label(next(r.label for r in model.data_rows() if r.key == row), words).strip()
    gap = actual - estimate
    return {"row": row, "label": text, "kind": what, "estimate": round(estimate, 6), "actual": round(actual, 6),
            "difference": round(gap, 6), "share": round(gap / estimate, 6) if estimate else None}


def _keys(model: Model, table: dict) -> list:
    have = {r.key for r in model.data_rows()}
    return [k for k in table if k in have]


def _compare_year(model, company, words, before, fill) -> dict:
    lines = []
    for what, table in (("flow", FLOWS), ("stock", STOCKS)):
        for row in _keys(model, table):
            line = _line(model, words, row, what, before.get(row, {}).get(CY), fill.values.get(row, {}).get(LY))
            if line:
                lines.append(line)
    return {"basis": AS_REPORTED, "period": fill.labels.get(LY, ""), "period_end": fill.as_of["ly_end"],
            "months": 12, "estimate_label": str(company.period_labels.get(CY, CY)), "lines": lines}


def _compare_quarter(model, company, words, before, old_values, fill, as_of) -> dict:
    old_days = old_values.get("days", {})
    rest_days = old_days.get(ROY)
    new_to_date = fill.values.get("days", {}).get(YTD)
    old_to_date = (old_days.get(YTD) or 0) if as_of["ytd_months"] else 0
    lines = []
    if rest_days and new_to_date is not None:
        share = (new_to_date - old_to_date) / rest_days
        for row in _keys(model, FLOWS):
            new = fill.values.get(row, {}).get(YTD)
            old = old_values.get(row, {}).get(YTD) if as_of["ytd_months"] else 0.0
            rest = before.get(row, {}).get(ROY)
            actual = None if new is None or old is None else new - old
            line = _line(model, words, row, "flow", None if rest is None else rest * share, actual)
            if line:
                lines.append(line)
    months = fill.as_of["ytd_months"] - as_of["ytd_months"]
    return {"basis": SHARED_BY_DAYS, "period": "%d months to %s" % (months, fill.as_of["ytd_end"]),
            "period_end": fill.as_of["ytd_end"], "months": months,
            "estimate_label": str(company.period_labels.get(ROY, ROY)), "lines": lines}


__all__ = ["AS_REPORTED", "NEW_YEAR", "RollRefused", "Rolled", "SHARED_BY_DAYS", "WITHIN_YEAR", "kind_of",
           "newer", "roll"]
