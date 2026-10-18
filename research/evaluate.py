"""Working a sector model out on the server (PLAN.md R4).

Every cell is worked out once, after the cells it reads, in plain floating
point with no rounding: the same arithmetic Excel does on the formulas the
workbook writer emits. A cell that is empty, or a figure nobody typed, reads
as nil to a formula that names it, as an empty cell does in Excel, and is
None in the answer.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from research import formula
from research.definition import Company, Model, ModelError, cells, missing
from research.formula import DivisionByNil, Ref


class EvaluationError(ArithmeticError):
    """The figures typed for a company make a row impossible to work out."""


@dataclass(frozen=True)
class Result:
    values: dict        # row -> period -> number, or None for an empty cell
    missing: tuple = () # assumptions not typed yet, as (row, period): the cells that read them are None
    undefined: tuple = ()   # with `partial`, cells that divide by a blank figure: None, as are those that read them

    def value(self, row: str, period: str) -> float | None:
        return self.values[row][period]


def target(model: Model, ref: Ref, period: str) -> tuple[str, str]:
    """The cell a name in a formula points at, from the period the formula stands in."""
    if ref.period is None:
        return ref.name, period
    return ref.name, model.period(period).prev if ref.period == "prev" else ref.period


def _order(model: Model, plan: dict) -> list:
    """The formula cells, each after the cells it reads. A circle is an error that names its cells."""
    reads = {}
    for key, cell in plan.items():
        if cell.tree is None:
            continue
        cells_read = [target(model, ref, key[1]) for ref in formula.refs(cell.tree) if ref.name not in model.constants]
        reads[key] = [t for t in cells_read if t in plan and plan[t].tree is not None]
    done, order = set(), []
    for start in reads:
        if start in done:
            continue
        path, on_path = [(start, iter(reads[start]))], {start}
        while path:
            key, rest = path[-1]
            for nxt in rest:
                if nxt in on_path:
                    circle = [k for k, _ in path[[k for k, _ in path].index(nxt):]] + [nxt]
                    raise ModelError("rows go in a circle: " + " -> ".join("%s[%s]" % k for k in circle))
                if nxt not in done:
                    path.append((nxt, iter(reads[nxt])))
                    on_path.add(nxt)
                    break
            else:
                path.pop()
                on_path.discard(key)
                done.add(key)
                order.append(key)
    return order


def evaluate(model: Model, company: Company, partial: bool = False, plan: dict | None = None) -> Result:
    """Every row of the model in every period, for one company's typed figures.

    With `partial`, assumptions not typed yet are allowed: every cell that
    reads one, however indirectly, is None (not nil), and the answer lists
    them, so the past works out before the forecast is filled in. A plain
    division by a blank figure is None too (listed as `undefined`), where
    without `partial` it is an error. `plan` is `cells`' answer when the
    caller has it already.
    """
    plan = plan if plan is not None else cells(model, company, allow_missing=partial)
    absent = missing(plan)
    unknown = set(absent)
    undefined: list = []
    known = {key: cell.value for key, cell in plan.items() if cell.tree is None}
    constants = {name: float(company.constants[name]) for name in model.constants}
    for key in _order(model, plan):
        if unknown and plan[key].tree is not None:
            reads = {target(model, ref, key[1]) for ref in formula.refs(plan[key].tree) if ref.name not in constants}
            if reads & unknown:
                unknown.add(key)
                known[key] = None
                continue

        def lookup(ref: Ref, period=key[1]) -> float:
            if ref.name in constants:
                return constants[ref.name]
            return known.get(target(model, ref, period)) or 0.0
        try:
            value = formula.compute(plan[key].tree, lookup)
        except DivisionByNil:
            if partial:                 # a figure not typed yet: blank, as are the cells that read it
                undefined.append(key)
                unknown.add(key)
                known[key] = None
                continue
            raise EvaluationError("row %r divides by nil in %r: a figure it divides by is nil or missing" % key) from None
        if not math.isfinite(value):
            raise EvaluationError("row %r in %r is too large to work out" % key)
        known[key] = value
    return Result({row.key: {p.name: known.get((row.key, p.name)) for p in model.periods} for row in model.data_rows()},
                  tuple(absent), tuple(undefined))
