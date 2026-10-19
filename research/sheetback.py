"""Saving a model back from Excel (PLAN.md R5): "Save to Variater" in the add-in.

The add-in sends what the analyst's Model sheet holds: its cells as Excel
answers them (from A1, numbers as numbers, blanks as "") and the sheet's
defined names with their addresses. Only the **named cells** are read back:
every assumption (``<row>_<period>``) and every constant. Typed figures in
blue and every formula stay as Variater holds them.

**A changed structure is refused, never guessed.** The sheet must still be
the one Variater wrote for this model: every name it wrote is there and at
its own address (inserting or deleting a row or a column moves them, which
Excel does by itself), and every row's words in column A are where they were.
A refusal names the first difference, so the analyst knows what to undo or
to load the model again. A named cell holding words is refused too.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

from research.definition import MAX_NUMBER, Company, Model, cells, label, words_for
from research.workbook import SAVED_NAME, layout, sheets

RESTRUCTURED = "restructured"
NOT_A_NUMBER = "not_a_number"
OTHER_MODEL = "other_model"
MAX_ROWS, MAX_COLUMNS = 1000, 60          # far beyond any model's sheet; a bigger grid is not ours
_ADDRESS = re.compile(r"^\$?([A-Z]{1,3})\$?([0-9]{1,7})$")


class SheetRefused(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class SavedBack:
    company: Company
    changed: list = field(default_factory=list)     # (row or constant, period or None, before, after)


def _place(address: str) -> tuple[int, int] | None:
    """'$C$12' or 'C12' as (row, column), 1-based."""
    match = _ADDRESS.match(str(address).split("!")[-1].strip().upper())
    if not match:
        return None
    column = 0
    for ch in match.group(1):
        column = column * 26 + ord(ch) - 64
    return int(match.group(2)), column


def _at(grid: list, row: int, column: int):
    if row - 1 >= len(grid) or not isinstance(grid[row - 1], list) or column - 1 >= len(grid[row - 1]):
        return ""
    return grid[row - 1][column - 1]


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value).strip()


def _number(value):
    """A cell's value as a number, None when blank; anything else is not a number."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError
    if not math.isfinite(value) or abs(value) > MAX_NUMBER:
        raise TypeError
    return float(value)


def read_back(model: Model, company: Company, saved: str, grid: list, names: dict) -> SavedBack:
    """The company with the sheet's assumptions and constants, or a refusal saying what changed."""
    if not isinstance(grid, list) or len(grid) > MAX_ROWS or any(
            not isinstance(r, list) or len(r) > MAX_COLUMNS for r in grid):
        raise SheetRefused(RESTRUCTURED, "the sheet is larger than a model Variater wrote")
    expected = sheets(model, company, allow_missing=True, saved=saved)[0]
    where = _place(names.get(SAVED_NAME, "")) if isinstance(names, dict) else None
    if where is None or not _text(_at(grid, *where)).endswith(str(saved)):
        raise SheetRefused(OTHER_MODEL, "this sheet is not the saved model of this company: load it again")
    # every row's words where Variater wrote them
    header = layout(model).header_row
    for cell in expected["cells"]:
        if cell["column"] == 1 and cell["row"] >= header and "text" in cell:
            found = _text(_at(grid, cell["row"], 1))
            if found != cell["text"].strip():
                raise SheetRefused(RESTRUCTURED, "row %d should read '%s' but reads '%s': rows were added, removed "
                                   "or moved" % (cell["row"], cell["text"].strip()[:80], found[:80]))
    for name, address in expected["names"].items():
        given = names.get(name)
        if given is None:
            raise SheetRefused(RESTRUCTURED, "the named cell %s is gone" % name)
        if _place(given) != _place(address):
            raise SheetRefused(RESTRUCTURED, "the named cell %s has moved from %s to %s" % (
                name, address.replace("$", ""), str(given).replace("$", "")))
    words = words_for(model, company)
    written = cells(model, company, allow_missing=True)
    rows = {r.key: r for r in model.data_rows()}
    values = {r: dict(by) for r, by in company.values.items()}
    constants = dict(company.constants)
    changed = []
    for name, address in expected["names"].items():
        if name == SAVED_NAME:
            continue
        raw = _at(grid, *_place(address))
        if name in model.constants:
            row, period, what = name, None, label(model.constants[name], words)
        else:
            row, period = _split(name, rows, model)
            what = "%s, %s" % (label(rows[row].label, words).strip(), company.period_labels.get(period, period))
        try:
            value = _number(raw)
        except TypeError:
            raise SheetRefused(NOT_A_NUMBER, "%s (%s) holds '%s', not a number" % (
                address.replace("$", ""), what, _text(raw)[:40])) from None
        if period is None:
            if value is None:
                raise SheetRefused(NOT_A_NUMBER, "%s (%s) is blank" % (address.replace("$", ""), what))
            if constants.get(name) != value:
                changed.append((name, None, constants.get(name), value))
            constants[name] = value
            continue
        was = written[(row, period)].value
        if value == was or (value is not None and was is not None and math.isclose(value, was, rel_tol=1e-12, abs_tol=1e-12)):
            continue                # as Variater wrote it (a line switched off is written as 0, not typed)
        before = values.get(row, {}).get(period)
        if before != value:
            changed.append((row, period, before, value))
        if value is None:
            values.get(row, {}).pop(period, None)
        else:
            values.setdefault(row, {})[period] = value
    out = replace(company, values={r: v for r, v in values.items() if v}, constants=constants)
    return SavedBack(out, changed)


def _split(name: str, rows: dict, model: Model) -> tuple[str, str]:
    """An assumption's name back into its row and period: the period is the part after the last row key."""
    for period in model.periods:
        suffix = "_" + period.name
        if name.endswith(suffix) and name[: -len(suffix)] in rows:
            return name[: -len(suffix)], period.name
    raise SheetRefused(RESTRUCTURED, "the named cell %s is not one of this model's" % name)


__all__ = ["NOT_A_NUMBER", "OTHER_MODEL", "RESTRUCTURED", "SavedBack", "SheetRefused", "read_back"]
