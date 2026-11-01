"""A downloaded deal workbook (PLAN.md 7.3a) worked out by the tests' own Excel
reader, for the browser tests (``web/e2e/money.spec.ts``)::

    python -m tests.deal_workbook_check deal.xlsx

prints ``{"irr": ..., "moic": ..., "money": "EUR thousands"}``: the IRR and MOIC
the workbook's formulas give, and what its money is counted in.
"""
from __future__ import annotations

import json
import sys

from openpyxl import load_workbook

from tests.research_support import workbook_values

SHEET = "Deal model"


def named(wb, name: str) -> str:
    """The address a defined name points at, on the model sheet."""
    return wb.defined_names[name].attr_text.split("!")[1].replace("$", "")


def worked_out(wb):
    """Any named cell's value, worked out by the tests' own reader from the workbook's formulas."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 50_000))     # a year reads the year before through many rows
    get = workbook_values(wb, SHEET)
    return lambda name: get(named(wb, name))


def row_of(wb, label: str) -> int:
    for (cell,) in wb[SHEET].iter_rows(min_col=1, max_col=1):
        if cell.value == label:
            return cell.row
    raise AssertionError("no row %r" % label)


def main(path: str) -> dict:
    wb = load_workbook(path)
    value = worked_out(wb)
    return {"irr": value("irr"), "moic": value("moic"), "money": wb[SHEET].cell(row_of(wb, "Revenue"), 2).value}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1])))
