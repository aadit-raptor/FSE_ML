"""The reference cases as a spreadsheet (PLAN.md 3.4).

``python -m tests.reference.workbook`` writes ``reference_cases.xlsx`` from
``cases.py``: an index sheet, then one sheet per case with its inputs and
every line of the working as a live Excel formula over the cells above it,
beside the value it comes to and the engine figure it must match. Open it in
Excel or LibreOffice and the formulas recalculate on their own.

``evaluate`` works the formulas out in Python, the way the spreadsheet does,
and refuses anything but plain arithmetic, so a formula here means the same
thing in both places. ``tests/test_reference_cases.py`` fails if the
committed workbook differs from what this module writes.
"""
import io
import sys
import tokenize
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font

from tests.reference.cases import CASES, Case

WORKBOOK = Path(__file__).with_name("reference_cases.xlsx")

FUNCTIONS = {"max": "MAX", "min": "MIN"}
OPERATORS = {"+", "-", "*", "/", "**", "(", ")", ","}
FORMATS = {"money": "#,##0.0000", "pct": "0.0000%", "x": '0.0000"x"'}
STAMP = datetime(2026, 10, 6)   # fixed, so rebuilding changes nothing that matters


def _tokens(formula: str) -> list:
    """The formula's tokens, refusing anything a spreadsheet would read
    differently: only numbers, names, the four operations, powers, brackets
    and commas. A leading minus is refused too (Excel's ``-x^2`` is
    ``(-x)^2``); write ``0 - x``."""
    out, previous = [], None
    for tok in tokenize.generate_tokens(io.StringIO(formula).readline):
        if tok.type in (tokenize.NEWLINE, tokenize.ENDMARKER, tokenize.NL):
            continue
        if tok.type == tokenize.OP and tok.string not in OPERATORS:
            raise ValueError(f"{formula!r}: {tok.string!r} is not allowed")
        if tok.type not in (tokenize.NUMBER, tokenize.NAME, tokenize.OP):
            raise ValueError(f"{formula!r}: {tok.string!r} is not allowed")
        if tok.string == "-" and (previous is None or previous.string in ("(", ",")):
            raise ValueError(f"{formula!r}: write 0 - x rather than a leading minus")
        out.append(tok)
        previous = tok
    return out


def evaluate(case: Case) -> dict:
    """Every line's value, in order, from the case's inputs."""
    values = dict(case.given)
    for line in case.lines:
        if line.key in values:
            raise ValueError(f"{case.id}: {line.key} is defined twice")
        names = {t.string for t in _tokens(line.formula) if t.type == tokenize.NAME}
        unknown = names - set(values) - set(FUNCTIONS)
        if unknown:
            raise ValueError(f"{case.id}: {line.key} uses {sorted(unknown)} before they exist")
        values[line.key] = float(eval(line.formula, {"__builtins__": {}}, {**values, "max": max, "min": min}))
    return values


def excel_formula(formula: str, cells: dict) -> str:
    """The formula with names replaced by their cells, as Excel writes it."""
    parts = []
    for tok in _tokens(formula):
        if tok.type == tokenize.NAME:
            parts.append(FUNCTIONS[tok.string] if tok.string in FUNCTIONS else cells[tok.string])
        elif tok.string == "**":
            parts.append("^")
        else:
            parts.append(tok.string)
    return "=" + "".join(parts)


def _case_sheet(wb: Workbook, number: int, case: Case) -> None:
    ws = wb.create_sheet(case.id[:31])
    values = evaluate(case)
    bold = Font(bold=True)
    ws["A1"] = f"{number}. {case.title}"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = f"Covers: {', '.join(case.covers)}"
    ws["A3"] = case.reasoning
    ws.merge_cells("A3:F3")
    ws["A3"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[3].height = 90

    row = 5
    ws.cell(row, 1, "Inputs").font = bold
    ws.cell(row, 2, "Name").font = bold
    ws.cell(row, 3, "Value").font = bold
    cells = {}
    for name, value in case.given.items():
        row += 1
        ws.cell(row, 1, name)
        ws.cell(row, 2, name)
        ws.cell(row, 3, value)
        cells[name] = f"$C${row}"

    row += 2
    for col, title in enumerate(("Working", "Name", "Formula", "Comes to", "Engine figure it must match",
                                 "Compared as"), start=1):
        ws.cell(row, col, title).font = bold
    for line in case.lines:
        row += 1
        cells[line.key] = f"$C${row}"
        ws.cell(row, 1, line.label)
        ws.cell(row, 2, line.key)
        ws.cell(row, 3, excel_formula(line.formula, cells)).number_format = FORMATS[line.kind]
        ws.cell(row, 4, values[line.key]).number_format = FORMATS[line.kind]
        if line.check:
            ws.cell(row, 5, line.check)
            ws.cell(row, 6, {"money": "money, to 0.01", "pct": "per cent, to 0.01",
                             "x": "multiple, to 0.01"}[line.kind])

    row += 2
    ws.cell(row, 1, "Headline answers, typed by hand").font = bold
    for key, value in case.answers.items():
        row += 1
        ws.cell(row, 1, next(line.label for line in case.lines if line.key == key))
        ws.cell(row, 2, key)
        ws.cell(row, 4, value).number_format = FORMATS["money"]

    for col, width in zip("ABCDEF", (62, 16, 16, 16, 46, 18)):
        ws.column_dimensions[col].width = width


def build() -> Workbook:
    wb = Workbook()
    index = wb.active
    index.title = "Cases"
    index["A1"] = "Variater: hand-checked reference cases (PLAN.md 3.4)"
    index["A1"].font = Font(bold=True, size=13)
    index["A2"] = ("Small deals worked out by hand from docs/methodology.md. Each sheet's formulas are the "
                   "working; tests/test_reference_cases.py requires the engine to match every line with an "
                   "engine figure to 0.01 (money in the deal's unit, IRR and rates in per cent). "
                   "Regenerate with: python -m tests.reference.workbook")
    index.merge_cells("A2:E2")
    index["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    index.row_dimensions[2].height = 60
    for col, title in enumerate(("#", "Case", "Covers", "Figures checked", "Sheet"), start=1):
        index.cell(4, col, title).font = Font(bold=True)
    for number, case in enumerate(CASES, start=1):
        _case_sheet(wb, number, case)
        r = 4 + number
        index.cell(r, 1, number)
        index.cell(r, 2, case.title)
        index.cell(r, 3, ", ".join(case.covers))
        index.cell(r, 4, sum(1 for line in case.lines if line.check))
        link = index.cell(r, 5, case.id[:31])
        link.hyperlink = f"#'{case.id[:31]}'!A1"
    for col, width in zip("ABCDE", (5, 62, 40, 16, 32)):
        index.column_dimensions[col].width = width
    wb.properties.creator = "variater.com"
    wb.properties.created = STAMP
    wb.properties.modified = STAMP
    return wb


def contents(wb) -> dict:
    """Every non-empty cell, sheet by sheet: what the staleness check compares.
    Numbers to nine decimals, since the file keeps them to fifteen digits."""
    def value(v):
        return round(v, 9) if isinstance(v, float) else v
    return {ws.title: {c.coordinate: value(c.value) for row in ws.iter_rows() for c in row if c.value is not None}
            for ws in wb.worksheets}


def is_current() -> bool:
    return WORKBOOK.exists() and contents(load_workbook(WORKBOOK)) == contents(build())


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(0 if is_current() else 1)
    build().save(WORKBOOK)
    print(f"wrote {WORKBOOK}")
