"""A sector model as a live Excel workbook (PLAN.md R4).

The same rows the server works out, written as Excel formulas from the same
parsed trees: plain arithmetic, MAX, MIN and one IF (for `div`). Formulas use
ordinary cell addresses, so Excel's trace arrows work; every assumption cell
and constant also has a defined name (`s1_occ_y1`, `unit_rupees`) by which
the add-in will find it. Typed cells are blue; assumptions are yellow; a
marked figure is amber or red with its reason in a comment and on the Notes
sheet.

Nothing is worked out here: the workbook holds formulas without stored
values, and the spreadsheet program that opens it does the arithmetic.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.workbook.properties import CalcProperties

from research import formula
from research.definition import ASSUME, MARKS, Company, Heading, InputError, Model, ModelError, cells, label, words_for
from research.evaluate import target
from research.formula import Ref

SOURCE = "variater.com"
SHEET = "Model"
FIRST_CONSTANT_ROW = 4
FIRST_PERIOD_COLUMN = 3          # A = the row's label, B = its unit

FORMATS = {
    "money": "#,##0.00;(#,##0.00);-", "count": "#,##0;(#,##0);-", "count3": "#,##0.000", "days": "0",
    "share": "0.0%;-0.0%;-", "rupees": "#,##0;(#,##0);-", "rupees2": "0.00", "times": "0.00", "check": "0.00;-0.00;0.00",
}
BLUE = Font(color="0B4F9E")
BOLD = Font(bold=True)
WHITE_BOLD = Font(bold=True, color="FFFFFF")
HEAD = PatternFill("solid", fgColor="1F2A30")
SECTION = PatternFill("solid", fgColor="DCE6EA")
ASSUMED = PatternFill("solid", fgColor="FFF8D6")
MARKED = {"amber": PatternFill("solid", fgColor="F6C566"), "red": PatternFill("solid", fgColor="F2A097")}
_LOOKS_LIKE_A_CELL = re.compile(r"[A-Za-z]{1,3}[0-9]+|[Rr][0-9]*[Cc][0-9]*")


@dataclass(frozen=True)
class Layout:
    """Where every row, period and constant of a model stands on its sheet."""
    header_row: int
    rows: dict              # row key -> sheet row
    columns: dict           # period name -> column letter
    constants: dict         # constant name -> address

    def cell(self, row: str, period: str) -> str:
        return "%s%d" % (self.columns[period], self.rows[row])

    def constant(self, name: str) -> str:
        return self.constants[name]


def layout(model: Model) -> Layout:
    constants = {name: "C%d" % (FIRST_CONSTANT_ROW + i) for i, name in enumerate(model.constants)}
    header = FIRST_CONSTANT_ROW + len(model.constants) + 1
    rows = {r.key: header + 1 + i for i, r in enumerate(model.rows) if not isinstance(r, Heading)}
    columns = {p.name: get_column_letter(FIRST_PERIOD_COLUMN + i) for i, p in enumerate(model.periods)}
    return Layout(header, rows, columns, constants)


def _absolute(address: str) -> str:
    return re.sub(r"([A-Z]+)([0-9]+)", r"$\1$\2", address)


def _name(wb: Workbook, name: str, address: str) -> None:
    if _LOOKS_LIKE_A_CELL.fullmatch(name):
        raise ModelError("%r cannot be a name in Excel: it reads as a cell" % name)
    if name in wb.defined_names:
        raise ModelError("two cells would both be named %r" % name)
    wb.defined_names[name] = DefinedName(name, attr_text="%s!%s" % (SHEET, _absolute(address)))


def _put(ws, row: int, column: int, text: str):
    """Words into a cell, as words: a name or label that starts with = must never become a formula."""
    cell = ws.cell(row, column)
    cell.value = text
    cell.data_type = "s"
    return cell


def _unit_text(unit: str, company: Company, words: dict) -> str:
    money = company.unit_name
    return {"money": money, "check": money, "share": "%", "rupees": words.get("currency", ""), "rupees2": words.get("currency", ""),
            "count": "number", "count3": "number", "days": "days", "times": "times"}[unit]


def write(model: Model, company: Company) -> Workbook:
    """The model for one company as a workbook: a Model sheet of live formulas and a Notes sheet."""
    plan = cells(model, company)
    words = words_for(model, company)
    where = layout(model)
    wb = Workbook()
    ws = wb.active
    ws.title = SHEET
    wb.properties.creator = wb.properties.lastModifiedBy = SOURCE
    wb.calculation = CalcProperties(fullCalcOnLoad=True)      # no values are stored: work everything out on opening

    width = FIRST_PERIOD_COLUMN + len(model.periods) - 1
    title = "%s: %s model%s" % (company.name, model.name, " (%s)" % company.sector if company.sector else "")
    _put(ws, 1, 1, title).font = Font(bold=True, size=13)
    vetted = ("Read by a practising analyst." if model.vetted else
              "NOT YET VETTED: no practising analyst has read this model yet.")
    _put(ws, 2, 1, vetted + " Yellow cells are assumptions to type; everything in black is a formula. Source: " + SOURCE
         ).font = Font(bold=True, color="1D6F42" if model.vetted else "B03A2E")
    _put(ws, 3, 1, "%sMoney in %s." % (company.basis + ". " if company.basis else "", company.unit_name))
    for name, text in model.constants.items():
        at = where.constant(name)
        _put(ws, int(at[1:]), 1, label(text, words))
        ws[at] = company.constants[name]
        ws[at].font, ws[at].number_format = BLUE, "#,##0.######"
        _name(wb, name, at)

    ws.cell(where.header_row, 1, "Row")
    ws.cell(where.header_row, 2, "Unit")
    for i, period in enumerate(model.periods):
        _put(ws, where.header_row, FIRST_PERIOD_COLUMN + i, str(company.period_labels.get(period.name, period.name)))
    for cell in ws[where.header_row][:width]:
        cell.font, cell.fill, cell.alignment = WHITE_BOLD, HEAD, Alignment(horizontal="right" if cell.column > 2 else "left", wrap_text=True)

    def address(ref: Ref, period: str) -> str:
        if ref.name in where.constants:
            return _absolute(where.constants[ref.name])
        row, column = target(model, ref, period)
        return where.cell(row, column)

    marked = []
    n = where.header_row
    for row in model.rows:
        n += 1
        text = label(row.label, words)
        if isinstance(row, Heading):
            _put(ws, n, 1, text).font = BOLD
            for cell in ws[n][:width]:
                cell.fill = SECTION
            continue
        _put(ws, n, 1, text).font = BOLD if row.bold else Font()
        _put(ws, n, 2, _unit_text(row.unit, company, words)).font = Font(color="888888")
        for period in model.periods:
            cell = ws[where.cell(row.key, period.name)]
            cell.number_format = FORMATS[row.unit]
            if row.bold:
                cell.font = BOLD
            held = plan.get((row.key, period.name))
            if held is None:
                continue
            if held.tree is not None:
                cell.value = "=" + formula.to_excel(held.tree, lambda ref, p=period.name: address(ref, p))
                continue
            cell.value, cell.font = held.value, BLUE
            if held.typed is ASSUME:
                cell.fill = ASSUMED
                _name(wb, "%s_%s" % (row.key, period.name), cell.coordinate)
            mark = company.marks.get(row.key, {}).get(period.name)
            if mark:
                if mark.get("level") not in MARKS:
                    raise InputError("a mark is amber or red, not %r (row %r)" % (mark.get("level"), row.key))
                cell.fill = MARKED[mark["level"]]
                cell.comment = Comment(str(mark.get("why", "")), SOURCE)
                marked.append(("%s!%s" % (SHEET, cell.coordinate), "%s, %s" % (text.strip(), company.period_labels.get(period.name, period.name)),
                               mark["level"], str(mark.get("why", ""))))

    ws.column_dimensions["A"].width = 74
    ws.column_dimensions["B"].width = 12
    for period in model.periods:
        ws.column_dimensions[where.columns[period.name]].width = 17
    ws.row_dimensions[where.header_row].height = 32
    ws.freeze_panes = "C%d" % (where.header_row + 1)
    _notes(wb, model, marked)
    return wb


def _notes(wb: Workbook, model: Model, marked: list) -> None:
    ws = wb.create_sheet("Notes")
    ws["A1"] = "What this model does and leaves out"
    ws["A1"].font = Font(bold=True, size=13)
    n = 2
    for note in model.notes:
        n += 1
        _put(ws, n, 1, note)
    n += 2
    ws.cell(n, 1, "Figures to look at again, and why").font = Font(bold=True, size=13)
    n += 1
    for j, head in enumerate(["Cell", "Figure", "Level", "Why"]):
        cell = ws.cell(n, 1 + j, head)
        cell.font, cell.fill = WHITE_BOLD, HEAD
    for at, figure, level, why in marked:
        n += 1
        ws.cell(n, 1, at)
        _put(ws, n, 2, figure)
        ws.cell(n, 3, level).fill = MARKED[level]
        _put(ws, n, 4, why)
    if not marked:
        ws.cell(n + 1, 1, "None.")
    ws.column_dimensions["A"].width = 18
    ws.column_dimensions["B"].width = 60
    ws.column_dimensions["C"].width = 9
    ws.column_dimensions["D"].width = 140
