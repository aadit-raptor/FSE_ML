"""A downloaded workbook's charts (PLAN.md 7.9), for the browser tests
(``web/e2e/charts.spec.ts``)::

    python -m tests.chart_check download.xlsx

prints ``{"charts": {title: {"sheet", "kind", "refs", "values"}}, "colour_scales":
{sheet: [ranges]}}``: every chart's title, the sheet its series read, the
chart types it holds, each range its series read and that range's values,
worked out by the tests' own Excel reader from the file's cells and formulas
(so a chart over helper formulas gives the figures it would draw).
"""
from __future__ import annotations

import html
import io
import json
import re
import sys
import zipfile

from openpyxl import load_workbook
from openpyxl.utils import range_boundaries

from tests.research_support import workbook_values

_REF = re.compile(r"^'?(?P<sheet>.+?)'?!(?P<range>\$?[A-Z]+\$?\d+(?::\$?[A-Z]+\$?\d+)?)$")


def _charts(data: bytes) -> list[tuple[str, list[str], list[str]]]:
    out = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in sorted(n for n in z.namelist() if re.match(r"xl/charts/chart\d+\.xml$", n)):
            xml = z.read(name).decode("utf-8")
            title = html.unescape("".join(re.findall(r"<a:t>([^<]*)</a:t>", re.split(r"</(?:c:)?title>", xml)[0])))
            kinds = re.findall(r"<(?:c:)?(\w+Chart)>", xml)
            refs = [html.unescape(f) for f in re.findall(r"<(?:c:)?f>([^<]*)</(?:c:)?f>", xml)]
            out.append((title, kinds, refs))
    return out


def _values(wb, ref: str):
    m = _REF.match(ref)
    sheet = m.group("sheet").replace("''", "'")
    get = workbook_values(wb, sheet)
    min_col, min_row, max_col, max_row = range_boundaries(m.group("range").replace("$", ""))
    cells = []
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            v = get("%s!%s" % (sheet, wb[sheet].cell(row=row, column=col).coordinate))
            cells.append(v if isinstance(v, (int, float, str)) or v is None else str(v))
    return sheet, cells


def main(path: str) -> dict:
    data = open(path, "rb").read()
    wb = load_workbook(io.BytesIO(data))
    charts = {}
    for title, kinds, refs in _charts(data):
        values, sheet = {}, None
        for ref in refs:
            sheet, values[ref] = _values(wb, ref)
        charts[title] = {"sheet": sheet, "kinds": kinds, "refs": refs, "values": values}
    scales = {ws.title: [str(r.sqref) for r in ws.conditional_formatting
                         if any(rule.type == "colorScale" for rule in r.rules)] for ws in wb.worksheets}
    return {"charts": charts, "colour_scales": {k: v for k, v in scales.items() if v}}


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1])))
