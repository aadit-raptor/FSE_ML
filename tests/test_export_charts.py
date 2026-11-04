"""Charts in every Excel download (PLAN.md 7.9).

A chart is a native Excel chart over the sheet's own cells, so these tests
read each chart's series ranges back (openpyxl, and the chart XML itself for
the parts openpyxl's reader drops) and check they point at the right cells;
the helper cells a waterfall or a fan needs are formulas over the table,
worked out here by the tests' own Excel reader, so changing a cell of the
table moves the chart.
"""
import html
import io
import re
import zipfile

import openpyxl
import pytest
from fastapi.testclient import TestClient

from api.charts import WATERFALL_HEAD
from api.main import app
from tests import research_support as support

client = TestClient(app)

YEARS = ["Y1", "Y2", "Y3"]
PL = {"name": "P&L", "columns": ["", *YEARS],
      "rows": [["Revenue", 100.0, 110.0, 121.0], ["Gross profit", 40.0, 44.0, 48.4], ["EBITDA", 20.0, 22.0, 25.0]]}
# The default deal's equity bridge shape: two totals, a fall, a rise, a fall that crosses zero
BRIDGE = {"name": "Bridge", "columns": ["Step", "Value"],
          "rows": [["Entry", 30.0], ["Fees", -5.0], ["Growth", 40.0], ["Write-down", -90.0], ["Exit", -25.0]]}
BRIDGE_TOTALS = [True, False, False, False, True]


def _post(sheets, **extra):
    r = client.post("/api/export/workbook", json={"sheets": sheets, **extra})
    assert r.status_code == 200, r.text
    return r.content


def _charts_xml(data: bytes) -> dict:
    """Each chart's title and every range its series read, from the file itself."""
    out = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in sorted(n for n in z.namelist() if re.match(r"xl/charts/chart\d+\.xml$", n)):
            xml = z.read(name).decode("utf-8")
            title = html.unescape("".join(re.findall(r"<a:t>([^<]*)</a:t>", re.split(r"</(?:c:)?title>", xml)[0])))
            out[title] = {"refs": [html.unescape(f) for f in re.findall(r"<(?:c:)?f>([^<]*)</(?:c:)?f>", xml)], "xml": xml}
    return out


def _cat(series) -> str:
    ref = series.cat.strRef or series.cat.numRef
    return ref.f


def _fill(series) -> str:
    """A series' fill colour as openpyxl reads it back."""
    colour = series.graphicalProperties.solidFill
    return colour if isinstance(colour, str) else (colour.srgbClr if isinstance(colour.srgbClr, str) else colour.srgbClr.val)


def _series_refs(chart) -> list:
    return [s.val.numRef.f for s in chart.series]


def test_a_bar_chart_by_rows_reads_the_rows_it_names_and_the_header_years():
    data = _post([{**PL, "charts": [{"kind": "bar", "title": "Revenue and EBITDA",
                                     "series": [{"index": 0, "tone": "context"}, {"index": 2, "tone": "answer"}]}]}])
    ws = openpyxl.load_workbook(io.BytesIO(data))["P&L"]
    [chart] = ws._charts
    assert type(chart).__name__ == "BarChart"
    assert _series_refs(chart) == ["'P&L'!$B$2:$D$2", "'P&L'!$B$4:$D$4"]       # Revenue and EBITDA, not gross profit
    assert {_cat(s) for s in chart.series} == {"'P&L'!$B$1:$D$1"}
    # The legend reads the row's own label cell, so renaming the row renames the series
    assert [s.tx.strRef.f for s in chart.series] == ["'P&L'!$A$2", "'P&L'!$A$4"]
    # Cyan for the answer, grey for context: the Tape palette
    assert _fill(chart.series[1]) == "62B6CB"
    assert _fill(chart.series[0]) == "7F8C94"


def test_a_chart_by_columns_reads_its_columns_and_the_categories_column():
    sheet = {"name": "Senior", "columns": ["Year", "Opening", "Closing"],
             "rows": [["Y1", 400.0, 380.0], ["Y2", 380.0, 350.0]],
             "charts": [{"kind": "bar", "by": "columns", "title": "Senior: closing", "series": [{"index": 2}]}]}
    [chart] = openpyxl.load_workbook(io.BytesIO(_post([sheet])))["Senior"]._charts
    assert _series_refs(chart) == ["'Senior'!$C$2:$C$3"]
    assert _cat(chart.series[0]) == "'Senior'!$A$2:$A$3"


def _waterfall_book():
    data = _post([{**BRIDGE, "charts": [{"kind": "waterfall", "by": "columns", "title": "Equity bridge",
                                         "series": [{"index": 1}], "totals": BRIDGE_TOTALS}]}])
    return data, openpyxl.load_workbook(io.BytesIO(data))


def _helper(get, first: int, n: int) -> list:
    return [{WATERFALL_HEAD[c - 1]: get("%s%d" % ("ABCDEFGHIJK"[c - 1], first + 1 + i)) for c in range(2, 12)}
            for i in range(n)]


def _first_helper_row(ws) -> int:
    return next(r for r in range(1, 40) if ws.cell(row=r, column=1).value == "Step" and r > 1)


def test_a_waterfall_stacks_each_step_on_a_hidden_base_worked_out_from_the_table():
    data, wb = _waterfall_book()
    ws = wb["Bridge"]
    first = _first_helper_row(ws)
    [chart] = ws._charts
    # Seven stacked series over the helper columns E:K, under the table; bases drawn without fill
    assert _series_refs(chart) == ["'Bridge'!$%s$%d:$%s$%d" % (c, first + 1, c, first + 5) for c in "EFGHIJK"]
    assert chart.grouping == "stacked" and chart.overlap == 100
    assert chart.series[0].graphicalProperties.noFill and chart.series[1].graphicalProperties.noFill
    assert [_fill(chart.series[i]) for i in (2, 4, 6)] == ["58B28A", "D8665A", "62B6CB"]
    # Labels are formulas over the table's own labels
    assert ws.cell(row=first + 2, column=1).value == "=A3"
    get = support.workbook_values(wb, "Bridge")
    rows = _helper(get, first, 5)
    # Entry 30 (total) -> Fees -5 -> Growth +40 -> Write-down -90 crosses zero -> Exit -25 (total)
    assert [r["Running total"] for r in rows] == [30, 25, 65, -25, -25]
    fees, growth, down = rows[1], rows[2], rows[3]
    assert (fees["Base above zero"], fees["Fall above zero"], fees["Rise above zero"]) == (25, 5, 0)
    assert (growth["Base above zero"], growth["Rise above zero"], growth["Fall above zero"]) == (25, 40, 0)
    # From 65 down to -25: 65 above zero and 25 below, both red, no base
    assert (down["Base above zero"], down["Base below zero"], down["Fall above zero"], down["Fall below zero"]) == (0, 0, 65, -25)
    assert rows[0]["Total"] == 30 and rows[4]["Total"] == -25 and growth["Total"] == 0
    # Every step's pieces reach exactly its two ends: the stack above zero to the higher, below to the lower
    for r, total in zip(rows, BRIDGE_TOTALS):
        above = r["Base above zero"] + r["Rise above zero"] + r["Fall above zero"]
        below = r["Base below zero"] + r["Rise below zero"] + r["Fall below zero"]
        if total:
            assert above == below == 0
        else:
            assert above == max(r["Starts at"], r["Ends at"], 0) and below == min(r["Starts at"], r["Ends at"], 0)


def test_changing_a_cell_of_the_table_moves_the_waterfall():
    _, wb = _waterfall_book()
    ws = wb["Bridge"]
    first = _first_helper_row(ws)
    ws["B4"] = 70.0                                         # growth 40 -> 70
    rows = _helper(support.workbook_values(wb, "Bridge"), first, 5)
    assert rows[2]["Rise above zero"] == 70 and rows[2]["Running total"] == 95
    assert rows[3]["Fall above zero"] == 90 and rows[3]["Fall below zero"] == 0      # no longer crosses zero


def test_a_fan_stacks_the_bands_between_percentile_rows_and_draws_its_lines_over_them():
    bands = {"name": "Revenue bands", "columns": ["Percentile", *YEARS],
             "rows": [["P5", 90.0, 95.0, 99.0], ["P25", 95.0, 102.0, 108.0], ["P50", 100.0, 108.0, 116.0],
                      ["P75", 104.0, 114.0, 125.0], ["P95", 110.0, 122.0, 136.0], ["Plan", 100.0, 107.0, 115.0]],
             "charts": [{"kind": "fan", "title": "Revenue fan", "series": [{"index": i} for i in range(5)],
                         "lines": [{"index": 2, "tone": "answer"}, {"index": 5, "tone": "plan"}]}]}
    data = _post([bands])
    wb = openpyxl.load_workbook(io.BytesIO(data))
    ws = wb["Revenue bands"]
    first = next(r for r in range(9, 20) if isinstance(ws.cell(row=r, column=2).value, str)
                 and ws.cell(row=r, column=2).value == "=B1")
    get = support.workbook_values(wb, "Revenue bands")
    stacked = [[get("%s%d" % (c, first + 1 + i)) for c in "BCD"] for i in range(5)]
    assert stacked[0] == [90, 95, 99]                                      # the lowest band, unfilled
    assert stacked[1] == [5, 7, 9] and stacked[4] == [6, 8, 11]            # P25-P5 ... P95-P75
    # Summed back, the bands give the top percentile: the fan is the table's
    assert [sum(col) for col in zip(*stacked)] == [110, 122, 136]
    refs = _charts_xml(data)["Revenue fan"]["refs"]
    bands_refs = ["'Revenue bands'!$B$%d:$D$%d" % (first + 1 + i, first + 1 + i) for i in range(5)]
    lines_refs = ["'Revenue bands'!$B$4:$D$4", "'Revenue bands'!$B$7:$D$7"]    # P50 and the plan, from the table
    assert all(ref in refs for ref in bands_refs + lines_refs)
    xml = _charts_xml(data)["Revenue fan"]["xml"]
    assert re.search(r"<(c:)?areaChart>", xml) and re.search(r"<(c:)?lineChart>", xml)
    assert 'prstDash val="dash"' in xml                                    # the plan is dashed, as on screen
    ws["B6"] = 140.0                                                       # P95 in Y1 moves the top band
    assert support.workbook_values(wb, "Revenue bands")("B%d" % (first + 5)) == 36


def test_a_heatmap_is_the_table_with_a_colour_scale():
    sens = {"name": "IRR sensitivity", "columns": ["Exit multiple", "3y", "5y"],
            "rows": [[8.0, 0.05, 0.08], [10.0, 0.15, 0.17], [12.0, 0.25, 0.24]],
            "charts": [{"kind": "heatmap", "title": "IRR by exit multiple and hold"}]}
    ws = openpyxl.load_workbook(io.BytesIO(_post([sens])))["IRR sensitivity"]
    ranges = {str(r.sqref): r.rules for r in ws.conditional_formatting}
    [rule] = ranges["B2:C4"]
    assert rule.type == "colorScale"
    assert [c.rgb[-6:] for c in rule.colorScale.color] == ["D8665A", "F2F2F2", "58B28A"]
    assert not ws._charts


def test_range_histogram_scatter_and_diverging_bars_read_their_cells():
    scen = {"name": "Scenarios", "columns": ["Scenario", "Mean", "P5", "P95"],
            "rows": [["Base", 0.2, 0.05, 0.33], ["Bear", 0.1, -0.05, 0.25]],
            "charts": [{"kind": "range", "by": "columns", "title": "Range",
                        "series": [{"index": 2}, {"index": 1, "tone": "answer"}, {"index": 3}]}]}
    hist = {"name": "Hist", "columns": ["From", "To", "Density"], "rows": [[0.0, 0.1, 2.0], [0.1, 0.2, 5.0]],
            "charts": [{"kind": "histogram", "by": "columns", "title": "Hist", "series": [{"index": 2}]}]}
    paths = {"name": "Paths", "columns": ["IRR", "Exit"], "rows": [[0.1 * i, 8.0 + i] for i in range(10)],
             "charts": [{"kind": "scatter", "by": "columns", "title": "Scatter", "categories": 1,
                         "series": [{"index": 0}], "points": 4}]}
    attr = {"name": "Attribution", "columns": ["Part", "Value"], "rows": [["EBITDA", 12.0], ["Multiple", -7.0]],
            "charts": [{"kind": "bar", "by": "columns", "horizontal": True, "title": "Missed", "series": [{"index": 1}]}]}
    data = _post([scen, hist, paths, attr])
    wb = openpyxl.load_workbook(io.BytesIO(data))
    [rng] = wb["Scenarios"]._charts
    assert _series_refs(rng) == ["'Scenarios'!$C$2:$C$3", "'Scenarios'!$B$2:$B$3", "'Scenarios'!$D$2:$D$3"]
    assert rng.hiLowLines is not None
    [h] = wb["Hist"]._charts
    assert _series_refs(h) == ["'Hist'!$C$2:$C$3"] and h.gapWidth == 5
    [sc] = wb["Paths"]._charts
    # A sampled scatter: the first four paths only, IRR against the exit multiple
    assert sc.series[0].yVal.numRef.f == "'Paths'!$A$2:$A$5"
    assert sc.series[0].xVal.numRef.f == "'Paths'!$B$2:$B$5"
    [bar] = wb["Attribution"]._charts
    assert bar.type == "bar" and _series_refs(bar) == ["'Attribution'!$B$2:$B$3"]
    # Gains green, losses red, as the screen's diverging bars
    assert [_fill(p) for p in bar.series[0].dPt] == ["58B28A", "D8665A"]


def test_charts_sit_beside_the_table_not_over_it():
    data, wb = _waterfall_book()
    [chart] = wb["Bridge"]._charts
    # The table and its helper cells run to column K; the chart starts after them
    assert chart.anchor._from.col >= len(WATERFALL_HEAD) + 1
    wb2 = openpyxl.load_workbook(io.BytesIO(_post([{**PL, "charts": [
        {"kind": "line", "title": "A", "series": [{"index": 0}]},
        {"kind": "line", "title": "B", "series": [{"index": 2}]}]}])))
    a, b = wb2["P&L"]._charts
    assert a.anchor._from.col == b.anchor._from.col == 5 and b.anchor._from.row > a.anchor._from.row + 15


@pytest.mark.parametrize("chart", [
    {"kind": "bar", "title": "x", "series": [{"index": 3}]},                                  # no fourth row
    {"kind": "waterfall", "title": "x", "series": [{"index": 1}]},                           # a waterfall reads columns
    {"kind": "waterfall", "by": "columns", "title": "x", "series": [{"index": 1}], "totals": [True]},
    {"kind": "range", "by": "columns", "title": "x", "series": [{"index": 1}]},
    {"kind": "line", "title": "x", "series": [{"index": 0}], "lines": [{"index": 1}]},
    {"kind": "bar", "by": "columns", "categories": 9, "title": "x", "series": [{"index": 1}]},
    {"kind": "bar", "title": "x", "series": []},
])
def test_a_chart_the_sheet_cannot_hold_is_refused(chart):
    r = client.post("/api/export/workbook", json={"sheets": [{**PL, "charts": [chart]}]})
    assert r.status_code == 422


def test_a_waterfall_of_thousands_of_steps_is_refused():
    rows = [["Step %d" % i, 1.0] for i in range(201)]
    r = client.post("/api/export/workbook", json={"sheets": [{"name": "Long", "columns": ["Step", "Value"], "rows": rows,
        "charts": [{"kind": "waterfall", "by": "columns", "title": "x", "series": [{"index": 1}]}]}]})
    assert r.status_code == 422 and "at most 200 steps" in r.text


def test_without_charts_the_workbook_is_as_before():
    wb = openpyxl.load_workbook(io.BytesIO(_post([PL])))
    assert not wb["P&L"]._charts and wb["P&L"].max_row == 4


def test_the_simulation_download_charts_the_same_figures_the_screen_draws():
    req = {"mc": {"n": 4000}, "seed": 7}
    run = client.post("/api/montecarlo/run", json=req).json()
    r = client.post("/api/export/montecarlo-sample", json=req)
    assert r.status_code == 200
    wb = openpyxl.load_workbook(io.BytesIO(r.content))
    assert wb.sheetnames == ["Paths", "IRR distribution", "MOIC distribution", "IRR by percentile", "Drivers",
                             "Heatmap", "About"]
    irr = wb["IRR distribution"]
    n = len(run["irr_histogram"]["density"])
    assert [irr.cell(row=i + 2, column=3).value for i in range(n)] == pytest.approx(run["irr_histogram"]["density"])
    assert [irr.cell(row=i + 2, column=1).value for i in range(n)] == pytest.approx(run["irr_histogram"]["edges"][:-1])
    [chart] = irr._charts
    assert _series_refs(chart) == ["'IRR distribution'!$C$2:$C$%d" % (n + 1)]
    moic = wb["MOIC distribution"]
    assert moic.cell(row=2, column=3).value == pytest.approx(run["moic_histogram"]["density"][0])
    cdf = wb["IRR by percentile"]
    assert [cdf.cell(row=i + 2, column=2).value for i in range(101)] == pytest.approx(run["irr_cdf"]["values"])
    [line] = cdf._charts
    assert _series_refs(line) == ["'IRR by percentile'!$B$2:$B$102"]
    drivers = wb["Drivers"]
    assert [drivers.cell(row=i + 2, column=1).value for i in range(len(run["drivers"]))] == [d["driver"] for d in run["drivers"]]
    assert drivers.cell(row=2, column=2).value == pytest.approx(run["drivers"][0]["spearman_rho"])
    heat = wb["Heatmap"]
    grid = run["heatmap"]["irr"]
    cells = [heat.cell(row=i + 2, column=j + 2).value for i in range(len(grid)) for j in range(len(grid[0]))]
    assert cells == pytest.approx([v for row in grid for v in row])
    assert any(r.type == "colorScale" for rules in heat.conditional_formatting for r in rules.rules)
    # The Drivers step's scatter: a sample of the paths in this very file
    paths = wb["Paths"]
    head = [c.value for c in paths[1]]
    scatters = {c.title.tx.rich.p[0].r[0].t: c for c in paths._charts}
    sc = scatters["IRR against Exit Multiple (sampled paths)"]
    irr_col = openpyxl.utils.get_column_letter(head.index("IRR") + 1)
    exit_col = openpyxl.utils.get_column_letter(head.index("Exit Multiple") + 1)
    assert sc.series[0].yVal.numRef.f == "'Paths'!$%s$2:$%s$2001" % (irr_col, irr_col)
    assert sc.series[0].xVal.numRef.f == "'Paths'!$%s$2:$%s$2001" % (exit_col, exit_col)


if support.soffice_wanted():
    def test_libreoffice_works_the_chart_cells_out_as_the_reader_does():
        """A real spreadsheet program agrees with the tests' reader on every helper cell."""
        _, wb = _waterfall_book()
        ours = support.workbook_values(wb, "Bridge")
        first = _first_helper_row(wb["Bridge"])
        theirs = support.recalculated_by_libreoffice(wb)["Bridge"]
        for i in range(5):
            for c in range(2, 12):
                address = "%s%d" % ("ABCDEFGHIJK"[c - 1], first + 1 + i)
                assert theirs[address].value == pytest.approx(ours(address)), address
