"""Native Excel charts in the downloads (PLAN.md 7.9).

One builder for every workbook: a chart is openpyxl's own chart object over
the sheet's cells, never a picture, so it stays editable and follows the
numbers when a cell changes. Where Excel has no form of its own (or openpyxl
can't write it) the nearest honest one is drawn and the sheet says so:

- a waterfall is stacked columns on hidden bases, from helper cells that are
  formulas over the table (so they move with it): Excel 2016's own waterfall
  is a chart type openpyxl can't write and older readers can't show;
- a fan is a stacked area chart whose bands are the differences between
  percentile rows (formulas again), with the median and the plan as lines;
- a heatmap is the table itself with a colour scale;
- a scatter of simulated paths is a sample of them.

The specs come from api.schemas.ChartSpec (the web describes which chart goes
with which sheet) or are built by the API for a workbook it writes itself.
This module must not import api.schemas (the schema imports `chart_problem`).
"""
from typing import Optional

from openpyxl.chart import AreaChart, BarChart, LineChart, Reference, ScatterChart, Series
from openpyxl.chart.axis import ChartLines
from openpyxl.chart.marker import DataPoint
from openpyxl.chart.series import SeriesLabel, StrRef
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter, quote_sheetname

# The Tape palette (web/src/app/globals.css), kept to colours that read on a
# light or a dark Excel: cyan is the answer, green and red gain and loss,
# grey the plan and context.
TONES = {
    "answer": "62B6CB",
    "gain": "58B28A",
    "loss": "D8665A",
    "attention": "D9A54A",
    "plan": "7F8C94",
    "context": "7F8C94",
}
# Series without a tone, in turn (a stack of facilities)
CYCLE = ("62B6CB", "7F8C94", "58B28A", "D9A54A", "D8665A", "3E8EA3", "B9C3C9", "9AD0DE")
# A fan's bands, outermost lightest, as the screen's opacities over the canvas
FAN_BANDS = ("B7DDE7", "86C6D6")

AXIS_FORMATS = {
    "money": "#,##0", "percent": "0%", "multiple": '0.0"x"', "integer": "#,##0", "number": "#,##0.00",
    "text": "General",
}
WIDTH_CM, HEIGHT_CM = 18.0, 9.0
SLOT_ROWS = 21            # rows each chart takes in the column of charts
# A waterfall writes ten helper formulas a step: a bridge has a handful, so a
# sheet of thousands is a mistake, not a bridge, and would bloat the file
MAX_WATERFALL_STEPS = 200


# ---------------------------------------------------------------------------
# Checking a spec against its sheet (the request's validator calls this)
# ---------------------------------------------------------------------------
def chart_problem(chart, n_rows: int, n_cols: int) -> Optional[str]:
    """Why this chart can't be drawn from a sheet of n_rows data rows and
    n_cols columns, or None. Words for a developer: the web builds specs."""
    kind, by = chart.kind, chart.by
    if n_rows < 1:
        return "the sheet has no rows"
    if kind == "heatmap":
        return None if n_cols >= 2 else "a heatmap needs labels and at least one column of figures"
    if kind == "waterfall" and n_rows > MAX_WATERFALL_STEPS:
        return "a waterfall takes at most %d steps" % MAX_WATERFALL_STEPS
    if kind in ("waterfall", "histogram", "scatter") and by != "columns":
        return "a %s reads columns" % kind
    if kind == "fan" and by != "rows":
        return "a fan reads rows"
    if chart.lines and kind != "fan":
        return "only a fan draws lines over its bands"
    if not chart.series:
        return "no series"
    needs = {"waterfall": 1, "histogram": 1, "scatter": 1, "range": 3}.get(kind)
    if needs and len(chart.series) != needs:
        return "a %s takes %d series" % (kind, needs)
    if kind == "fan" and len(chart.series) < 2:
        return "a fan takes its percentile rows, lowest first"
    if chart.totals is not None and (kind != "waterfall" or len(chart.totals) != n_rows):
        return "totals are a waterfall's, one per row"
    limit = n_rows if by == "rows" else n_cols
    for s in [*chart.series, *chart.lines]:
        if s.index >= limit:
            return "series %d is outside the sheet" % s.index
    if by == "rows" and n_cols < 2:
        return "a chart by rows needs a column of figures"
    if by == "columns" and chart.categories >= n_cols:
        return "the categories column is outside the sheet"
    return None


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def _get(chart, name, default=None):
    return getattr(chart, name, default)


def _ref(ws, col, row, max_col=None, max_row=None):
    return Reference(ws, min_col=col, min_row=row, max_col=max_col or col, max_row=max_row or row)


def _cell(ws, col, row) -> str:
    return "%s!$%s$%d" % (quote_sheetname(ws.title), get_column_letter(col), row)


def _series(values, title_cell: Optional[str], name: Optional[str]) -> Series:
    s = Series(values)
    if name:
        s.tx = SeriesLabel(v=name)
    elif title_cell:
        s.tx = SeriesLabel(strRef=StrRef(title_cell))
    return s


def _paint(series, colour: str, *, line: bool = False, dashed: bool = False, width: int = 22000) -> None:
    gp = series.graphicalProperties or GraphicalProperties()
    series.graphicalProperties = gp
    if line:
        gp.line.solidFill = colour
        gp.line.width = width
        if dashed:
            gp.line.prstDash = "dash"
        series.smooth = False
    else:
        gp.solidFill = colour
        gp.line.solidFill = colour


def _hide_line(series) -> None:
    gp = series.graphicalProperties or GraphicalProperties()
    series.graphicalProperties = gp
    gp.line.noFill = True
    series.marker.symbol = "none"
    series.smooth = False


def _colour(spec_series, i: int) -> str:
    tone = _get(spec_series, "tone")
    return TONES[tone] if tone else CYCLE[i % len(CYCLE)]


def _finish(chart, spec) -> None:
    """Title, axes and size shared by every chart."""
    chart.title = spec.title
    chart.width, chart.height = WIDTH_CM, HEIGHT_CM
    chart.style = 2
    for axis in (chart.x_axis, chart.y_axis):
        axis.delete = False                    # openpyxl 3.1 hides axes unless told
        axis.majorTickMark = "out"
    chart.y_axis.majorGridlines = ChartLines()
    chart.y_axis.majorGridlines.spPr = GraphicalProperties(ln=None)
    chart.y_axis.majorGridlines.spPr.line.solidFill = "D5DDE1"
    if spec.y_title:
        chart.y_axis.title = spec.y_title
    if spec.x_title:
        chart.x_axis.title = spec.x_title
    if spec.y_format:
        chart.y_axis.number_format = AXIS_FORMATS[spec.y_format]
        chart.y_axis.numFmt.sourceLinked = False
    if chart.legend is not None:
        chart.legend.position = "b"


class _Layout:
    """Where a sheet's helper cells and charts go: helpers under the table, charts in a column to its right."""

    def __init__(self, ws, n_rows: int, n_cols: int, helper_cols: int = 0):
        self.ws, self.n_rows, self.n_cols = ws, n_rows, n_cols
        self.helper_row = n_rows + 3          # header row, the data, one blank row
        self.chart_col = max(n_cols, helper_cols) + 2
        self.chart_row = 1

    def helper_block(self, note: str, height: int) -> int:
        """Reserve height rows under a note; returns the block's first row."""
        self.ws.cell(row=self.helper_row, column=1, value=note).data_type = "s"
        first = self.helper_row + 1
        self.helper_row = first + height + 1
        return first

    def place(self, chart, note: Optional[str]) -> None:
        if note:
            self.ws.cell(row=self.chart_row, column=self.chart_col, value=note).data_type = "s"
        self.ws.add_chart(chart, "%s%d" % (get_column_letter(self.chart_col), self.chart_row + 1))
        self.chart_row += SLOT_ROWS


def _by_rows(ws, layout, spec, chart_cls, **kwargs):
    """A bar, line or area chart of rows: the header's columns are the categories."""
    chart = chart_cls(**kwargs)
    last = layout.n_cols
    for i, s in enumerate(spec.series):
        row = s.index + 2
        series = _series(_ref(ws, 2, row, max_col=last), _cell(ws, 1, row), s.name)
        chart.series.append(series)
    chart.set_categories(_ref(ws, 2, 1, max_col=last))
    return chart


def _by_columns(ws, layout, spec, chart_cls, series_list=None, points=None, **kwargs):
    """A chart of columns: the categories column gives the labels."""
    chart = chart_cls(**kwargs)
    last = (min(layout.n_rows, points) if points else layout.n_rows) + 1
    for s in series_list if series_list is not None else spec.series:
        col = s.index + 1
        chart.series.append(_series(_ref(ws, col, 2, max_row=last), _cell(ws, col, 1), s.name))
    chart.set_categories(_ref(ws, spec.categories + 1, 2, max_row=last))
    return chart


def _bars(ws, layout, spec):
    stacked = spec.kind == "stacked_bar"
    make = _by_rows if spec.by == "rows" else _by_columns
    chart = make(ws, layout, spec, BarChart)
    chart.type = "bar" if spec.horizontal else "col"
    chart.grouping = "stacked" if stacked else "clustered"
    if stacked:
        chart.overlap = 100
    chart.gapWidth = 60
    for i, (series, s) in enumerate(zip(chart.series, spec.series)):
        _paint(series, _colour(s, i))
    if spec.horizontal and len(spec.series) == 1:
        # Diverging bars: each bar coloured by its sign when the file was made
        _sign_points(ws, layout, spec, chart.series[0])
        chart.legend = None
    return chart


def _sign_points(ws, layout, spec, series) -> None:
    col = spec.series[0].index + 1
    for i in range(layout.n_rows):
        v = ws.cell(row=i + 2, column=col).value
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            point = DataPoint(idx=i)
            point.graphicalProperties = GraphicalProperties(solidFill=TONES["gain" if v >= 0 else "loss"])
            series.dPt.append(point)


def _lines(ws, layout, spec):
    make = _by_rows if spec.by == "rows" else _by_columns
    chart = make(ws, layout, spec, LineChart)
    for i, (series, s) in enumerate(zip(chart.series, spec.series)):
        _paint(series, _colour(s, i), line=True, dashed=_get(s, "tone") == "plan")
        series.marker.symbol = "none"
    if len(spec.series) == 1:
        chart.legend = None
    return chart


def _histogram(ws, layout, spec):
    chart = _by_columns(ws, layout, spec, BarChart)
    chart.type, chart.grouping, chart.gapWidth = "col", "clustered", 5
    _paint(chart.series[0], _colour(spec.series[0], 0))
    chart.legend = None
    return chart


def _range(ws, layout, spec):
    """Low, middle and high as markers joined by a high-low line (the screen's range rows)."""
    make = _by_rows if spec.by == "rows" else _by_columns
    chart = make(ws, layout, spec, LineChart)
    for series, s, symbol in zip(chart.series, spec.series, ("dash", "diamond", "dash")):
        _hide_line(series)
        series.marker.symbol = symbol
        series.marker.size = 9
        colour = _colour(s, 0) if _get(s, "tone") else TONES["answer" if symbol == "diamond" else "plan"]
        series.marker.graphicalProperties = GraphicalProperties(solidFill=colour)
        series.marker.graphicalProperties.line.solidFill = colour
    chart.hiLowLines = ChartLines()
    chart.hiLowLines.spPr = GraphicalProperties()
    chart.hiLowLines.spPr.line.solidFill = TONES["answer"]
    chart.hiLowLines.spPr.line.width = 28000
    return chart


def _scatter(ws, layout, spec):
    chart = ScatterChart()
    last = (min(layout.n_rows, spec.points) if _get(spec, "points") else layout.n_rows) + 1
    s = spec.series[0]
    col, x_col = s.index + 1, spec.categories + 1
    series = Series(_ref(ws, col, 2, max_row=last), _ref(ws, x_col, 2, max_row=last))
    series.tx = SeriesLabel(v=s.name) if s.name else SeriesLabel(strRef=StrRef(_cell(ws, col, 1)))
    _hide_line(series)
    series.marker.symbol = "circle"
    series.marker.size = 3
    series.marker.graphicalProperties = GraphicalProperties(solidFill=_colour(s, 0))
    series.marker.graphicalProperties.line.noFill = True
    chart.series.append(series)
    chart.legend = None
    return chart


# A waterfall's helper columns after Step: the running total, each bar's two
# ends, then the stacked pieces (bases hidden, then rises, falls and totals)
WATERFALL_HEAD = ("Step", "Running total", "Starts at", "Ends at", "Base above zero", "Base below zero",
                  "Rise above zero", "Rise below zero", "Fall above zero", "Fall below zero", "Total")


def _waterfall_row(ws, k: int, value: str, before: Optional[str], total: bool) -> None:
    """One step's helper formulas in row k. Each bar from lo to hi is split at zero, so a step crossing
    zero stacks right: Excel stacks values above zero and below zero apart."""
    put = lambda col, formula: ws.cell(row=k, column=col, value="=" + formula)   # noqa: E731
    if total:
        put(2, value)
        put(3, "0")
        put(4, value)
        for col in range(5, 11):
            put(col, "0")
        put(11, value)
        return
    put(2, "%s+%s" % (before or "0", value))
    put(3, before or "0")
    put(4, "B%d" % k)
    lo, hi = "MIN(C{0},D{0})".format(k), "MAX(C{0},D{0})".format(k)
    up = "MAX(%s,0)-MAX(%s,0)" % (hi, lo)          # the part above zero
    down = "MIN(%s,0)-MIN(%s,0)" % (lo, hi)        # the part below zero (negative)
    put(5, "MAX(%s,0)" % lo)
    put(6, "MIN(%s,0)" % hi)
    put(7, "IF(D{0}>=C{0},{1},0)".format(k, up))
    put(8, "IF(D{0}>=C{0},{1},0)".format(k, down))
    put(9, "IF(D{0}<C{0},{1},0)".format(k, up))
    put(10, "IF(D{0}<C{0},{1},0)".format(k, down))
    put(11, "0")


def _waterfall(ws, layout, spec):
    """Totals and steps as stacked columns over hidden bases, from helper cells that are formulas over the
    table: rises green, falls red, totals cyan, as the screen's waterfall."""
    n = layout.n_rows
    value_col = get_column_letter(spec.series[0].index + 1)
    label_col = get_column_letter(spec.categories + 1)
    totals = list(spec.totals or [False] * n)
    first = layout.helper_block(
        "For the chart \"%s\": Excel has no waterfall that every version reads, so these cells, worked out "
        "from the table, stack each step on a hidden base." % spec.title, n + 1)
    for j, word in enumerate(WATERFALL_HEAD):
        ws.cell(row=first, column=j + 1, value=word)
    for i in range(n):
        k = first + 1 + i
        ws.cell(row=k, column=1, value="=%s%d" % (label_col, i + 2))
        _waterfall_row(ws, k, "%s%d" % (value_col, i + 2), "B%d" % (k - 1) if i else None, totals[i])
        for c in range(2, 12):
            ws.cell(row=k, column=c).number_format = AXIS_FORMATS["money"]
    rows = (first + 1, first + n)
    chart = BarChart()
    chart.type, chart.grouping, chart.overlap, chart.gapWidth = "col", "stacked", 100, 60
    fills = (None, None, "gain", "gain", "loss", "loss", "answer")
    for col, tone in zip(range(5, 12), fills):
        s = _series(_ref(ws, col, rows[0], max_row=rows[1]), _cell(ws, col, first), None)
        if tone is None:
            s.graphicalProperties = GraphicalProperties(noFill=True)
            s.graphicalProperties.line.noFill = True
        else:
            _paint(s, TONES[tone])
        chart.series.append(s)
    chart.set_categories(_ref(ws, 1, rows[0], max_row=rows[1]))
    chart.legend = None
    return chart


def _fan(ws, layout, spec):
    """Stacked areas between percentile rows (lowest first), the lines drawn over them."""
    last = layout.n_cols
    rows = [s.index + 2 for s in spec.series]
    first = layout.helper_block(
        "For the chart \"%s\": Excel draws no fan, so these cells, worked out from the table, stack the bands "
        "between the percentiles; the lowest is drawn without fill." % spec.title, len(rows) + 1)
    for c in range(2, last + 1):
        ws.cell(row=first, column=c, value="=%s1" % get_column_letter(c))
    for i, r in enumerate(rows):
        k = first + 1 + i
        low, high = ws.cell(row=rows[i - 1] if i else r, column=1).value, ws.cell(row=r, column=1).value
        ws.cell(row=k, column=1, value=str(high) if i == 0 else "%s to %s" % (low, high)).data_type = "s"
        for c in range(2, last + 1):
            col = get_column_letter(c)
            formula = "=%s%d" % (col, r) if i == 0 else "=%s%d-%s%d" % (col, r, col, rows[i - 1])
            ws.cell(row=k, column=c, value=formula).number_format = AXIS_FORMATS["money"]

    bands = AreaChart()
    bands.grouping = "stacked"
    middle = (len(rows) - 1) / 2
    for i in range(len(rows)):
        k = first + 1 + i
        s = _series(_ref(ws, 2, k, max_col=last), _cell(ws, 1, k), None)
        if i == 0:
            s.graphicalProperties = GraphicalProperties(noFill=True)
            s.graphicalProperties.line.noFill = True
        else:
            # Inner bands darker, as the screen's
            inner = abs((i - 0.5) - middle) < 1
            s.graphicalProperties = GraphicalProperties(solidFill=FAN_BANDS[1 if inner else 0])
            s.graphicalProperties.line.noFill = True
        bands.series.append(s)
    bands.set_categories(_ref(ws, 2, 1, max_col=last))

    lines = LineChart()
    for i, s in enumerate(spec.lines):
        row = s.index + 2
        series = _series(_ref(ws, 2, row, max_col=last), _cell(ws, 1, row), s.name)
        _paint(series, _colour(s, 0) if _get(s, "tone") else TONES["answer"], line=True,
               dashed=_get(s, "tone") == "plan")
        series.marker.symbol = "none"
        lines.series.append(series)
    lines.set_categories(_ref(ws, 2, 1, max_col=last))
    bands += lines
    return bands


def _heatmap(ws, layout, spec) -> None:
    """The table's figures coloured from loss (lowest) to gain (highest)."""
    area = "B2:%s%d" % (get_column_letter(layout.n_cols), layout.n_rows + 1)
    ws.conditional_formatting.add(area, ColorScaleRule(
        start_type="min", start_color=TONES["loss"],
        mid_type="percentile", mid_value=50, mid_color="F2F2F2",
        end_type="max", end_color=TONES["gain"]))
    note = spec.note or ("\"%s\": Excel draws no heatmap, so the table's own cells carry a colour scale, "
                         "lowest red to highest green." % spec.title)
    ws.cell(row=layout.chart_row, column=layout.chart_col, value=note).data_type = "s"
    layout.chart_row += 2


DRAW = {
    "bar": _bars, "stacked_bar": _bars, "line": _lines, "histogram": _histogram, "range": _range,
    "scatter": _scatter, "waterfall": _waterfall, "fan": _fan,
}


def draw(ws, charts, n_rows: int, n_cols: int) -> None:
    """Add each chart to the sheet holding its table (header in row 1, data from row 2)."""
    waterfall = any(spec.kind == "waterfall" for spec in charts or [])
    layout = _Layout(ws, n_rows, n_cols, len(WATERFALL_HEAD) if waterfall else 0)
    for spec in charts or []:
        problem = chart_problem(spec, n_rows, n_cols)
        if problem:
            raise ValueError("chart %r: %s" % (spec.title, problem))
        if spec.kind == "heatmap":
            _heatmap(ws, layout, spec)
            continue
        chart = DRAW[spec.kind](ws, layout, spec)
        _finish(chart, spec)
        layout.place(chart, spec.note)
