"""Excel downloads.

The web app sends the tables it is showing; the API writes them to a workbook.
Replaces the Streamlit app's download buttons (pandas + openpyxl there too).
"""
import io
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from api.auth import AuthUser, require_user

from api.deps import resolve_settings
from api.limits import simulation_slot
from api.routers.deals import store_errors
from api import charts, deal_workbook
from api.schemas import (ChartSeries, ChartSpec, DealInputsIn, DealModelRequest, MonteCarloRequest,
                         WorkbookRequest)
from api.serialize import histogram, percentile_curve
from core.deal import DealInputs, in_millions, run_deal
from core.money import in_unit
from core.model_version import stamp
from core.montecarlo import (MCInputs, analysis_sample, apply_scenario, build_sim_params, driver_sensitivity,
                             growth_exit_heatmap, mc_in_millions)
from db import deals as deals_store
from simulation.vectorized_simulation import run_vectorized_simulation_full

router = APIRouter(prefix="/export", tags=["export"])

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
# Who made every workbook: the About sheet's first row and the file's author
SOURCE = "variater.com"
SAMPLE_ROWS = 10_000
SCATTER_POINTS = 2_000        # the Drivers step's scatter shows as many
# Excel forbids these in sheet names
_BAD_SHEET_CHARS = re.compile(r"[\[\]:*?/\\]")


def _sheet_name(name: str, used: set) -> str:
    base = _BAD_SHEET_CHARS.sub(" ", name).strip()[:31] or "Sheet"
    candidate, n = base, 2
    while candidate.lower() in used:
        suffix = f" {n}"
        candidate, n = base[: 31 - len(suffix)] + suffix, n + 1
    used.add(candidate.lower())
    return candidate


def _safe_filename(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "export"
    return stem if stem.lower().endswith(".xlsx") else f"{stem}.xlsx"


# Per-cell number formats (PLAN.md 2.3a). The cells hold numbers; Excel shows
# these standard formats with the reader's own separators (1.234,5 in a
# German Excel), so the workbook itself never needs a locale.
NUMBER_FORMATS = {
    "money": "#,##0.0",
    "percent": "0.0%",           # the cell holds a fraction: 0.2116 shows 21.2%
    "multiple": '0.00"x"',
    "integer": "#,##0",
    "number": "#,##0.00",
    "text": "@",
}
# Formats whose grouping follows the account's lakh and crore choice
_GROUPED_DECIMALS = {"money": 1, "integer": 0, "number": 2}
_LAKH = 100_000


def lakh_pattern(value: float, decimals: int) -> str:
    """An Excel format grouping this value in lakh and crore: 1,23,45,678.9.

    Excel has no Indian grouping and its conditional formats drop the minus
    sign, so each cell gets a single-section pattern sized to its own digits
    (a minus sign then shows as usual).
    """
    digits = len(str(int(round(abs(value), decimals))))
    frac = "." + "0" * decimals if decimals else ""
    if digits <= 5:                      # below a lakh both groupings agree
        return "#,##0" + frac
    groups = ["##0"]
    remaining = digits - 3
    while remaining > 0:
        groups.insert(0, "#" * min(2, remaining))
        remaining -= 2
    return "\\,".join(groups) + frac


def _number_format(kind: Optional[str], value, grouping: str) -> Optional[str]:
    if kind is None or value is None:
        return None
    if kind == "text":
        return NUMBER_FORMATS["text"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None                       # labels keep Excel's default
    if grouping == "lakh" and kind in _GROUPED_DECIMALS and abs(value) >= _LAKH:
        return lakh_pattern(value, _GROUPED_DECIMALS[kind])
    return NUMBER_FORMATS[kind]


def _apply_formats(ws, df: pd.DataFrame, column_formats, row_formats, grouping: str) -> None:
    """Set each data cell's number format: its row's format, else its column's."""
    column_formats, row_formats = list(column_formats or []), list(row_formats or [])
    for r in range(len(df)):
        row_kind = row_formats[r] if r < len(row_formats) else None
        for c in range(df.shape[1]):
            kind = row_kind or (column_formats[c] if c < len(column_formats) else None)
            if kind is None:
                continue
            cell = ws.cell(row=r + 2, column=c + 1)     # row 1 is the header
            number_format = _number_format(kind, cell.value, grouping)
            if number_format:
                cell.number_format = number_format


def workbook_bytes(sheets, grouping: str = "locale") -> bytes:
    """sheets: iterable of (name, DataFrame), (name, DataFrame, column
    formats, row formats) or that and a list of charts (api/charts.py, PLAN.md
    7.9) drawn from the sheet's own cells. Always writes at least one sheet."""
    buf = io.BytesIO()
    used: set = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        wrote = False
        for name, df, *rest in sheets:
            formats, chart_specs = rest[:2], (rest[2] if len(rest) > 2 else None)
            sheet_name = _sheet_name(name, used)
            df.to_excel(w, sheet_name=sheet_name, index=False)
            if any(formats):
                _apply_formats(w.sheets[sheet_name], df, *formats, grouping)
            if chart_specs:
                charts.draw(w.sheets[sheet_name], chart_specs, len(df), df.shape[1])
            wrote = True
        if not wrote:
            pd.DataFrame({"Note": ["No data"]}).to_excel(w, sheet_name="Data", index=False)
        # What Excel shows under File > Info: made by variater.com, not by a library
        w.book.properties.creator = SOURCE
        w.book.properties.lastModifiedBy = SOURCE
    return buf.getvalue()


def source_rows() -> tuple[list, list]:
    """Items and values opening every About sheet: where and when it was made."""
    made = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return ["Source", "Generated (UTC)"], [SOURCE, made]


# The simulated paths' columns: rates are fractions, multiples times EBITDA
SAMPLE_FORMATS = {
    "IRR": "percent", "Growth": "percent", "Interest": "percent", "Gross Margin": "percent",
    "EBITDA Shock": "percent", "MOIC": "multiple", "Exit Multiple": "multiple",
    "Exit Equity": "money", "Exit EV": "money", "Exit EBITDA": "money", "Net Debt Exit": "money",
}

UNIT_WORDS = {"thousands": "thousands", "lakhs": "lakhs", "millions": "millions", "crores": "crores",
              "billions": "billions"}


def money_rows(money) -> tuple[list, list]:
    """Items and values saying what a workbook's money columns are counted in."""
    return (["Currency", "Money unit"], [money.currency, UNIT_WORDS[money.unit]])


def model_rows(model: dict) -> tuple[list, list]:
    """Items and values saying which model made a workbook's figures (PLAN.md 3.1)."""
    items = ["Model engine version", "Model commit", "Settings fingerprint", "Data vintage"]
    values = [model["engine_version"], model.get("commit") or "local build",
              model.get("settings_fingerprint") or "none", model["data_vintage"]]
    for name, edition in (model.get("data_sets") or {}).items():
        items.append(f"Data: {name}")
        values.append(edition)
    return items, values


def _xlsx_response(data: bytes, filename: str) -> Response:
    return Response(data, media_type=XLSX, headers={
        "Content-Disposition": f'attachment; filename="{_safe_filename(filename)}"'})


DealFrom = Query(None, description="The saved deal the figures come from: the export is then an entry "
                                    "in its history (PLAN.md 3.3). Must be the caller's.")


def _record_export(user: AuthUser, deal_id: Optional[uuid.UUID], export: str) -> None:
    """An export from a saved deal goes in its history before the file is
    handed over, so every file a deal gave has its entry."""
    if deal_id is None:
        return
    with store_errors():
        deals_store.record_export(user.subject, deal_id, export)


@router.post("/workbook", response_class=Response,
             responses={200: {"content": {XLSX: {}}, "description": "Excel workbook"}})
def post_workbook(req: WorkbookRequest, deal_id: Optional[uuid.UUID] = DealFrom,
                  user: AuthUser = Depends(require_user)):
    """Write the given tables to an Excel workbook, one sheet per table."""
    frames = []
    for sheet in req.sheets:
        rows = [row + [None] * (len(sheet.columns) - len(row)) for row in sheet.rows]
        frames.append((sheet.name, pd.DataFrame(rows, columns=sheet.columns),
                       sheet.column_formats, sheet.row_formats, sheet.charts))
    source_items, source_values = source_rows()
    items, values = money_rows(req.money) if req.money is not None else ([], [])
    # The stamp of the result on screen; an older caller sends none, and then
    # the workbook says which model this API runs
    model_items, model_values = model_rows(req.model.model_dump() if req.model else stamp(None))
    frames.append(("About", pd.DataFrame({"Item": source_items + items + model_items,
                                          "Value": source_values + values + model_values})))
    data = workbook_bytes(frames, req.grouping)
    _record_export(user, deal_id, "workbook")
    return _xlsx_response(data, req.filename)


# What the About sheet of a deal's live workbook says beside the usual rows
DEAL_MODEL_NOTES = (
    ("What this is", "The deal model as live formulas: change a blue input on the Deal model sheet and every "
                     "line follows, worked out by the spreadsheet program, not by Variater."),
    ("Rounding", "Variater rounds money to two decimals of a million at each line; this sheet does not, so "
                 "a very small deal can differ in the last digits."),
    ("Hold", "The columns are laid out for the deal's holding period; another hold needs a new download."),
)


def deal_model(inputs: DealInputsIn, settings: dict, *, name: str, grouping: str = "locale",
               saved: Optional[str] = None):
    """A deal's live-formula workbook as cells (api/deal_workbook.py), the app's
    run of the same deal beside it, the money and the model stamp."""
    cfg = resolve_settings(settings)
    model = stamp(cfg)
    deal = DealInputs(**inputs.model_dump())
    result = run_deal(*in_millions(deal, cfg))
    money = inputs.money()
    label = "%s %s" % (money.currency, UNIT_WORDS[money.unit])
    source_items, source_values = source_rows()
    money_items, money_values = money_rows(money)
    model_items, model_values = model_rows(model)
    about = ([*source_items, "Deal", *money_items, *(k for k, _ in DEAL_MODEL_NOTES), *model_items],
             [*source_values, name, *money_values, *(v for _, v in DEAL_MODEL_NOTES), *model_values])
    sheets = deal_workbook.sheets(deal, cfg, title=name, money_label=label, result=result,
                                  factor=in_unit(1.0, inputs.unit), app_irr=result.irr, grouping=grouping,
                                  saved=saved, about=about)
    return sheets, result, money, model


@router.post("/deal-model", response_class=Response,
             responses={200: {"content": {XLSX: {}}, "description": "Excel workbook of live formulas"}})
def post_deal_model(req: DealModelRequest, deal_id: Optional[uuid.UUID] = DealFrom,
                    user: AuthUser = Depends(require_user)):
    """The deal as a workbook whose cells are working formulas (PLAN.md 7.3a): every debt type, the tax
    rules and the money in the deal's own currency and unit; recalculated, it gives the app's IRR."""
    sheets, *_ = deal_model(req.inputs, req.settings, name=req.name, grouping=req.grouping,
                            saved=str(deal_id) if deal_id else None)
    buf = io.BytesIO()
    deal_workbook.write(sheets).save(buf)
    _record_export(user, deal_id, "deal_model")
    return _xlsx_response(buf.getvalue(), "%s model.xlsx" % req.name)


def _histogram_sheet(name: str, values, bins: int, fmt: str, title: str):
    h = histogram(values, bins)
    edges = h["edges"]
    df = pd.DataFrame({"From": edges[:-1], "To": edges[1:], "Density": h["density"]})
    chart = ChartSpec(kind="histogram", by="columns", title=title, categories=0,
                      series=[ChartSeries(index=2, tone="answer")], y_title="Density", x_title="From")
    return (name, df, [fmt, fmt, "number"], None, [chart])


def simulation_sheets(sim, req: MonteCarloRequest, params, mc, deal) -> list:
    """The Distribution, Drivers and Heatmap steps' charts over cells holding the same figures the screen
    draws (PLAN.md 7.9): the run's own histograms, percentiles, rank correlations and IRR grid."""
    sheets = [
        _histogram_sheet("IRR distribution", sim.irr, req.histogram_bins, "percent", "IRR distribution"),
        _histogram_sheet("MOIC distribution", sim.moic, req.histogram_bins, "multiple", "MOIC distribution"),
    ]
    cdf = percentile_curve(sim.irr)
    sheets.append(("IRR by percentile", pd.DataFrame({"Percentile": cdf["percentiles"], "IRR": cdf["values"]}),
                   ["integer", "percent"], None,
                   [ChartSpec(kind="line", by="columns", title="IRR by percentile", categories=0,
                              series=[ChartSeries(index=1, tone="answer")], x_title="Percentile",
                              y_format="percent")]))
    drivers = driver_sensitivity(analysis_sample(sim))
    sheets.append(("Drivers", pd.DataFrame({"Driver": [d for d, _ in drivers],
                                            "Rank correlation with IRR": [float(r) for _, r in drivers]}),
                   ["text", "number"], None,
                   [ChartSpec(kind="bar", by="columns", horizontal=True, title="What moves the IRR",
                              categories=0, series=[ChartSeries(index=1)],
                              note="Spearman rank correlation of each draw with IRR, as on the Drivers step.")]))
    growth, multiples, grid = growth_exit_heatmap(params, mc, deal)
    heat = pd.DataFrame([[float(m), *(float(v) for v in row)] for m, row in zip(multiples, grid)],
                        columns=["Exit multiple", *("Growth %.1f%%" % (g * 100) for g in growth)])
    sheets.append(("Heatmap", heat, ["multiple", *(["percent"] * len(growth))], None,
                   [ChartSpec(kind="heatmap", title="IRR by growth and exit multiple",
                              note="IRR by growth (across) and exit multiple (down): each cell a full deal-model "
                                   "run at the simulation's mean assumptions. Excel draws no heatmap, so the "
                                   "cells carry a colour scale, lowest red to highest green.")]))
    return sheets


@router.post("/montecarlo-sample", response_class=Response,
             responses={200: {"content": {XLSX: {}}, "description": "Excel workbook"}})
@simulation_slot
def post_montecarlo_sample(req: MonteCarloRequest, deal_id: Optional[uuid.UUID] = DealFrom,
                           user: AuthUser = Depends(require_user)):
    """Simulate with the same inputs as /montecarlo/run and export up to 10,000 paths.

    With a fixed seed the paths are exactly those behind the on-screen results.
    """
    cfg = resolve_settings(req.settings, check_correlations=True)
    model_items, model_values = model_rows(stamp(cfg))
    # The paths hold IRR, MOIC and the drawn rates and multiples: no money to convert back
    mc, deal, cfg = mc_in_millions(MCInputs(**req.mc.model_dump()), DealInputs(**req.deal.model_dump()), cfg)
    params = build_sim_params(mc, deal, cfg)
    if req.scenario:
        params = apply_scenario(req.scenario, params, cfg)
    t0 = time.perf_counter()
    sim = run_vectorized_simulation_full(params, seed=req.seed)
    elapsed = time.perf_counter() - t0
    df = sim.df
    sample = df.sample(min(SAMPLE_ROWS, len(df)), random_state=42).reset_index(drop=True)
    money_items, money_values = money_rows(req.deal.money())
    source_items, source_values = source_rows()
    about = pd.DataFrame({
        "Item": [*source_items, "Paths simulated", "Paths in this file", "Seed", "Scenario", "Run time (s)", *money_items, *model_items],
        "Value": [*source_values, len(df), len(sample), "random" if req.seed is None else req.seed,
                  req.scenario or "none", round(elapsed, 3), *money_values, *model_values],
    })
    formats = [SAMPLE_FORMATS.get(c) for c in sample.columns]
    paths_charts = [ChartSpec(kind="scatter", by="columns", title="IRR against %s (sampled paths)" % driver,
                              categories=list(sample.columns).index(driver),
                              series=[ChartSeries(index=list(sample.columns).index("IRR"), tone="answer")],
                              points=SCATTER_POINTS, x_title=driver, y_title="IRR", y_format="percent",
                              note="A sample of %d simulated paths, as on the Drivers step." % SCATTER_POINTS)
                    for driver in ("Exit Multiple", "Growth") if driver in sample.columns]
    charted = simulation_sheets(sim, req, params, mc, deal)
    data = workbook_bytes([("Paths", sample, formats, None, paths_charts), *charted, ("About", about)])
    _record_export(user, deal_id, "simulation_sample")
    return _xlsx_response(data, "mc_simulation.xlsx")
