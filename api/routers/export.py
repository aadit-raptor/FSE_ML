"""Excel downloads.

The web app sends the tables it is showing; the API writes them to a workbook.
Replaces the Streamlit app's download buttons (pandas + openpyxl there too).
"""
import io
import re
import time
from typing import Optional

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import Response

from api.deps import resolve_settings
from api.limits import simulation_slot
from api.schemas import MonteCarloRequest, WorkbookRequest
from core.deal import DealInputs
from core.montecarlo import MCInputs, apply_scenario, build_sim_params, mc_in_millions
from simulation.vectorized_simulation import run_vectorized_simulation_full

router = APIRouter(prefix="/export", tags=["export"])

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
SAMPLE_ROWS = 10_000
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


def _apply_formats(ws, df: pd.DataFrame, columns, rows, grouping: str) -> None:
    """Set each data cell's number format: its row's format, else its column's."""
    columns, rows = list(columns or []), list(rows or [])
    for r in range(len(df)):
        row_kind = rows[r] if r < len(rows) else None
        for c in range(df.shape[1]):
            kind = row_kind or (columns[c] if c < len(columns) else None)
            if kind is None:
                continue
            cell = ws.cell(row=r + 2, column=c + 1)     # row 1 is the header
            number_format = _number_format(kind, cell.value, grouping)
            if number_format:
                cell.number_format = number_format


def workbook_bytes(sheets, grouping: str = "locale") -> bytes:
    """sheets: iterable of (name, DataFrame) or (name, DataFrame, column
    formats, row formats). Always writes at least one sheet."""
    buf = io.BytesIO()
    used: set = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        wrote = False
        for name, df, *formats in sheets:
            sheet_name = _sheet_name(name, used)
            df.to_excel(w, sheet_name=sheet_name, index=False)
            if any(formats):
                _apply_formats(w.sheets[sheet_name], df, *formats, grouping)
            wrote = True
        if not wrote:
            pd.DataFrame({"Note": ["No data"]}).to_excel(w, sheet_name="Data", index=False)
    return buf.getvalue()


# The simulated paths' columns: rates are fractions, multiples times EBITDA
SAMPLE_FORMATS = {
    "IRR": "percent", "Growth": "percent", "Interest": "percent", "Gross Margin": "percent",
    "EBITDA Shock": "percent", "MOIC": "multiple", "Exit Multiple": "multiple",
    "Exit Equity": "money", "Exit EV": "money", "Exit EBITDA": "money", "Net Debt Exit": "money",
}

UNIT_WORDS = {"thousands": "thousands", "millions": "millions", "billions": "billions"}


def money_rows(money) -> tuple[list, list]:
    """Items and values saying what a workbook's money columns are counted in."""
    return (["Currency", "Money unit"], [money.currency, UNIT_WORDS[money.unit]])


def _xlsx_response(data: bytes, filename: str) -> Response:
    return Response(data, media_type=XLSX, headers={
        "Content-Disposition": f'attachment; filename="{_safe_filename(filename)}"'})


@router.post("/workbook", response_class=Response,
             responses={200: {"content": {XLSX: {}}, "description": "Excel workbook"}})
def post_workbook(req: WorkbookRequest):
    """Write the given tables to an Excel workbook, one sheet per table."""
    frames = []
    for sheet in req.sheets:
        rows = [row + [None] * (len(sheet.columns) - len(row)) for row in sheet.rows]
        frames.append((sheet.name, pd.DataFrame(rows, columns=sheet.columns),
                       sheet.column_formats, sheet.row_formats))
    if req.money is not None:
        items, values = money_rows(req.money)
        frames.append(("About", pd.DataFrame({"Item": items, "Value": values})))
    return _xlsx_response(workbook_bytes(frames, req.grouping), req.filename)


@router.post("/montecarlo-sample", response_class=Response,
             responses={200: {"content": {XLSX: {}}, "description": "Excel workbook"}})
@simulation_slot
def post_montecarlo_sample(req: MonteCarloRequest):
    """Simulate with the same inputs as /montecarlo/run and export up to 10,000 paths.

    With a fixed seed the paths are exactly those behind the on-screen results.
    """
    cfg = resolve_settings(req.settings, check_correlations=True)
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
    about = pd.DataFrame({
        "Item": ["Paths simulated", "Paths in this file", "Seed", "Scenario", "Run time (s)", *money_items],
        "Value": [len(df), len(sample), "random" if req.seed is None else req.seed,
                  req.scenario or "none", round(elapsed, 3), *money_values],
    })
    formats = [SAMPLE_FORMATS.get(c) for c in sample.columns]
    return _xlsx_response(workbook_bytes([("Paths", sample, formats, None), ("About", about)]),
                          "mc_simulation.xlsx")
