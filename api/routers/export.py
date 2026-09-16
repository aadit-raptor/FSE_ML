"""Excel downloads.

The web app sends the tables it is showing; the API writes them to a workbook.
Replaces the Streamlit app's download buttons (pandas + openpyxl there too).
"""
import io
import re
import time

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import Response

from api.deps import resolve_settings
from api.limits import simulation_slot
from api.schemas import MonteCarloRequest, WorkbookRequest
from core.deal import DealInputs
from core.montecarlo import MCInputs, apply_scenario, build_sim_params
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


def workbook_bytes(sheets) -> bytes:
    """sheets: iterable of (name, DataFrame). Always writes at least one sheet."""
    buf = io.BytesIO()
    used: set = set()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        wrote = False
        for name, df in sheets:
            df.to_excel(w, sheet_name=_sheet_name(name, used), index=False)
            wrote = True
        if not wrote:
            pd.DataFrame({"Note": ["No data"]}).to_excel(w, sheet_name="Data", index=False)
    return buf.getvalue()


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
        frames.append((sheet.name, pd.DataFrame(rows, columns=sheet.columns)))
    return _xlsx_response(workbook_bytes(frames), req.filename)


@router.post("/montecarlo-sample", response_class=Response,
             responses={200: {"content": {XLSX: {}}, "description": "Excel workbook"}})
@simulation_slot
def post_montecarlo_sample(req: MonteCarloRequest):
    """Simulate with the same inputs as /montecarlo/run and export up to 10,000 paths.

    With a fixed seed the paths are exactly those behind the on-screen results.
    """
    cfg = resolve_settings(req.settings, check_correlations=True)
    params = build_sim_params(MCInputs(**req.mc.model_dump()), DealInputs(**req.deal.model_dump()), cfg)
    if req.scenario:
        params = apply_scenario(req.scenario, params, cfg)
    t0 = time.perf_counter()
    sim = run_vectorized_simulation_full(params, seed=req.seed)
    elapsed = time.perf_counter() - t0
    df = sim.df
    sample = df.sample(min(SAMPLE_ROWS, len(df)), random_state=42).reset_index(drop=True)
    about = pd.DataFrame({
        "Item": ["Paths simulated", "Paths in this file", "Seed", "Scenario", "Run time (s)"],
        "Value": [len(df), len(sample), "random" if req.seed is None else req.seed,
                  req.scenario or "none", round(elapsed, 3)],
    })
    return _xlsx_response(workbook_bytes([("Paths", sample), ("About", about)]), "mc_simulation.xlsx")
