"""Sector models (PLAN.md R4): a model picked by sector, shaped for the
company, worked out on the server and written as live Excel formulas.

    GET  /api/research/models                    the models, their blocks and the examples solved by hand
    POST /api/research/models/change             add, drop or swap a line of revenue (the structure mended)
    POST /api/research/models/run                the model worked out for the company typed
    POST /api/research/models/workbook           the same as a workbook of live formulas
    POST /api/research/models/cells              the same workbook as cells, for the Excel add-in
    GET  /api/research/companies/{id}/fill       the model's past from the company's accepted quarters

Nothing is stored here: the company's figures and assumptions come with each
request (saving a model per company is R5). A run is a hundred-odd rows of
plain arithmetic, so it is neither a model run (``RUN_PATHS``) nor slotted.
Refusals answer 422 with ``{"code", "message"}``: the message names the row
and period at fault and is shown as the model's own words.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
import io
import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from api.auth import AuthUser, require_user
from api.routers.coverage import store_errors
from api.routers.export import _xlsx_response
from api.schemas import (
    ModelCatalogueOut, ModelCellsOut, ModelChangeIn, ModelChangeOut, ModelFillOut, ModelRunIn, ModelRunOut,
)
from db import coverage as rows
from research import actuals, blocks, compose
from research.definition import (
    ASSUME, Company, Heading, InputError, ModelError, Pick, cells, label, rule_for, words_for,
)
from research.evaluate import EvaluationError, evaluate
from research.models import CATALOGUE
from research.workbook import _unit_text, sheets, write

router = APIRouter(prefix="/research", tags=["research"])

EXAMPLES = json.loads((Path(compose.__file__).parent / "models" / "examples.json").read_text(encoding="utf-8"))
MODEL_REFUSED = "model_refused"
COMPANY_REFUSED = "company_refused"
CANNOT_WORK_OUT = "cannot_work_out"


@contextmanager
def refusals():
    try:
        yield
    except ModelError as exc:
        raise HTTPException(422, {"code": MODEL_REFUSED, "message": str(exc)}) from None
    except InputError as exc:
        raise HTTPException(422, {"code": COMPANY_REFUSED, "message": str(exc)}) from None
    except EvaluationError as exc:
        raise HTTPException(422, {"code": CANNOT_WORK_OUT, "message": str(exc)}) from None


def _structure(data) -> compose.Structure:
    return compose.read(data.model_dump(), CATALOGUE.specs)


def _company(data) -> Company:
    raw = data.model_dump()
    raw["marks"] = {row: {p: dict(m) for p, m in by.items()} for row, by in raw["marks"].items()}
    raw["off"] = tuple(raw["off"])
    return Company(**raw)


@router.get("/models", response_model=ModelCatalogueOut)
def get_models(user: AuthUser = Depends(require_user)):
    return {
        "models": [{"id": s.id, "name": s.name, "about": s.about, "offered": s.offered, "vetted": s.vetted,
                    "sectors": sorted(s.sectors), "default": s.default.to_dict()} for s in CATALOGUE.specs.values()],
        "blocks": [{"id": b, "counts_places": b in blocks.COUNTING} for b in blocks.BLOCKS],
        "examples": EXAMPLES,
    }


@router.post("/models/change", response_model=ModelChangeOut)
def post_change(body: ModelChangeIn, user: AuthUser = Depends(require_user)):
    """The structure after one change, mended so it always builds (research/compose.py `normalise`)."""
    with refusals():
        structure = _structure(body.structure)
        if body.action in ("drop", "swap") and body.key is None:
            raise ModelError("say which line to %s" % body.action)
        if body.action == "add":
            changed = compose.add_segment(structure, body.block or "growth")
        elif body.action == "drop":
            changed = compose.drop_segment(structure, body.key)
        elif body.action == "swap":
            if body.block is None:
                raise ModelError("say which block line %s is to be worked out by" % body.key)
            changed = compose.swap_block(structure, body.key, body.block)
        else:                           # costs per place follow this line, or (no key) none
            changed = compose.Structure(structure.model, structure.segments, body.key, structure.forecast_years)
        CATALOGUE.build(changed)
    return {"structure": changed.to_dict()}


def _cells_out(model, company, plan, values) -> list:
    words = words_for(model, company)
    out = []
    for row in model.rows:
        if isinstance(row, Heading):
            out.append({"key": None, "label": label(row.label, words)})
            continue
        row_cells = {}
        for period in model.periods:
            held = plan.get((row.key, period.name))
            if held is None:
                continue
            rule = rule_for(row, period)
            cell = {"kind": "formula" if held.tree is not None else ("assumption" if held.typed is ASSUME else "figure"),
                    "value": values[row.key][period.name]}
            if isinstance(rule, Pick):
                cell["choice"] = rule.choice
                rule = rule.options[company.choices[rule.choice]]
            if isinstance(rule, str):
                cell["formula"] = rule
            mark = company.marks.get(row.key, {}).get(period.name)
            if mark and held.tree is None:
                cell["mark"] = mark
            row_cells[period.name] = cell
        out.append({"key": row.key, "label": label(row.label, words), "unit": row.unit,
                    "unit_text": _unit_text(row.unit, company, words), "bold": row.bold, "segment": row.segment,
                    "cells": row_cells})
    return out


@router.post("/models/run", response_model=ModelRunOut)
def post_run(body: ModelRunIn, user: AuthUser = Depends(require_user)):
    """The model worked out. Assumptions not typed yet are allowed: what reads them is null, and listed."""
    with refusals():
        structure = _structure(body.structure)
        spec = CATALOGUE.spec(structure.model)
        model = CATALOGUE.build(structure)
        company, dropped = compose.fit(model, _company(body.company))
        plan = cells(model, company, allow_missing=True)
        result = evaluate(model, company, partial=True, plan=plan)
        words = words_for(model, company)
        return {
            "model": spec.id, "name": spec.name, "vetted": model.vetted, "offered": spec.offered,
            "structure": structure.to_dict(),
            "periods": [{"name": p.name, "kind": p.kind, "label": str(company.period_labels.get(p.name, p.name))}
                        for p in model.periods],
            "constants": [{"key": k, "label": label(v, words), "value": company.constants.get(k)}
                          for k, v in model.constants.items()],
            "choices": {k: sorted(v) for k, v in model.choices().items()},
            "segments": sorted(model.segments()),
            "rows": _cells_out(model, company, plan, result.values),
            "missing": [{"row": r, "period": p} for r, p in result.missing],
            "undefined": [{"row": r, "period": p} for r, p in result.undefined],
            "dropped": [{"row": r, "period": p} for r, p in dropped],
            "choices_made": dict(company.choices),
            "notes": list(model.notes),
        }


def _filename(company: Company, model) -> str:
    return "%s_%s_model.xlsx" % (company.name or "company", model.id)


@router.post("/models/workbook", response_class=Response, responses={
    200: {"content": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {}},
          "description": "The model as live Excel formulas: assumptions yellow and named, typed figures blue"}})
def post_workbook(body: ModelRunIn, user: AuthUser = Depends(require_user)):
    with refusals():
        structure = _structure(body.structure)
        model = CATALOGUE.build(structure)
        company, _ = compose.fit(model, _company(body.company))
        wb = write(model, company, allow_missing=True)
    buf = io.BytesIO()
    wb.save(buf)
    return _xlsx_response(buf.getvalue(), _filename(company, model))


@router.post("/models/cells", response_model=ModelCellsOut)
def post_cells(body: ModelRunIn, user: AuthUser = Depends(require_user)):
    """What the Excel add-in writes for "Load model": the download's own cells, from the same `sheets`."""
    with refusals():
        structure = _structure(body.structure)
        model = CATALOGUE.build(structure)
        company, _ = compose.fit(model, _company(body.company))
        return {"company_name": company.name, "vetted": model.vetted,
                "sheets": sheets(model, company, allow_missing=True)}


@router.get("/companies/{company_id}/fill", response_model=ModelFillOut)
def get_fill(company_id: uuid.UUID, scope: str = Query("consolidated", pattern="^(consolidated|standalone)$"),
             user: AuthUser = Depends(require_user)):
    """The model's past from the periods the analyst accepted for this company (research/actuals.py)."""
    with store_errors():
        rows.get_company(user.subject, company_id)
        stored = rows.company_figures(user.subject, company_id)
    return actuals.fill(stored, scope).to_dict()

