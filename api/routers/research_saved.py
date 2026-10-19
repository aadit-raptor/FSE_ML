"""Saved sector models (PLAN.md R5): one per covered company, with versions,
saved back from Excel and rolled forward on new results.

    GET  /api/research/companies/{id}/model                         the saved model, and newer results if any
    PUT  /api/research/companies/{id}/model                         autosave (the first save creates it)
    GET  /api/research/companies/{id}/model/versions                its history, rolled versions with their comparison
    POST /api/research/companies/{id}/model/versions                keep the working copy as a version
    POST /api/research/companies/{id}/model/versions/{n}/restore    an earlier version back
    POST /api/research/companies/{id}/model/roll                    roll forward on the accepted results
    GET  /api/research/companies/{id}/model/cells                   the saved model's cells, for the add-in
    GET  /api/research/companies/{id}/model/workbook                the same as a workbook
    POST /api/research/companies/{id}/model/sheet                   "Save to Variater": the named cells read back

The results a roll brings are read here from the caller's accepted periods
(research/actuals.py), never taken from the browser. Refusals are 422 (or
409 for a conflict) with ``{"code", "message"}``; another account's company
is a 404, as a missing one is. Nothing here runs a model in the sense of
``RUN_PATHS``: a sector model is plain arithmetic (as research_models.py).
"""
from __future__ import annotations

import io
import uuid
from contextlib import contextmanager
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Response

from api.auth import AuthUser, require_user
from api.routers.coverage import _iso, store_errors
from api.routers.export import _xlsx_response
from api.routers.research_models import _company, _structure, refusals
from api.schemas import (
    ModelCellsOut, ModelRollOut, ModelSheetIn, ModelSheetOut, ModelVersionIn, ModelVersionList, SavedModelIn,
    SavedModelOut,
)
from db import coverage as coverage_rows
from db import research_models as rows
from research import actuals, compose
from research.definition import Company, cells
from research.models import CATALOGUE
from research.rollforward import RollRefused, newer, roll
from research.sheetback import SheetRefused, read_back
from research.workbook import sheets, write

router = APIRouter(prefix="/research", tags=["research"])


@contextmanager
def saved_errors():
    with store_errors():
        try:
            yield
        except rows.NoModel:
            raise HTTPException(404, {"code": rows.NO_MODEL, "message": "this company has no saved model"}) from None
        except rows.VersionNotFound:
            raise HTTPException(404, "not_found") from None
        except rows.Refused as exc:
            status = 409 if exc.code in (rows.CHANGED_MEANWHILE, coverage_rows.COVERAGE_FULL) else 422
            raise HTTPException(status, {"code": exc.code, "message": ""}) from None


def _fill(subject: str, company_id: uuid.UUID, scope: str):
    return actuals.fill(coverage_rows.company_figures(subject, company_id), scope)


def _out(record: rows.ModelRecord, fill=None, changed=()) -> dict:
    ahead = fill.as_of if fill is not None and newer(record.as_of, fill.as_of) else None
    return {"company_id": str(record.company_id), "company_name": record.company_name, "scope": record.scope,
            "structure": record.structure, "company": record.company, "as_of": record.as_of,
            "latest_version": record.latest_version, "created_at": _iso(record.created_at),
            "updated_at": _iso(record.updated_at), "newer_results": ahead, "changed": list(changed)}


def _built(record: rows.ModelRecord):
    """The saved model and company as research/ takes them."""
    structure = compose.read(record.structure, CATALOGUE.specs)
    model = CATALOGUE.build(structure)
    raw = dict(record.company)
    raw["off"] = tuple(raw.get("off") or ())
    return model, Company(**raw)


def _stored(company: Company) -> dict:
    out = asdict(company)
    out["off"] = list(out["off"])
    return out


@router.get("/companies/{company_id}/model", response_model=SavedModelOut)
def get_saved(company_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with saved_errors():
        record = rows.get_model(user.subject, company_id)
        return _out(record, _fill(user.subject, company_id, record.scope))


@router.put("/companies/{company_id}/model", response_model=SavedModelOut)
def put_saved(company_id: uuid.UUID, body: SavedModelIn, user: AuthUser = Depends(require_user)):
    """Autosave: only a model that builds and a company that fits it are kept, its choices settled and figures
    for cells it no longer has left out, as a run takes them (research/compose.py `fit`)."""
    with refusals():
        model = CATALOGUE.build(_structure(body.structure))
        company, _ = compose.fit(model, _company(body.company))
        cells(model, company, allow_missing=True)
    with saved_errors():
        record, changed = rows.save_model(
            user.subject, company_id, body.structure.model_dump(), _stored(company), body.scope,
            body.as_of.model_dump() if body.as_of else None)
        return _out(record, _fill(user.subject, company_id, record.scope), changed)


def _version_out(v: rows.VersionRecord) -> dict:
    return {"number": v.number, "kind": v.kind, "label": v.label, "created_at": _iso(v.created_at),
            "as_of": v.as_of, "comparison": v.comparison}


@router.get("/companies/{company_id}/model/versions", response_model=ModelVersionList)
def get_versions(company_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with saved_errors():
        return {"versions": [_version_out(v) for v in rows.list_versions(user.subject, company_id)]}


@router.post("/companies/{company_id}/model/versions", status_code=201)
def post_version(company_id: uuid.UUID, body: ModelVersionIn, user: AuthUser = Depends(require_user)):
    with saved_errors():
        return _version_out(rows.save_version(user.subject, company_id, body.label))


@router.post("/companies/{company_id}/model/versions/{number}/restore", response_model=SavedModelOut)
def post_restore(company_id: uuid.UUID, number: int, user: AuthUser = Depends(require_user)):
    with saved_errors():
        record = rows.restore_version(user.subject, company_id, number)
        return _out(record, _fill(user.subject, company_id, record.scope))


@router.post("/companies/{company_id}/model/roll", response_model=ModelRollOut)
def post_roll(company_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """Roll the model forward to the accepted results: kept as it stood (with its estimate beside the results),
    then moved on."""
    with saved_errors():
        record = rows.get_model(user.subject, company_id)
        fill = _fill(user.subject, company_id, record.scope)
        with refusals():
            model, company = _built(record)
            try:
                rolled = roll(model, company, record.as_of, fill)
            except RollRefused as exc:
                raise HTTPException(422, {"code": exc.code, "message": ""}) from None
        label = "Before rolling forward to %s" % rolled.comparison["period"]
        after, before = rows.roll(user.subject, company_id, record.updated_at, _stored(rolled.company),
                                  rolled.as_of, rolled.comparison, label)
        return {"model": _out(after, fill), "kind": rolled.kind, "version": before.number,
                "comparison": rolled.comparison,
                "carried": [{"row": r, "period": p} for r, p in rolled.carried],
                "to_type": [{"row": r, "period": p} for r, p in rolled.to_type]}


@router.get("/companies/{company_id}/model/cells", response_model=ModelCellsOut)
def get_saved_cells(company_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    """What the add-in writes for a saved model: the download's cells, with the model's reference beside the title."""
    with saved_errors():
        record = rows.get_model(user.subject, company_id)
    with refusals():
        model, company = _built(record)
        return {"company_name": company.name, "vetted": model.vetted,
                "sheets": sheets(model, company, allow_missing=True, saved=str(company_id))}


@router.get("/companies/{company_id}/model/workbook", response_class=Response, responses={
    200: {"content": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {}},
          "description": "The saved model as live Excel formulas, which the add-in can save back"}})
def get_saved_workbook(company_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with saved_errors():
        record = rows.get_model(user.subject, company_id)
    with refusals():
        model, company = _built(record)
        wb = write(model, company, allow_missing=True, saved=str(company_id))
    buf = io.BytesIO()
    wb.save(buf)
    return _xlsx_response(buf.getvalue(), "%s_%s_model.xlsx" % (company.name or "company", model.id))


@router.post("/companies/{company_id}/model/sheet", response_model=ModelSheetOut)
def post_sheet(company_id: uuid.UUID, body: ModelSheetIn, user: AuthUser = Depends(require_user)):
    """"Save to Variater": the sheet's named cells (assumptions and constants) read back and saved, as a version.
    A sheet whose structure was changed is refused, saying what changed."""
    with saved_errors():
        record = rows.get_model(user.subject, company_id)
        with refusals():
            model, company = _built(record)
            try:
                back = read_back(model, company, str(company_id), body.grid, body.names)
            except SheetRefused as exc:
                raise HTTPException(422, {"code": exc.code, "message": str(exc)}) from None
            cells(model, back.company, allow_missing=True)
        saved, _ = rows.save_model(user.subject, company_id, record.structure, _stored(back.company), record.scope,
                                   record.as_of, version_label="Saved from Excel", worked_from=record.updated_at)
        return {"model": _out(saved, _fill(user.subject, company_id, saved.scope)),
                "changed": [{"row": r, "period": p, "before": b, "after": a} for r, p, b, a in back.changed]}
