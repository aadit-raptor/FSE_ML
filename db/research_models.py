"""Saved sector models, one per covered company, with versions (PLAN.md R5).

The pattern is saved deals' (db/deals.py, PLAN.md 1.5): the working copy
lives on the model's row and autosave overwrites it; a version is written
only when the content differs from the latest one, automatic checkpoints at
most one per ``AUTO_CHECKPOINT_S`` and the newest ``KEEP_AUTO_VERSIONS``
kept; saved, restored and rolled versions are all kept, up to
``MAX_VERSIONS`` a model. A restore first keeps unsaved edits as a version.

**A roll forward** (research/rollforward.py) keeps the model as it stood as a
``rolled`` version holding the estimate set beside the new results, then
makes the rolled model the working copy, in one transaction, and only if the
working copy is still the one the roll was worked out from.

**Whose model.** Every function takes the caller's subject and matches it in
the statement that reads or writes, through the covered company: another
account's model is ``NotFound``, as a missing one is.

**Room.** These tables count against coverage's budget (db/coverage.py
``BUDGET_BYTES``, reported as ``coverage_data``); a model's content is at
most ``MAX_CONTENT_BYTES``.

**Audit history.** Each change writes one entry in its own transaction, with
the names of the rows changed and never a figure: ``model_saved`` (a change
to the working copy, on screen or from Excel), ``model_versioned``,
``model_restored``, ``model_rolled``. Never log a company's figures.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import delete, func, insert, select, update

from db import audit, coverage
from db.coverage import NoAccount, NotFound
from db.engine import connect, transaction
from db.models import CoveredCompany, ResearchModel, ResearchModelVersion, User, utc_now

AUTO_CHECKPOINT_S = 15 * 60
KEEP_AUTO_VERSIONS = 20
MAX_VERSIONS = 200
MAX_CONTENT_BYTES = 256 * 1024
MAX_LABEL_LENGTH = 120
MAX_FIELDS = 50                 # row names an audit entry lists at most

# Refusals, as codes the screen words (web/messages/en.json models.error_*)
NO_MODEL = "no_model"
TOO_LARGE = "model_too_large"
TOO_MANY_VERSIONS = "too_many_versions"
CHANGED_MEANWHILE = "changed_meanwhile"


class NoModel(LookupError):
    """The company is the caller's but has no saved model yet."""


class VersionNotFound(LookupError):
    pass


class Refused(ValueError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ModelRecord:
    company_id: uuid.UUID
    company_name: str
    scope: str
    structure: dict
    company: dict
    as_of: Optional[dict]
    latest_version: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class VersionRecord:
    number: int
    kind: str
    label: Optional[str]
    created_at: datetime
    as_of: Optional[dict] = None
    structure: Optional[dict] = None
    company: Optional[dict] = None
    comparison: Optional[dict] = None


_MODEL = (ResearchModel.id, ResearchModel.company_id, CoveredCompany.name.label("company_name"), ResearchModel.scope,
          ResearchModel.structure, ResearchModel.company, ResearchModel.as_of, ResearchModel.latest_version,
          ResearchModel.created_at, ResearchModel.updated_at)
_SUMMARY = (ResearchModelVersion.number, ResearchModelVersion.kind, ResearchModelVersion.label,
            ResearchModelVersion.created_at, ResearchModelVersion.as_of)


def _record(row) -> ModelRecord:
    return ModelRecord(company_id=row.company_id, company_name=row.company_name, scope=row.scope,
                       structure=row.structure, company=row.company, as_of=row.as_of,
                       latest_version=row.latest_version, created_at=row.created_at, updated_at=row.updated_at)


def _owner_id(subject: str):
    return select(User.id).where(User.subject == subject).scalar_subquery()


def _content_size(structure: dict, company: dict) -> int:
    return len(json.dumps([structure, company], separators=(",", ":")))


def _clean_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    label = " ".join(str(label).split())
    return label[:MAX_LABEL_LENGTH] or None


def _owned_company(conn, subject: str, company_id: uuid.UUID, *, lock: bool = False):
    stmt = select(CoveredCompany.id, CoveredCompany.owner_id, CoveredCompany.name).where(
        CoveredCompany.id == company_id, CoveredCompany.owner_id == _owner_id(subject))
    row = conn.execute(stmt.with_for_update() if lock else stmt).first()
    if row is None:
        raise NotFound(str(company_id))
    return row


def _model_row(conn, subject: str, company_id: uuid.UUID, *, lock: bool = False):
    stmt = (select(*_MODEL).join(CoveredCompany, CoveredCompany.id == ResearchModel.company_id)
            .where(ResearchModel.company_id == company_id, ResearchModel.owner_id == _owner_id(subject)))
    return conn.execute(stmt.with_for_update(of=ResearchModel) if lock else stmt).first()


def _changed_rows(old: dict, new: dict) -> list[str]:
    """Names of what differs between two companies' typed content: row keys, and the other parts by name."""
    names = set()
    old_values, new_values = old.get("values") or {}, new.get("values") or {}
    names |= {k for k in set(old_values) | set(new_values) if old_values.get(k) != new_values.get(k)}
    for part in ("constants", "choices", "off", "words", "period_labels", "name", "unit_name", "marks"):
        if old.get(part) != new.get(part):
            names.add(part)
    return sorted(names)[:MAX_FIELDS]


def _add_version(conn, model_id: uuid.UUID, kind: str, label: Optional[str], structure: dict, company: dict,
                 as_of: Optional[dict], now: datetime, comparison: Optional[dict] = None) -> VersionRecord:
    if kind != "auto":
        held = conn.execute(select(func.count()).where(ResearchModelVersion.model_id == model_id)).scalar_one()
        if held >= MAX_VERSIONS:
            raise Refused(TOO_MANY_VERSIONS)
        if coverage._used(conn) >= coverage.BUDGET_BYTES:
            raise Refused(coverage.COVERAGE_FULL)
    number = conn.execute(update(ResearchModel).where(ResearchModel.id == model_id)
                          .values(latest_version=ResearchModel.latest_version + 1)
                          .returning(ResearchModel.latest_version)).scalar_one()
    conn.execute(insert(ResearchModelVersion).values(
        model_id=model_id, number=number, kind=kind, label=label, structure=structure, company=company,
        as_of=as_of, comparison=comparison, created_at=now))
    return VersionRecord(number=number, kind=kind, label=label, created_at=now, as_of=as_of)


def _latest(conn, model_id: uuid.UUID):
    return conn.execute(select(*_SUMMARY, ResearchModelVersion.structure, ResearchModelVersion.company)
                        .where(ResearchModelVersion.model_id == model_id)
                        .order_by(ResearchModelVersion.number.desc()).limit(1)).first()


def _prune(conn, model_id: uuid.UUID) -> None:
    keep = (select(ResearchModelVersion.number)
            .where(ResearchModelVersion.model_id == model_id, ResearchModelVersion.kind == "auto")
            .order_by(ResearchModelVersion.number.desc()).limit(KEEP_AUTO_VERSIONS))
    conn.execute(delete(ResearchModelVersion).where(
        ResearchModelVersion.model_id == model_id, ResearchModelVersion.kind == "auto",
        ResearchModelVersion.number.not_in(keep)))


# ---------------------------------------------------------------------------
def get_model(subject: str, company_id: uuid.UUID) -> ModelRecord:
    with connect() as conn:
        row = _model_row(conn, subject, company_id)
        if row is None:
            _owned_company(conn, subject, company_id)
            raise NoModel(str(company_id))
    return _record(row)


def save_model(subject: str, company_id: uuid.UUID, structure: dict, company: dict, scope: str,
               as_of: Optional[dict], *, now: Optional[datetime] = None, version_label: Optional[str] = None,
               budget: Optional[int] = None, worked_from: Optional[datetime] = None) -> tuple[ModelRecord, list[str]]:
    """Autosave the working copy (created on the first save), adding an automatic checkpoint at most once every
    ``AUTO_CHECKPOINT_S``; with ``version_label``, keep it as a saved version too (a save from Excel). Answers the
    record and the names of the rows that changed (empty when nothing did: no write, no entry)."""
    now = now or utc_now()
    if _content_size(structure, company) > MAX_CONTENT_BYTES:
        raise Refused(TOO_LARGE)
    label = _clean_label(version_label)
    with transaction() as conn:
        owned = _owned_company(conn, subject, company_id, lock=True)
        row = _model_row(conn, subject, company_id, lock=True)
        if row is None:
            if coverage._used(conn) >= (coverage.BUDGET_BYTES if budget is None else budget):
                raise Refused(coverage.COVERAGE_FULL)
            model_id = uuid.uuid4()
            conn.execute(insert(ResearchModel).values(
                id=model_id, company_id=company_id, owner_id=owned.owner_id, scope=scope, structure=structure,
                company=company, as_of=as_of, created_at=now, updated_at=now))
            _add_version(conn, model_id, "created", label, structure, company, as_of, now)
            audit.record(conn, owned.owner_id, "model_saved", now=now, detail={"fields": ["created"]})
            return _record(_model_row(conn, subject, company_id)), ["created"]
        model_id = row.id
        if worked_from is not None and row.updated_at != worked_from:
            raise Refused(CHANGED_MEANWHILE)        # a save from Excel read an older working copy
        same = (row.structure == structure and row.company == company and row.scope == scope
                and row.as_of == as_of)
        if same and label is None:
            return _record(row), []
        changed = [] if same else _changed_rows(row.company, company)
        if row.structure != structure:
            changed = ["structure"] + changed[: MAX_FIELDS - 1]
        if not same:
            conn.execute(update(ResearchModel).where(ResearchModel.id == model_id).values(
                structure=structure, company=company, scope=scope, as_of=as_of, updated_at=now))
            audit.record(conn, owned.owner_id, "model_saved", now=now,
                         detail={"fields": changed or ["scope"]})
        latest = _latest(conn, model_id)
        differs = latest is None or latest.structure != structure or latest.company != company
        if label is not None:
            if differs:
                version = _add_version(conn, model_id, "saved", label, structure, company, as_of, now)
                if same:
                    audit.record(conn, owned.owner_id, "model_versioned", now=now,
                                 detail={"version": version.number})
        elif differs and (latest is None or now - latest.created_at >= timedelta(seconds=AUTO_CHECKPOINT_S)):
            _add_version(conn, model_id, "auto", None, structure, company, as_of, now)
            _prune(conn, model_id)
        return _record(_model_row(conn, subject, company_id)), changed


def list_versions(subject: str, company_id: uuid.UUID) -> list[VersionRecord]:
    """The model's versions, newest first, without their content (a rolled one with its comparison)."""
    with connect() as conn:
        row = _model_row(conn, subject, company_id)
        if row is None:
            _owned_company(conn, subject, company_id)
            raise NoModel(str(company_id))
        rows = conn.execute(select(*_SUMMARY, ResearchModelVersion.comparison)
                            .where(ResearchModelVersion.model_id == row.id)
                            .order_by(ResearchModelVersion.number.desc())).all()
    return [VersionRecord(number=r.number, kind=r.kind, label=r.label, created_at=r.created_at, as_of=r.as_of,
                          comparison=r.comparison) for r in rows]


def save_version(subject: str, company_id: uuid.UUID, label: Optional[str] = None,
                 *, now: Optional[datetime] = None) -> VersionRecord:
    """Keep the working copy as a version; when nothing changed since the latest, that one is kept (and named)."""
    now = now or utc_now()
    label = _clean_label(label)
    with transaction() as conn:
        row = _model_row(conn, subject, company_id, lock=True)
        if row is None:
            _owned_company(conn, subject, company_id)
            raise NoModel(str(company_id))
        latest = _latest(conn, row.id)
        if latest is not None and latest.structure == row.structure and latest.company == row.company:
            kind = "saved" if latest.kind == "auto" else latest.kind
            keeps_its_label = latest.kind in ("rolled", "restored")      # its label says what it is
            new_label = latest.label if keeps_its_label or label is None else label
            if kind == latest.kind and new_label == latest.label:
                return VersionRecord(number=latest.number, kind=kind, label=new_label, created_at=latest.created_at,
                                     as_of=latest.as_of)
            conn.execute(update(ResearchModelVersion).where(
                ResearchModelVersion.model_id == row.id, ResearchModelVersion.number == latest.number,
            ).values(kind=kind, label=new_label))
            version = VersionRecord(number=latest.number, kind=kind, label=new_label, created_at=latest.created_at,
                                    as_of=latest.as_of)
        else:
            version = _add_version(conn, row.id, "saved", label, row.structure, row.company, row.as_of, now)
        audit.record(conn, audit.subject_id(subject), "model_versioned", now=now, detail={"version": version.number})
    return version


def restore_version(subject: str, company_id: uuid.UUID, number: int,
                    *, now: Optional[datetime] = None) -> ModelRecord:
    """An earlier version back as the working copy, its ``as_of`` with it. Unsaved edits are kept first."""
    now = now or utc_now()
    with transaction() as conn:
        row = _model_row(conn, subject, company_id, lock=True)
        if row is None:
            _owned_company(conn, subject, company_id)
            raise NoModel(str(company_id))
        target = conn.execute(select(ResearchModelVersion.structure, ResearchModelVersion.company,
                                     ResearchModelVersion.as_of)
                              .where(ResearchModelVersion.model_id == row.id,
                                     ResearchModelVersion.number == number)).first()
        if target is None:
            raise VersionNotFound(str(number))
        latest = _latest(conn, row.id)
        if latest is None or latest.structure != row.structure or latest.company != row.company:
            _add_version(conn, row.id, "saved", "Before restoring version %d" % number, row.structure, row.company,
                         row.as_of, now)
        conn.execute(update(ResearchModel).where(ResearchModel.id == row.id).values(
            structure=target.structure, company=target.company, as_of=target.as_of, updated_at=now))
        _add_version(conn, row.id, "restored", "Restored version %d" % number, target.structure, target.company,
                     target.as_of, now)
        audit.record(conn, audit.subject_id(subject), "model_restored", now=now, detail={"version": number})
        return _record(_model_row(conn, subject, company_id))


def roll(subject: str, company_id: uuid.UUID, worked_from: datetime, company: dict, as_of: dict,
         comparison: dict, label: str, *, now: Optional[datetime] = None) -> tuple[ModelRecord, VersionRecord]:
    """Keep the model as it stood as a ``rolled`` version holding `comparison`, then make the rolled `company` the
    working copy. Refused when the working copy changed after `worked_from` (its ``updated_at``)."""
    now = now or utc_now()
    if _content_size({}, company) > MAX_CONTENT_BYTES:
        raise Refused(TOO_LARGE)
    with transaction() as conn:
        row = _model_row(conn, subject, company_id, lock=True)
        if row is None:
            _owned_company(conn, subject, company_id)
            raise NoModel(str(company_id))
        if row.updated_at != worked_from:
            raise Refused(CHANGED_MEANWHILE)
        before = _add_version(conn, row.id, "rolled", _clean_label(label), row.structure, row.company, row.as_of,
                              now, comparison)
        conn.execute(update(ResearchModel).where(ResearchModel.id == row.id).values(
            company=company, as_of=as_of, updated_at=now))
        audit.record(conn, audit.subject_id(subject), "model_rolled", now=now, detail={"version": before.number})
        return _record(_model_row(conn, subject, company_id)), before


__all__ = ["AUTO_CHECKPOINT_S", "CHANGED_MEANWHILE", "KEEP_AUTO_VERSIONS", "MAX_VERSIONS", "ModelRecord", "NoAccount",
           "NoModel", "NotFound", "Refused", "VersionNotFound", "VersionRecord", "get_model", "list_versions",
           "restore_version", "roll", "save_model", "save_version"]
