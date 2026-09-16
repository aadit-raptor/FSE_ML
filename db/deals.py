"""Saved deals, their version history and the account's settings (PLAN.md 1.5).

**Whose deal.** Every function takes the caller's subject (from the verified
token, never from a request body) and matches it in the same statement that
reads or writes the deal. A deal someone else owns is indistinguishable from
one that doesn't exist: both raise ``DealNotFound``, so ids can't be probed.

**What a deal holds.** The deal inputs, complete, and the Settings overrides in
effect: together they decide every number the deal screens show, so a deal
reopened on another device -- or a version restored -- gives the same IRR.

**Staying small** (the free Neon plan has 0.5 GB):

- the working copy (the draft) lives on the deal row and autosave overwrites
  it, so typing never adds rows;
- a version is only written when its content differs from the latest one
  (saving twice without edits returns the same version);
- automatic checkpoints are at most one per ``AUTO_CHECKPOINT_S`` per deal,
  and only the newest ``KEEP_AUTO_VERSIONS`` are kept; versions the user saved
  or that mark a restore are kept;
- settings are stored as overrides only, usually ``{}``.

A version is a few hundred bytes (``tests/test_deals.py`` measures it).

**Never log deal contents**: names, inputs and settings stay out of logs and
error reports, as everywhere else in the API.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional

from pydantic import ValidationError
from sqlalchemy import and_, delete, func, insert, select, update

from core.config import DEFAULTS
from db.engine import connect, transaction
from db.models import Deal, DealVersion, User, utc_now

MAX_NAME_LENGTH = 120
MAX_LABEL_LENGTH = 120
# At most one automatic checkpoint per deal in this window
AUTO_CHECKPOINT_S = 15 * 60
# Automatic checkpoints kept per deal (saved and restore versions are all kept)
KEEP_AUTO_VERSIONS = 20


class NoAccount(LookupError):
    """The caller hasn't finished their account, so there is no one to own a deal."""


class DealNotFound(LookupError):
    """No such deal for this caller (it may exist and belong to someone else)."""


class VersionNotFound(LookupError):
    """No such version of the deal."""


class InvalidDeal(ValueError):
    """Content the app refuses to store."""


@dataclass(frozen=True)
class DealRecord:
    id: uuid.UUID
    name: str
    latest_version: int
    archived: bool
    created_at: datetime
    updated_at: datetime
    inputs: Optional[dict] = None
    settings: Optional[dict] = None


@dataclass(frozen=True)
class VersionRecord:
    number: int
    kind: str
    label: Optional[str]
    created_at: datetime
    inputs: Optional[dict] = None
    settings: Optional[dict] = None


# ---------------------------------------------------------------------------
# Validation: nothing invalid reaches the database, whichever caller writes it
# ---------------------------------------------------------------------------
def clean_name(name: Any, *, what: str = "name", limit: int = MAX_NAME_LENGTH) -> str:
    text = " ".join(str(name or "").split())
    if not text:
        raise InvalidDeal(f"{what} can't be empty")
    if len(text) > limit:
        raise InvalidDeal(f"{what} can be at most {limit} characters")
    return text


def clean_inputs(inputs: Mapping) -> dict:
    """Deal inputs, complete: missing fields take the API's defaults now, so
    the stored deal doesn't change if those defaults do later."""
    from api.schemas import DealInputsIn  # the one definition of a deal's inputs

    try:
        return DealInputsIn.model_validate(dict(inputs or {})).model_dump()
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first.get("loc", ())) or "inputs"
        raise InvalidDeal(f"inputs.{where}: {first.get('msg', 'invalid')}") from None


def clean_settings(settings: Optional[Mapping]) -> dict:
    """Settings overrides: known keys, the default's type, and only the keys
    that differ from the defaults (so an untouched deal stores ``{}``)."""
    out = {}
    for key, value in dict(settings or {}).items():
        if key not in DEFAULTS:
            raise InvalidDeal(f"unknown setting: {key}")
        default = DEFAULTS[key]
        if isinstance(default, bool):
            if not isinstance(value, bool):
                raise InvalidDeal(f"setting {key} must be true or false")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise InvalidDeal(f"setting {key} must be a number")
        if value != default:
            out[key] = value
    return out


def clean_content(inputs: Mapping, settings: Optional[Mapping]) -> tuple[dict, dict]:
    return clean_inputs(inputs), clean_settings(settings)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def _owner_id(subject: str):
    return select(User.id).where(User.subject == subject).scalar_subquery()


def _owned(subject: str, deal_id: uuid.UUID):
    return and_(Deal.id == deal_id, Deal.owner_id == _owner_id(subject))


def _record(row, *, content: bool = True) -> DealRecord:
    return DealRecord(
        id=row.id, name=row.name, latest_version=row.latest_version,
        archived=row.archived_at is not None, created_at=row.created_at, updated_at=row.updated_at,
        inputs=row.inputs if content else None, settings=row.settings if content else None,
    )


_DEAL_COLUMNS = (Deal.id, Deal.name, Deal.latest_version, Deal.archived_at, Deal.created_at,
                 Deal.updated_at, Deal.inputs, Deal.settings)
_VERSION_SUMMARY = (DealVersion.number, DealVersion.kind, DealVersion.label, DealVersion.created_at)


def _lock_deal(conn, subject: str, deal_id: uuid.UUID):
    """The caller's deal row, locked for this transaction, or ``DealNotFound``."""
    row = conn.execute(
        select(*_DEAL_COLUMNS).where(_owned(subject, deal_id)).with_for_update()
    ).first()
    if row is None:
        raise DealNotFound(str(deal_id))
    return row


def _add_version(conn, deal_id: uuid.UUID, kind: str, label: Optional[str], inputs: dict,
                 settings: dict, now: datetime) -> VersionRecord:
    number = conn.execute(
        update(Deal).where(Deal.id == deal_id)
        .values(latest_version=Deal.latest_version + 1).returning(Deal.latest_version)
    ).scalar_one()
    conn.execute(insert(DealVersion).values(
        deal_id=deal_id, number=number, kind=kind, label=label, inputs=inputs, settings=settings,
        created_at=now))
    return VersionRecord(number=number, kind=kind, label=label, created_at=now)


def _latest_version(conn, deal_id: uuid.UUID):
    return conn.execute(
        select(*_VERSION_SUMMARY, DealVersion.inputs, DealVersion.settings)
        .where(DealVersion.deal_id == deal_id)
        .order_by(DealVersion.number.desc()).limit(1)
    ).first()


def _prune_auto_versions(conn, deal_id: uuid.UUID) -> None:
    keep = (select(DealVersion.number)
            .where(DealVersion.deal_id == deal_id, DealVersion.kind == "auto")
            .order_by(DealVersion.number.desc()).limit(KEEP_AUTO_VERSIONS))
    conn.execute(delete(DealVersion).where(
        DealVersion.deal_id == deal_id, DealVersion.kind == "auto",
        DealVersion.number.not_in(keep)))


def list_deals(subject: str, *, include_archived: bool = False) -> list[DealRecord]:
    """The caller's deals, most recently edited first (no contents)."""
    query = select(*_DEAL_COLUMNS).where(Deal.owner_id == _owner_id(subject))
    if not include_archived:
        query = query.where(Deal.archived_at.is_(None))
    with connect() as conn:
        rows = conn.execute(query.order_by(Deal.updated_at.desc(), Deal.name)).all()
    return [_record(r, content=False) for r in rows]


def create_deal(subject: str, name: str, inputs: Mapping, settings: Optional[Mapping] = None,
                *, now: Optional[datetime] = None, label: Optional[str] = None) -> DealRecord:
    """A new deal owned by the caller, with its first version."""
    name = clean_name(name)
    inputs, settings = clean_content(inputs, settings)
    now = now or utc_now()
    with transaction() as conn:
        owner = conn.execute(select(User.id).where(User.subject == subject)).scalar()
        if owner is None:
            raise NoAccount(subject)
        deal_id = uuid.uuid4()
        conn.execute(insert(Deal).values(
            id=deal_id, owner_id=owner, name=name, inputs=inputs, settings=settings,
            latest_version=0, created_at=now, updated_at=now))
        _add_version(conn, deal_id, "created", label, inputs, settings, now)
        row = conn.execute(select(*_DEAL_COLUMNS).where(Deal.id == deal_id)).one()
    return _record(row)


def get_deal(subject: str, deal_id: uuid.UUID) -> DealRecord:
    with connect() as conn:
        row = conn.execute(select(*_DEAL_COLUMNS).where(_owned(subject, deal_id))).first()
    if row is None:
        raise DealNotFound(str(deal_id))
    return _record(row)


def update_deal(subject: str, deal_id: uuid.UUID, *, name: Optional[str] = None,
                archived: Optional[bool] = None, now: Optional[datetime] = None) -> DealRecord:
    """Rename, archive or unarchive. Neither touches the content or its history."""
    now = now or utc_now()
    values: dict = {}
    if name is not None:
        values["name"] = clean_name(name)
    if archived is not None:
        # Archiving again keeps the original time
        values["archived_at"] = func.coalesce(Deal.archived_at, now) if archived else None
    with transaction() as conn:
        if values:
            values["updated_at"] = now
            row = conn.execute(update(Deal).where(_owned(subject, deal_id)).values(**values)
                               .returning(*_DEAL_COLUMNS)).first()
        else:
            row = conn.execute(select(*_DEAL_COLUMNS).where(_owned(subject, deal_id))).first()
    if row is None:
        raise DealNotFound(str(deal_id))
    return _record(row)


def delete_deal(subject: str, deal_id: uuid.UUID) -> None:
    """Delete the deal and every version of it, for good."""
    with transaction() as conn:
        gone = conn.execute(delete(Deal).where(_owned(subject, deal_id)).returning(Deal.id)).first()
    if gone is None:
        raise DealNotFound(str(deal_id))


def save_draft(subject: str, deal_id: uuid.UUID, inputs: Mapping, settings: Optional[Mapping],
               *, now: Optional[datetime] = None) -> DealRecord:
    """Autosave: overwrite the working copy, adding an automatic checkpoint at
    most once every ``AUTO_CHECKPOINT_S``."""
    inputs, settings = clean_content(inputs, settings)
    now = now or utc_now()
    with transaction() as conn:
        row = _lock_deal(conn, subject, deal_id)
        if row.inputs == inputs and row.settings == settings:
            return _record(row)  # nothing changed: no write, no new time
        conn.execute(update(Deal).where(Deal.id == deal_id)
                     .values(inputs=inputs, settings=settings, updated_at=now))
        latest = _latest_version(conn, deal_id)
        differs = latest is None or latest.inputs != inputs or latest.settings != settings
        due = latest is None or now - latest.created_at >= timedelta(seconds=AUTO_CHECKPOINT_S)
        if differs and due:
            _add_version(conn, deal_id, "auto", None, inputs, settings, now)
            _prune_auto_versions(conn, deal_id)
        row = conn.execute(select(*_DEAL_COLUMNS).where(Deal.id == deal_id)).one()
    return _record(row)


def duplicate_deal(subject: str, deal_id: uuid.UUID, name: Optional[str] = None,
                   *, now: Optional[datetime] = None) -> DealRecord:
    """A new deal from this one's working copy (its history stays behind)."""
    source = get_deal(subject, deal_id)
    if name is None:
        suffix = " (copy)"
        name = source.name[: MAX_NAME_LENGTH - len(suffix)] + suffix
    return create_deal(subject, name, source.inputs or {}, source.settings, now=now)


def list_versions(subject: str, deal_id: uuid.UUID) -> list[VersionRecord]:
    """The deal's versions, newest first (no contents)."""
    with connect() as conn:
        rows = conn.execute(
            select(*_VERSION_SUMMARY, Deal.id.label("owned"))
            .select_from(Deal).outerjoin(DealVersion, DealVersion.deal_id == Deal.id)
            .where(_owned(subject, deal_id))
            .order_by(DealVersion.number.desc())
        ).all()
    if not rows:
        raise DealNotFound(str(deal_id))
    return [VersionRecord(number=r.number, kind=r.kind, label=r.label, created_at=r.created_at)
            for r in rows if r.number is not None]


def get_version(subject: str, deal_id: uuid.UUID, number: int) -> VersionRecord:
    with connect() as conn:
        rows = conn.execute(
            select(Deal.id, DealVersion.number, DealVersion.kind, DealVersion.label,
                   DealVersion.created_at, DealVersion.inputs, DealVersion.settings)
            .select_from(Deal)
            .outerjoin(DealVersion, and_(DealVersion.deal_id == Deal.id, DealVersion.number == number))
            .where(_owned(subject, deal_id))
        ).all()
    if not rows:
        raise DealNotFound(str(deal_id))
    r = rows[0]
    if r.number is None:
        raise VersionNotFound(str(number))
    return VersionRecord(number=r.number, kind=r.kind, label=r.label, created_at=r.created_at,
                         inputs=r.inputs, settings=r.settings)


def save_version(subject: str, deal_id: uuid.UUID, label: Optional[str] = None,
                 *, now: Optional[datetime] = None) -> VersionRecord:
    """Keep the working copy as a version.

    When nothing changed since the latest version, no row is added: that
    version is kept (and named, when a label is given) and returned.
    """
    label = clean_name(label, what="label", limit=MAX_LABEL_LENGTH) if label is not None else None
    now = now or utc_now()
    with transaction() as conn:
        row = _lock_deal(conn, subject, deal_id)
        latest = _latest_version(conn, deal_id)
        if latest is not None and latest.inputs == row.inputs and latest.settings == row.settings:
            if label is None and latest.kind != "auto":
                return VersionRecord(number=latest.number, kind=latest.kind, label=latest.label,
                                     created_at=latest.created_at)
            kind = "saved" if latest.kind == "auto" else latest.kind
            new_label = label if label is not None else latest.label
            conn.execute(update(DealVersion)
                         .where(DealVersion.deal_id == deal_id, DealVersion.number == latest.number)
                         .values(kind=kind, label=new_label))
            return VersionRecord(number=latest.number, kind=kind, label=new_label,
                                 created_at=latest.created_at)
        return _add_version(conn, deal_id, "saved", label, row.inputs, row.settings, now)


def restore_version(subject: str, deal_id: uuid.UUID, number: int,
                    *, now: Optional[datetime] = None) -> DealRecord:
    """Make an earlier version the working copy again.

    Nothing is lost: unsaved edits are kept as a version first, and the
    restore itself is recorded, so it can be undone from the history.
    """
    now = now or utc_now()
    with transaction() as conn:
        row = _lock_deal(conn, subject, deal_id)
        target = conn.execute(
            select(DealVersion.inputs, DealVersion.settings)
            .where(DealVersion.deal_id == deal_id, DealVersion.number == number)
        ).first()
        if target is None:
            raise VersionNotFound(str(number))
        latest = _latest_version(conn, deal_id)
        if latest is None or latest.inputs != row.inputs or latest.settings != row.settings:
            _add_version(conn, deal_id, "saved", f"Before restoring version {number}",
                         row.inputs, row.settings, now)
        conn.execute(update(Deal).where(Deal.id == deal_id)
                     .values(inputs=target.inputs, settings=target.settings, updated_at=now))
        _add_version(conn, deal_id, "restored", f"Restored version {number}",
                     target.inputs, target.settings, now)
        row = conn.execute(select(*_DEAL_COLUMNS).where(Deal.id == deal_id)).one()
    return _record(row)


# ---------------------------------------------------------------------------
# The account's settings
# ---------------------------------------------------------------------------
def get_settings(subject: str) -> dict:
    """The caller's Settings overrides; ``NoAccount`` before sign-up is finished."""
    with connect() as conn:
        row = conn.execute(select(User.settings).where(User.subject == subject)).first()
    if row is None:
        raise NoAccount(subject)
    return dict(row.settings or {})


def save_settings(subject: str, settings: Optional[Mapping]) -> dict:
    """Replace the caller's Settings overrides."""
    settings = clean_settings(settings)
    with transaction() as conn:
        done = conn.execute(update(User).where(User.subject == subject)
                            .values(settings=settings).returning(User.id)).first()
    if done is None:
        raise NoAccount(subject)
    return settings
