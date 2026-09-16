"""Tables and the rules every table follows.

Rules (checked by ``check_conventions``, run in the test suite):

- **Times are UTC.** Every date-time column is ``UTCDateTime``: stored as
  ``timestamptz``, refuses naive datetimes on write, and reads back in UTC.
  Screens convert to the viewer's time zone.
- **Money always has a currency.** A money amount is a ``MoneyAmount`` column
  named ``<name>_amount`` with a ``CurrencyCode`` column ``<name>_currency``
  (ISO 4217, e.g. ``EUR``) in the same table. No bare float money.
- Constraint and index names follow ``NAMING_CONVENTION`` so migrations can
  drop them by name.

To add a table: define it here, then generate and review a migration (see
CLAUDE.md "Adding a table").
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    BigInteger, CheckConstraint, DateTime, ForeignKey, Identity, Index, Integer, MetaData, Numeric,
    String, UniqueConstraint, Uuid, func, text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """A timezone-aware timestamp, always handed back in UTC."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: Optional[datetime], dialect):
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetime: pass a timezone-aware datetime (UTC)")
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: Optional[datetime], dialect):
        if value is None:
            return None
        if value.tzinfo is None:  # a database without time zones; values were written as UTC
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


_CURRENCY_RE = re.compile(r"^[A-Z]{3}$")


class CurrencyCode(TypeDecorator):
    """An ISO 4217 currency code: three capital letters."""

    impl = String(3)
    cache_ok = True

    def process_bind_param(self, value: Optional[str], dialect):
        if value is None:
            return None
        if not isinstance(value, str) or not _CURRENCY_RE.match(value):
            raise ValueError("currency must be an ISO 4217 code such as USD, EUR or JPY")
        return value


class MoneyAmount(TypeDecorator):
    """A money amount, exact (never float). Pair it with a ``CurrencyCode`` column."""

    impl = Numeric(24, 6)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if isinstance(value, float):
            raise ValueError("money amounts must be Decimal or int, not float")
        return value


def check_conventions(metadata: MetaData) -> list[str]:
    """Rule breaks in ``metadata``; empty when every table follows the rules."""
    problems = []
    for table in metadata.sorted_tables:
        columns = {c.name: c for c in table.columns}
        for col in table.columns:
            where = f"{table.name}.{col.name}"
            is_datetime = isinstance(col.type, DateTime) or (
                isinstance(col.type, TypeDecorator) and isinstance(col.type.impl, DateTime))
            if is_datetime and not isinstance(col.type, UTCDateTime):
                problems.append(f"{where}: date-time columns must use UTCDateTime")
            if isinstance(col.type, MoneyAmount):
                if not col.name.endswith("_amount"):
                    problems.append(f"{where}: money columns are named <name>_amount")
                    continue
                pair = col.name[: -len("_amount")] + "_currency"
                if pair not in columns:
                    problems.append(f"{where}: money needs a currency column {pair}")
                elif not isinstance(columns[pair].type, CurrencyCode):
                    problems.append(f"{table.name}.{pair}: must use CurrencyCode")
                elif columns[pair].nullable and not col.nullable:
                    problems.append(f"{table.name}.{pair}: can't be empty while {col.name} is required")
            elif isinstance(col.type, (Numeric,)) and col.name.endswith("_amount"):
                problems.append(f"{where}: money amounts must use MoneyAmount")
    return problems


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
class StorageCheck(Base):
    """One reading of the database's size against the free plan's storage limit.

    Written by the database health check (``db.health``), at most every few
    minutes; readings older than ``db.health.KEEP_READINGS_DAYS`` are pruned.
    """

    __tablename__ = "storage_checks"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    checked_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False,
                                                 server_default=func.now(), index=True)
    environment: Mapped[str] = mapped_column(String(20), nullable=False)
    database_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    limit_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)


class User(Base):
    """One signed-in person, and how they want figures shown (PLAN.md 1.4).

    ``subject`` is the account id from the identity provider (Clerk's
    ``user_…``), the only link back to them. Names, email addresses and
    sign-in details stay with the provider: nothing personal is stored here,
    so the database holds no contact details to leak and none reach the logs.

    The four preferences are asked at sign-up and used everywhere figures are
    shown or defaults chosen (PLAN.md 2.2 and 2.3 build on them):
    ``country`` ISO 3166-1 alpha-2, ``preferred_currency`` ISO 4217,
    ``locale`` a BCP 47 tag, ``time_zone`` an IANA name.
    """

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    subject: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    country: Mapped[str] = mapped_column(String(2), nullable=False)
    preferred_currency: Mapped[str] = mapped_column(CurrencyCode, nullable=False)
    locale: Mapped[str] = mapped_column(String(35), nullable=False)
    time_zone: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False,
                                                 server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False,
                                                 server_default=func.now())
    # The account's Settings overrides (PLAN.md 1.5): only the keys that
    # differ from core/config.py DEFAULTS, so an untouched account stores {}
    settings: Mapped[dict] = mapped_column(JSONB, nullable=False,
                                           server_default=text("'{}'::jsonb"))


class Deal(Base):
    """One saved deal and its working copy (PLAN.md 1.5).

    ``inputs`` and ``settings`` are the draft: what the deal screens show now.
    Autosave overwrites them in place, so editing never grows the database;
    history lives in ``deal_versions``. ``inputs`` is always complete (every
    field of ``DealInputsIn``), so a deal reopens with the same numbers even
    if the API's input defaults change; ``settings`` holds only overrides.

    ``id`` is a random UUID: it appears in addresses, and can't be guessed or
    counted. Every read and write also matches ``owner_id``, so a deal id
    alone never opens someone else's deal (db/deals.py).
    """

    __tablename__ = "deals"
    __table_args__ = (
        Index("ix_deals_owner_id_updated_at", "owner_id", "updated_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    inputs: Mapped[dict] = mapped_column(JSONB, nullable=False)
    settings: Mapped[dict] = mapped_column(JSONB, nullable=False,
                                           server_default=text("'{}'::jsonb"))
    # Highest version number handed out; numbers are never reused
    latest_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    archived_at: Mapped[Optional[datetime]] = mapped_column(UTCDateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False,
                                                 server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False,
                                                 server_default=func.now())


VERSION_KINDS = ("created", "saved", "auto", "restored")


class DealVersion(Base):
    """A point in a deal's history the user can go back to.

    Kept small so the free 0.5 GB lasts (db/deals.py): a version is written
    only when its content differs from the one before; automatic checkpoints
    are spaced out and capped per deal; settings are overrides only. A
    version is a few hundred bytes.
    """

    __tablename__ = "deal_versions"
    __table_args__ = (
        UniqueConstraint("deal_id", "number", name="uq_deal_versions_deal_id_number"),
        CheckConstraint("kind IN ('created', 'saved', 'auto', 'restored')", name="kind"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    deal_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("deals.id", ondelete="CASCADE"), nullable=False)
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(10), nullable=False)
    label: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    inputs: Mapped[dict] = mapped_column(JSONB, nullable=False)
    settings: Mapped[dict] = mapped_column(JSONB, nullable=False,
                                           server_default=text("'{}'::jsonb"))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False,
                                                 server_default=func.now())


class UsageCounter(Base):
    """A shared usage counter, when Upstash Redis can't be used (PLAN.md 1.6).

    The fallback store for api/usage.py: one row per limit window, e.g. one
    user's runs today. ``key`` names the limit, a hash of the user and the
    window, never the user id itself. Rows are deleted once ``expires_at`` has
    passed, so the table stays a few kilobytes.
    """

    __tablename__ = "usage_counters"

    key: Mapped[str] = mapped_column(String(160), primary_key=True)
    count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime, nullable=False, index=True)


__all__ = ["Base", "CurrencyCode", "Deal", "DealVersion", "MoneyAmount", "StorageCheck", "UsageCounter",
           "User", "UTCDateTime", "VERSION_KINDS", "check_conventions", "utc_now"]
