"""Reading and writing account profiles (PLAN.md 1.4).

One row per signed-in person (``db.models.User``), found by the identity
provider's subject. The preferences -- country, currency, locale, time zone
and digit grouping -- are validated here rather than at the edge, so nothing invalid can
reach the database whichever caller writes it.

Validation is deliberately generic, not a list of "supported" countries: the
app must work for any country and currency (PLAN.md guiding principle 2).
Currency codes are checked for shape by ``CurrencyCode``; time zones must be
real IANA names.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional
from zoneinfo import ZoneInfo, available_timezones

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from db.engine import connect, transaction
from db.models import User, utc_now

# ISO 3166-1 alpha-2, plus the user-assigned codes; two capitals either way
COUNTRY_RE = re.compile(r"^[A-Z]{2}$")
CURRENCY_RE = re.compile(r"^[A-Z]{3}$")
# BCP 47: language, optional script, optional region, optional variants
LOCALE_RE = re.compile(r"^[a-z]{2,3}(-[A-Z][a-z]{3})?(-([A-Z]{2}|\d{3}))?(-[A-Za-z0-9]{3,8})*$")
# How long numbers are grouped (PLAN.md 2.3a): as the locale does, always in
# thousands (1,000,000), or in lakh and crore (10,00,000) whatever the locale
DIGIT_GROUPINGS = ("locale", "thousands", "lakh")


class InvalidProfile(ValueError):
    """A profile field the app refuses to store."""


@dataclass(frozen=True)
class Profile:
    """How one person wants money, dates and numbers shown."""

    country: str
    preferred_currency: str
    locale: str
    time_zone: str
    digit_grouping: str = "locale"

    def as_dict(self) -> dict:
        return asdict(self)


def clean(country: str, preferred_currency: str, locale: str, time_zone: str,
          digit_grouping: str = "locale") -> Profile:
    """A validated profile, or ``InvalidProfile`` naming the field at fault."""
    country = (country or "").strip().upper()
    currency = (preferred_currency or "").strip().upper()
    locale = (locale or "").strip()
    time_zone = (time_zone or "").strip()
    if not COUNTRY_RE.match(country):
        raise InvalidProfile("country must be a two-letter ISO 3166-1 code, such as GB or JP")
    if not CURRENCY_RE.match(currency):
        raise InvalidProfile("currency must be a three-letter ISO 4217 code, such as EUR or INR")
    if not LOCALE_RE.match(locale):
        raise InvalidProfile("locale must be a language tag such as en-GB, fr or pt-BR")
    if time_zone not in available_timezones():
        raise InvalidProfile("time zone must be an IANA name such as Europe/London")
    try:
        ZoneInfo(time_zone)
    except Exception:  # noqa: BLE001 - a name the tz database can't load
        raise InvalidProfile("time zone must be an IANA name such as Europe/London") from None
    if digit_grouping not in DIGIT_GROUPINGS:
        raise InvalidProfile("digit grouping must be one of: " + ", ".join(DIGIT_GROUPINGS))
    return Profile(country=country, preferred_currency=currency, locale=locale, time_zone=time_zone,
                   digit_grouping=digit_grouping)


def get_profile(subject: str) -> Optional[Profile]:
    """The person's saved profile, or None when they haven't set one yet."""
    with connect() as conn:
        row = conn.execute(
            select(User.country, User.preferred_currency, User.locale, User.time_zone,
                   User.digit_grouping)
            .where(User.subject == subject)
        ).first()
    if row is None:
        return None
    return Profile(country=row.country, preferred_currency=row.preferred_currency,
                   locale=row.locale, time_zone=row.time_zone, digit_grouping=row.digit_grouping)


def save_profile(subject: str, profile: Profile) -> Profile:
    """Store the profile for ``subject``, inserting the row the first time.

    One statement, so a person signing in from two tabs at once can't create
    two rows (the unique index on ``subject`` decides, and the later write
    wins).
    """
    now = utc_now()
    values = {"subject": subject, "created_at": now, "updated_at": now, **profile.as_dict()}
    statement = insert(User).values(**values)
    statement = statement.on_conflict_do_update(
        index_elements=[User.subject],
        set_={**profile.as_dict(), "updated_at": now},
    )
    with transaction() as conn:
        conn.execute(statement)
    return profile
