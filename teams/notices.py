"""Emails to people mentioned in a comment (PLAN.md 7.2), in batches inside the mail service's free plan.

``send_pending`` works through the ``mention_notices`` not yet sent:

- **one email per person per batch**: everything waiting for them, at most one
  email every ``BATCH_HOURS``; a mention made sooner waits for the next run;
- **inside the free plan**: at most ``DAILY_EMAILS`` a UTC day and
  ``MONTHLY_EMAILS`` a calendar month for the whole service (Resend's free plan
  allows 100 and 3,000); past either, notices wait;
- **in the recipient's language and time zone** (their account's locale and
  time zone): the words are ``email.*`` in ``web/messages/<language>.json``,
  English underneath a missing key;
- **saying who and when, never what**: the email names the person who wrote
  the comment and when, and links to the deal; never the comment, the deal's
  name or a figure (deal contents don't leave the app);
- a notice not sent within ``EXPIRE_DAYS`` is dropped (``expired``), so mail
  switched on later never sends a backlog, and so is one whose recipient can no
  longer read the deal; one whose recipient has no verified
  address is ``no_address``; one the mail service refuses outright ``failed``.

It runs as job ``mentions.send`` after a comment that mentions someone, and in
the nightly ``job-maintenance`` task for anything left waiting. Without a mail
service or a directory nothing is sent and nothing is marked, but expiry.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from sqlalchemy import func, select, update

from api.observability import log_event
from db.engine import connect, transaction
from db.models import Deal, DealComment, DealShare, MentionNotice, TeamMember, User, utc_now
from teams import directory as people
from teams import mail as mailer

BATCH_HOURS = 1
DAILY_EMAILS = 90
MONTHLY_EMAILS = 2900
EXPIRE_DAYS = 7
# Recipients one run works through at most (the rest wait for the next)
MAX_PER_RUN = 50
MESSAGES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web", "messages")
DEFAULT_LANGUAGE = "en"
SLOT = re.compile(r"\{([A-Za-z]+)\}")


def app_url() -> str:
    return (os.environ.get("FSE_APP_URL", "").strip() or "https://variater.com").rstrip("/")


def words(language: str) -> dict:
    """The ``email`` messages of a language, English underneath any it lacks."""
    def read(lang: str) -> dict:
        path = os.path.join(MESSAGES, f"{lang}.json")
        if not re.fullmatch(r"[a-z]{2,3}", lang) or not os.path.isfile(path):
            return {}
        with open(path, encoding="utf-8") as handle:
            return json.load(handle).get("email") or {}
    return {**read(DEFAULT_LANGUAGE), **read(language)}


def words_for(locale: Optional[str]) -> dict:
    from ml.memo import language_for
    return words(language_for(locale))


def fill(template: str, **values: str) -> str:
    """A message with its ``{slot}``s filled (never ``str.format``: a name may hold braces)."""
    return SLOT.sub(lambda m: values.get(m.group(1), m.group(0)), template)


def local_time(when: datetime, zone: Optional[str]) -> str:
    try:
        tz = ZoneInfo(zone) if zone else ZoneInfo("UTC")
    except (ZoneInfoNotFoundError, ValueError):
        tz = ZoneInfo("UTC")
    return when.astimezone(tz).strftime("%Y-%m-%d %H:%M") + f" ({tz.key})"


def compose(*, locale: Optional[str], zone: Optional[str],
            mentions: list[tuple[Optional[str], datetime, str]]) -> tuple[str, str]:
    """An email's subject and text in the locale's language: per mention, who (a name, or None for "a team
    member"), when in ``zone``, and the deal's link."""
    w = words_for(locale)
    lines = [w["greeting"], ""]
    for name, when, deal_id in mentions:
        lines.append(fill(w["mentionedYou"], name=name or w["someone"], time=local_time(when, zone)))
        lines.append(fill(w["openDeal"], link=f"{app_url()}/deal/team?deal={deal_id}"))
        lines.append("")
    lines.append(w["why"])
    subject = w["subjectOne"] if len(mentions) == 1 else fill(w["subjectMany"], count=str(len(mentions)))
    return subject, "\n".join(lines)


def _sent_since(conn, since: datetime) -> int:
    """Emails sent since ``since``: one per recipient and send time."""
    batches = (select(MentionNotice.recipient_id, MentionNotice.sent_at)
               .where(MentionNotice.outcome == "sent", MentionNotice.sent_at >= since)
               .group_by(MentionNotice.recipient_id, MentionNotice.sent_at).subquery())
    return conn.execute(select(func.count()).select_from(batches)).scalar_one()


def _mark(notice_ids: list[int], outcome: str, now: datetime) -> None:
    with transaction() as conn:
        conn.execute(update(MentionNotice).where(MentionNotice.id.in_(notice_ids), MentionNotice.sent_at.is_(None))
                     .values(sent_at=now, outcome=outcome))


def send_pending(now: Optional[datetime] = None, *, mail: Optional[mailer.Mail] = None,
                 directory: Optional[people.Directory] = None) -> dict:
    """Send what is due; a flat summary of what happened."""
    now = now or utc_now()
    mail = mail if mail is not None else mailer.get_mail()
    directory = directory if directory is not None else people.get_directory()
    summary = {"mentions_mail": getattr(mail, "kind", "none"), "mentions_emails_sent": 0,
               "mentions_expired": 0, "mentions_no_address": 0, "mentions_failed": 0, "mentions_waiting": 0}
    # Someone who can no longer read the deal (removed from the team, the deal unshared) is told nothing
    comment_deal = (select(DealComment.deal_id).where(DealComment.id == MentionNotice.comment_id)
                    .correlate(MentionNotice).scalar_subquery())
    can_read = (select(Deal.id).where(Deal.id == comment_deal, Deal.owner_id == MentionNotice.recipient_id)
                .correlate(MentionNotice).exists()
                | select(DealShare.deal_id).join(TeamMember, TeamMember.team_id == DealShare.team_id)
                .where(DealShare.deal_id == comment_deal, TeamMember.user_id == MentionNotice.recipient_id)
                .correlate(MentionNotice).exists())
    with transaction() as conn:
        summary["mentions_expired"] = conn.execute(
            update(MentionNotice).where(MentionNotice.sent_at.is_(None),
                                        (MentionNotice.created_at < now - timedelta(days=EXPIRE_DAYS)) | ~can_read)
            .values(sent_at=now, outcome="expired")).rowcount
    with connect() as conn:
        pending = conn.execute(
            select(MentionNotice.id, MentionNotice.recipient_id, MentionNotice.created_at,
                   DealComment.created_at.label("written"), DealComment.deal_id,
                   User.subject.label("author"))
            .join(DealComment, DealComment.id == MentionNotice.comment_id)
            .join(User, User.id == DealComment.author_id)
            .where(MentionNotice.sent_at.is_(None))
            .order_by(MentionNotice.created_at, MentionNotice.id)).all()
        if not pending:
            return summary
        recipients = list(dict.fromkeys(r.recipient_id for r in pending))
        last = dict(conn.execute(
            select(MentionNotice.recipient_id, func.max(MentionNotice.sent_at))
            .where(MentionNotice.recipient_id.in_(recipients), MentionNotice.outcome == "sent")
            .group_by(MentionNotice.recipient_id)).all())
        accounts = {r.id: r for r in conn.execute(
            select(User.id, User.subject, User.locale, User.time_zone).where(User.id.in_(recipients))).all()}
        day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today = _sent_since(conn, day)
        month = _sent_since(conn, day.replace(day=1))
    if mail is None or isinstance(directory, people.NoDirectory):
        summary["mentions_waiting"] = len(pending)
        return summary
    names = directory.names({r.author for r in pending})
    sent_to = 0
    for recipient in recipients:
        theirs = [r for r in pending if r.recipient_id == recipient]
        held = last.get(recipient)
        due = held is None or now - held >= timedelta(hours=BATCH_HOURS)
        if not due or sent_to >= MAX_PER_RUN or today >= DAILY_EMAILS or month >= MONTHLY_EMAILS:
            summary["mentions_waiting"] += len(theirs)
            continue
        ids = [r.id for r in theirs]
        account = accounts[recipient]
        try:
            address = directory.email(account.subject)
        except people.DirectoryUnavailable:
            summary["mentions_waiting"] += len(theirs)
            continue
        if not address:
            _mark(ids, "no_address", now)
            summary["mentions_no_address"] += len(ids)
            continue
        subject, text = compose(locale=account.locale, zone=account.time_zone,
                                mentions=[(names.get(r.author), r.written, str(r.deal_id)) for r in theirs])
        try:
            mail.send(mailer.Email(to=address, subject=subject, text=text, key=mailer.batch_key(recipient, ids)))
        except mailer.MailError as exc:
            log_event("mention_email_failed", logging.WARNING, reason=exc.reason, status=exc.status)
            if exc.reason == "rejected":
                _mark(ids, "failed", now)
                summary["mentions_failed"] += len(ids)
                continue
            # The service is over a limit or down: everything left waits for the next run
            left = set(recipients[recipients.index(recipient):])
            summary["mentions_waiting"] += sum(1 for r in pending if r.recipient_id in left)
            break
        _mark(ids, "sent", now)
        summary["mentions_emails_sent"] += 1
        sent_to += 1
        today += 1
        month += 1
    return summary

