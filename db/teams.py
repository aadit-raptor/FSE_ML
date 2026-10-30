"""Teams, invitations, shared deals and comments (PLAN.md 7.2).

**Who.** A person is their user id here and nothing else: names and email
addresses stay with the identity provider (teams/directory.py reads them when a
screen or an email needs them). Every function takes the caller's subject from
the verified token and matches it in the statement that reads or writes, as
db/deals.py does: a team the caller isn't in, or a deal they can't read,
answers ``TeamNotFound`` / ``DealNotFound`` (404), never "it exists".

**Roles.** In a team: ``owner`` runs it (renames it, invites, changes roles,
removes people, deletes it) and edits its deals; ``editor`` edits its deals;
``viewer`` reads them. Everyone in it may comment and share their own deals with
it. A team always has an owner: the last one can't leave, be removed or step
down (``LastOwner``); they delete the team instead.

**Sharing.** A deal's owner shares it with a team they are in; its members then
reach it through db/deals.py by their role. Leaving or being removed from a team
takes back every deal the person shared with it.

**Comments** are the writer's own words on a deal anyone with access can read:
never logged, never in an audit entry, never in an email. A comment names the
people it mentions by user id; each must be able to read the deal, and each gets
a ``mention_notices`` row the mail sender (teams/notices.py) works through.

Every change is one audit entry in its own transaction; one that changes
nothing (a role set to the role held) writes none.
"""
from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Sequence

from sqlalchemy import and_, delete, func, insert, or_, select, update

from db import audit
from db.deals import EDIT_ROLES, READ, DealNotFound, NotPermitted, _access, _allowed, _reachable
from db.engine import connect, transaction
from db.models import (
    INVITE_ROLES, TEAM_ROLES, Deal, DealComment, DealShare, MentionNotice, Team, TeamInvite, TeamMember, User,
    utc_now,
)

MAX_TEAM_NAME = 80
# A person's teams, and a team's people: well inside the free database
MAX_TEAMS_PER_PERSON = 20
MAX_MEMBERS = 50
# Live invitations a team holds at once; how long one lasts and how many may join by it
MAX_LIVE_INVITES = 20
INVITE_DAYS = 7
INVITE_USES = 10
MAX_INVITE_USES = 50
# A comment's length, its mentions, and the comments a deal holds
MAX_COMMENT = 2000
MAX_MENTIONS = 10
MAX_COMMENTS_PER_DEAL = 2000
TOKEN_PREFIX = "inv_"


class TeamNotFound(LookupError):
    """No such team for this caller (it may exist without them)."""


class MemberNotFound(LookupError):
    """No such person in the team."""


class InviteInvalid(LookupError):
    """An invitation that doesn't exist, has ended or was used up."""


class CommentNotFound(LookupError):
    """No such comment on the deal."""


class LastOwner(ValueError):
    """The change would leave the team without an owner."""


class TeamFull(ValueError):
    """A limit on teams, people, invitations or comments was reached; the code says which."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class InvalidTeamInput(ValueError):
    """A name, role, comment or mention the app refuses to store."""


@dataclass(frozen=True)
class TeamRecord:
    id: uuid.UUID
    name: str
    role: str
    members: int
    created_at: datetime


@dataclass(frozen=True)
class MemberRecord:
    user_id: int
    subject: str
    role: str
    joined_at: datetime


@dataclass(frozen=True)
class InviteRecord:
    id: uuid.UUID
    role: str
    uses_left: int
    expires_at: datetime
    token: Optional[str] = None


@dataclass(frozen=True)
class CommentRecord:
    id: uuid.UUID
    author_id: int
    author_subject: str
    body: str
    mentions: list[int]
    created_at: datetime


@dataclass(frozen=True)
class PersonRecord:
    """Someone who can read a deal: its owner, or a member of a team it is shared with."""
    user_id: int
    subject: str
    access: str


# ---------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------
def _me(subject: str):
    return select(User.id).where(User.subject == subject).scalar_subquery()


def clean_team_name(name) -> str:
    if not isinstance(name, str):
        raise InvalidTeamInput("name")
    name = " ".join(name.split())
    if not name or len(name) > MAX_TEAM_NAME or any(ord(c) < 32 for c in name):
        raise InvalidTeamInput("name")
    return name


def clean_comment(body) -> str:
    if not isinstance(body, str):
        raise InvalidTeamInput("body")
    body = body.strip()
    # Line breaks are kept; other control characters are not text
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    if not body or len(body) > MAX_COMMENT or any(ord(c) < 32 and c not in "\n\t" for c in body):
        raise InvalidTeamInput("body")
    return body


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _member_row(conn, subject: str, team_id: uuid.UUID, *, lock: bool = False):
    query = (select(TeamMember.user_id, TeamMember.role)
             .where(TeamMember.team_id == team_id, TeamMember.user_id == _me(subject)))
    row = conn.execute(query.with_for_update() if lock else query).first()
    if row is None:
        raise TeamNotFound(str(team_id))
    return row


def _owner_row(conn, subject: str, team_id: uuid.UUID):
    """The caller's membership, which must be an owner's (else ``NotPermitted``)."""
    # Lock the team so two owners stepping down at once can't both leave it ownerless
    conn.execute(select(Team.id).where(Team.id == team_id).with_for_update()).first()
    row = _member_row(conn, subject, team_id)
    if row.role != "owner":
        raise NotPermitted("team_owner")
    return row


def _owners(conn, team_id: uuid.UUID) -> int:
    return conn.execute(select(func.count()).select_from(TeamMember)
                        .where(TeamMember.team_id == team_id, TeamMember.role == "owner")).scalar_one()


def _user_id(conn, subject: str) -> int:
    from db.deals import NoAccount
    found = conn.execute(select(User.id).where(User.subject == subject)).scalar()
    if found is None:
        raise NoAccount(subject)
    return found


def subjects_of(user_ids) -> dict[int, str]:
    """The subjects of these user ids (to ask the directory for names): never sent to a browser."""
    ids = sorted({int(i) for i in user_ids if i is not None})
    if not ids:
        return {}
    with connect() as conn:
        return dict(conn.execute(select(User.id, User.subject).where(User.id.in_(ids))).all())


def my_id(subject: str) -> Optional[int]:
    with connect() as conn:
        return conn.execute(select(User.id).where(User.subject == subject)).scalar()


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------
def list_teams(subject: str) -> list[TeamRecord]:
    """The caller's teams, by name, with their role and how many people each has."""
    counts = (select(TeamMember.team_id, func.count().label("n"))
              .group_by(TeamMember.team_id).subquery())
    query = (select(Team.id, Team.name, Team.created_at, TeamMember.role, counts.c.n)
             .join(TeamMember, TeamMember.team_id == Team.id)
             .join(counts, counts.c.team_id == Team.id)
             .where(TeamMember.user_id == _me(subject))
             .order_by(Team.name, Team.id))
    with connect() as conn:
        rows = conn.execute(query).all()
    return [TeamRecord(id=r.id, name=r.name, role=r.role, members=r.n, created_at=r.created_at) for r in rows]


def create_team(subject: str, name, *, now: Optional[datetime] = None) -> TeamRecord:
    name = clean_team_name(name)
    now = now or utc_now()
    with transaction() as conn:
        me = _user_id(conn, subject)
        held = conn.execute(select(func.count()).select_from(TeamMember)
                            .where(TeamMember.user_id == me)).scalar_one()
        if held >= MAX_TEAMS_PER_PERSON:
            raise TeamFull("teams")
        team_id = uuid.uuid4()
        conn.execute(insert(Team).values(id=team_id, name=name, created_at=now))
        conn.execute(insert(TeamMember).values(team_id=team_id, user_id=me, role="owner", joined_at=now))
        audit.record(conn, me, "team_created", now=now, detail={"team": str(team_id)})
    return TeamRecord(id=team_id, name=name, role="owner", members=1, created_at=now)


def get_team(subject: str, team_id: uuid.UUID) -> tuple[TeamRecord, list[MemberRecord]]:
    """The team and its people (owners first, then by when they joined), for its members only."""
    with connect() as conn:
        mine = _member_row(conn, subject, team_id)
        team = conn.execute(select(Team.id, Team.name, Team.created_at).where(Team.id == team_id)).one()
        rows = conn.execute(
            select(TeamMember.user_id, TeamMember.role, TeamMember.joined_at, User.subject)
            .join(User, User.id == TeamMember.user_id).where(TeamMember.team_id == team_id)
            .order_by((TeamMember.role != "owner"), TeamMember.joined_at, TeamMember.user_id)).all()
    members = [MemberRecord(user_id=r.user_id, subject=r.subject, role=r.role, joined_at=r.joined_at) for r in rows]
    return (TeamRecord(id=team.id, name=team.name, role=mine.role, members=len(members),
                       created_at=team.created_at), members)


def rename_team(subject: str, team_id: uuid.UUID, name, *, now: Optional[datetime] = None) -> None:
    name = clean_team_name(name)
    with transaction() as conn:
        me = _owner_row(conn, subject, team_id).user_id
        old = conn.execute(select(Team.name).where(Team.id == team_id)).scalar_one()
        if old == name:
            return
        conn.execute(update(Team).where(Team.id == team_id).values(name=name))
        audit.record(conn, me, "team_renamed", now=now, detail={"team": str(team_id)})


def delete_team(subject: str, team_id: uuid.UUID, *, now: Optional[datetime] = None) -> None:
    """Delete the team: its invitations and shares go with it; the deals stay their owners'."""
    with transaction() as conn:
        me = _owner_row(conn, subject, team_id).user_id
        conn.execute(delete(Team).where(Team.id == team_id))
        audit.record(conn, me, "team_deleted", now=now, detail={"team": str(team_id)})


def set_role(subject: str, team_id: uuid.UUID, user_id: int, role: str,
             *, now: Optional[datetime] = None) -> None:
    """An owner gives someone in the team another role (themselves too, unless they are the last owner)."""
    if role not in TEAM_ROLES:
        raise InvalidTeamInput("role")
    with transaction() as conn:
        me = _owner_row(conn, subject, team_id).user_id
        held = conn.execute(select(TeamMember.role).where(TeamMember.team_id == team_id,
                                                          TeamMember.user_id == user_id)).scalar()
        if held is None:
            raise MemberNotFound(str(user_id))
        if held == role:
            return
        if held == "owner" and _owners(conn, team_id) <= 1:
            raise LastOwner()
        conn.execute(update(TeamMember).where(TeamMember.team_id == team_id, TeamMember.user_id == user_id)
                     .values(role=role))
        audit.record(conn, me, "team_role_changed", now=now,
                     detail={"team": str(team_id), "member": user_id, "role": role})


def remove_member(subject: str, team_id: uuid.UUID, user_id: int, *, now: Optional[datetime] = None) -> None:
    """Take someone out of the team: an owner may remove anyone, anyone may remove themselves (leave).
    Every deal they shared with the team is no longer shared with it."""
    with transaction() as conn:
        mine = _member_row(conn, subject, team_id)
        leaving = mine.user_id == user_id
        if not leaving:
            mine = _owner_row(conn, subject, team_id)
        else:
            conn.execute(select(Team.id).where(Team.id == team_id).with_for_update()).first()
        held = conn.execute(select(TeamMember.role).where(TeamMember.team_id == team_id,
                                                          TeamMember.user_id == user_id)).scalar()
        if held is None:
            raise MemberNotFound(str(user_id))
        if held == "owner" and _owners(conn, team_id) <= 1:
            raise LastOwner()
        conn.execute(delete(TeamMember).where(TeamMember.team_id == team_id, TeamMember.user_id == user_id))
        theirs = select(Deal.id).where(Deal.owner_id == user_id)
        conn.execute(delete(DealShare).where(DealShare.team_id == team_id, DealShare.deal_id.in_(theirs)))
        if leaving:
            audit.record(conn, mine.user_id, "team_left", now=now, detail={"team": str(team_id)})
        else:
            audit.record(conn, mine.user_id, "team_member_removed", now=now,
                         detail={"team": str(team_id), "member": user_id})


# ---------------------------------------------------------------------------
# Invitations
# ---------------------------------------------------------------------------
def create_invite(subject: str, team_id: uuid.UUID, role: str, *, uses: int = INVITE_USES,
                  days: int = INVITE_DAYS, now: Optional[datetime] = None) -> InviteRecord:
    """A new invitation link's token, shown once: only its hash is kept."""
    if role not in INVITE_ROLES:
        raise InvalidTeamInput("role")
    if not 1 <= uses <= MAX_INVITE_USES:
        raise InvalidTeamInput("uses")
    now = now or utc_now()
    with transaction() as conn:
        me = _owner_row(conn, subject, team_id).user_id
        live = conn.execute(select(func.count()).select_from(TeamInvite).where(
            TeamInvite.team_id == team_id, TeamInvite.expires_at > now, TeamInvite.uses_left > 0)).scalar_one()
        if live >= MAX_LIVE_INVITES:
            raise TeamFull("invites")
        # Ended invitations are of no use to anyone: they go when a new one is made
        conn.execute(delete(TeamInvite).where(TeamInvite.team_id == team_id,
                                              or_(TeamInvite.expires_at <= now, TeamInvite.uses_left <= 0)))
        token = TOKEN_PREFIX + secrets.token_urlsafe(32)
        invite_id = uuid.uuid4()
        expires = now + timedelta(days=days)
        conn.execute(insert(TeamInvite).values(id=invite_id, team_id=team_id, token_hash=token_hash(token),
                                               role=role, uses_left=uses, expires_at=expires, created_at=now))
        audit.record(conn, me, "team_invited", now=now, detail={"team": str(team_id), "role": role})
    return InviteRecord(id=invite_id, role=role, uses_left=uses, expires_at=expires, token=token)


def list_invites(subject: str, team_id: uuid.UUID, *, now: Optional[datetime] = None) -> list[InviteRecord]:
    """The team's live invitations (without their links, which aren't kept), for its owners."""
    now = now or utc_now()
    with connect() as conn:
        if _member_row(conn, subject, team_id).role != "owner":
            raise NotPermitted("team_owner")
        rows = conn.execute(select(TeamInvite.id, TeamInvite.role, TeamInvite.uses_left, TeamInvite.expires_at)
                            .where(TeamInvite.team_id == team_id, TeamInvite.expires_at > now,
                                   TeamInvite.uses_left > 0)
                            .order_by(TeamInvite.expires_at)).all()
    return [InviteRecord(id=r.id, role=r.role, uses_left=r.uses_left, expires_at=r.expires_at) for r in rows]


def withdraw_invite(subject: str, team_id: uuid.UUID, invite_id: uuid.UUID) -> None:
    with transaction() as conn:
        _owner_row(conn, subject, team_id)
        gone = conn.execute(delete(TeamInvite).where(TeamInvite.team_id == team_id, TeamInvite.id == invite_id)
                            .returning(TeamInvite.id)).first()
    if gone is None:
        raise InviteInvalid(str(invite_id))


def join(subject: str, token, *, now: Optional[datetime] = None) -> TeamRecord:
    """Join the team an invitation is for, in its role. Someone already in it keeps their role (and uses
    nothing up)."""
    if not isinstance(token, str) or not token.startswith(TOKEN_PREFIX) or len(token) > 100:
        raise InviteInvalid("token")
    now = now or utc_now()
    with transaction() as conn:
        me = _user_id(conn, subject)
        invite = conn.execute(select(TeamInvite.id, TeamInvite.team_id, TeamInvite.role, TeamInvite.uses_left,
                                     TeamInvite.expires_at)
                              .where(TeamInvite.token_hash == token_hash(token)).with_for_update()).first()
        if invite is None or invite.expires_at <= now:
            raise InviteInvalid("token")
        team_id = invite.team_id
        held = conn.execute(select(TeamMember.role).where(TeamMember.team_id == team_id,
                                                          TeamMember.user_id == me)).scalar()
        if held is None and invite.uses_left <= 0:
            raise InviteInvalid("token")
        if held is None:
            people = conn.execute(select(func.count()).select_from(TeamMember)
                                  .where(TeamMember.team_id == team_id)).scalar_one()
            if people >= MAX_MEMBERS:
                raise TeamFull("members")
            teams = conn.execute(select(func.count()).select_from(TeamMember)
                                 .where(TeamMember.user_id == me)).scalar_one()
            if teams >= MAX_TEAMS_PER_PERSON:
                raise TeamFull("teams")
            conn.execute(insert(TeamMember).values(team_id=team_id, user_id=me, role=invite.role, joined_at=now))
            conn.execute(update(TeamInvite).where(TeamInvite.id == invite.id)
                         .values(uses_left=TeamInvite.uses_left - 1))
            audit.record(conn, me, "team_joined", now=now, detail={"team": str(team_id), "role": invite.role})
    team, _ = get_team(subject, team_id)
    return team


# ---------------------------------------------------------------------------
# Sharing a deal
# ---------------------------------------------------------------------------
def _own_deal(conn, subject: str, deal_id: uuid.UUID, *, lock: bool = False):
    query = select(Deal.id, Deal.owner_id, _access(subject)).where(_reachable(subject, deal_id))
    row = conn.execute(query.with_for_update(of=Deal) if lock else query).first()
    row = _allowed(row, deal_id, READ)
    if row.access != "owner":
        raise NotPermitted("deal_owner")
    return row


def deal_teams(subject: str, deal_id: uuid.UUID) -> list[uuid.UUID]:
    """The teams the deal is shared with, for anyone who can read it."""
    with connect() as conn:
        _allowed(conn.execute(select(Deal.id, _access(subject)).where(_reachable(subject, deal_id))).first(),
                 deal_id, READ)
        rows = conn.execute(select(DealShare.team_id).where(DealShare.deal_id == deal_id)
                            .order_by(DealShare.shared_at)).all()
    return [r.team_id for r in rows]


def share(subject: str, deal_id: uuid.UUID, team_id: uuid.UUID, *, now: Optional[datetime] = None) -> bool:
    """Share the caller's deal with one of their teams; False if it already was."""
    now = now or utc_now()
    with transaction() as conn:
        owner = _own_deal(conn, subject, deal_id, lock=True).owner_id
        _member_row(conn, subject, team_id)
        if conn.execute(select(DealShare.deal_id).where(DealShare.deal_id == deal_id,
                                                        DealShare.team_id == team_id)).first():
            return False
        conn.execute(insert(DealShare).values(deal_id=deal_id, team_id=team_id, shared_at=now))
        audit.record(conn, owner, "shared", deal_id=deal_id, now=now, detail={"team": str(team_id)})
    return True


def unshare(subject: str, deal_id: uuid.UUID, team_id: uuid.UUID, *, now: Optional[datetime] = None) -> bool:
    """Stop sharing the caller's deal with a team; False if it wasn't."""
    with transaction() as conn:
        owner = _own_deal(conn, subject, deal_id, lock=True).owner_id
        gone = conn.execute(delete(DealShare).where(DealShare.deal_id == deal_id, DealShare.team_id == team_id)
                            .returning(DealShare.team_id)).first()
        if gone is None:
            return False
        audit.record(conn, owner, "unshared", deal_id=deal_id, now=now, detail={"team": str(team_id)})
    return True


def deal_people(subject: str, deal_id: uuid.UUID) -> list[PersonRecord]:
    """Everyone who can read the deal (its owner first), with their access: who may be mentioned."""
    with connect() as conn:
        deal = _allowed(conn.execute(select(Deal.id, Deal.owner_id, _access(subject))
                                     .where(_reachable(subject, deal_id))).first(), deal_id, READ)
        return _people(conn, deal_id, deal.owner_id)


def _people(conn, deal_id: uuid.UUID, owner_id: int) -> list[PersonRecord]:
    rows = conn.execute(
        select(TeamMember.user_id, TeamMember.role, User.subject)
        .join(DealShare, DealShare.team_id == TeamMember.team_id)
        .join(User, User.id == TeamMember.user_id)
        .where(DealShare.deal_id == deal_id)).all()
    owner_subject = conn.execute(select(User.subject).where(User.id == owner_id)).scalar_one()
    best: dict[int, PersonRecord] = {owner_id: PersonRecord(owner_id, owner_subject, "owner")}
    for r in rows:
        access = "editor" if r.role in EDIT_ROLES else "viewer"
        held = best.get(r.user_id)
        if held is None or (held.access == "viewer" and access == "editor"):
            best[r.user_id] = PersonRecord(r.user_id, r.subject, access)
    order = {"owner": 0, "editor": 1, "viewer": 2}
    return sorted(best.values(), key=lambda p: (order[p.access], p.user_id))


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------
def list_comments(subject: str, deal_id: uuid.UUID) -> list[CommentRecord]:
    """The deal's comments, oldest first, for anyone who can read it."""
    with connect() as conn:
        _allowed(conn.execute(select(Deal.id, _access(subject)).where(_reachable(subject, deal_id))).first(),
                 deal_id, READ)
        rows = conn.execute(select(DealComment.id, DealComment.author_id, DealComment.body, DealComment.mentions,
                                   DealComment.created_at, User.subject)
                            .join(User, User.id == DealComment.author_id)
                            .where(DealComment.deal_id == deal_id)
                            .order_by(DealComment.created_at, DealComment.id)).all()
    return [CommentRecord(id=r.id, author_id=r.author_id, author_subject=r.subject, body=r.body,
                          mentions=list(r.mentions or []), created_at=r.created_at) for r in rows]


def add_comment(subject: str, deal_id: uuid.UUID, body, mentions: Sequence[int] = (),
                *, now: Optional[datetime] = None) -> CommentRecord:
    """A comment by the caller on a deal they can read. Each mention must be someone else who can read it;
    each gets a notice for the mail sender."""
    body = clean_comment(body)
    wanted = list(dict.fromkeys(int(m) for m in mentions))
    if len(wanted) > MAX_MENTIONS:
        raise InvalidTeamInput("mentions")
    now = now or utc_now()
    with transaction() as conn:
        deal = _allowed(conn.execute(select(Deal.id, Deal.owner_id, _access(subject))
                                     .where(_reachable(subject, deal_id)).with_for_update(of=Deal)).first(),
                        deal_id, READ)
        me = _user_id(conn, subject)
        held = conn.execute(select(func.count()).select_from(DealComment)
                            .where(DealComment.deal_id == deal_id)).scalar_one()
        if held >= MAX_COMMENTS_PER_DEAL:
            raise TeamFull("comments")
        readers = {p.user_id for p in _people(conn, deal_id, deal.owner_id)}
        if any(m == me or m not in readers for m in wanted):
            raise InvalidTeamInput("mentions")
        comment_id = uuid.uuid4()
        conn.execute(insert(DealComment).values(id=comment_id, deal_id=deal_id, author_id=me, body=body,
                                                mentions=wanted, created_at=now))
        if wanted:
            conn.execute(insert(MentionNotice), [{"comment_id": comment_id, "recipient_id": m, "created_at": now}
                                                 for m in wanted])
        audit.record(conn, me, "commented", deal_id=deal_id, now=now)
    return CommentRecord(id=comment_id, author_id=me, author_subject=subject, body=body, mentions=wanted,
                         created_at=now)


def delete_comment(subject: str, deal_id: uuid.UUID, comment_id: uuid.UUID,
                   *, now: Optional[datetime] = None) -> None:
    """Delete a comment: its writer may, and so may the deal's owner. Its unsent notices go with it."""
    with transaction() as conn:
        deal = _allowed(conn.execute(select(Deal.id, Deal.owner_id, _access(subject))
                                     .where(_reachable(subject, deal_id))).first(), deal_id, READ)
        me = _user_id(conn, subject)
        author = conn.execute(select(DealComment.author_id).where(DealComment.id == comment_id,
                                                                  DealComment.deal_id == deal_id)
                              .with_for_update()).scalar()
        if author is None:
            raise CommentNotFound(str(comment_id))
        if author != me and deal.access != "owner":
            raise NotPermitted("comment")
        conn.execute(delete(DealComment).where(and_(DealComment.id == comment_id, DealComment.deal_id == deal_id)))
        audit.record(conn, me, "comment_deleted", deal_id=deal_id, now=now)


__all__ = ["CommentNotFound", "CommentRecord", "InvalidTeamInput", "InviteInvalid", "InviteRecord", "LastOwner",
           "MemberNotFound", "MemberRecord", "PersonRecord", "TeamFull", "TeamNotFound", "TeamRecord",
           "add_comment", "create_invite", "create_team", "deal_people", "deal_teams", "delete_comment",
           "delete_team", "get_team", "join", "list_comments", "list_invites", "list_teams", "remove_member",
           "my_id", "rename_team", "set_role", "share", "subjects_of", "unshare", "withdraw_invite"]
