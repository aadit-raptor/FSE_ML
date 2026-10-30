"""Teams, sharing a deal with one, and comments on a deal (PLAN.md 7.2).

    GET    /api/teams                                  the caller's teams (and whether names and mail are set up)
    POST   /api/teams                                  make a team; the caller is its owner
    GET    /api/teams/{id}                             the team and its people, for its members
    PATCH  /api/teams/{id}                             rename it (owners)
    DELETE /api/teams/{id}                             delete it (owners)
    PUT    /api/teams/{id}/members/{member}            give someone another role (owners)
    DELETE /api/teams/{id}/members/{member}            remove someone (owners), or leave (anyone, themselves)
    GET    /api/teams/{id}/invites                     live invitation links, without their tokens (owners)
    POST   /api/teams/{id}/invites                     a new invitation link: its token is shown once (owners)
    DELETE /api/teams/{id}/invites/{invite}            withdraw one (owners)
    POST   /api/teams/join                             join by an invitation's token
    GET    /api/deals/{id}/shares                      the teams a deal is shared with
    PUT    /api/deals/{id}/shares                      share the caller's deal with one of their teams
    DELETE /api/deals/{id}/shares/{team}               stop sharing it
    GET    /api/deals/{id}/people                      who can read the deal (who may be mentioned)
    GET    /api/deals/{id}/comments                    its comments
    POST   /api/deals/{id}/comments                    comment, mentioning people by member id
    DELETE /api/deals/{id}/comments/{comment}          delete a comment (its writer, or the deal's owner)

A team or deal the caller can't reach is a 404, as a missing one is; one they
are in but may not change this way is a 403 ``{"code"}``. People are member ids
(user ids) here, never their sign-in ids; names come from the sign-in provider
per request (teams/directory.py) and are never stored. A comment's words are
deal content: never logged.
"""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Iterable

from fastapi import APIRouter, Depends, HTTPException, Response

from api.auth import AuthUser, require_user
from api.schemas import (
    CommentCreate, CommentList, CommentOut, DealPeople, DealShares, InviteCreate, InviteList, InviteOut,
    JoinRequest, RoleChange, ShareRequest, TeamDetail, TeamList, TeamName, TeamSummary,
)
from db import deals
from db import teams as store
from db.engine import is_configured as database_configured
from teams import directory, mail

router = APIRouter(tags=["teams"])


def _iso(value) -> str:
    return value.isoformat().replace("+00:00", "Z")


@contextmanager
def team_errors():
    if not database_configured():
        raise HTTPException(503, {"code": "no_database"})
    try:
        yield
    except (store.TeamNotFound, deals.DealNotFound):
        raise HTTPException(404, {"code": "not_found"}) from None
    except (store.MemberNotFound, store.CommentNotFound):
        raise HTTPException(404, {"code": "not_found"}) from None
    except store.InviteInvalid:
        raise HTTPException(404, {"code": "invite_invalid"}) from None
    except deals.NotPermitted as exc:
        raise HTTPException(403, {"code": f"not_{exc.args[0] if exc.args else 'permitted'}"}) from None
    except store.LastOwner:
        raise HTTPException(409, {"code": "last_owner"}) from None
    except store.TeamFull as exc:
        raise HTTPException(409, {"code": f"full_{exc.code}"}) from None
    except store.InvalidTeamInput as exc:
        raise HTTPException(422, {"code": f"invalid_{exc.args[0]}"}) from None
    except deals.NoAccount:
        raise HTTPException(409, {"code": "no_account"}) from None


def _names(subjects: Iterable[str]) -> dict[str, str]:
    return directory.get_directory().names(subjects)


def _team(t: store.TeamRecord) -> dict:
    return {"id": str(t.id), "name": t.name, "role": t.role, "members": t.members, "created_at": _iso(t.created_at)}


def _invite(i: store.InviteRecord) -> dict:
    return {"id": str(i.id), "role": i.role, "uses_left": i.uses_left, "expires_at": _iso(i.expires_at),
            "token": i.token}


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------
@router.get("/teams", response_model=TeamList)
def list_teams(user: AuthUser = Depends(require_user)):
    with team_errors():
        found = store.list_teams(user.subject)
    sender = mail.get_mail()
    return {"teams": [_team(t) for t in found], "directory": directory.configured_kind(),
            "mail": sender.kind if sender else "none"}


@router.post("/teams", response_model=TeamSummary, status_code=201)
def create_team(req: TeamName, user: AuthUser = Depends(require_user)):
    with team_errors():
        return _team(store.create_team(user.subject, req.name))


@router.post("/teams/join", response_model=TeamSummary)
def join_team(req: JoinRequest, user: AuthUser = Depends(require_user)):
    with team_errors():
        return _team(store.join(user.subject, req.token))


@router.get("/teams/{team_id}", response_model=TeamDetail)
def get_team(team_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with team_errors():
        team, members = store.get_team(user.subject, team_id)
    names = _names(m.subject for m in members)
    people = [{"member_id": m.user_id, "name": names.get(m.subject), "you": m.subject == user.subject,
               "role": m.role, "joined_at": _iso(m.joined_at)} for m in members]
    return {**_team(team), "people": people}


@router.patch("/teams/{team_id}", status_code=204)
def rename_team(team_id: uuid.UUID, req: TeamName, user: AuthUser = Depends(require_user)):
    with team_errors():
        store.rename_team(user.subject, team_id, req.name)
    return Response(status_code=204)


@router.delete("/teams/{team_id}", status_code=204)
def delete_team(team_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with team_errors():
        store.delete_team(user.subject, team_id)
    return Response(status_code=204)


@router.put("/teams/{team_id}/members/{member_id}", status_code=204)
def set_role(team_id: uuid.UUID, member_id: int, req: RoleChange, user: AuthUser = Depends(require_user)):
    with team_errors():
        store.set_role(user.subject, team_id, member_id, req.role)
    return Response(status_code=204)


@router.delete("/teams/{team_id}/members/{member_id}", status_code=204)
def remove_member(team_id: uuid.UUID, member_id: int, user: AuthUser = Depends(require_user)):
    with team_errors():
        store.remove_member(user.subject, team_id, member_id)
    return Response(status_code=204)


@router.get("/teams/{team_id}/invites", response_model=InviteList)
def list_invites(team_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with team_errors():
        return {"invites": [_invite(i) for i in store.list_invites(user.subject, team_id)]}


@router.post("/teams/{team_id}/invites", response_model=InviteOut, status_code=201)
def create_invite(team_id: uuid.UUID, req: InviteCreate, user: AuthUser = Depends(require_user)):
    with team_errors():
        return _invite(store.create_invite(user.subject, team_id, req.role, uses=req.uses))


@router.delete("/teams/{team_id}/invites/{invite_id}", status_code=204)
def withdraw_invite(team_id: uuid.UUID, invite_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with team_errors():
        store.withdraw_invite(user.subject, team_id, invite_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Sharing a deal
# ---------------------------------------------------------------------------
@router.get("/deals/{deal_id}/shares", response_model=DealShares)
def deal_shares(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with team_errors():
        shared = store.deal_teams(user.subject, deal_id)
        mine = {t.id: t.name for t in store.list_teams(user.subject)}
    return {"teams": [{"team_id": str(t), "name": mine.get(t)} for t in shared]}


@router.put("/deals/{deal_id}/shares", response_model=DealShares)
def share_deal(deal_id: uuid.UUID, req: ShareRequest, user: AuthUser = Depends(require_user)):
    with team_errors():
        store.share(user.subject, deal_id, req.team_id)
    return deal_shares(deal_id, user)


@router.delete("/deals/{deal_id}/shares/{team_id}", response_model=DealShares)
def unshare_deal(deal_id: uuid.UUID, team_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with team_errors():
        store.unshare(user.subject, deal_id, team_id)
    return deal_shares(deal_id, user)


@router.get("/deals/{deal_id}/people", response_model=DealPeople)
def deal_people(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with team_errors():
        people = store.deal_people(user.subject, deal_id)
    names = _names(p.subject for p in people)
    return {"people": [{"member_id": p.user_id, "name": names.get(p.subject), "you": p.subject == user.subject,
                        "access": p.access} for p in people]}


# ---------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------
def _comments(subject: str, deal_id: uuid.UUID, found: list[store.CommentRecord]) -> list[dict]:
    owner = deals.access_of(subject, deal_id) == "owner"
    mentioned = {m for c in found for m in c.mentions}
    subjects = store.subjects_of(mentioned) | {c.author_id: c.author_subject for c in found}
    names = _names(subjects.values())

    def person(user_id: int) -> dict:
        who = subjects.get(user_id, "")
        return {"member_id": user_id, "name": names.get(who), "you": who == subject}
    return [{"id": str(c.id), "author": person(c.author_id), "body": c.body,
             "mentions": [person(m) for m in c.mentions], "created_at": _iso(c.created_at),
             "can_delete": owner or c.author_subject == subject} for c in found]


@router.get("/deals/{deal_id}/comments", response_model=CommentList)
def list_comments(deal_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with team_errors():
        found = store.list_comments(user.subject, deal_id)
        return {"comments": _comments(user.subject, deal_id, found)}


@router.post("/deals/{deal_id}/comments", response_model=CommentOut, status_code=201)
def add_comment(deal_id: uuid.UUID, req: CommentCreate, user: AuthUser = Depends(require_user)):
    with team_errors():
        made = store.add_comment(user.subject, deal_id, req.body, req.mentions)
        out = _comments(user.subject, deal_id, [made])[0]
    if made.mentions:
        _queue_mentions()
    return out


@router.delete("/deals/{deal_id}/comments/{comment_id}", status_code=204)
def delete_comment(deal_id: uuid.UUID, comment_id: uuid.UUID, user: AuthUser = Depends(require_user)):
    with team_errors():
        store.delete_comment(user.subject, deal_id, comment_id)
    return Response(status_code=204)


MENTIONS_OWNER = "system:mentions"
MENTIONS_KIND = "mentions.send"


def _queue_mentions() -> None:
    """Ask the runner to send what is due. One queued run is enough for any number of comments; a full queue
    leaves the mentions to the next comment or the nightly run."""
    from jobs import config
    from jobs.queue import QueueFull

    try:
        # A run already going may have read the pending mentions before this one: queue another behind it
        config.get_queue().submit(MENTIONS_OWNER, MENTIONS_KIND, {}, max_active_per_owner=2, max_queued=200)
    except QueueFull:
        return
    config.wake_runner()
