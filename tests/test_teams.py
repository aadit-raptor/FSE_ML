"""Teams, sharing, comments and mention emails (PLAN.md 7.2).

Real rows in a real database (``fresh_db``) through the real endpoints. The
permission matrix is the heart of it: every deal endpoint for the deal's owner,
a team owner, an editor, a viewer, someone removed, and a stranger. Emails go to
an in-memory outbox, names and addresses come from the development directory.
"""
from __future__ import annotations

import json
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from api.auth import AuthUser, require_user
from api.main import app
from db import engine as db_engine
from db import teams as store
from db.models import AuditEvent, DealComment, MentionNotice, TeamInvite, utc_now
from teams import directory, mail, notices

client = TestClient(app)

INPUTS = {"ebitda": 240.0, "entry_mult": 9.0, "exit_mult": 10.5, "hold": 6, "growth": 7.5,
          "currency": "USD", "unit": "millions"}
ACTUALS = {"currency": "USD", "unit": "millions", "years": [{"ebitda": 250.0}]}
PROFILES = {
    "dev:olivia": {"country": "GB", "preferred_currency": "GBP", "locale": "en-GB", "time_zone": "Europe/London"},
    "dev:tara": {"country": "IN", "preferred_currency": "INR", "locale": "en-IN", "time_zone": "Asia/Kolkata"},
    "dev:eddie": {"country": "DE", "preferred_currency": "EUR", "locale": "de-DE", "time_zone": "Europe/Berlin"},
    "dev:vera": {"country": "US", "preferred_currency": "USD", "locale": "en-US", "time_zone": "America/New_York"},
    "dev:stan": {"country": "FR", "preferred_currency": "EUR", "locale": "fr-FR", "time_zone": "Europe/Paris"},
}
# olivia owns the deal and the team; tara is a second team owner; eddie edits, vera views, stan is outside
OWNER, TEAM_OWNER, EDITOR, VIEWER, STRANGER = "dev:olivia", "dev:tara", "dev:eddie", "dev:vera", "dev:stan"


def as_user(subject: str) -> None:
    app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject, is_dev=True)


@pytest.fixture
def people(fresh_db):  # noqa: ARG001
    """Five finished accounts, the development directory and an in-memory outbox."""
    for subject, profile in PROFILES.items():
        as_user(subject)
        assert client.post("/api/account", json=profile).status_code == 200
    outbox = mail.OutboxMail()
    mail.use_mail(outbox)
    directory.use_directory(directory.DevDirectory())
    yield outbox
    mail.use_mail(None, choose_again=True)
    directory.use_directory(None)


def make_team(name="Deal team") -> str:
    as_user(OWNER)
    made = client.post("/api/teams", json={"name": name})
    assert made.status_code == 201, made.text
    return made.json()["id"]


def invite(team: str, role: str, uses: int = 10) -> str:
    as_user(OWNER)
    made = client.post(f"/api/teams/{team}/invites", json={"role": role, "uses": uses})
    assert made.status_code == 201, made.text
    return made.json()["token"]


def join(subject: str, token: str):
    as_user(subject)
    return client.post("/api/teams/join", json={"token": token})


def member_id(team: str, subject: str) -> int:
    as_user(OWNER)
    name = subject.split(":", 1)[1]
    return next(p["member_id"] for p in client.get(f"/api/teams/{team}").json()["people"] if p["name"] == name)


@pytest.fixture
def shared(people):
    """A deal of olivia's shared with a team in which tara is an owner, eddie an editor and vera a viewer."""
    team = make_team()
    assert join(TEAM_OWNER, invite(team, "editor")).status_code == 200
    assert join(EDITOR, invite(team, "editor")).status_code == 200
    assert join(VIEWER, invite(team, "viewer")).status_code == 200
    as_user(OWNER)
    assert client.put(f"/api/teams/{team}/members/{member_id(team, TEAM_OWNER)}",
                      json={"role": "owner"}).status_code == 204
    deal = client.post("/api/deals", json={"name": "Project Heron", "inputs": INPUTS, "settings": {}}).json()["id"]
    assert client.put(f"/api/deals/{deal}/shares", json={"team_id": team}).status_code == 200
    return {"team": team, "deal": deal, "outbox": people}


# ---------------------------------------------------------------------------
# Who may do what to a shared deal
# ---------------------------------------------------------------------------
def _edit_inputs(deal: str, growth: float):
    return client.put(f"/api/deals/{deal}/draft", json={"inputs": {**INPUTS, "growth": growth}, "settings": {}})


@pytest.mark.parametrize("subject,access", [(OWNER, "owner"), (TEAM_OWNER, "editor"), (EDITOR, "editor"),
                                            (VIEWER, "viewer")])
def test_everyone_in_the_team_reads_the_shared_deal_with_their_access(shared, subject, access):
    as_user(subject)
    got = client.get(f"/api/deals/{shared['deal']}")
    assert got.status_code == 200 and got.json()["access"] == access
    assert got.json()["inputs"]["growth"] == 7.5
    listed = {d["id"]: d["access"] for d in client.get("/api/deals").json()["deals"]}
    assert listed[shared["deal"]] == access
    for path in ("versions", "history", "comments", "people", "shares", "actuals"):
        assert client.get(f"/api/deals/{shared['deal']}/{path}").status_code == 200, path


@pytest.mark.parametrize("subject", [TEAM_OWNER, EDITOR])
def test_a_team_owner_or_editor_edits_the_deal_and_the_owner_sees_it(shared, subject):
    deal = shared["deal"]
    as_user(subject)
    assert _edit_inputs(deal, 9.0).status_code == 200
    assert client.post(f"/api/deals/{deal}/versions", json={"label": "Their case"}).status_code == 201
    assert client.patch(f"/api/deals/{deal}", json={"name": "Project Heron II"}).status_code == 200
    assert client.put(f"/api/deals/{deal}/actuals", json=ACTUALS).status_code == 200
    as_user(OWNER)
    mine = client.get(f"/api/deals/{deal}").json()
    assert mine["inputs"]["growth"] == 9.0 and mine["name"] == "Project Heron II" and mine["access"] == "owner"


@pytest.mark.parametrize("subject", [TEAM_OWNER, EDITOR, VIEWER])
def test_only_the_deals_owner_archives_deletes_shares_or_opts_it_into_validation(shared, subject):
    deal, team = shared["deal"], shared["team"]
    as_user(subject)
    assert client.patch(f"/api/deals/{deal}", json={"archived": True}).status_code == 403
    assert client.delete(f"/api/deals/{deal}").status_code == 403
    assert client.put(f"/api/deals/{deal}/validation", json={"opt_in": True}).status_code == 403
    assert client.delete(f"/api/deals/{deal}/shares/{team}").status_code == 403
    as_user(OWNER)
    assert client.get(f"/api/deals/{deal}").json()["archived"] is False
    assert client.get(f"/api/deals/{deal}/shares").json()["teams"][0]["team_id"] == team


def test_a_viewer_reads_but_changes_nothing(shared):
    deal = shared["deal"]
    as_user(VIEWER)
    assert _edit_inputs(deal, 12.0).status_code == 403
    assert client.post(f"/api/deals/{deal}/versions", json={"label": "Mine"}).status_code == 403
    assert client.post(f"/api/deals/{deal}/versions/1/restore").status_code == 403
    assert client.patch(f"/api/deals/{deal}", json={"name": "Renamed"}).status_code == 403
    assert client.put(f"/api/deals/{deal}/actuals", json=ACTUALS).status_code == 403
    # A copy of their own is theirs to change
    copy = client.post(f"/api/deals/{deal}/duplicate", json={})
    assert copy.status_code == 201 and copy.json()["access"] == "owner"
    as_user(OWNER)
    held = client.get(f"/api/deals/{deal}").json()
    assert held["inputs"]["growth"] == 7.5 and held["name"] == "Project Heron" and held["latest_version"] == 1


def test_a_stranger_finds_nothing_at_all(shared):
    deal, team = shared["deal"], shared["team"]
    as_user(STRANGER)
    for path in ("", "/versions", "/history", "/comments", "/people", "/shares", "/actuals", "/versions/1"):
        assert client.get(f"/api/deals/{deal}{path}").status_code == 404, path
    assert _edit_inputs(deal, 1.0).status_code == 404
    assert client.post(f"/api/deals/{deal}/comments", json={"body": "hello"}).status_code == 404
    assert client.get(f"/api/teams/{team}").status_code == 404
    assert shared["deal"] not in {d["id"] for d in client.get("/api/deals").json()["deals"]}


def test_removing_a_member_or_unsharing_takes_the_deal_away(shared):
    deal, team = shared["deal"], shared["team"]
    as_user(OWNER)
    assert client.delete(f"/api/teams/{team}/members/{member_id(team, EDITOR)}").status_code == 204
    as_user(EDITOR)
    assert client.get(f"/api/deals/{deal}").status_code == 404
    assert _edit_inputs(deal, 3.0).status_code == 404
    as_user(OWNER)
    assert client.delete(f"/api/deals/{deal}/shares/{team}").status_code == 200
    as_user(VIEWER)
    assert client.get(f"/api/deals/{deal}").status_code == 404


def test_a_deals_owner_leaving_the_team_takes_their_deals_with_them(shared):
    deal, team = shared["deal"], shared["team"]
    as_user(OWNER)
    assert client.delete(f"/api/teams/{team}/members/{member_id(team, OWNER)}").status_code == 204
    as_user(VIEWER)
    assert client.get(f"/api/deals/{deal}").status_code == 404
    as_user(OWNER)
    assert client.get(f"/api/teams/{team}").status_code == 404
    assert client.get(f"/api/deals/{deal}").json()["access"] == "owner"


def test_a_deal_is_shared_only_by_its_owner_and_only_with_their_own_team(shared, people):  # noqa: ARG001
    as_user(STRANGER)
    theirs = client.post("/api/teams", json={"name": "Other"}).json()["id"]
    as_user(OWNER)
    assert client.put(f"/api/deals/{shared['deal']}/shares", json={"team_id": theirs}).status_code == 404


# ---------------------------------------------------------------------------
# Teams: roles, owners, invitations
# ---------------------------------------------------------------------------
def test_only_owners_run_the_team(shared):
    team = shared["team"]
    vera = member_id(team, VIEWER)
    for subject in (EDITOR, VIEWER):
        as_user(subject)
        assert client.patch(f"/api/teams/{team}", json={"name": "Mine now"}).status_code == 403
        assert client.post(f"/api/teams/{team}/invites", json={"role": "editor"}).status_code == 403
        assert client.put(f"/api/teams/{team}/members/{vera}", json={"role": "editor"}).status_code == 403
        assert client.get(f"/api/teams/{team}/invites").status_code == 403
        assert client.delete(f"/api/teams/{team}").status_code == 403
    # A second owner can, and every member sees the team and everyone's role
    as_user(TEAM_OWNER)
    assert client.put(f"/api/teams/{team}/members/{vera}", json={"role": "editor"}).status_code == 204
    as_user(VIEWER)
    detail = client.get(f"/api/teams/{team}").json()
    roles = {p["name"]: p["role"] for p in detail["people"]}
    assert roles == {"olivia": "owner", "tara": "owner", "eddie": "editor", "vera": "editor"}
    assert detail["role"] == "editor" and [p["you"] for p in detail["people"]].count(True) == 1
    assert _edit_inputs(shared["deal"], 4.0).status_code == 200


def test_a_team_always_keeps_an_owner(people):  # noqa: ARG001
    team = make_team()
    me = member_id(team, OWNER)
    as_user(OWNER)
    assert client.delete(f"/api/teams/{team}/members/{me}").status_code == 409
    assert client.put(f"/api/teams/{team}/members/{me}", json={"role": "viewer"}).status_code == 409
    assert join(EDITOR, invite(team, "editor")).status_code == 200
    as_user(OWNER)
    assert client.put(f"/api/teams/{team}/members/{member_id(team, EDITOR)}",
                      json={"role": "owner"}).status_code == 204
    # With a second owner, the first may step down
    assert client.put(f"/api/teams/{team}/members/{me}", json={"role": "viewer"}).status_code == 204


def test_an_invitation_lets_in_as_many_as_it_says_for_as_long_as_it_says(people):  # noqa: ARG001
    team = make_team()
    token = invite(team, "viewer", uses=1)
    assert join(EDITOR, token).json()["role"] == "viewer"
    # Joining again changes nothing and spends nothing; a second person finds it used up
    assert join(EDITOR, token).status_code == 200
    assert join(VIEWER, token).status_code == 404
    late = invite(team, "editor")
    with db_engine.transaction() as conn:
        conn.execute(TeamInvite.__table__.update().values(expires_at=utc_now() - timedelta(minutes=1)))
    assert join(VIEWER, late).status_code == 404
    assert join(VIEWER, "inv_" + "x" * 40).status_code == 404


def test_an_invitation_is_kept_only_as_its_hash_and_can_be_withdrawn(people):  # noqa: ARG001
    team = make_team()
    token = invite(team, "editor")
    with db_engine.connect() as conn:
        stored = conn.execute(select(TeamInvite.token_hash)).scalars().all()
    assert stored == [store.token_hash(token)] and token not in stored
    as_user(OWNER)
    listed = client.get(f"/api/teams/{team}/invites").json()["invites"]
    assert len(listed) == 1 and listed[0]["token"] is None
    assert client.delete(f"/api/teams/{team}/invites/{listed[0]['id']}").status_code == 204
    assert join(EDITOR, token).status_code == 404


def test_deleting_a_team_unshares_its_deals_and_keeps_them(shared):
    as_user(OWNER)
    assert client.delete(f"/api/teams/{shared['team']}").status_code == 204
    assert client.get(f"/api/deals/{shared['deal']}").status_code == 200
    as_user(EDITOR)
    assert client.get(f"/api/deals/{shared['deal']}").status_code == 404
    assert client.get("/api/teams").json()["teams"] == []


# ---------------------------------------------------------------------------
# Comments and activity
# ---------------------------------------------------------------------------
def test_comments_name_their_writer_and_mention_only_people_who_can_read_the_deal(shared):
    deal, team = shared["deal"], shared["team"]
    eddie, stan = member_id(team, EDITOR), None
    as_user(STRANGER)
    client.post("/api/teams", json={"name": "Elsewhere"})
    as_user(OWNER)
    with db_engine.connect() as conn:
        stan = store.my_id(STRANGER)
    me = member_id(team, OWNER)
    assert client.post(f"/api/deals/{deal}/comments", json={"body": "x", "mentions": [stan]}).status_code == 422
    assert client.post(f"/api/deals/{deal}/comments", json={"body": "x", "mentions": [me]}).status_code == 422
    made = client.post(f"/api/deals/{deal}/comments", json={"body": "Check the exit multiple", "mentions": [eddie]})
    assert made.status_code == 201
    as_user(VIEWER)
    listed = client.get(f"/api/deals/{deal}/comments").json()["comments"]
    assert [(c["author"]["name"], c["body"], [m["name"] for m in c["mentions"]], c["can_delete"])
            for c in listed] == [("olivia", "Check the exit multiple", ["eddie"], False)]
    assert client.post(f"/api/deals/{deal}/comments", json={"body": "Agreed"}).status_code == 201


def test_a_comment_is_deleted_by_its_writer_or_the_deals_owner_only(shared):
    deal = shared["deal"]
    as_user(VIEWER)
    vera = client.post(f"/api/deals/{deal}/comments", json={"body": "Mine"}).json()["id"]
    as_user(EDITOR)
    eddie = client.post(f"/api/deals/{deal}/comments", json={"body": "Also mine"}).json()["id"]
    assert client.delete(f"/api/deals/{deal}/comments/{vera}").status_code == 403
    assert client.delete(f"/api/deals/{deal}/comments/{eddie}").status_code == 204
    as_user(OWNER)
    assert client.delete(f"/api/deals/{deal}/comments/{vera}").status_code == 204
    assert client.get(f"/api/deals/{deal}/comments").json()["comments"] == []


def test_the_deals_history_is_the_teams_and_says_who_did_what_without_the_words(shared):
    deal = shared["deal"]
    as_user(EDITOR)
    assert _edit_inputs(deal, 8.0).status_code == 200
    assert client.post(f"/api/deals/{deal}/comments", json={"body": "A secret plan"}).status_code == 201
    as_user(VIEWER)
    entries = client.get(f"/api/deals/{deal}/history").json()["entries"]
    seen = [(e["action"], e["by_name"], e["by_you"]) for e in entries]
    assert seen[:3] == [("commented", "eddie", False), ("edited", "eddie", False), ("shared", "olivia", False)]
    assert entries[2]["team"] == shared["team"]
    with db_engine.connect() as conn:
        details = conn.execute(select(AuditEvent.detail)).scalars().all()
    assert "A secret plan" not in json.dumps(details) and "Project Heron" not in json.dumps(details)


def test_sharing_twice_or_setting_the_same_role_writes_no_history(shared):
    deal, team = shared["deal"], shared["team"]
    with db_engine.connect() as conn:
        before = conn.execute(select(func.count()).select_from(AuditEvent)).scalar_one()
    as_user(OWNER)
    assert client.put(f"/api/deals/{deal}/shares", json={"team_id": team}).status_code == 200
    assert client.put(f"/api/teams/{team}/members/{member_id(team, VIEWER)}", json={"role": "viewer"}).status_code == 204
    with db_engine.connect() as conn:
        assert conn.execute(select(func.count()).select_from(AuditEvent)).scalar_one() == before


# ---------------------------------------------------------------------------
# Mention emails
# ---------------------------------------------------------------------------
def _mention(deal: str, author: str, *people: int, body="Have a look"):
    as_user(author)
    made = client.post(f"/api/deals/{deal}/comments", json={"body": body, "mentions": list(people)})
    assert made.status_code == 201, made.text


def test_a_mention_queues_the_mail_job_and_the_job_emails_the_person(shared, job_queue):
    tara = member_id(shared["team"], TEAM_OWNER)
    _mention(shared["deal"], OWNER, tara, body="Confidential: bid 9.5x")
    queued = [j for j in job_queue.list("system:mentions")]
    assert [j.kind for j in queued] == ["mentions.send"]
    from jobs.runner import Runner
    Runner(lambda: job_queue).run_next()
    sent = shared["outbox"].sent
    assert [e.to for e in sent] == ["tara@example.test"]
    text = sent[0].text
    assert "olivia mentioned you" in text and f"/deal/team?deal={shared['deal']}" in text
    # Never what was said, and never the deal's name
    assert "Confidential" not in text and "9.5" not in text and "Project Heron" not in text
    with db_engine.connect() as conn:
        assert conn.execute(select(MentionNotice.outcome)).scalars().all() == ["sent"]


def test_the_email_is_in_the_recipients_time_zone(shared):
    tara = member_id(shared["team"], TEAM_OWNER)
    _mention(shared["deal"], OWNER, tara)
    with db_engine.connect() as conn:
        written = conn.execute(select(DealComment.created_at)).scalar_one()
    notices.send_pending()
    local = written.astimezone(__import__("zoneinfo").ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M")
    assert f"{local} (Asia/Kolkata)" in shared["outbox"].sent[0].text


def test_mentions_are_batched_one_email_per_person_per_hour(shared):
    team, deal, outbox = shared["team"], shared["deal"], shared["outbox"]
    eddie, vera = member_id(team, EDITOR), member_id(team, VIEWER)
    _mention(deal, OWNER, eddie)
    _mention(deal, TEAM_OWNER, eddie, vera)
    first = notices.send_pending()
    assert first["mentions_emails_sent"] == 2
    to_eddie = [e for e in outbox.sent if e.to == "eddie@example.test"]
    assert len(to_eddie) == 1 and "olivia mentioned you" in to_eddie[0].text and "tara mentioned you" in to_eddie[0].text
    assert to_eddie[0].subject == "You were mentioned in 2 comments on Variater"
    _mention(deal, OWNER, eddie)
    assert notices.send_pending()["mentions_waiting"] == 1 and len(outbox.sent) == 2
    later = notices.send_pending(utc_now() + timedelta(hours=1, minutes=1))
    assert later["mentions_emails_sent"] == 1 and len(outbox.sent) == 3


def test_mail_stays_inside_the_free_daily_sends(shared, monkeypatch):
    team, deal, outbox = shared["team"], shared["deal"], shared["outbox"]
    monkeypatch.setattr(notices, "DAILY_EMAILS", 1)
    _mention(deal, OWNER, member_id(team, EDITOR), member_id(team, VIEWER))
    summary = notices.send_pending()
    assert summary["mentions_emails_sent"] == 1 and summary["mentions_waiting"] == 1 and len(outbox.sent) == 1


def test_without_mail_mentions_wait_and_then_expire_rather_than_pile_up(shared):
    deal = shared["deal"]
    _mention(deal, OWNER, member_id(shared["team"], EDITOR))
    mail.use_mail(None)
    assert notices.send_pending()["mentions_waiting"] == 1
    with db_engine.connect() as conn:
        assert conn.execute(select(MentionNotice.sent_at)).scalar_one() is None
    mail.use_mail(shared["outbox"])
    late = notices.send_pending(utc_now() + timedelta(days=notices.EXPIRE_DAYS, minutes=1))
    assert late["mentions_expired"] == 1 and shared["outbox"].sent == []


@pytest.mark.parametrize("reason,outcome,waiting", [("refused", None, 1), ("unreachable", None, 1),
                                                     ("rejected", "failed", 0)])
def test_a_refused_email_waits_and_a_rejected_one_is_given_up(shared, reason, outcome, waiting):
    _mention(shared["deal"], OWNER, member_id(shared["team"], EDITOR))
    shared["outbox"].fail = mail.MailError(reason, 500)
    assert notices.send_pending()["mentions_waiting"] == waiting
    with db_engine.connect() as conn:
        assert conn.execute(select(MentionNotice.outcome)).scalar_one() == outcome


def test_someone_with_no_address_is_noted_and_a_directory_outage_waits(shared):
    _mention(shared["deal"], OWNER, member_id(shared["team"], EDITOR))

    class Down(directory.DevDirectory):
        def email(self, subject):
            raise directory.DirectoryUnavailable()

    class Nobody(directory.DevDirectory):
        def email(self, subject):
            return None
    assert notices.send_pending(directory=Down())["mentions_waiting"] == 1
    assert notices.send_pending(directory=Nobody())["mentions_no_address"] == 1
    assert shared["outbox"].sent == []


def test_the_email_reads_in_the_recipients_language(tmp_path, monkeypatch):
    english = {"email": {"subjectOne": "You were mentioned", "subjectMany": "{count} mentions", "greeting": "Hello,",
                         "mentionedYou": "{name} mentioned you at {time}.", "openDeal": "Open: {link}",
                         "someone": "A team member", "why": "Because."}}
    german = {"email": {"subjectOne": "Sie wurden erwähnt", "mentionedYou": "{name} hat Sie um {time} erwähnt."}}
    (tmp_path / "en.json").write_text(json.dumps(english), encoding="utf-8")
    (tmp_path / "de.json").write_text(json.dumps(german), encoding="utf-8")
    monkeypatch.setattr(notices, "MESSAGES", str(tmp_path))
    from ml import memo
    monkeypatch.setattr(memo, "MESSAGES", tmp_path)
    when = utc_now().replace(hour=12, minute=0, second=0, microsecond=0)
    subject, text = notices.compose(locale="de-DE", zone="Europe/Berlin", mentions=[(None, when, "d1")])
    assert subject == "Sie wurden erwähnt"
    assert "A team member hat Sie um" in text and "(Europe/Berlin)" in text and "Open: " in text
    assert notices.compose(locale="xx-YY", zone=None, mentions=[("Ann", when, "d1")])[0] == "You were mentioned"


# ---------------------------------------------------------------------------
# The directory and the mail service, against their recorded shapes
# ---------------------------------------------------------------------------
CLERK_USERS = [
    {"id": "user_a", "first_name": "Asha", "last_name": "Rao", "primary_email_address_id": "e1",
     "email_addresses": [{"id": "e0", "email_address": "old@example.com", "verification": {"status": "verified"}},
                         {"id": "e1", "email_address": "asha@example.com", "verification": {"status": "verified"}}]},
    {"id": "user_b", "first_name": None, "last_name": None, "username": "bo", "primary_email_address_id": "e2",
     "email_addresses": [{"id": "e2", "email_address": "bo@example.com",
                          "verification": {"status": "unverified"}}]},
]


def test_clerk_names_come_in_one_call_and_are_kept_a_while():
    calls = []

    def get(url, params, headers):
        calls.append((url, params, headers))
        wanted = {v for k, v in params if k == "user_id"}
        return 200, [u for u in CLERK_USERS if u["id"] in wanted]
    clerk = directory.ClerkDirectory("sk_test_secret", get=get)
    assert clerk.names(["user_a", "user_b", "dev:x"]) == {"user_a": "Asha Rao", "user_b": "bo"}
    assert clerk.names(["user_a"]) == {"user_a": "Asha Rao"} and len(calls) == 1
    url, params, headers = calls[0]
    assert url == "https://api.clerk.com/v1/users" and headers["Authorization"] == "Bearer sk_test_secret"
    assert "sk_test_secret" not in url and all("secret" not in v for _, v in params)


def test_clerk_gives_only_a_verified_primary_address_and_an_outage_is_said():
    clerk = directory.ClerkDirectory("sk", get=lambda u, p, h: (200, [x for x in CLERK_USERS
                                                                    if ("user_id", x["id"]) in p]))
    assert clerk.email("user_a") == "asha@example.com"
    assert clerk.email("user_b") is None
    down = directory.ClerkDirectory("sk", get=lambda u, p, h: (503, None))
    assert down.names(["user_a"]) == {}
    with pytest.raises(directory.DirectoryUnavailable):
        down.email("user_a")


def test_resend_gets_the_email_with_its_key_in_a_header_and_an_idempotency_key():
    posted = []
    resend = mail.ResendMail("re_secret", "Variater <notify@send.variater.com>",
                             post=lambda url, body, headers: posted.append((url, body, headers)) or 200)
    resend.send(mail.Email(to="a@example.com", subject="S", text="T", key="k1"))
    url, body, headers = posted[0]
    assert url == "https://api.resend.com/emails" and body["to"] == ["a@example.com"]
    assert headers == {"Authorization": "Bearer re_secret", "Idempotency-Key": "k1"}
    for status, reason in ((429, "refused"), (503, "refused"), (422, "rejected")):
        with pytest.raises(mail.MailError) as caught:
            mail.ResendMail("re", post=lambda *a, s=status: s).send(mail.Email("a@b.c", "S", "T", "k"))
        assert caught.value.reason == reason and "a@b.c" not in str(caught.value)


def test_an_owner_renames_the_team_and_its_members_see_the_new_name(shared):
    as_user(TEAM_OWNER)
    assert client.patch(f"/api/teams/{shared['team']}", json={"name": "  Heron   desk "}).status_code == 204
    assert client.patch(f"/api/teams/{shared['team']}", json={"name": "Bad\u0007name"}).status_code == 422
    as_user(VIEWER)
    assert [t["name"] for t in client.get("/api/teams").json()["teams"]] == ["Heron desk"]
    as_user(OWNER)
    shares = client.get(f"/api/deals/{shared['deal']}/shares").json()["teams"]
    assert shares == [{"team_id": shared["team"], "name": "Heron desk"}]


def test_the_mail_sender_comes_from_the_environment_and_is_never_a_folder_in_production(tmp_path, monkeypatch):
    monkeypatch.delenv("RESEND_API_KEY", raising=False)
    monkeypatch.setenv("FSE_MAIL_DIR", str(tmp_path))
    monkeypatch.setenv("FSE_ENV", "local")
    try:
        mail.use_mail(None, choose_again=True)
        folder = mail.get_mail()
        assert folder.kind == "folder"
        folder.send(mail.Email(to="a@example.test", subject="S", text="T", key="k"))
        assert json.loads((tmp_path / "k.json").read_text())["to"] == "a@example.test"
        monkeypatch.setenv("FSE_ENV", "production")
        mail.use_mail(None, choose_again=True)
        assert mail.get_mail() is None
        monkeypatch.setenv("RESEND_API_KEY", "re_x")
        mail.use_mail(None, choose_again=True)
        assert mail.get_mail().kind == "resend"
    finally:
        mail.use_mail(None, choose_again=True)


def test_names_come_from_clerk_when_its_secret_is_set_else_from_the_development_sign_in(monkeypatch):
    monkeypatch.setenv("CLERK_SECRET_KEY", "sk_test_x")
    assert directory.configured_kind() == "clerk"
    monkeypatch.delenv("CLERK_SECRET_KEY")
    monkeypatch.setenv("FSE_AUTH_DEV", "1")
    monkeypatch.setenv("FSE_ENV", "local")
    assert directory.configured_kind() == "dev"
    assert directory.DevDirectory().names(["dev:ann", "user_1"]) == {"dev:ann": "ann"}
    assert directory.NoDirectory().email("user_1") is None


def test_an_editor_renames_even_when_the_archive_flag_comes_along_unchanged(shared):
    as_user(EDITOR)
    assert client.patch(f"/api/deals/{shared['deal']}", json={"name": "Kept", "archived": False}).status_code == 200
    assert client.patch(f"/api/deals/{shared['deal']}", json={"archived": True}).status_code == 403


def test_someone_who_lost_access_is_not_emailed(shared):
    team = shared["team"]
    _mention(shared["deal"], OWNER, member_id(team, EDITOR))
    as_user(OWNER)
    assert client.delete(f"/api/teams/{team}/members/{member_id(team, EDITOR)}").status_code == 204
    assert notices.send_pending()["mentions_expired"] == 1
    assert shared["outbox"].sent == []
