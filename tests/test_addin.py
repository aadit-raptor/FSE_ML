"""The Excel add-in's sign-in (PLAN.md R3): a pass made for a user signed in
on the site, which reaches the research endpoints and nothing else, ends
when the pane signs out or its time is up, and is never stored as itself.

Real authentication throughout (the development sign-in, as local runs and
the browser tests use it), on a throwaway database.
"""
from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from api import auth
from api.main import app
from db import addin
from db.engine import connect
from db.models import AddinSession, utc_now
from jobs import scheduled
from tests.test_deals import PROFILE

client = TestClient(app)


@pytest.fixture
def site(fresh_db, monkeypatch, signed_out):  # noqa: ARG001 -- real sign-in on a database
    for name in ("CLERK_ISSUER", "CLERK_PUBLISHABLE_KEY", "RENDER_SERVICE_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FSE_AUTH_DEV", "1")
    monkeypatch.setenv("FSE_ENV", "local")
    auth.reset_addin_cache()
    for who in ("alice", "bob"):
        assert client.post("/api/account", json=PROFILE, headers=bearer(f"dev:{who}")).status_code == 200
    yield
    auth.reset_addin_cache()


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def new_pass(who: str = "alice") -> str:
    made = client.post("/api/addin/sessions", headers=bearer(f"dev:{who}"))
    assert made.status_code == 201, made.text
    body = made.json()
    assert body["token"].startswith("addin:") and body["lifetime_s"] == 12 * 3600
    return body["token"]


def test_a_pass_reaches_coverage_as_the_person_who_made_it(site):  # noqa: ARG001
    token = new_pass("alice")
    made = client.post("/api/research/companies", json={"name": "DMart"}, headers=bearer(token))
    assert made.status_code == 201, made.text
    listed = client.get("/api/research/companies", headers=bearer(token)).json()
    assert [c["name"] for c in listed["companies"]] == ["DMart"]
    # the site's own sign-in sees the same company: it is the same person
    assert [c["name"] for c in client.get("/api/research/companies", headers=bearer("dev:alice")).json()["companies"]] \
        == ["DMart"]
    # and someone else's pass sees none of it
    other = new_pass("bob")
    assert client.get("/api/research/companies", headers=bearer(other)).json()["companies"] == []
    assert client.get("/api/files", headers=bearer(token)).status_code == 200
    assert client.get("/api/addin/session", headers=bearer(token)).status_code == 200


@pytest.mark.parametrize("method, path", [
    ("get", "/api/deals"), ("post", "/api/deal/run"), ("get", "/api/account"), ("get", "/api/account/settings"),
    ("get", "/api/account/history"), ("post", "/api/montecarlo/run"), ("post", "/api/addin/sessions"),
    ("get", "/api/companies/sources"), ("get", "/api/jobs/00000000-0000-4000-8000-000000000000"),
])
def test_a_pass_reaches_nothing_but_research(site, method, path):  # noqa: ARG001
    refused = client.request(method.upper(), path, json={}, headers=bearer(new_pass()))
    assert (refused.status_code, refused.json()["detail"]) == (403, "addin_scope")


def test_a_pass_cannot_make_another(site):  # noqa: ARG001
    token = new_pass()
    assert client.post("/api/addin/sessions", headers=bearer(token)).status_code == 403
    with connect() as conn:
        assert len(conn.execute(select(AddinSession.id)).all()) == 1


def test_signing_out_ends_the_pass_at_once(site):  # noqa: ARG001
    token = new_pass()
    assert client.get("/api/research/companies", headers=bearer(token)).status_code == 200  # cached now
    assert client.delete("/api/addin/session", headers=bearer(token)).status_code == 204
    gone = client.get("/api/research/companies", headers=bearer(token))
    assert gone.status_code == 401 and gone.headers["WWW-Authenticate"] == "Bearer"
    # the site's own sign-in has no add-in pass to end
    assert client.delete("/api/addin/session", headers=bearer("dev:alice")).json()["detail"] == "not_addin"


def test_a_pass_ends_with_its_time(site):  # noqa: ARG001
    old = addin.create("dev:alice", now=utc_now() - addin.LIFETIME - timedelta(seconds=1))
    assert client.get("/api/research/companies", headers=bearer(old.token)).status_code == 401
    # and the nightly maintenance deletes it
    assert scheduled.job_maintenance()["addin_sessions_deleted"] == 1


def test_only_the_hash_is_kept_and_a_person_holds_at_most_five(site):  # noqa: ARG001
    tokens = [new_pass() for _ in range(addin.MAX_PER_USER + 2)]
    with connect() as conn:
        hashes = set(conn.execute(select(AddinSession.token_hash)).scalars())
    assert len(hashes) == addin.MAX_PER_USER
    assert not any(t in h or t[len("addin:"):] in h for t in tokens for h in hashes)
    assert hashes == {addin.token_hash(t) for t in tokens[-addin.MAX_PER_USER:]}
    # the oldest two ended
    assert [client.get("/api/files", headers=bearer(t)).status_code for t in tokens[:2]] == [401, 401]


@pytest.mark.parametrize("token", ["addin:", "addin:short", "addin:" + "x" * 43 + "!", "addin:" + "A" * 43])
def test_made_up_passes_are_refused(site, token):  # noqa: ARG001
    assert client.get("/api/research/companies", headers=bearer(token)).status_code == 401


def test_a_pass_needs_a_finished_account(site):  # noqa: ARG001
    refused = client.post("/api/addin/sessions", headers=bearer("dev:newcomer"))
    assert (refused.status_code, refused.json()["detail"]) == (409, "no_account")


def test_a_pass_works_whichever_sign_in_the_site_uses(site, monkeypatch):  # noqa: ARG001
    """With Clerk configured the development tokens stop working, but a pass still does."""
    token = new_pass()
    monkeypatch.setenv("CLERK_ISSUER", "https://example.clerk.accounts.dev")
    assert client.get("/api/research/companies", headers=bearer("dev:alice")).status_code == 401
    auth.reset_addin_cache()
    assert client.get("/api/research/companies", headers=bearer(token)).status_code == 200
