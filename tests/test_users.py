"""Accounts: the profile asked at sign-up, and who can read it (PLAN.md 1.4).

Real rows in a real database (``fresh_db``), through the real endpoints. The
point of every test here is that the answer changes with the data: a saved
profile comes back as saved, one account never sees another's, and a country
or time zone the app can't honour is refused rather than stored.
"""
import pytest
from fastapi.testclient import TestClient

from api.auth import AuthUser, require_user
from api.main import app
from db import users as user_store

client = TestClient(app)

LONDON = {"country": "GB", "preferred_currency": "GBP", "locale": "en-GB",
          "time_zone": "Europe/London", "digit_grouping": "locale"}
TOKYO = {"country": "JP", "preferred_currency": "JPY", "locale": "ja-JP",
         "time_zone": "Asia/Tokyo", "digit_grouping": "locale"}


@pytest.fixture
def as_user():
    """Sign in as a given subject for the next request."""
    def sign_in(subject: str):
        app.dependency_overrides[require_user] = lambda: AuthUser(subject=subject)
    yield sign_in


# ---------------------------------------------------------------------------
# No database needed
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field, value", [
    ("country", "gb"),          # accepted, upper-cased
    ("preferred_currency", "gbp"),
    ("locale", " en-GB "),
    ("time_zone", " Europe/London "),
])
def test_profiles_are_tidied_before_they_are_stored(field, value):
    cleaned = user_store.clean(**{**LONDON, field: value})
    assert cleaned.as_dict() == LONDON


@pytest.mark.parametrize("field, value, why", [
    ("country", "GBR", "three-letter country"),
    ("country", "G1", "digit in a country"),
    ("country", "", "no country"),
    ("preferred_currency", "GB", "two-letter currency"),
    ("preferred_currency", "POUND", "a name, not a code"),
    ("locale", "english", "not a language tag"),
    ("locale", "en_GB", "underscore, not a hyphen"),
    ("time_zone", "GMT+1", "not an IANA name"),
    ("time_zone", "Europe/Atlantis", "a place with no time zone"),
    ("time_zone", "", "no time zone"),
])
def test_impossible_profiles_are_refused(field, value, why):
    with pytest.raises(user_store.InvalidProfile):
        user_store.clean(**{**LONDON, field: value})


def test_any_country_and_currency_work_not_just_a_supported_list():
    """Guiding principle 2: global by design, no US (or G7) default."""
    for profile in (
        {"country": "IN", "preferred_currency": "INR", "locale": "hi-IN", "time_zone": "Asia/Kolkata"},
        {"country": "BR", "preferred_currency": "BRL", "locale": "pt-BR", "time_zone": "America/Sao_Paulo"},
        {"country": "NG", "preferred_currency": "NGN", "locale": "en-NG", "time_zone": "Africa/Lagos"},
        {"country": "KZ", "preferred_currency": "KZT", "locale": "kk-Cyrl-KZ", "time_zone": "Asia/Almaty"},
    ):
        assert user_store.clean(**profile).as_dict() == {**profile, "digit_grouping": "locale"}


# ---------------------------------------------------------------------------
# With a database
# ---------------------------------------------------------------------------
def test_the_profile_comes_back_as_it_was_saved(fresh_db, as_user):  # noqa: ARG001
    as_user("user_first")
    assert client.get("/api/account").json() == {"subject": "user_first", "profile": None}

    saved = client.post("/api/account", json=LONDON)
    assert saved.status_code == 200, saved.text
    assert saved.json() == {"subject": "user_first", "profile": LONDON}
    # Read back in a new request: it is the database answering, not a cache
    assert client.get("/api/account").json()["profile"] == LONDON


def test_saving_again_replaces_the_profile_without_a_second_row(fresh_db, as_user):  # noqa: ARG001
    as_user("user_first")
    client.post("/api/account", json=LONDON)
    client.post("/api/account", json=TOKYO)
    assert client.get("/api/account").json()["profile"] == TOKYO
    with user_store.connect() as conn:
        from sqlalchemy import func, select

        from db.models import User
        assert conn.execute(select(func.count()).select_from(User)).scalar() == 1


def test_accounts_never_see_each_others_profiles(fresh_db, as_user):  # noqa: ARG001
    as_user("user_first")
    client.post("/api/account", json=LONDON)
    as_user("user_second")
    assert client.get("/api/account").json() == {"subject": "user_second", "profile": None}
    client.post("/api/account", json=TOKYO)
    assert client.get("/api/account").json()["profile"] == TOKYO
    # The first account is untouched
    as_user("user_first")
    assert client.get("/api/account").json()["profile"] == LONDON


def test_the_subject_in_the_body_is_ignored(fresh_db, as_user):  # noqa: ARG001
    """A caller can't write someone else's profile by naming them."""
    as_user("user_first")
    resp = client.post("/api/account", json={**LONDON, "subject": "user_second"})
    assert resp.status_code == 422  # extra fields are refused outright
    as_user("user_second")
    assert client.get("/api/account").json()["profile"] is None


@pytest.mark.parametrize("bad", [
    {**LONDON, "time_zone": "Europe/Atlantis"},
    {**LONDON, "country": "G1"},
    {**LONDON, "preferred_currency": "gb"},
])
def test_the_endpoint_refuses_an_impossible_profile(fresh_db, as_user, bad):  # noqa: ARG001
    as_user("user_first")
    assert client.post("/api/account", json=bad).status_code == 422
    assert client.get("/api/account").json()["profile"] is None


def test_without_a_database_the_account_says_so(monkeypatch, as_user):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    as_user("user_first")
    resp = client.get("/api/account")
    assert resp.status_code == 503 and "DATABASE_URL" in resp.json()["detail"]
