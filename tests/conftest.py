"""Shared test setup: a signed-in caller, and a database to work in.

Every endpoint except the health checks needs a signed-in user (PLAN.md 1.4),
so by default the tests here run as one. ``tests/test_auth.py`` is what proves
the real thing works: it drops this override and drives the token path --
signature, issuer, expiry -- end to end, including that a call without a token
is refused.

Tests that need Postgres take ``fresh_db``: an empty database of their own on
``TEST_DATABASE_URL`` (``python -m db.local`` prints one), dropped afterwards.
Without that variable they skip, except in CI, where ``FSE_REQUIRE_DB=1``
turns a skip into a failure.
"""
import os
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from api import usage
from api.auth import AuthUser, require_user
from api.main import app
from db import engine as db_engine
from db import health as db_health
from db import migrate

TEST_USER = AuthUser(subject="test:pytest", is_dev=True)


@pytest.fixture(autouse=True)
def signed_in():
    """Run tests as ``TEST_USER`` unless they ask for the real check."""
    app.dependency_overrides[require_user] = lambda: TEST_USER
    yield TEST_USER
    app.dependency_overrides.pop(require_user, None)


@pytest.fixture(autouse=True)
def usage_counters():
    """Fresh usage counters for each test, kept in memory: limits apply as in
    production, but no test sends counts to Upstash or the database unless it
    builds a store itself (tests/test_limits.py)."""
    fresh = usage.UsageCounters([], background=False)
    usage.set_counters(fresh)
    yield fresh
    usage.set_counters(None)


@pytest.fixture
def signed_out(signed_in):  # noqa: ARG001 - replaces the autouse override
    """Real authentication: tokens are verified as they are in production."""
    app.dependency_overrides.pop(require_user, None)
    yield


@pytest.fixture(scope="session")
def admin_url():
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        if os.environ.get("FSE_REQUIRE_DB") == "1":
            pytest.fail("TEST_DATABASE_URL is required (FSE_REQUIRE_DB=1)")
        pytest.skip("TEST_DATABASE_URL not set (python -m db.local prints one)")
    return url


@pytest.fixture
def fresh_db(admin_url, monkeypatch):
    """An empty database of its own, set as DATABASE_URL, dropped afterwards."""
    name = f"fse_test_{uuid.uuid4().hex[:10]}"
    admin = create_engine(db_engine.sqlalchemy_url(admin_url), isolation_level="AUTOCOMMIT")
    with admin.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{name}"'))
    url = make_url(db_engine.sqlalchemy_url(admin_url)).set(database=name) \
        .render_as_string(hide_password=False).replace("postgresql+psycopg://", "postgresql://", 1)
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setattr(db_engine, "RETRY_DELAYS_S", (0.05, 0.1, 0.2, 0.4, 0.8))
    db_health.reset()
    migrate.reset_state()
    yield url
    db_engine.dispose_engines()
    db_health.reset()
    migrate.reset_state()
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    admin.dispose()
