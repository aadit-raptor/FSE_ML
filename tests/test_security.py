"""Security hardening (PLAN.md 1.7): response headers, CORS locked to the app's
own origins, TLS to the database, and a database role that can only read and
write rows.

The database tests connect as a real login role in ``fse_app`` (migration
0005) and drive the API through it, so they prove both halves: the app works
with only those rights, and those rights stop schema changes.
"""
import base64
import hashlib
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ProgrammingError

import api.main as api_main
from api import security
from api.auth import AuthUser, require_user
from db import engine as db_engine
from db import migrate

client = TestClient(api_main.app)


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------
# Written out, not read from api/security.py, so dropping a header there fails here
EXPECTED_HEADERS = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
}


def assert_hardened(resp, csp=security.API_CSP):
    for name, value in EXPECTED_HEADERS.items():
        assert resp.headers.get(name) == value, (name, resp.headers.get(name))
    assert resp.headers["Content-Security-Policy"] == csp
    assert resp.headers["Cache-Control"] == "no-store"


def test_health_answers_carry_the_security_headers():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert_hardened(resp)
    assert "default-src 'none'" in resp.headers["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in resp.headers["Content-Security-Policy"]


def test_model_answers_and_refusals_carry_them_too(signed_out):  # noqa: ARG001
    refused = client.post("/api/deal/run", json={})
    assert refused.status_code == 401
    assert_hardened(refused)
    missing = client.get("/api/no-such-route")
    assert missing.status_code == 404
    assert_hardened(missing)


def test_deal_results_are_never_cached():
    resp = client.post("/api/deal/run", json={})
    assert resp.status_code == 200 and resp.json()["returns"]["irr"] == pytest.approx(0.2116, abs=5e-5)
    assert_hardened(resp)


def test_the_docs_page_may_run_only_its_own_inline_script():
    resp = client.get("/api/docs")
    assert resp.status_code == 200 and "swagger-ui" in resp.text
    csp = resp.headers["Content-Security-Policy"]
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp
    # FastAPI writes its inline script as a bare <script> tag; the ones with src have attributes
    inline = [part.split("</script>", 1)[0] for part in resp.text.split("<script>")[1:]]
    assert inline, "the docs page has an inline script to allow"
    assert security.inline_scripts(resp.content) == inline
    # Upper-case tags and spaced end tags are found; scripts with src aren't hashed
    assert security.inline_scripts(b'<script src="x.js"></script><SCRIPT>a()</script ><script>b()</script>') \
        == ["a()", "b()"]
    for body in inline:
        digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in csp
    # A different script would not match any allowed hash
    assert security.docs_csp(b"<script>alert(1)</script>") != csp
    assert int(resp.headers["Content-Length"]) == len(resp.content)
    for name, value in EXPECTED_HEADERS.items():
        assert resp.headers[name] == value


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
def app_client(monkeypatch, environment, origins=None):
    monkeypatch.setenv("FSE_ENV", environment)
    if origins is None:
        monkeypatch.delenv("FSE_CORS_ORIGINS", raising=False)
    else:
        monkeypatch.setenv("FSE_CORS_ORIGINS", origins)
    return TestClient(api_main.create_app())


def preflight(c, origin, headers="authorization,content-type"):
    return c.options("/api/deal/run", headers={
        "Origin": origin, "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": headers})


def test_production_allows_only_the_app_origin(monkeypatch):
    c = app_client(monkeypatch, "production")
    ok = preflight(c, "https://fse-ml.vercel.app")
    assert ok.status_code == 200
    assert ok.headers["access-control-allow-origin"] == "https://fse-ml.vercel.app"
    assert "access-control-allow-credentials" not in ok.headers
    for origin in ("https://evil.example", "http://fse-ml.vercel.app", "http://localhost:3000",
                   "https://fse-ml.vercel.app.evil.example"):
        refused = preflight(c, origin)
        assert refused.status_code == 400, origin
        assert "access-control-allow-origin" not in refused.headers
    # A header the app doesn't send isn't allowed cross-origin
    assert preflight(c, "https://fse-ml.vercel.app", "x-anything").status_code == 400


def test_staging_allows_no_cross_origin_calls(monkeypatch):
    c = app_client(monkeypatch, "staging")
    assert preflight(c, "https://fse-ml.vercel.app").status_code == 400


def test_local_allows_the_dev_server(monkeypatch):
    c = app_client(monkeypatch, "local")
    assert preflight(c, "http://localhost:3000").headers["access-control-allow-origin"] == "http://localhost:3000"


@pytest.mark.parametrize("value", ["*", "https://*.vercel.app", "http://fse-ml.vercel.app",
                                   "https://fse-ml.vercel.app/", "https://fse-ml.vercel.app/path",
                                   "https://user:pw@fse-ml.vercel.app", "fse-ml.vercel.app",
                                   "http://localhost:3000"])
def test_a_bad_origin_setting_locks_the_api_down_rather_than_opening_it(monkeypatch, value):
    monkeypatch.setenv("FSE_CORS_ORIGINS", value)
    assert security.cors_origins("production") == []
    c = app_client(monkeypatch, "production", value)
    assert "access-control-allow-origin" not in preflight(c, "https://evil.example").headers


def test_the_origin_setting_replaces_the_default(monkeypatch):
    monkeypatch.setenv("FSE_CORS_ORIGINS", "https://app.example.com, *, https://app.example.com")
    assert security.cors_origins("production") == ["https://app.example.com"]


# ---------------------------------------------------------------------------
# Encrypted database connections
# ---------------------------------------------------------------------------
NEON = "postgresql://u:p@ep-x-pooler.us-east-2.aws.neon.tech/neondb"


@pytest.mark.parametrize("environment", ["production", "staging"])
def test_deployed_copies_always_use_tls(environment):
    assert db_engine.require_tls(NEON, environment).endswith("?sslmode=require")
    kept = NEON + "?sslmode=verify-full&channel_binding=require"
    assert db_engine.require_tls(kept, environment) == kept
    for mode in ("disable", "allow", "prefer"):
        with pytest.raises(db_engine.DatabaseUnavailable, match="TLS"):
            db_engine.require_tls(f"{NEON}?sslmode={mode}", environment)


def test_tls_is_enforced_when_the_engine_url_is_built(monkeypatch):
    monkeypatch.setenv("FSE_ENV", "production")
    assert "sslmode=require" in db_engine.sqlalchemy_url(NEON)
    with pytest.raises(db_engine.DatabaseUnavailable):
        db_engine.sqlalchemy_url(NEON + "?sslmode=disable")


def test_local_runs_and_ci_keep_their_plain_postgres():
    local = "postgresql://postgres:postgres@localhost:5432/postgres"
    assert db_engine.require_tls(local, "local") == local
    remote = "postgresql://postgres:postgres@host.docker.internal:5432/postgres"
    assert db_engine.require_tls(remote, "local") == remote
    # Loopback traffic never leaves the machine, even on a deployed copy;
    # any other host does
    assert db_engine.require_tls(local, "production") == local
    assert db_engine.require_tls(remote, "production").endswith("?sslmode=require")


# ---------------------------------------------------------------------------
# Least-privilege database role
# ---------------------------------------------------------------------------
@pytest.fixture
def app_role(fresh_db, monkeypatch):
    """A login role in fse_app on the migrated test database. DATABASE_URL
    becomes that role; the owner moves to DATABASE_MIGRATION_URL."""
    migrate.upgrade(fresh_db)
    name = f"fse_test_{uuid.uuid4().hex[:10]}"
    password = uuid.uuid4().hex
    owner = create_engine(db_engine.sqlalchemy_url(fresh_db), isolation_level="AUTOCOMMIT")
    with owner.connect() as conn:
        conn.execute(text(f"CREATE ROLE {name} LOGIN PASSWORD '{password}' IN ROLE {migrate_app_role()}"))
    url = make_url(db_engine.sqlalchemy_url(fresh_db)).set(username=name, password=password) \
        .render_as_string(hide_password=False).replace("postgresql+psycopg://", "postgresql://", 1)
    monkeypatch.setenv("DATABASE_MIGRATION_URL", fresh_db)
    monkeypatch.setenv("DATABASE_URL", url)
    yield url
    db_engine.dispose_engines()
    with owner.connect() as conn:
        conn.execute(text(f"DROP ROLE IF EXISTS {name}"))
    owner.dispose()


def migrate_app_role():
    from importlib import import_module
    return import_module("db.migrations.versions.0005_app_role").APP_ROLE


def as_role(url, sql):
    engine = create_engine(db_engine.sqlalchemy_url(url))
    try:
        with engine.begin() as conn:
            return conn.execute(text(sql))
    finally:
        engine.dispose()


def test_the_api_works_as_the_restricted_role(app_role, monkeypatch):  # noqa: ARG001
    api_main.app.dependency_overrides[require_user] = lambda: AuthUser(subject="user_role_test")
    profile = {"country": "GB", "preferred_currency": "GBP", "locale": "en-GB", "time_zone": "Europe/London"}
    assert client.post("/api/account", json=profile).status_code == 200
    run = client.post("/api/deal/run", json={"inputs": {"exit_mult": 12.0}})
    created = client.post("/api/deals", json={"name": "Role test", "inputs": {"exit_mult": 12.0},
                                              "settings": {}})
    assert created.status_code == 201, created.text
    deal_id = created.json()["id"]
    assert client.put(f"/api/deals/{deal_id}/draft",
                      json={"inputs": {"exit_mult": 12.0}, "settings": {}}).status_code == 200
    opened = client.get(f"/api/deals/{deal_id}").json()
    rerun = client.post("/api/deal/run", json={"inputs": opened["inputs"], "settings": opened["settings"]})
    assert rerun.json()["returns"]["irr"] == run.json()["returns"]["irr"]
    assert client.delete(f"/api/deals/{deal_id}").status_code == 204

    health = client.get("/api/health/database").json()
    assert health["status"] == "ok" and health["migrations"]["status"] == "current", health
    assert health["role"] == {"status": "restricted", "privileges": []}


@pytest.mark.parametrize("sql", [
    "CREATE TABLE sneaky (id int)",
    "DROP TABLE deal_versions",
    "ALTER TABLE deals ADD COLUMN sneaky int",
    "TRUNCATE deals",
    "UPDATE alembic_version SET version_num = '0001'",
    "DELETE FROM alembic_version",
    "CREATE ROLE sneaky LOGIN",
    "ALTER ROLE fse_app CREATEROLE",
])
def test_the_restricted_role_cant_change_the_schema(app_role, sql):
    with pytest.raises(ProgrammingError, match="permission denied|must be owner|must have admin"):
        as_role(app_role, sql)


def test_tables_added_by_later_migrations_are_usable_by_the_role(app_role, fresh_db):
    as_role(fresh_db, "CREATE TABLE later_table (id int)")
    as_role(app_role, "INSERT INTO later_table VALUES (1)")
    assert as_role(app_role, "SELECT count(*) FROM later_table").scalar() == 1
    with pytest.raises(ProgrammingError):
        as_role(app_role, "DROP TABLE later_table")


def test_the_health_check_flags_an_owner_connection(fresh_db):
    health = client.get("/api/health/database").json()
    assert health["status"] == "ok"
    assert health["role"]["status"] == "privileged" and health["role"]["privileges"]


def test_new_schema_changes_still_run_as_the_owner(app_role, fresh_db):  # noqa: ARG001
    migrate.downgrade(fresh_db, "0004")
    migrate.reset_state()
    # DATABASE_URL is the restricted role; migrating on first use goes through the owner
    migrate.ensure_migrated()
    assert migrate.current(fresh_db) == migrate.head_revision()


# ---------------------------------------------------------------------------
# The header scan the deployed checks run (ops/check_headers.py)
# ---------------------------------------------------------------------------
GOOD_PAGE = {
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "camera=()",
    "Content-Security-Policy": "default-src 'self'; script-src 'self' 'nonce-abc123' 'strict-dynamic'; "
                               "style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self'; "
                               "form-action 'self'; frame-ancestors 'none'",
}


def test_the_scan_passes_the_apis_real_headers():
    from ops import check_headers
    resp = client.get("/api/health")
    assert check_headers.api_problems(resp.headers) == []
    refused = TestClient(api_main.create_app()).options("/api/deal/run", headers={
        "Origin": check_headers.UNKNOWN_ORIGIN, "Access-Control-Request-Method": "POST"})
    assert check_headers.cors_problems(refused.status_code, refused.headers) == []


def test_the_scan_passes_a_strict_page():
    from ops import check_headers
    assert check_headers.page_problems(GOOD_PAGE) == []
    assert check_headers.nonce(GOOD_PAGE) == "abc123"


@pytest.mark.parametrize("change, finding", [
    ({"Content-Security-Policy": None}, "Content-Security-Policy is missing"),
    ({"Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; object-src 'none'; "
                                 "base-uri 'self'; frame-ancestors 'none'"}, "no nonce"),
    ({"Content-Security-Policy": "default-src 'self'; script-src 'nonce-a' 'strict-dynamic' https:; "
                                 "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"}, "allows https:"),
    ({"Content-Security-Policy": "default-src 'self'; script-src 'nonce-a' 'strict-dynamic' 'unsafe-eval'; "
                                 "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"}, "'unsafe-eval'"),
    ({"Content-Security-Policy": "default-src 'self'; script-src 'nonce-a' 'strict-dynamic'; "
                                 "base-uri 'self'; frame-ancestors 'none'"}, "object-src"),
    ({"Content-Security-Policy": "default-src 'self'; script-src 'nonce-a' 'strict-dynamic'; "
                                 "object-src 'none'; base-uri 'self'"}, "frame-ancestors"),
    ({"Strict-Transport-Security": "max-age=3600"}, "Strict-Transport-Security"),
    ({"X-Content-Type-Options": None}, "nosniff"),
    ({"X-Frame-Options": "SAMEORIGIN"}, "X-Frame-Options"),
    ({"Referrer-Policy": "unsafe-url"}, "Referrer-Policy"),
    ({"Permissions-Policy": None}, "Permissions-Policy"),
    ({"X-Powered-By": "Next.js"}, "X-Powered-By"),
])
def test_the_scan_flags_each_weakness(change, finding):
    from ops import check_headers
    headers = {k: v for k, v in {**GOOD_PAGE, **change}.items() if v is not None}
    problems = check_headers.page_problems(headers)
    assert any(finding in p for p in problems), problems


def test_the_scan_flags_cacheable_or_loadable_api_answers_and_open_cors():
    from ops import check_headers
    good = dict(client.get("/api/health").headers)
    assert any("no-store" in p for p in check_headers.api_problems({**good, "cache-control": "max-age=60"}))
    assert any("default-src" in p for p in check_headers.api_problems(
        {**good, "content-security-policy": "default-src 'self'; frame-ancestors 'none'"}))
    assert check_headers.cors_problems(200, {"Access-Control-Allow-Origin": "*"})
    assert check_headers.cors_problems(200, {"Access-Control-Allow-Origin": check_headers.UNKNOWN_ORIGIN})


def test_the_database_check_warns_then_fails_on_a_privileged_role(capsys, monkeypatch):
    from ops import check_database
    healthy = {"status": "ok", "migrations": {"status": "current"}, "storage": {"warning": False},
               "role": {"status": "restricted", "privileges": []}}
    privileged = {**healthy, "role": {"status": "privileged", "privileges": ["owns_tables"]}}
    assert check_database.role_problem(healthy) is None
    assert "owns_tables" in check_database.role_problem(privileged)
    monkeypatch.setattr(check_database, "fetch", lambda api: privileged)
    assert check_database.main(["https://api.example"]) == 0
    assert "::warning::" in capsys.readouterr().out
    assert check_database.main(["https://api.example", "--require-restricted-role"]) == 1
