"""Sign-in on the API (PLAN.md 1.4).

Nothing here is mocked at the boundary that matters: tokens are real RS256
JWTs signed with a key generated for the test, verified through the same
``api.auth`` code production uses. Only the network fetch of the signing keys
is replaced, by a JWKS built from that key.

Covered: a signed-out call is refused on every route; a valid token gets real
model output; a token that is expired, unsigned, signed by another key, from
another issuer or altered is refused; and the development sign-in refuses to
work in production or beside a real Clerk instance.
"""
import base64
import json
import time

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

import api.auth as auth
from api.main import app

client = TestClient(app)

ISSUER = "https://sample-instance-42.clerk.accounts.dev"
OTHER_ISSUER = "https://someone-else.clerk.accounts.dev"
KEY_ID = "ins_test_key_1"

# A route from each router: all of them must need a login
PROTECTED = [
    ("POST", "/api/deal/run", {}),
    ("POST", "/api/deal/sources-and-uses", {"ebitda": 100, "entry_mult": 10, "senior_x": 3.4, "mezz_x": 0.8}),
    ("POST", "/api/montecarlo/run", {"mc": {"n": 200}, "seed": 1}),
    ("POST", "/api/forecasting/seed", {"history": {}}),
    ("POST", "/api/backtesting/run", {"deal": "Burger King"}),
    ("GET", "/api/settings/defaults", None),
    ("GET", "/api/capabilities", None),
    ("POST", "/api/export/workbook", {"sheets": [{"name": "t", "columns": ["a"], "rows": [[1]]}]}),
    ("GET", "/api/account", None),
]


# ---------------------------------------------------------------------------
# A Clerk-shaped instance, signed with a key this test owns
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def signing_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def other_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def jwks_for(key, kid=KEY_ID) -> dict:
    """The public half of ``key`` as a JSON Web Key Set, as Clerk serves it."""
    numbers = key.public_key().public_numbers()

    def b64(value: int) -> str:
        raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    return {"keys": [{"kty": "RSA", "use": "sig", "alg": "RS256", "kid": kid,
                      "n": b64(numbers.n), "e": b64(numbers.e)}]}


def token_for(key, *, subject="user_2abcDEF", issuer=ISSUER, kid=KEY_ID,
              lifetime=60, algorithm="RS256", **claims) -> str:
    now = int(time.time())
    payload = {"sub": subject, "iss": issuer, "iat": now, "nbf": now,
               "exp": now + lifetime, "sid": "sess_1", **claims}
    return jwt.encode(payload, key, algorithm=algorithm, headers={"kid": kid})


@pytest.fixture
def clerk(monkeypatch, signing_key, signed_out):  # noqa: ARG001 - signed_out drops the test override
    """A configured Clerk instance whose keys come from ``signing_key``."""
    monkeypatch.setenv("CLERK_ISSUER", ISSUER)
    monkeypatch.delenv("CLERK_PUBLISHABLE_KEY", raising=False)
    monkeypatch.delenv("FSE_AUTH_DEV", raising=False)
    auth.reset_key_cache()
    served = jwks_for(signing_key)
    # Only the HTTP hop to Clerk is replaced: the key set is still parsed,
    # the key still rebuilt from n/e, the signature still checked for real.
    monkeypatch.setattr("jwt.PyJWKClient.fetch_data", lambda self: json.loads(json.dumps(served)))
    yield served
    auth.reset_key_cache()


def bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Signed out
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method, path, body", PROTECTED)
def test_every_endpoint_needs_a_login(clerk, method, path, body):  # noqa: ARG001
    resp = client.request(method, path, json=body)
    assert resp.status_code == 401, f"{path} answered {resp.status_code}"
    assert resp.headers["WWW-Authenticate"] == "Bearer"


def test_no_route_is_left_unprotected(clerk):  # noqa: ARG001
    """Every route in the published schema is either public on purpose or needs a login.

    Reading the schema rather than a list means a route added later is
    covered the day it appears.
    """
    schema = client.get("/api/openapi.json").json()
    checked = 0
    for path, operations in schema["paths"].items():
        if path in auth.PUBLIC_PATHS:
            continue
        # Path parameters are filled with something harmless
        url = (path.replace("{ticker}", "AAPL")
               .replace("{deal_id}", "00000000-0000-4000-8000-000000000000").replace("{number}", "1"))
        for method in operations:
            if method.upper() not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                continue
            resp = client.request(method.upper(), url, json={})
            assert resp.status_code == 401, f"{method} {url} answered {resp.status_code}"
            checked += 1
    assert checked >= len(PROTECTED)


@pytest.mark.parametrize("path", ["/api/health", "/api/openapi.json"])
def test_health_and_schema_stay_public(clerk, path):  # noqa: ARG001
    assert client.get(path).status_code == 200


def test_malformed_authorization_headers_are_refused(clerk):  # noqa: ARG001
    for header in ({"Authorization": "Bearer"}, {"Authorization": "Basic abc"},
                   {"Authorization": "Bearer not-a-jwt"}, {"Authorization": "Bearer "}):
        assert client.post("/api/deal/run", json={}, headers=header).status_code in (401, 403)


# ---------------------------------------------------------------------------
# Signed in
# ---------------------------------------------------------------------------
def test_valid_token_gets_real_model_output(clerk, signing_key):  # noqa: ARG001
    resp = client.post("/api/deal/run", json={}, headers=bearer(token_for(signing_key)))
    assert resp.status_code == 200, resp.text
    # The same default-deal IRR the rest of the suite pins: auth changes nothing
    assert round(resp.json()["returns"]["irr"], 4) == 0.2116


def test_the_user_is_the_one_in_the_token(clerk, signing_key):  # noqa: ARG001
    token = token_for(signing_key, subject="user_someone_else")
    assert auth.authenticate(token).subject == "user_someone_else"
    assert auth.authenticate(token).session_id == "sess_1"


@pytest.mark.parametrize("make, why", [
    (lambda key: token_for(key, lifetime=-120), "expired"),
    (lambda key: token_for(key, issuer=OTHER_ISSUER), "another issuer"),
    (lambda key: token_for(key, kid="unknown_kid"), "unknown key id"),
    (lambda key: jwt.encode({"sub": "user_1", "iss": ISSUER, "exp": int(time.time()) + 60},
                            "secret", algorithm="HS256", headers={"kid": KEY_ID}), "symmetric algorithm"),
    (lambda key: jwt.encode({"sub": "user_1", "iss": ISSUER, "exp": int(time.time()) + 60},
                            None, algorithm="none", headers={"kid": KEY_ID}), "unsigned"),
    (lambda key: token_for(key, subject=""), "no user"),
    (lambda key: token_for(key)[:-4] + "AAAA", "altered signature"),
])
def test_bad_tokens_are_refused(clerk, signing_key, make, why):  # noqa: ARG001
    resp = client.post("/api/deal/run", json={}, headers=bearer(make(signing_key)))
    assert resp.status_code == 401, f"{why} was accepted"


def test_token_signed_by_another_key_is_refused(clerk, other_key):  # noqa: ARG001
    """A well-formed token for the right issuer, signed by a key we don't trust."""
    resp = client.post("/api/deal/run", json={}, headers=bearer(token_for(other_key)))
    assert resp.status_code == 401


def test_a_token_without_an_expiry_is_refused(clerk, signing_key):  # noqa: ARG001
    forever = jwt.encode({"sub": "user_1", "iss": ISSUER, "iat": int(time.time())},
                         signing_key, algorithm="RS256", headers={"kid": KEY_ID})
    assert client.post("/api/deal/run", json={}, headers=bearer(forever)).status_code == 401


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_issuer_comes_from_the_publishable_key(monkeypatch):
    encoded = base64.b64encode(b"sample-instance-42.clerk.accounts.dev$").decode().rstrip("=")
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    monkeypatch.setenv("CLERK_PUBLISHABLE_KEY", f"pk_test_{encoded}")
    assert auth.clerk_issuer() == ISSUER
    monkeypatch.setenv("CLERK_ISSUER", "https://accounts.example.com/")
    assert auth.clerk_issuer() == "https://accounts.example.com"


@pytest.mark.parametrize("key", ["", "pk_test_!!!not-base64!!!", "sk_test_secret", "pk_test_" + base64.b64encode(b"a b$").decode()])
def test_unusable_publishable_keys_give_no_issuer(monkeypatch, key):
    monkeypatch.delenv("CLERK_ISSUER", raising=False)
    monkeypatch.setenv("CLERK_PUBLISHABLE_KEY", key)
    assert auth.clerk_issuer() is None


def test_without_any_configuration_the_api_refuses_rather_than_opens_up(monkeypatch, signed_out):  # noqa: ARG001
    for name in ("CLERK_ISSUER", "CLERK_PUBLISHABLE_KEY", "FSE_AUTH_DEV"):
        monkeypatch.delenv(name, raising=False)
    assert client.post("/api/deal/run", json={}).status_code == 401
    # Even with a token: no configuration means nobody is let in
    assert client.post("/api/deal/run", json={}, headers=bearer("dev:someone")).status_code == 503


# ---------------------------------------------------------------------------
# Development sign-in (local runs and the browser tests)
# ---------------------------------------------------------------------------
@pytest.fixture
def dev_auth(monkeypatch, signed_out):  # noqa: ARG001
    for name in ("CLERK_ISSUER", "CLERK_PUBLISHABLE_KEY", "RENDER_SERVICE_NAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("FSE_AUTH_DEV", "1")
    monkeypatch.setenv("FSE_ENV", "local")
    yield


def test_development_sign_in_works_locally(dev_auth):  # noqa: ARG001
    resp = client.post("/api/deal/run", json={}, headers=bearer("dev:e2e"))
    assert resp.status_code == 200, resp.text
    assert round(resp.json()["returns"]["irr"], 4) == 0.2116
    assert auth.authenticate("dev:e2e").subject == "dev:e2e"
    assert auth.authenticate("dev:e2e").is_dev


def test_development_sign_in_still_needs_a_token(dev_auth):  # noqa: ARG001
    assert client.post("/api/deal/run", json={}).status_code == 401
    assert client.post("/api/deal/run", json={}, headers=bearer("not-a-dev-token")).status_code == 401
    assert client.post("/api/deal/run", json={}, headers=bearer("dev:")).status_code == 401
    assert client.post("/api/deal/run", json={}, headers=bearer("dev:has spaces")).status_code == 401


def test_development_sign_in_is_off_in_production(monkeypatch, dev_auth):  # noqa: ARG001
    monkeypatch.setenv("FSE_ENV", "production")
    assert not auth.dev_auth_enabled()
    assert client.post("/api/deal/run", json={}, headers=bearer("dev:e2e")).status_code == 503
    # Render names the service when FSE_ENV isn't set
    monkeypatch.delenv("FSE_ENV")
    monkeypatch.setenv("RENDER_SERVICE_NAME", "fse-api")
    assert not auth.dev_auth_enabled()


def test_development_sign_in_is_off_when_clerk_is_configured(monkeypatch, dev_auth):  # noqa: ARG001
    monkeypatch.setenv("CLERK_ISSUER", ISSUER)
    assert not auth.dev_auth_enabled()
    assert client.post("/api/deal/run", json={}, headers=bearer("dev:e2e")).status_code == 401


def test_development_sign_in_is_off_unless_asked_for(monkeypatch, dev_auth):  # noqa: ARG001
    monkeypatch.setenv("FSE_AUTH_DEV", "0")
    assert not auth.dev_auth_enabled()
    monkeypatch.delenv("FSE_AUTH_DEV")
    assert not auth.dev_auth_enabled()
