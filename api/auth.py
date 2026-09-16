"""Who is calling the API (PLAN.md 1.4).

Every endpoint except the health checks needs a signed-in user. The browser
gets a short-lived JWT from Clerk and sends it as ``Authorization: Bearer
<token>``; this module verifies it and hands the route an ``AuthUser``.

**Verification is local.** The token is signed with RS256 by the Clerk
instance; its public keys come from the instance's JWKS endpoint and are
cached, so a request costs no network call and no Clerk API quota. Checked on
every token: signature, algorithm (RS256 only, never ``none`` or a symmetric
algorithm), issuer, expiry and not-before. Nothing here calls Clerk's backend
API, so no secret key is needed on the API side.

**Configuration** (Render, and ``.env`` locally) -- either one is enough:

- ``CLERK_ISSUER``: ``https://<instance>.clerk.accounts.dev`` (or the custom
  domain of a Clerk production instance), or
- ``CLERK_PUBLISHABLE_KEY``: ``pk_test_…`` / ``pk_live_…``. The instance's
  domain is base64 inside it, so the issuer is derived from it.

**Development sign-in.** Local runs and the browser tests have no Clerk
instance, so ``FSE_AUTH_DEV=1`` accepts ``Bearer dev:<name>`` as the user
``dev:<name>``. It is fail-closed, and refuses to work:

- when the environment is production (``FSE_ENV``/Render service name), or
- when Clerk is configured -- a real deployment never also accepts dev tokens.

``tests/test_auth.py`` pins all of this, including that every route outside
``PUBLIC_PATHS`` answers 401 without a token.
"""
from __future__ import annotations

import base64
import binascii
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

# Endpoints that work signed out: the health checks Render, the uptime monitor
# and the status bar poll, and the schema the typed client is generated from.
PUBLIC_PATHS = frozenset({
    "/api/health", "/api/health/database", "/api/health/limits", "/api/openapi.json", "/api/docs",
    "/api/docs/oauth2-redirect", "/api/debug/error",
})

# Clock skew allowed on exp/nbf/iat, in seconds
LEEWAY_S = 30
# How long a fetched JWKS is reused before it is fetched again
JWKS_CACHE_S = 3600
DEV_TOKEN_PREFIX = "dev:"

_bearer = HTTPBearer(auto_error=False, description="Clerk session token")


@dataclass(frozen=True)
class AuthUser:
    """The caller. ``subject`` is Clerk's user id (``user_…``), stable forever.

    It is the only thing stored about a user in this database; names and email
    addresses stay in Clerk. Never log it (api/observability.py keeps user
    details out of logs and Sentry events).
    """

    subject: str
    session_id: Optional[str] = None
    is_dev: bool = False


def deploy_environment() -> str:
    from api.main import deploy_environment as _env  # avoids a circular import at module load

    return _env()


def issuer_from_publishable_key(key: str) -> Optional[str]:
    """The Clerk instance's issuer URL encoded in a publishable key.

    ``pk_test_<base64 of "your-app-12.clerk.accounts.dev$">``.
    """
    key = key.strip()
    for prefix in ("pk_test_", "pk_live_"):
        if key.startswith(prefix):
            encoded = key[len(prefix):]
            break
    else:
        return None
    try:
        decoded = base64.b64decode(encoded + "=" * (-len(encoded) % 4)).decode("ascii")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    domain = decoded.rstrip("$").strip()
    if not domain or "/" in domain or " " in domain:
        return None
    return f"https://{domain}"


def clerk_issuer() -> Optional[str]:
    """The issuer this API trusts, or None when Clerk isn't configured."""
    explicit = (os.environ.get("CLERK_ISSUER") or "").strip().rstrip("/")
    if explicit:
        return explicit
    key = (os.environ.get("CLERK_PUBLISHABLE_KEY") or "").strip()
    return issuer_from_publishable_key(key) if key else None


def is_configured() -> bool:
    return clerk_issuer() is not None


def dev_auth_enabled() -> bool:
    """Whether ``Bearer dev:<name>`` is accepted (never in production, never with Clerk)."""
    if (os.environ.get("FSE_AUTH_DEV") or "").strip() not in ("1", "true", "True"):
        return False
    if deploy_environment() == "production" or is_configured():
        return False
    return True


# ---------------------------------------------------------------------------
# Clerk keys
# ---------------------------------------------------------------------------
_jwks_lock = threading.Lock()
_jwks: dict[str, tuple[float, PyJWKClient]] = {}


def _jwk_client(issuer: str) -> PyJWKClient:
    """A JWKS client for ``issuer``, rebuilt at most once an hour.

    PyJWKClient caches keys itself and refetches when it meets an unknown key
    id, so a Clerk key rotation is picked up without a restart.
    """
    now = time.monotonic()
    with _jwks_lock:
        cached = _jwks.get(issuer)
        if cached is None or now - cached[0] > JWKS_CACHE_S:
            client = PyJWKClient(f"{issuer}/.well-known/jwks.json", cache_keys=True, timeout=5)
            _jwks[issuer] = (now, client)
            return client
        return cached[1]


def reset_key_cache() -> None:
    """Forget cached signing keys (tests, and after changing configuration)."""
    with _jwks_lock:
        _jwks.clear()


def _unauthorized(detail: str) -> HTTPException:
    # WWW-Authenticate tells a browser client the token was the problem
    return HTTPException(status_code=401, detail=detail,
                         headers={"WWW-Authenticate": "Bearer"})


def verify_clerk_token(token: str, issuer: str) -> AuthUser:
    """Check a Clerk session token's signature and claims, or raise 401."""
    try:
        signing_key = _jwk_client(issuer).get_signing_key_from_jwt(token)
    except jwt.PyJWKClientConnectionError:
        # The key set itself is unreachable: the caller's token may be fine,
        # so this is the server's problem, not a refusal
        raise HTTPException(status_code=503,
                            detail="Sign-in keys unavailable; try again shortly.") from None
    except (jwt.PyJWKClientError, jwt.PyJWTError):
        # Not a JWT, or signed with a key this instance doesn't publish
        raise _unauthorized("token key not recognised") from None
    try:
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],   # never "none", never HS256 with a public key as the secret
            issuer=issuer,
            leeway=LEEWAY_S,
            options={"require": ["exp", "iat", "sub", "iss"], "verify_aud": False},
        )
    except jwt.ExpiredSignatureError:
        raise _unauthorized("session expired") from None
    except jwt.InvalidTokenError as exc:
        raise _unauthorized(f"invalid session token: {exc}") from None
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise _unauthorized("token has no user")
    session = claims.get("sid")
    return AuthUser(subject=subject, session_id=session if isinstance(session, str) else None)


def verify_dev_token(token: str) -> AuthUser:
    name = token[len(DEV_TOKEN_PREFIX):].strip()
    if not name or len(name) > 64 or not name.replace("_", "").replace("-", "").isalnum():
        raise _unauthorized("development token must be dev:<name>")
    return AuthUser(subject=f"{DEV_TOKEN_PREFIX}{name}", is_dev=True)


def authenticate(token: Optional[str]) -> AuthUser:
    """The user a bearer token stands for, or 401."""
    if not token:
        raise _unauthorized("sign in to use the API")
    issuer = clerk_issuer()
    if issuer:
        return verify_clerk_token(token, issuer)
    if dev_auth_enabled():
        if not token.startswith(DEV_TOKEN_PREFIX):
            raise _unauthorized("development sign-in expects dev:<name>")
        return verify_dev_token(token)
    # Nothing configured: refuse rather than let everyone in
    raise HTTPException(
        status_code=503,
        detail="Sign-in is not configured on this server (set CLERK_ISSUER).")


def require_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> AuthUser:
    """FastAPI dependency: the signed-in user, or 401."""
    token = credentials.credentials if credentials else None
    user = authenticate(token)
    # Routes read it from the request too (middleware and handlers)
    request.state.user = user
    return user


CurrentUser = Depends(require_user)
