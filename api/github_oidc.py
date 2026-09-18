"""Which scheduled workflow is calling (PLAN.md 1.9), with no shared secret.

The free engine has no scheduler, so GitHub Actions is the scheduler: its
workflows call ``/api/scheduled/...``. They prove who they are with the
**OpenID Connect token GitHub mints for each workflow run**
(``permissions: id-token: write``): a short-lived RS256 JWT, signed by GitHub,
naming the repository, the workflow file, the branch and the run. The API
verifies it locally against GitHub's published keys, exactly as it verifies
Clerk's session tokens (api/auth.py), so there is no password to create,
store, copy into Render or leak.

Checked on every token: signature (RS256 only), issuer, expiry, and

- ``aud`` is this environment's audience (``fse-scheduler:production`` or
  ``fse-scheduler:staging``), so a token minted for staging can't be replayed
  against production;
- ``repository_id`` is this repository's (the name alone could be re-used
  after a rename), and ``repository`` its name;
- ``job_workflow_ref`` is an allowed workflow file **on an allowed branch**:
  production accepts only ``scheduled.yml`` on ``main``; staging also accepts
  ``staging.yml`` and ``scheduled.yml`` on ``staging``. Both branches are
  protected, so only reviewed code can mint an accepted token, and pull
  requests from forks get no token at all.

Configuration (defaults fit this repository): ``FSE_SCHEDULER_REPOSITORY``,
``FSE_SCHEDULER_REPOSITORY_ID``.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from api.observability import deploy_environment

GITHUB_ISSUER = "https://token.actions.githubusercontent.com"
GITHUB_JWKS_URL = f"{GITHUB_ISSUER}/.well-known/jwks"
DEFAULT_REPOSITORY = "aadit-raptor/FSE_ML"
DEFAULT_REPOSITORY_ID = "1214945471"
AUDIENCE_PREFIX = "fse-scheduler:"
LEEWAY_S = 30
JWKS_CACHE_S = 3600

# (workflow file, ref) pairs whose tokens each environment accepts
ALLOWED_WORKFLOWS: dict[str, frozenset[tuple[str, str]]] = {
    "production": frozenset({("scheduled.yml", "refs/heads/main")}),
    "staging": frozenset({("scheduled.yml", "refs/heads/main"), ("scheduled.yml", "refs/heads/staging"),
                          ("staging.yml", "refs/heads/staging")}),
}
# Local runs and CI behave like staging
DEFAULT_ALLOWED = ALLOWED_WORKFLOWS["staging"]

_bearer = HTTPBearer(auto_error=False, description="GitHub Actions OIDC token")


@dataclass(frozen=True)
class Workflow:
    """The workflow run that called."""

    workflow: str          # file name, e.g. scheduled.yml
    ref: str               # refs/heads/main
    event: str             # schedule, workflow_dispatch, push
    run_id: Optional[int]
    sha: Optional[str]


def audience(environment: Optional[str] = None) -> str:
    return AUDIENCE_PREFIX + (environment or deploy_environment())


def repository() -> tuple[str, str]:
    name = (os.environ.get("FSE_SCHEDULER_REPOSITORY") or DEFAULT_REPOSITORY).strip()
    repo_id = (os.environ.get("FSE_SCHEDULER_REPOSITORY_ID") or DEFAULT_REPOSITORY_ID).strip()
    return name, repo_id


def allowed_workflows(environment: Optional[str] = None) -> frozenset[tuple[str, str]]:
    return ALLOWED_WORKFLOWS.get(environment or deploy_environment(), DEFAULT_ALLOWED)


_lock = threading.Lock()
_client: list = [0.0, None]


def _jwk_client() -> PyJWKClient:
    now = time.monotonic()
    with _lock:
        if _client[1] is None or now - _client[0] > JWKS_CACHE_S:
            _client[:] = [now, PyJWKClient(GITHUB_JWKS_URL, cache_keys=True, timeout=5)]
        return _client[1]


def reset_key_cache() -> None:
    with _lock:
        _client[:] = [0.0, None]


def _refused(detail: str) -> HTTPException:
    return HTTPException(status_code=401, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def verify_workflow_token(token: str, environment: Optional[str] = None) -> Workflow:
    """Check a GitHub Actions OIDC token, or raise 401 (503 if GitHub's keys are unreachable)."""
    try:
        key = _jwk_client().get_signing_key_from_jwt(token)
    except jwt.PyJWKClientConnectionError:
        raise HTTPException(503, "GitHub's signing keys are unavailable; try again shortly.") from None
    except (jwt.PyJWKClientError, jwt.PyJWTError):
        raise _refused("token key not recognised") from None
    try:
        claims = jwt.decode(
            token, key.key, algorithms=["RS256"], issuer=GITHUB_ISSUER,
            audience=audience(environment), leeway=LEEWAY_S,
            options={"require": ["exp", "iat", "iss", "aud", "repository", "repository_id",
                                 "job_workflow_ref"]})
    except jwt.ExpiredSignatureError:
        raise _refused("token expired") from None
    except jwt.InvalidTokenError as exc:
        raise _refused(f"invalid workflow token: {exc}") from None

    name, repo_id = repository()
    if claims.get("repository") != name or str(claims.get("repository_id")) != repo_id:
        raise _refused("token is from another repository")
    workflow_ref = str(claims.get("job_workflow_ref"))
    path, _, ref = workflow_ref.partition("@")
    prefix = f"{name}/.github/workflows/"
    if not path.startswith(prefix):
        raise _refused("token is not from a workflow of this repository")
    workflow = path[len(prefix):]
    if (workflow, ref) not in allowed_workflows(environment):
        raise _refused(f"{workflow} on {ref} may not run scheduled jobs here")
    run_id = claims.get("run_id")
    return Workflow(workflow=workflow, ref=ref, event=str(claims.get("event_name") or "unknown")[:20],
                    run_id=int(run_id) if str(run_id or "").isdigit() else None,
                    sha=claims.get("sha") if isinstance(claims.get("sha"), str) else None)


def require_workflow(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> Workflow:
    """FastAPI dependency: the calling workflow run, or 401."""
    if credentials is None:
        raise _refused("scheduled endpoints need a GitHub Actions token")
    workflow = verify_workflow_token(credentials.credentials)
    request.state.workflow = workflow
    return workflow
