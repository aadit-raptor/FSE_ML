"""FastAPI application serving the LBO model to the web frontend.

    uvicorn api.main:app --reload

Interactive docs at /api/docs; the OpenAPI schema at /api/openapi.json is what
the frontend's typed client is generated from.
"""
import os
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.observability import (
    REQUEST_ID_HEADER, RequestContextMiddleware, configure_logging, deploy_environment, init_sentry,
    utc_now_iso,
)
from api.auth import require_user
from api import usage
from api.limits import LimitExceeded, LimitRefusal, LimitsMiddleware, enforce_user_limits
from api.security import CORS_ALLOW_HEADERS, SecurityHeadersMiddleware, cors_origins
from api.github_oidc import require_workflow
from api.routers import (
    account, backtesting, deal, deals, export, forecasting, integrations, jobs, montecarlo,
    scheduled,
)
from db import DatabaseUnavailable
from db import health as db_health

API_VERSION = "0.1.0"


# /api/debug/error: at most one deliberate error per this many seconds, so a
# public URL can't burn the free Sentry event allowance
TEST_ERROR_INTERVAL_S = 60.0
_last_test_error = [0.0]


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Render stops a sleeping free instance with SIGTERM: send shared usage
    # counts before the process goes (api/usage.py)
    usage.counters().close()


def create_app() -> FastAPI:
    configure_logging()
    init_sentry(os.environ.get("SENTRY_DSN"), environment=deploy_environment(),
                release=os.environ.get("RENDER_GIT_COMMIT") or None)
    app = FastAPI(
        title="Simulation Model API",
        version=API_VERSION,
        description="LBO modelling: deal wizard, Monte Carlo, backtesting and "
                    "3-statement forecasting.",
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # Browser origins allowed to call the API: the app's own domains only
    # (api/security.py; FSE_CORS_ORIGINS replaces the default)
    origins = cors_origins(deploy_environment())
    # Innermost: limit refusals are logged with a request ID and carry CORS
    # headers, so the browser can read the message (api/limits.py)
    app.add_middleware(LimitsMiddleware)
    # Inside CORS, so its 500 responses still get CORS headers
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                       allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
                       allow_headers=list(CORS_ALLOW_HEADERS),
                       expose_headers=["Content-Disposition", REQUEST_ID_HEADER, "Server-Timing"])
    # Security headers on every response, refusals and preflights included;
    # inside gzip so the docs page can be hashed uncompressed
    app.add_middleware(SecurityHeadersMiddleware)
    # Simulation responses carry chart data (histograms, scatter samples)
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.get("/api/health", tags=["meta"])
    def health():
        # commit: the git SHA Render built this deploy from (None locally), so a
        # check can tell which change each environment is running
        # time: the server clock in UTC; screens show it in the viewer's time zone
        return {"status": "ok", "version": API_VERSION,
                "environment": deploy_environment(),
                "commit": os.environ.get("RENDER_GIT_COMMIT") or None,
                "time": utc_now_iso(),
                # last known state only: querying here would keep Neon awake
                "database": db_health.summary()}

    @app.get("/api/health/database", tags=["meta"])
    def health_database(response: Response):
        """Connect to the database (waiting for it to wake), apply pending
        migrations and report storage use against the free limit. 503 when
        the database can't be reached."""
        result = db_health.check(deploy_environment())
        if result["status"] == "error":
            response.status_code = 503
        return result

    @app.get("/api/health/limits", tags=["meta"])
    def health_limits():
        """Where usage counters are shared (Upstash, the database or memory
        only), whether Upstash answers, and today's Redis command use. Never
        queries the database; the Upstash check is a PING (not billed),
        reused for 10 minutes."""
        return usage.status()

    @app.get("/api/health/jobs", tags=["meta"])
    def health_jobs(response: Response):
        """The job queue (PLAN.md 1.9): which queue and runner are configured,
        whether the runner is running, jobs per status, and the newest run of
        each scheduled task. Queries the database, so it is cached for a
        minute and nothing polls it often."""
        result = jobs_health()
        if result.get("status") == "error":
            response.status_code = 503
        return result

    @app.exception_handler(LimitExceeded)
    async def limit_exceeded(request: Request, exc: LimitExceeded):
        return JSONResponse(exc.body, status_code=429, headers=exc.headers)

    @app.exception_handler(DatabaseUnavailable)
    async def database_unavailable(request: Request, exc: DatabaseUnavailable):
        return JSONResponse({"detail": "The database is unavailable; try again shortly."},
                            status_code=503)

    @app.get("/api/debug/error", include_in_schema=False)
    def debug_error():
        """Raise a deliberate error to prove error tracking works (not in production)."""
        if deploy_environment() == "production":
            raise HTTPException(404, "Not Found")
        now = time.monotonic()
        if now - _last_test_error[0] < TEST_ERROR_INTERVAL_S:
            raise HTTPException(429, "one test error per minute")
        _last_test_error[0] = now
        raise RuntimeError("Deliberate test error (PLAN.md 1.2)")

    # Everything except the health checks needs a signed-in user (api/auth.py)
    # and counts against that user's limits (api/limits.py). Applying both
    # here, not endpoint by endpoint, means a new route is protected by
    # default -- forgetting is impossible rather than unlikely.
    for module in (deal, deals, montecarlo, forecasting, backtesting, integrations, export, account,
                   jobs):
        app.include_router(module.router, prefix="/api",
                           dependencies=[Depends(require_user), Depends(enforce_user_limits)],
                           responses={429: {"model": LimitRefusal,
                                            "description": "A usage limit was reached"}})
    # The scheduler's endpoints take a GitHub Actions token instead of a user
    # (api/github_oidc.py); the per-address limits still apply
    app.include_router(scheduled.router, prefix="/api", dependencies=[Depends(require_workflow)])
    return app


JOBS_HEALTH_CACHE_S = 60.0
_jobs_health: list = [0.0, None]


def jobs_health() -> dict:
    from jobs import config
    from jobs import scheduled as scheduled_tasks

    now = time.monotonic()
    if _jobs_health[1] is not None and now - _jobs_health[0] < JOBS_HEALTH_CACHE_S:
        return {**_jobs_health[1], "runner_alive": config.runner_alive()}
    out = {"status": "ok", "queue": config.queue_mode(), "runner": config.runner_mode(),
           "runner_alive": config.runner_alive(), "checked_at": utc_now_iso()}
    try:
        out["counts"] = config.get_queue().counts()
        out["scheduled"] = {name: {k: run[k] for k in ("status", "trigger", "workflow", "github_run_id",
                                                        "started_at", "finished_at")}
                            for name, run in scheduled_tasks.latest_runs().items()}             if config.queue_mode() == "database" else {}
    except DatabaseUnavailable:
        out.update(status="error", detail="The database is unavailable.")
    _jobs_health[:] = [now, out]
    return out


app = create_app()
