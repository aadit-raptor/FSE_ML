"""FastAPI application serving the LBO model to the web frontend.

    uvicorn api.main:app --reload

Interactive docs at /api/docs; the OpenAPI schema at /api/openapi.json is what
the frontend's typed client is generated from.
"""
import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.observability import (
    REQUEST_ID_HEADER, RequestContextMiddleware, configure_logging, init_sentry, utc_now_iso,
)
from api.routers import backtesting, deal, export, forecasting, integrations, montecarlo

API_VERSION = "0.1.0"


def deploy_environment() -> str:
    """Which copy of the API this is: production, staging or local.

    FSE_ENV wins when set. Otherwise Render's RENDER_SERVICE_NAME decides:
    a service whose name ends in -staging (fse-api-staging) is staging, any
    other Render service is production. Read per request so tests can set it.
    """
    explicit = os.environ.get("FSE_ENV", "").strip().lower()
    if explicit:
        return explicit
    service = os.environ.get("RENDER_SERVICE_NAME", "")
    if not service:
        return "local"
    return "staging" if service.endswith("-staging") else "production"


# /api/debug/error: at most one deliberate error per this many seconds, so a
# public URL can't burn the free Sentry event allowance
TEST_ERROR_INTERVAL_S = 60.0
_last_test_error = [0.0]


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
    )

    # Browser origins allowed to call the API, comma-separated.
    origins = [o.strip() for o in os.environ.get(
        "FSE_CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
    # Innermost of the three, so its 500 responses still get CORS headers
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"],
                       expose_headers=["Content-Disposition", REQUEST_ID_HEADER, "Server-Timing"])
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
                "time": utc_now_iso()}

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

    for module in (deal, montecarlo, forecasting, backtesting, integrations, export):
        app.include_router(module.router, prefix="/api")
    return app


app = create_app()
