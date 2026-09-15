"""FastAPI application serving the LBO model to the web frontend.

    uvicorn api.main:app --reload

Interactive docs at /api/docs; the OpenAPI schema at /api/openapi.json is what
the frontend's typed client is generated from.
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from api.routers import backtesting, deal, export, forecasting, integrations, montecarlo

API_VERSION = "0.1.0"


def create_app() -> FastAPI:
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
    app.add_middleware(CORSMiddleware, allow_origins=origins,
                       allow_methods=["GET", "POST"], allow_headers=["*"],
                       expose_headers=["Content-Disposition"])
    # Simulation responses carry chart data (histograms, scatter samples)
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    @app.get("/api/health", tags=["meta"])
    def health():
        return {"status": "ok", "version": API_VERSION}

    for module in (deal, montecarlo, forecasting, backtesting, integrations, export):
        app.include_router(module.router, prefix="/api")
    return app


app = create_app()
