"""What a job can be: the long runs, each the same code as its endpoint.

A job of kind ``montecarlo.run`` runs exactly what ``POST
/api/montecarlo/run`` runs -- the same function, with the same request model
and the same response model -- so a seeded result is identical whichever way
it was asked for (``tests/test_jobs.py`` checks it). The endpoint's
one-at-a-time wrapper is left out: the runner takes the same simulation slot
itself, and waits for it instead of giving up after 30 seconds.

Progress comes from the ``model_timer`` sections the run already has: when
one starts, the job's stage changes; when it ends, its progress moves on.

To add a kind: register it here with its request and response models and the
timer names its run passes through. Quick runs (the deal model) stay plain
endpoints: a job only pays off when the wait is long.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Optional

from pydantic import BaseModel, TypeAdapter

from api.routers import backtesting, forecasting, montecarlo
from api.schemas import (
    BacktestRequest, BacktestResponse, ForecastRunRequest, ForecastRunResponse,
    DocumentReadResult, MemoDrafted, MonteCarloRequest, MonteCarloResponse, ReadResult, ScenariosRequest, ScenariosResponse,
)


@dataclass(frozen=True)
class Stage:
    started: str            # the stage shown while this model section runs
    done: float             # progress once it ends
    after: str              # the stage shown after it


@dataclass(frozen=True)
class JobKind:
    name: str
    title: str
    request: type[BaseModel]
    response: type[BaseModel]
    run: Callable[[Any], Any]
    stages: dict[str, Stage] = field(default_factory=dict)
    # Takes the one simulation slot while it runs (api/limits.py): the slot is
    # "one memory-heavy thing at a time", which a file scan is too
    simulation: bool = True

    def parse(self, payload: dict) -> BaseModel:
        return self.request.model_validate(payload)

    def execute(self, request: BaseModel) -> Any:
        """Run it and return the JSON the endpoint would have answered with."""
        adapter = TypeAdapter(self.response)
        return adapter.dump_python(adapter.validate_python(self.run(request)), mode="json")


def _unwrapped(endpoint):
    # simulation_slot wraps with functools.wraps, which keeps the original here
    return getattr(endpoint, "__wrapped__", endpoint)


class ReadRequest(BaseModel):
    owner: str
    file_id: uuid.UUID
    quarter_end: Optional[date] = None
    company_id: Optional[uuid.UUID] = None


def _read_results(request: ReadRequest) -> dict:
    from research.reader import service

    return service.read_file(request.owner, request.file_id, request.quarter_end, company_id=request.company_id)


class DocumentRequest(BaseModel):
    owner: str
    file_id: uuid.UUID


def _read_document(request: DocumentRequest) -> dict:
    from ml import document_reader

    return document_reader.extract_file(request.owner, request.file_id)


class MemoRequest(BaseModel):
    owner: str
    deal_id: uuid.UUID
    version: int


def _draft_memo(request: MemoRequest) -> dict:
    from ml import memo

    return memo.draft(request.owner, request.deal_id, request.version)


class ScanRequest(BaseModel):
    file_id: uuid.UUID


class ScanOutcome(BaseModel):
    status: str
    reason: str | None = None


def _scan_upload(request: ScanRequest) -> dict:
    from uploads import service

    return service.scan_file(request.file_id)


KINDS: dict[str, JobKind] = {k.name: k for k in (
    JobKind("montecarlo.run", "Monte Carlo simulation", MonteCarloRequest, MonteCarloResponse,
            _unwrapped(montecarlo.post_run),
            {"montecarlo.run": Stage("Simulating paths", 0.8, "Summarising results")}),
    JobKind("montecarlo.scenarios", "Scenario simulations", ScenariosRequest, ScenariosResponse,
            _unwrapped(montecarlo.post_scenarios),
            {"montecarlo.scenarios": Stage("Simulating four scenarios", 0.9, "Summarising results")}),
    JobKind("backtesting.run", "Backtest", BacktestRequest, BacktestResponse,
            _unwrapped(backtesting.post_run),
            {"backtest.run": Stage("Running the backtest", 0.9, "Summarising results")}),
    JobKind("forecasting.run", "Forecast", ForecastRunRequest, ForecastRunResponse,
            _unwrapped(forecasting.post_run),
            {"forecast.run": Stage("Running the three statements", 0.2, "Preparing the simulation"),
             "forecast.simulation": Stage("Simulating the forecast", 0.9, "Summarising results")}),
    # Queued by the API when an upload arrives (uploads/service.py), never by a
    # caller: it is not in api.schemas.JobSubmit, so /api/jobs refuses it
    JobKind("upload.scan", "File scan", ScanRequest, ScanOutcome, _scan_upload),
    # Queued by POST /api/research/reads under the caller's name, never through
    # /api/jobs (not in JobSubmit): its endpoint checks the file, the
    # confirmation and the limits first. Mostly a wait on the AI provider, with
    # a few tens of megabytes at most held, so it takes no simulation slot
    JobKind("research.read", "Results read", ReadRequest, ReadResult, _read_results,
            {"reader.read_a": Stage("Reading the whole filing", 0.55, "Reading the statement pages again"),
             "reader.read_b": Stage("Reading the statement pages again", 0.95, "Checking the figures")},
            simulation=False),
    # Queued by POST /api/documents/reads (PLAN.md 6.2), never through /api/jobs, for the same reasons
    JobKind("document.extract", "Document read", DocumentRequest, DocumentReadResult, _read_document,
            {"document.read_a": Stage("Reading the whole document", 0.6, "Reading the cited pages again"),
             "document.read_b": Stage("Reading the cited pages again", 0.95, "Checking every quote")},
            simulation=False),
    # Queued by POST /api/deals/{id}/memos (PLAN.md 6.3), never through /api/jobs: its endpoint checks the
    # version is the caller's first. Three simulations of the deal, so it takes the simulation slot
    JobKind("memo.draft", "Investment memo", MemoRequest, MemoDrafted, _draft_memo,
            {"montecarlo.run": Stage("Simulating the deal", 0.5, "Simulating the scenarios"),
             "montecarlo.scenarios": Stage("Simulating the scenarios", 0.85, "Setting the plan against actual"),
             "backtest.plan_vs_actual": Stage("Setting the plan against actual", 0.95, "Keeping the memo")}),
)}
