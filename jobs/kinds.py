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

from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, TypeAdapter

from api.routers import backtesting, forecasting, montecarlo
from api.schemas import (
    BacktestRequest, BacktestResponse, ForecastRunRequest, ForecastRunResponse,
    MonteCarloRequest, MonteCarloResponse, ScenariosRequest, ScenariosResponse,
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
    # Takes the one simulation slot while it runs (api/limits.py)
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
)}
