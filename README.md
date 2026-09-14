---
title: Simulation Model
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---
# Simulation Model — LBO analysis platform

Deal wizard, Monte Carlo simulation, backtesting against historical deals,
3-statement company forecasting (with SEC EDGAR autofill) and settings.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## API

The model is also served over HTTP for the new web frontend. The logic lives
in `core/` (shared with the Streamlit app, no Streamlit dependency) and
`api/` exposes it:

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive docs: http://localhost:8000/api/docs. The OpenAPI schema at
`/api/openapi.json` is what the frontend's typed client is generated from.
Allowed browser origins come from `FSE_CORS_ORIGINS` (comma-separated,
default `http://localhost:3000`).

| Area | Endpoints |
|---|---|
| Deal | `POST /api/deal/run`, `POST /api/deal/sources-and-uses` |
| Monte Carlo | `POST /api/montecarlo/run`, `POST /api/montecarlo/scenarios` |
| Forecasting | `GET /api/forecasting/defaults`, `POST /api/forecasting/seed`, `POST /api/forecasting/run` |
| Backtesting | `GET /api/backtesting/deals`, `POST /api/backtesting/run` |
| Settings | `GET /api/settings/defaults`, `POST /api/settings/validate` |
| Optional | `GET /api/capabilities`, `POST /api/ml/deal-risk`, `POST /api/ml/surrogate`, `POST /api/ml/macro-regime`, `GET /api/edgar/{ticker}` |

Inputs use the model's units: percentages as numbers like `60.0`, money in $M.
Optional features return `503` with the reason when they are not installed or
trained.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Optional ML layer (`ml/`)

The app runs without it. Each feature is behind a guarded import and appears
only when its dependencies and trained model files are present:

| Feature | Page | Enable |
|---|---|---|
| Deal risk score (anomaly detector) | Deal inputs | `pip install -r requirements-ml.txt` (model files are committed) |
| Live IRR sliders (surrogate network) | Monte Carlo | install as above (model files are committed); retrain with `python -m ml.surrogate.generate_data && python -m ml.surrogate.train` after changing the simulation engine or its default fees |
| Macro regime detection | Monte Carlo | install as above, set `FRED_API_KEY`, then `python -m ml.macro_regime` |

The surrogate is trained on a fixed deal (10x entry, 5-year hold, default
fees and operating terms — see `TRAINING_FIXED` in
`ml/surrogate/generate_data.py`); the page warns when a deal differs, and
RUN SIMULATION is always exact. Measured against fresh simulations on 200
random deals, median-IRR error is 0.19pp (worst 0.86pp); near a 5% wipeout
rate the 5th percentile is unreliable, so the page hides it there. `requirements-ml.txt` pins scikit-learn to
1.8 because the committed anomaly-detector models were saved with 1.8.0.
