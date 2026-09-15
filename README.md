# FSE/ML — LBO analysis platform

Deal model, Monte Carlo simulation, backtesting against historical LBOs,
3-statement company forecasting (with SEC EDGAR autofill) and settings.

- `web/` — Next.js app (the interface)
- `api/` — FastAPI app serving the model
- `core/`, `lbo_engine/`, `simulation/`, `analytics/` — the model
- `ml/` — optional ML features

Deployment (Vercel + Render): see [DEPLOY.md](DEPLOY.md).

## Run locally

Python 3.12 and Node 24.

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000      # API; docs at http://localhost:8000/api/docs

cd web
npm ci
npm run dev                                    # http://localhost:3000
```

The web app proxies `/api` to `FSE_API_URL` (default `http://127.0.0.1:8000`).

## API

| Area | Endpoints |
|---|---|
| Deal | `POST /api/deal/run`, `POST /api/deal/sources-and-uses` |
| Monte Carlo | `POST /api/montecarlo/run`, `POST /api/montecarlo/scenarios` |
| Forecasting | `GET /api/forecasting/defaults`, `POST /api/forecasting/seed`, `POST /api/forecasting/run` |
| Backtesting | `GET /api/backtesting/deals`, `POST /api/backtesting/run` |
| Settings | `GET /api/settings/defaults`, `POST /api/settings/validate` |
| Export | `POST /api/export/workbook`, `POST /api/export/montecarlo-sample` (Excel) |
| Optional | `GET /api/capabilities`, `POST /api/ml/deal-risk`, `POST /api/ml/surrogate`, `POST /api/ml/macro-regime`, `GET /api/edgar/{ticker}` |

Inputs use the model's units: percentages as numbers like `60.0`, money in $M.
Engine outputs keep engine units (IRR `0.157` = 15.7%). Optional features
return `503` with the reason when they are not installed or trained. Allowed
browser origins for direct API calls come from `FSE_CORS_ORIGINS`
(comma-separated, default `http://localhost:3000`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest                                         # model, API, parity with the golden snapshot

cd web
npm run build
npx playwright test                            # browser tests; they start the API and web app
```

CI runs `core`, `ml`, `web`, `e2e` and `docker` jobs.

## Optional ML layer (`ml/`)

The app runs without it; each feature appears only when the API can run it.

| Feature | Where | Enable |
|---|---|---|
| Deal risk score (anomaly detector) | Deal inputs | `pip install -r requirements-ml.txt` (model files are committed) |
| Live IRR sliders (surrogate network) | Monte Carlo → Live | as above; retrain with `python -m ml.surrogate.generate_data && python -m ml.surrogate.train` after changing the simulation engine or its default fees |
| Macro regime detection | Monte Carlo → Scenarios | as above, set `FRED_API_KEY`, then `python -m ml.macro_regime` |

The surrogate is trained on a fixed deal (10x entry, 5-year hold, default fees
and operating terms — see `TRAINING_FIXED` in `ml/surrogate/generate_data.py`);
the Live screen says when a deal differs. Measured against fresh simulations
on 200 random deals, median-IRR error is 0.19pp (worst 0.86pp); near a 5%
wipeout rate the 5th percentile is unreliable, so the screen hides it there.
`requirements-ml.txt` pins scikit-learn to 1.8 because the committed
anomaly-detector models were saved with 1.8.0.

## History

The first version was a Streamlit app, retired once the web app matched it.
`tests/golden/golden.json` is a snapshot of that app's outputs, still used as
the parity baseline; each deliberate model change since is pinned in
`tests/test_model_fixes.py`.
