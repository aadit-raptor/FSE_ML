<picture>
  <source media="(prefers-color-scheme: dark)" srcset="web/public/brand/logo-horizontal-dark.png">
  <img alt="Variater" src="web/public/brand/logo-horizontal-light.png" width="320">
</picture>

# Variater — LBO analysis platform

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
| Optional | `GET /api/capabilities`, `POST /api/ml/deal-risk`, `POST /api/montecarlo/live`, `POST /api/ml/macro-regime`, `GET /api/edgar/{ticker}` |

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

## Models (`ml/`)

Nothing here needs ML packages or trained files: every model reads stored,
sourced data or reruns the simulation, and each has a card in
[docs/model-cards/](docs/model-cards/) that CI re-evaluates.

The Live sliders (Monte Carlo → Live) rerun the simulation itself on up to
8,000 fixed paths per move (`core/live.py`), so the estimate is the deal's
own, whatever its facilities, tax rules or leases; on 230 varied regional
deals its median IRR is 0.13 points from a 100,000-path run
([docs/model-cards/surrogate.md](docs/model-cards/surrogate.md)).

The deal risk score (Deal → Inputs) since PLAN.md 5.2
compares the deal with its industry's companies in its region, from the stored
industry averages ([docs/model-cards/deal_risk.md](docs/model-cards/deal_risk.md)).

## History

The first version was a Streamlit app, retired once the web app matched it.
`tests/golden/golden.json` is a snapshot of that app's outputs, still used as
the parity baseline; each deliberate model change since is pinned in
`tests/test_model_fixes.py`.
