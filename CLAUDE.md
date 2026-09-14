# CLAUDE.md — FSE_ML

LBO modelling platform: deal wizard, Monte Carlo simulation, backtesting against
historical deals, 3-statement forecasting (with SEC EDGAR autofill) and settings.
Currently a Streamlit app being rebuilt as a FastAPI backend + Next.js frontend.

**Keep this file current.** It is how a new session (or a new device) knows where
the project stands. When you finish a step, fix a finding, or make a decision
the user should not have to repeat, update the relevant section **in the same
PR** as the work.

---

## Current status — read this first

Last updated: 2026-09-15, after PR #2 merged (`afa256c`).

### Main goal: frontend rebuild (Streamlit → FastAPI + Next.js)

The user wants a full, scalable, "anti-slop" frontend rebuilt from scratch, with
**the model's logic kept as is**. Streamlit is retired at the end.

| Step | Status |
|---|---|
| 1. Design direction — mockups of key screens for user approval | **Next.** Use the design skills (see Tooling). Needs the user's call on look & feel |
| 2. API layer — `core/` + `api/` | ✅ Done (PR #2) |
| 3. App shell — Next.js in `web/`, design system, layout, navigation, TS client generated from `/api/openapi.json` | Not started |
| 4. Rebuild the five screens: deal wizard, Monte Carlo, backtesting, forecasting, settings | Not started |
| 5. Browser tests in CI proving every control changes its output (Playwright) | Not started |
| 6. Deploy (frontend + API) and retire Streamlit | Not started — needs the user's hosting choice |

Agreed stack: **Next.js (App Router) + TypeScript + Tailwind + Radix + Motion**
for `web/`; FastAPI for `api/`; one repo.

Suggested design direction, not yet confirmed by the user: a data-dense
financial tool — high visual density, moderate boldness, light motion, calm
references (e.g. Linear, Stripe). Confirm before building.

### Model findings — recorded, NOT fixed (need the user's go-ahead)

1. **Minimum cash inflates returns.** The debt model opens with the minimum cash
   balance, but sponsor equity doesn't fund it, so the equity bridge residual
   equals the minimum cash and IRR is overstated when it is > 0. Most important.
   `tests/test_api.py::test_deal_run_matches_streamlit_snapshot` documents this;
   a fix must update that test deliberately.
2. **Settings that do nothing:** `mc_*` defaults, `def_senior_amort`,
   `mc_clip_irr`, sensitivity ranges (`sens_*`).
3. **Monte Carlo heatmap** (`core/montecarlo.py::growth_exit_heatmap`) is a
   fee-free closed-form approximation, not the simulation.
4. **Forecast target labels look inverted** (above deterministic EBITDA is
   labelled "Bear") — `core/forecasting.py::simulation_summary`.
5. **Backtest error attribution** uses arbitrary factors (0.3, 0.1, 0.05, 0.75).
6. **Streamlit backtesting page only ever runs Burger King's inputs** (keyed
   widgets keep the first deal's values). Not worth fixing — the page is being
   replaced; the API is unaffected.

### Waiting on the user

- Look & feel decision for step 1.
- A FRED API key (`FRED_API_KEY`) to enable macro regime detection.
- Whether to wire up the unused `ml/` modules (distress model, SHAP drivers,
  multiple predictor, growth calibrator, NLP extractor, correlation updater,
  personalization). SHAP and distress need no keys.

---

## Repo layout

| Path | What |
|---|---|
| `core/` | **All model logic**, no Streamlit dependency. Functions take inputs and settings explicitly. Shared by Streamlit and the API |
| `api/` | FastAPI app (`api/main.py`), routers per mode, schemas generated from engine dataclasses |
| `lbo_engine/` | Deterministic LBO engine (operating model, cash flow, debt, returns) |
| `simulation/vectorized_simulation.py` | Vectorized Monte Carlo engine |
| `analytics/` | Risk metrics |
| `ml/` | Optional ML: anomaly detector, surrogate network, macro regime, EDGAR extractor, plus unused modules |
| `app.py`, `pages/` | Streamlit app (to be retired). Pages call `core/` |
| `tests/golden/` | Snapshot of the Streamlit app's outputs taken **before** logic moved into `core/`; the parity baseline |
| `tests/`, `test_*.py` | Test suite (67 tests) |

## Commands (Windows, from the repo root)

```bash
.venv/Scripts/python.exe -m pytest                       # all tests
.venv/Scripts/python.exe -m uvicorn api.main:app --reload --port 8000   # API; docs at /api/docs
.venv/Scripts/python.exe -m streamlit run app.py         # legacy Streamlit app
```

Setup on a fresh machine: Python 3.12, then
`pip install -r requirements-ml.txt -r requirements-dev.txt` (or just
`requirements-dev.txt` without ML). Trained model files are committed.

## Working rules

- **`main` is protected.** Every change: branch → PR → CI (`core` and `ml` jobs)
  green → merge with **"Create a merge commit"**. Direct pushes to `main` fail.
  The GitHub CLI is not installed; open and merge PRs through the browser.
- **Keep model logic as is** unless the user approves a change. Record findings
  in this file instead of silently fixing them.
- **Parity:** `tests/test_core_parity.py` and `tests/test_api.py` pin results to
  `tests/golden/golden.json` (1e-9 relative tolerance — bit-for-bit locally,
  but Linux CI numpy can differ in the last bits). Never regenerate the golden
  file to make a test pass; a deliberate logic change updates tests explicitly.
- **Verify by output, not by render.** Several past bugs were controls that
  rendered fine and did nothing (scenario presets, WSP toggle, EDGAR, fee
  settings). Check that changing an input changes the result.
- **Mutation-check new tests**: break the logic on purpose, confirm the test fails.
- **Units:** API inputs use percentages as numbers (`60.0`) and money in $M;
  engine outputs keep engine units (IRR `0.157` = 15.7%).
- **Optional ML** stays behind guarded, lazy imports; the app and CI's `core`
  job must work without ML packages.
- Commit messages explain *why* and how it was verified.

## Tooling

Installed at **user level on the original Windows machine** — on another device,
reinstall:

| Tool | Install | Notes |
|---|---|---|
| `design-taste-frontend` (+ companions) | `npx skills add https://github.com/Leonxlnx/taste-skill` | Dials: `DESIGN_VARIANCE`, `MOTION_INTENSITY`, `VISUAL_DENSITY` |
| `web-design-guidelines` | `npx skills add vercel-labs/agent-skills --skill web-design-guidelines` | Fetches Vercel's guidelines from GitHub each run |
| `image-to-code` | `npx skills add https://github.com/Leonxlnx/taste-skill --skill image-to-code` | Written for Codex; expects to generate images, which Claude Code can't |
| Playwright CLI | `npm install -g @playwright/cli@latest` then `playwright-cli install --skills --global` | No Chrome on the original machine: use `--browser=msedge`. Writes to `.playwright-cli/` |
| awesome-design-md | `git clone https://github.com/VoltAgent/awesome-design-md` | 74 brand `DESIGN.md` files — inspiration only, don't clone a real brand's identity |

Skills load when a session starts: install first, then open a new session.

## Gotchas

- Node may be installed but missing from PATH in an older session:
  it lives at `C:\Program Files\nodejs`.
- The Windows console is cp1252 — printing `≥`, `→` etc. from Python fails;
  write to a file or use ASCII.
- Streamlit keyed widgets keep their value when their `value=` changes; to
  re-seed, assign through `st.session_state`.
- `tests/golden/generate_golden.py` drives the Streamlit app headlessly; run it
  against `main` in a separate worktree, never to "fix" a failing test.
- Deal defaults differ: the API uses stored 60% debt / 70% senior; the
  Streamlit wizard derives 42% / ~81% from its 3.4x + 0.8x debt multiples.
