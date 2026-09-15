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

Last updated: 2026-09-15, step 1 agreed and step 3 (web shell) in PR on
`feat/web-shell`.

### Main goal: frontend rebuild (Streamlit → FastAPI + Next.js)

The user wants a full, scalable, "anti-slop" frontend rebuilt from scratch, with
**the model's logic kept as is**. Streamlit is retired at the end.

| Step | Status |
|---|---|
| 1. Design direction — mockups of key screens for user approval | ✅ Agreed — see "Design system" below |
| 2. API layer — `core/` + `api/` | ✅ Done (PR #2) |
| 3. App shell — Next.js in `web/`, design system, layout, navigation, TS client generated from `/api/openapi.json` | ✅ Built on `feat/web-shell` (PR open). Every step route exists and says "Not built yet" |
| 4. Rebuild the five screens: deal wizard, Monte Carlo, backtesting, forecasting, settings | **Next.** Replace `StepPending` per step, matching the final-look mockup |
| 5. Browser tests in CI proving every control changes its output (Playwright) | Not started |
| 6. Deploy (frontend + API) and retire Streamlit | Not started — needs the user's hosting choice |

Agreed stack: **Next.js (App Router) + TypeScript + Tailwind + Radix + Motion**
for `web/`; FastAPI for `api/`; one repo.

### Design system (agreed with the user in step 1 — don't reopen)

Reference mockup of all five screens with real API numbers:
https://claude.ai/artifact/ALQqM9gaEQxs14oi2Devwg (private to the user).
Earlier rounds: directions, navigation options, mixes, type weights.

- **Look: "Tape".** Dark, dense workstation. Canvas `#0C1012`, panel `#12181B`,
  line `#222B30`, ink `#D5DDE1`, accent cyan `#62B6CB`; gain `#58B28A`,
  loss `#D8665A`, attention amber `#D9A54A`. Square corners, 1px hairline
  tile grids, no cards. Motion only for state changes. Tokens live in
  `web/src/app/globals.css`.
- **Type:** the user wanted Microgramma / Eurostile Extended. Free stand-ins:
  **Michroma** (text, labels) and **Orbitron** (weights 500–900, structure);
  **JetBrains Mono** for every figure so columns align.
  **Zoned** hierarchy — each area reads differently: top menu Orbitron heavy
  wide caps; steps light Michroma sentence case with underline; **inputs use
  Michroma thickened with `-webkit-text-stroke`** (group 0.45px, label 0.2px);
  result titles Orbitron 700 bright; alerts Orbitron 800 amber; actions 900.
  Use the `type-*` classes, not ad-hoc font styles.
- **Charts:** categories Michroma 8 caps muted; ticks mono dim; values mono
  ink, totals bright; reference lines Orbitron 700 caps (amber for thresholds);
  cyan = the answer, green/red = gain/loss, grey = context.
- **Navigation: "Guided" (mix 1).** Mode tabs across the top, a step row
  below, Ctrl K search on the right. No Run button in the top bar. The deal
  model reruns automatically on edit; Monte Carlo is marked **stale** (tab
  chip + dimmed tiles) and rerun from a changes bar above the results.
  Shortcuts: Ctrl K, Alt 1–5, `[` `]`.
- Screens show open model findings as visible markers rather than hiding them.

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

- Whether to fix finding 1 (minimum cash) before or during step 4.
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
| `web/` | Next.js 16 frontend. `src/lib/nav.ts` lists every mode and step (drives tabs, steps, search, routes); `src/components/shell/` is the Guided shell; `src/app/[mode]/[step]/` the screens; `src/lib/api/` the typed client |
| `web/openapi.json` | Snapshot of the API schema; `src/lib/api/schema.d.ts` is generated from it |
| `app.py`, `pages/` | Streamlit app (to be retired). Pages call `core/` |
| `tests/golden/` | Snapshot of the Streamlit app's outputs taken **before** logic moved into `core/`; the parity baseline |
| `tests/`, `test_*.py` | Test suite (68 tests) |

## Commands (Windows, from the repo root)

```bash
.venv/Scripts/python.exe -m pytest                       # all tests
.venv/Scripts/python.exe -m uvicorn api.main:app --reload --port 8000   # API; docs at /api/docs
.venv/Scripts/python.exe -m streamlit run app.py         # legacy Streamlit app

# web/ (run the API too; Next proxies /api to FSE_API_URL, default 127.0.0.1:8000)
npm --prefix web run dev                                 # http://localhost:3000
npm --prefix web run lint
npm --prefix web run typecheck
npm --prefix web run build

# After changing API schemas: refresh the snapshot, then the TS types
.venv/Scripts/python.exe -m api.export_openapi web/openapi.json
npm --prefix web run api:types
```

CI (`.github/workflows/tests.yml`) runs `core`, `ml` and `web` jobs.
`tests/test_openapi_snapshot.py` fails when `web/openapi.json` is stale; the
`web` job fails when `schema.d.ts` doesn't match the snapshot.

Setup on a fresh machine: Python 3.12, then
`pip install -r requirements-ml.txt -r requirements-dev.txt` (or just
`requirements-dev.txt` without ML). Trained model files are committed.

## Working rules

- **`main` is protected.** Every change: branch → PR → CI (`core`, `ml`, `web` jobs)
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
- **Next.js 16 differs from older versions.** Read `web/node_modules/next/dist/docs/`
  before using an API (e.g. `params` is a Promise; `PageProps`/`LayoutProps`
  are global types generated by `next typegen`). `web/AGENTS.md` is
  re-created by `next dev`; keep it committed.
- Components that use context or Motion (`MotionConfig`, `motion.*`) must be
  client components; the shell keeps them in `workspace.tsx` and the bars.
- Browser-automation key presses: send `Enter` and `]`, not `Return` or
  `bracketright`, or shortcuts appear broken when they aren't.
- Deal defaults differ: the API uses stored 60% debt / 70% senior; the
  Streamlit wizard derives 42% / ~81% from its 3.4x + 0.8x debt multiples.
