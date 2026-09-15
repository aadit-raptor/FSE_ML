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

Last updated: 2026-09-15. Work is on stacked branches, each containing the
previous: `feat/remaining-screens` (steps 3–4, PR #4 open) →
`feat/e2e-tests` (step 5) → `feat/model-fixes` (findings fixed).
`feat/model-fixes` holds everything; merging it covers all of them.

### Main goal: frontend rebuild (Streamlit → FastAPI + Next.js)

The user wants a full, scalable, "anti-slop" frontend rebuilt from scratch, with
the model's logic kept as is (the user later approved fixing the findings
below). Streamlit is retired at the end.

| Step | Status |
|---|---|
| 1. Design direction — mockups of key screens for user approval | ✅ Agreed — see "Design system" below |
| 2. API layer — `core/` + `api/` | ✅ Done (PR #2) |
| 3. App shell — Next.js in `web/`, design system, layout, navigation, TS client generated from `/api/openapi.json` | ✅ Built (`feat/web-shell`) |
| 4. Rebuild the five screens: deal wizard, Monte Carlo, backtesting, forecasting, settings | ✅ All five built and verified by output in the browser (`feat/deal-screens`, `feat/remaining-screens`). Not yet merged |
| 5. Browser tests in CI proving every control changes its output (Playwright) | ✅ `web/e2e/` (27 tests, CI job `e2e`). Mutation-checked |
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

### Model findings

Fixed on `feat/model-fixes` (user said "you do it all", 2026-09-15). Each fix
has a test in `tests/test_model_fixes.py` that fails on the old code; the
golden snapshot is untouched and parity tests explain every departure.

1. ✅ **Minimum cash** is now funded by sponsor equity (`run_lbo`) and listed
   in sources and uses; the bridge residual is 0. Parity: `assert_min_cash_funded`.
2. ✅ **Settings the model ignored** — `def_senior_amort`, `mc_clip_irr`,
   `sens_*` — are read by the deal model, simulation and backtest.
3. ✅ **Monte Carlo heatmap** cells are full `run_lbo` runs at the
   simulation's mean assumptions (was a fee-free closed form).
4. ✅ **Forecast target labels**: above plan is Bull, below plan Bear.
5. ✅ **Backtest** predicted exit values come from a deal-model run (was exit
   debt = 75% of entry debt); attribution is an exact split of the exit
   equity gap into exit EBITDA, exit multiple and net debt.
6. **Streamlit backtesting page only ever runs Burger King's inputs.**
   Streamlit-only, not fixed; the web app is unaffected. Moot once Streamlit
   is retired.
7. ✅ **Exit sensitivity grid**: each hold column is a full run for that hold;
   ranges come from `sens_*` (default rows now 6–12x, not 0.6–1.4x of exit).
8. ✅ **Interest circularity** iterates to $1k (`core/deal.py`
   `MAX_INTEREST_PASSES`/`INTEREST_TOLERANCE`). `test_core_parity` runs the
   engine with the snapshot's 3 passes to stay exact; `test_api` compares the
   API with the converged core run.
9. ✅ **Seeded Monte Carlo** uses a local `RandomState(seed)`, identical
   stream to the old global seed, safe under concurrent requests.

### Waiting on the user

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
| `web/` | Next.js 16 frontend. `src/lib/nav.ts` lists every mode and step (tabs, step row, search). `src/app/<mode>/<step>/page.tsx` are thin route files; screens live in `src/components/<mode>/`. State per mode sits in a provider mounted in `components/shell/AppShell.tsx` (Settings → Deal → Monte Carlo → Backtest → Forecast), so it survives mode switches. Settings overrides persist in localStorage and go into every run. `components/charts/` and `components/ui/` are shared; `src/lib/api/` the typed client |
| `web/openapi.json` | Snapshot of the API schema; `src/lib/api/schema.d.ts` is generated from it |
| `app.py`, `pages/` | Streamlit app (to be retired). Pages call `core/` |
| `tests/golden/` | Snapshot of the Streamlit app's outputs taken **before** logic moved into `core/`; the parity baseline |
| `tests/`, `test_*.py` | Test suite (85 tests); `tests/test_model_fixes.py` pins each finding fix |

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

Browser tests (`web/e2e/`, Playwright). They start uvicorn and `next start`
themselves, so build first. No bundled browser on this machine: use Edge.

```bash
npm --prefix web run build
PW_CHANNEL=msedge npm --prefix web run test:e2e          # PowerShell: $env:PW_CHANNEL="msedge"
```

Tests assert real model output (IRR, MOIC, golden backtest values), not just
rendering. Add one for every new control, and mutation-check it.

CI (`.github/workflows/tests.yml`) runs `core`, `ml`, `web` and `e2e` jobs.
`tests/test_openapi_snapshot.py` fails when `web/openapi.json` is stale; the
`web` job fails when `schema.d.ts` doesn't match the snapshot.

Setup on a fresh machine: Python 3.12, then
`pip install -r requirements-ml.txt -r requirements-dev.txt` (or just
`requirements-dev.txt` without ML). Trained model files are committed.

## Working rules

- **`main` is protected.** Every change: branch → PR → CI (`core`, `ml`, `web`, `e2e` jobs)
  green → merge with **"Create a merge commit"**. Direct pushes to `main` fail.
  The GitHub CLI is not installed; open and merge PRs through the browser.
- **Keep model logic as is** unless the user approves a change. Record new
  findings in this file instead of silently fixing them.
- **Parity:** `tests/test_core_parity.py` and `tests/test_api.py` pin results to
  `tests/golden/golden.json` (1e-9 relative tolerance — bit-for-bit locally,
  but Linux CI numpy can differ in the last bits). Never regenerate the golden
  file to make a test pass; a deliberate logic change updates tests explicitly
  (see how findings 1, 5, 7 and 8 did it).
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
- **Drive C: on the original machine is nearly full.** It hit 0 bytes during
  step 4 (npm cache, `.next`, an old scratch venv were cleared, leaving ~1 GB).
  Check `Get-PSDrive C` before builds; "No space left on device" / npm
  `nospc` errors mean this, not a code problem.
- Playwright: open the app at `localhost`, not `127.0.0.1` (Next's dev server
  blocks its client scripts for other hosts; the page never hydrates). Next.js
  renders a hidden `role="alert"` route announcer, so scope alert locators to
  `#content`. `.next/dev` grows to ~400 MB; delete it when disk is tight.
- The Browser pane's screenshots time out when the Claude window isn't drawn;
  verify with `javascript_tool` / `find` / `form_input` instead.
- Browser-automation key presses: send `Enter` and `]`, not `Return` or
  `bracketright`, or shortcuts appear broken when they aren't.
- Deal defaults differ: the API uses stored 60% debt / 70% senior; the
  Streamlit wizard derives 42% / ~81% from its 3.4x + 0.8x debt multiples.
