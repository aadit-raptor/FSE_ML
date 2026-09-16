# CLAUDE.md — FSE_ML

LBO modelling platform: deal wizard, Monte Carlo simulation, backtesting against
historical deals, 3-statement forecasting (with SEC EDGAR autofill) and settings.
A FastAPI backend (`api/`) and a Next.js frontend (`web/`); the original
Streamlit app is retired. Deployed on Render (API) and Vercel (web): see DEPLOY.md.

**Keep this file current.** It is how a new session (or a new device) knows where
the project stands. When you finish a step, fix a finding, or make a decision
the user should not have to repeat, update the relevant section **in the same
PR** as the work.

---

## Current status — read this first

Last updated: 2026-09-16 (PLAN.md 1.3). Steps 1–5 and the model finding fixes are merged
(PR #5). Step 6 is on `feat/deploy`: Streamlit parity (Excel downloads,
schedules, ML panels), Streamlit removed, and deploy config for the user's
choice of **Vercel (web) + Render (API)**. The user must create the accounts
and connect the repo themselves (DEPLOY.md); Claude can't create accounts or
sign in.

### Live site

| What | URL | Host (free plan) |
|---|---|---|
| Web app | https://fse-ml.vercel.app | Vercel Hobby, project `fse-ml`, root directory `web`, `FSE_API_URL` set |
| API | https://fse-api.onrender.com (health: `/api/health`) | Render free web service `fse-api` from `render.yaml` |
| Staging web | Vercel preview of the `staging` branch (behind Vercel login) | same Vercel project; previews always use the staging API (`web/next.config.ts`) |
| Staging API | https://fse-api-staging.onrender.com | Render free web service `fse-api-staging`, made by hand, deploys from `staging` |

Production deploys automatically from `main`, staging from `staging` (PLAN.md
1.1, DEPLOY.md "Environments"). `/api/health` reports `environment` and
`commit`, and the status bar says `· staging` on staging. After a push to
`staging`, `.github/workflows/staging.yml` waits for both staging copies to run
that commit and runs the live browser checks there (needs the GitHub secret
`VERCEL_AUTOMATION_BYPASS_SECRET`). After merging to `main`, bring staging
level: `git push origin main:staging`. Rollback: DEPLOY.md "Rollback" (default
is a git revert PR). The free API sleeps after 15 minutes
idle and takes about a minute to wake. Read-only checks against the live site:
`E2E_LIVE=1 npm --prefix web run test:live` (PowerShell: `$env:E2E_LIVE="1"`),
also run daily by `.github/workflows/live.yml`.

Monitoring (PLAN.md 1.2, DEPLOY.md "Monitoring"): Sentry in the API and the
web app (`SENTRY_DSN`), a shared `X-Request-ID` per API call, JSON request
logs in UTC with model-run timings, Better Stack uptime monitors and status
page as code in `ops/betterstack.py` (GitHub secret `BETTERSTACK_API_TOKEN`).
Staging alerts come from `staging.yml`, not a scheduled check, to keep the
free service asleep. **Never log or send deal contents**: no request bodies,
query strings or local variables in logs or Sentry events (tests check it).

Database (PLAN.md 1.3, DEPLOY.md "Database"): Neon free project `fse-ml`,
region us-east-2 (Ohio). Its default branch is named **`production`** (not
main); branch `staging` never auto-deletes. `DATABASE_URL` (pooled string) is
set on `fse-api` (production branch) and `fse-api-staging` (staging branch).
The API opens no connection at start-up and `/api/health` never queries the
database (either would burn Neon's 100 free compute hours);
`/api/health/database` connects, migrates on first use and reports storage
against the free 0.5 GB (warning at 80%). It runs daily from `live.yml` and
after each staging deploy.

Accounts (PLAN.md 1.4, DEPLOY.md "Accounts and sign-in"): **Clerk** free
development instance, email and Google sign-in. **Every API call except
`/api/health*` and the schema needs a signed-in user** — the dependency is on
`include_router`, so a new route is protected unless it is added to
`api.auth.PUBLIC_PATHS`. The browser sends Clerk's session token as a bearer
token; `api/auth.py` verifies it locally against the instance's JWKS (RS256,
issuer, expiry), so no Clerk secret key and no Clerk API quota are used.
Vercel needs `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` and `CLERK_SECRET_KEY`;
Render needs `CLERK_PUBLISHABLE_KEY` (or `CLERK_ISSUER`). Without a Clerk key
the app and the API use the **development sign-in** (`Bearer dev:<name>`,
`FSE_AUTH_DEV=1`), which is what local runs and the browser tests use and
which is refused in production or whenever Clerk is configured. The `users`
table stores only Clerk's user id plus country, currency, locale and time
zone, asked at sign-up on `/account`; **never store or log names, emails or
anything else personal**.

### What's next: PLAN.md

The rebuild is done. **PLAN.md** is the roadmap: software only (the user set
aside market research, customers, company, legal and pricing), **global and
universal by design** (any country, currency and deal; no dependence on the
four inception-era US deals), and **free first**. Phases 0-11 use only free
plans and free data (Vercel Hobby, Render free web, Neon, Upstash, GitHub
Actions for scheduled jobs, Supabase Storage, Clerk, Sentry, Better Stack,
Resend, PostHog, Stripe test mode, recorded or free-tier AI on public documents
only). Phase 12 switches on paid tiers once the product is functional. The
user's phase order puts data, ML, AI and features before scaling work, with
background jobs moved into foundations. Tasks carry dependencies, what the
user must do first (outside accounts and keys only) and a "done when" test;
the appendix lists every US-specific and deal-dependent assumption in the
code. Work one task per session and per PR, lowest open number first, tick it
in PLAN.md in the same PR. 0.1 is done (live site above); 2.1 is done (labels
below); 1.1 is done (staging, rollback drill in DEPLOY.md); 1.2 is done (monitoring, alert drill in DEPLOY.md); 1.3 is done (database); 1.4 is done (accounts and
sign-in, below); next is 1.5. End every task session with the handoff described in PLAN.md: tell the
user to start a new session and give the ready-to-paste prompt for the next
task.

### Main goal: frontend rebuild (Streamlit → FastAPI + Next.js)

The user wants a full, scalable, "anti-slop" frontend rebuilt from scratch, with
the model's logic kept as is (the user later approved fixing the findings
below). Streamlit is retired (removed in step 6).

| Step | Status |
|---|---|
| 1. Design direction — mockups of key screens for user approval | ✅ Agreed — see "Design system" below |
| 2. API layer — `core/` + `api/` | ✅ Done (PR #2) |
| 3. App shell — Next.js in `web/`, design system, layout, navigation, TS client generated from `/api/openapi.json` | ✅ Merged (PR #5) |
| 4. Rebuild the five screens: deal wizard, Monte Carlo, backtesting, forecasting, settings | ✅ All five built, verified by output, merged (PR #5) |
| 5. Browser tests in CI proving every control changes its output (Playwright) | ✅ `web/e2e/` (38 tests, CI job `e2e`). Mutation-checked |
| 6. Deploy (frontend + API) and retire Streamlit | Config done on `feat/deploy` (root `Dockerfile`, `render.yaml`, Vercel via `FSE_API_URL`, CI `docker` job). Streamlit removed. **Waiting on the user** to connect Render and Vercel (DEPLOY.md) |

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
- Screens show open model findings as visible markers rather than hiding them
  (none are open in the web app now).
- **Sign-in, sign-up and the account screen** use the same tokens: Clerk's
  components are themed through `appearance` in
  `components/auth/AuthScreens.tsx` (square corners, Tape colours), and the
  account screen is an ordinary form in the `type-*` classes. The top bar's
  right-hand link shows who is signed in.
- **Honest labels (PLAN.md 2.1):** unsourced inception-era numbers carry a
  visible label until sourced data replaces them. Wording lives in
  `web/src/lib/provenance.ts`: Settings (every step) and the Monte Carlo rail say
  "Illustrative defaults — not market data"; Backtest counts its example deals
  from the deal list; the risk score's sample size comes from
  `historical_sample` in `/api/ml/deal-risk`; Live lists the surrogate's fixed
  training deal from `training_deal` in `/api/ml/surrogate`. Keep these when
  editing those screens; `web/e2e/provenance.spec.ts` checks them (risk and
  Live replay `web/e2e/fixtures/ml-responses.json`, recorded from a real ML
  server, because CI's e2e job has no ML layer — re-record it if those
  responses change).

### Model findings

Fixed and merged in PR #5 (user said "you do it all", 2026-09-15). Each fix
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
| `core/` | **All model logic**, no web framework dependency. Functions take inputs and settings explicitly |
| `api/` | FastAPI app (`api/main.py`), routers per mode (plus `export.py` for Excel downloads), schemas generated from engine dataclasses |
| `api/auth.py` | Who is calling: Clerk session tokens verified locally against the instance's JWKS, plus the development sign-in for local runs and CI. `PUBLIC_PATHS` lists what works signed out |
| `lbo_engine/` | Deterministic LBO engine (operating model, cash flow, debt, returns) |
| `simulation/vectorized_simulation.py` | Vectorized Monte Carlo engine |
| `analytics/` | Risk metrics |
| `ml/` | Optional ML: anomaly detector, surrogate network, macro regime, EDGAR extractor, plus unused modules |
| `web/` | Next.js 16 frontend. `src/proxy.ts` sends signed-out visitors to `/sign-in`; `components/auth/` holds the session, the account screen and the sign-in pages; `lib/auth/` decides Clerk or development sign-in. `src/lib/nav.ts` lists every mode and step (tabs, step row, search). `src/app/<mode>/<step>/page.tsx` are thin route files; screens live in `src/components/<mode>/`. State per mode sits in a provider mounted in `components/shell/AppShell.tsx` (Settings → Deal → Monte Carlo → Backtest → Forecast), so it survives mode switches. Settings overrides persist in localStorage and go into every run. `components/charts/` and `components/ui/` are shared; `src/lib/api/` the typed client |
| `web/openapi.json` | Snapshot of the API schema; `src/lib/api/schema.d.ts` is generated from it |
| `Dockerfile`, `render.yaml` | API image and Render blueprint. `INSTALL_ML=true` build arg adds the ML layer |
| `DEPLOY.md` | Vercel + Render setup steps, environments, rollback, monitoring |
| `db/` | Database layer: `engine.py` (Neon-aware connections and retries), `models.py` (tables and column rules), `migrations/` (Alembic, numbered `0001_…`), `migrate.py` (CLI and migrate-on-first-use), `health.py` (status and storage check), `local.py` (local Postgres) |
| `db/users.py` | Account profiles: validating and storing country, currency, locale and time zone (`api/routers/account.py` serves them) |
| `api/observability.py`, `web/src/lib/monitoring.ts` | Request IDs, JSON logs, model-run timings, Sentry (with privacy scrubbing) |
| `ops/betterstack.py` | Better Stack uptime monitors, status page and incidents, as code (`monitoring.yml` syncs it) |
| `ops/check_database.py` | Deployed database check (reachable, migrations current, storage under 80%) for `live.yml` and `staging.yml` |
| `tests/golden/` | Snapshot of the retired Streamlit app's outputs; the parity baseline. Its generator was removed with Streamlit (see git history) |
| `tests/`, `test_*.py` | Test suite (195 tests with a database; database tests skip without `TEST_DATABASE_URL`); `tests/test_model_fixes.py` pins each finding fix, `tests/test_database.py` the database layer, `tests/test_auth.py` sign-in, `tests/test_users.py` accounts. `tests/conftest.py` signs every other test in and hands out throwaway databases |

## Commands (Windows, from the repo root)

```bash
.venv/Scripts/python.exe -m pytest                       # all tests
.venv/Scripts/python.exe -m uvicorn api.main:app --reload --port 8000   # API; docs at /api/docs

# web/ (run the API too; Next proxies /api to FSE_API_URL, default 127.0.0.1:8000)
npm --prefix web run dev                                 # http://localhost:3000
npm --prefix web run lint
npm --prefix web run typecheck
npm --prefix web run build

# After changing API schemas: refresh the snapshot, then the TS types
.venv/Scripts/python.exe -m api.export_openapi web/openapi.json
npm --prefix web run api:types
```

Database (optional locally; the API runs without one):

```bash
.venv/Scripts/python.exe -m pip install pgserver         # once: Postgres in a wheel, no Docker needed
.venv/Scripts/python.exe -m db.local                     # starts it (data in .localdb/), prints DATABASE_URL and TEST_DATABASE_URL
# set both in the shell (PowerShell: $env:DATABASE_URL="..."), then:
.venv/Scripts/python.exe -m pytest tests/test_database.py
.venv/Scripts/python.exe -m db.migrate upgrade | downgrade -1 | current
.venv/Scripts/python.exe -m db.local stop
```

### Adding a table

1. Define it in `db/models.py` on `Base`. Follow the column rules (tests
   enforce them): date-times are `UTCDateTime`; money is a `MoneyAmount`
   column `<name>_amount` beside a `CurrencyCode` column `<name>_currency`;
   no float money. Never store deal contents, names or email addresses
   anywhere they could be logged; a person is `users.subject` and nothing
   else.
2. With a local database up to date (`python -m db.migrate upgrade`), run
   `python -m db.migrate revision -m "add deals"`. It writes
   `db/migrations/versions/000N_add_deals.py`.
3. Read and fix the file: autogenerate misses renames, check constraints and
   data changes. Write a real `downgrade()`.
4. Make it safe while the previous deploy still runs: add first, drop in a
   later release, backfill before `NOT NULL`. Mind the free 0.5 GB.
5. Run `tests/test_database.py`: migrations must go up, all the way down and
   up again, and match the models exactly. Add tests for the new table's
   behaviour (real rows in, real rows out).
6. Deploys apply it automatically on first database use; `staging.yml` checks
   staging is at the new revision before the PR merges to `main`.

Browser tests (`web/e2e/`, Playwright). They start uvicorn and `next start`
themselves, so build first. They also need a **database** (accounts) and the
**development sign-in**: start `python -m db.local`, then set `DATABASE_URL`
and `FSE_AUTH_DEV=1` in the shell. No bundled browser on this machine: use
Edge.

```bash
npm --prefix web run build
PW_CHANNEL=msedge FSE_AUTH_DEV=1 npm --prefix web run test:e2e   # PowerShell: $env:PW_CHANNEL="msedge"
```

`e2e/auth.setup.ts` signs in once as `dev:e2e` and saves the browser state
every other spec reuses (`web/e2e/.auth/`, git-ignored); `auth.spec.ts` drops
it to check what a signed-out visitor can reach.

Tests assert real model output (IRR, MOIC, golden backtest values), not just
rendering. Add one for every new control, and mutation-check it.

CI (`.github/workflows/tests.yml`) runs `core`, `ml`, `web`, `e2e` and `docker`
jobs. `core`, `ml`, `e2e` and `docker` get a Postgres 17 service;
`FSE_REQUIRE_DB=1` makes a skipped database test fail. `docker` builds the root Dockerfile, runs it with a host-assigned PORT
and checks the default deal IRR (0.2116), Monte Carlo and an Excel export.
`tests/test_openapi_snapshot.py` fails when `web/openapi.json` is stale; the
`web` job fails when `schema.d.ts` doesn't match the snapshot.

Setup on a fresh machine: Python 3.12, then
`pip install -r requirements-ml.txt -r requirements-dev.txt` (or just
`requirements-dev.txt` without ML). Trained model files are committed.

## Working rules

- **`main` is protected.** Every change: branch → PR → CI (`core`, `ml`, `web`, `e2e`, `docker` jobs)
  green → merge with **"Create a merge commit"**. Direct pushes to `main` fail.
  The GitHub CLI is installed but not signed in (`gh auth status`), and
  signing it in needs the user's credentials: open and merge PRs through the
  browser, or run `gh auth login` first.
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
- The golden snapshot can't be regenerated any more (Streamlit is gone). Pin
  deliberate model changes with explicit tests instead.
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
  Playwright reuses servers already on ports 3000/8000 locally: stop any
  preview or stray uvicorn first, or tests hit stale code (a 404 on a new
  endpoint means this). A `next start` left over from an interrupted run keeps
  serving the old build after a rebuild: the page renders but never hydrates,
  so clicks do nothing and the console shows `ChunkLoadError`. Kill whatever
  holds the port (`netstat -ano | grep LISTENING | grep :3000`).
- The Browser pane's screenshots time out when the Claude window isn't drawn;
  verify with `javascript_tool` / `find` / `form_input` instead.
- Browser-automation key presses: send `Enter` and `]`, not `Return` or
  `bracketright`, or shortcuts appear broken when they aren't.
- Deal defaults: the API uses stored 60% debt / 70% senior; the retired
  Streamlit wizard derived 42% / ~81% from 3.4x + 0.8x debt multiples, which is
  what the golden `defaults` case records.
- Never put secrets in Playwright `extraHTTPHeaders`: they go to every host
  the page calls (Sentry included). `web/e2e/live.spec.ts` scopes the Vercel
  bypass header to the site's own origin.
- Vercel's Environment Variables page is in the project's main left menu,
  not under Settings. `NEXT_PUBLIC_*` values (e.g. the Sentry DSN) are baked
  in at build time, so a changed `SENTRY_DSN` needs a new deploy.
- Unauthenticated GitHub API calls allow 60 an hour; poll workflow runs
  sparingly (or read the run page in the browser).
- Neon: the app uses the pooled URL (`-pooler` host, PgBouncer transaction
  mode), so no session state (`SET`, advisory locks, `LISTEN`, prepared
  statements) across statements on app connections; migrations use the
  direct host. The pooler rejects the libpq `options` startup parameter, so
  don't set the time zone there: `UTCDateTime` converts instead.
- Anything polled often (`/api/health`, the uptime monitor, Render's health
  check) must not touch the database, or Neon never scales to zero.
- API in Render Oregon, database in Neon Ohio: ~50–70 ms a round trip. Batch
  queries per request.
- Vercel resolves Next rewrites at build time: changing `FSE_API_URL` needs a
  redeploy. `next.config.ts` fails the Vercel build if it's unset.
