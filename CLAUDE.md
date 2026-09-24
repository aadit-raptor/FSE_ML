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

Last updated: 2026-09-24 (PLAN.md 0.3: CI gates for the development cycle). Steps 1–5 and the model finding fixes are merged
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
level with a pull request from `main` to `staging` titled `chore: bring
staging level with main`, which the user merges (Claude can't push to
`staging` or merge). Rollback: DEPLOY.md "Rollback" (default
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

Saved deals (PLAN.md 1.5): tables `deals` (the working copy: complete inputs
plus Settings overrides, overwritten by autosave) and `deal_versions`
(history), and `users.settings` (the account's Settings overrides; **Settings
no longer live in localStorage** — an old browser copy is imported once).
`db/deals.py` matches the caller's subject in the same statement as every
read and write, so another account's deal answers **404**, never 403.
Compact by design for the free 0.5 GB: a version is written only when content
differs from the latest one, automatic checkpoints are one per 15 minutes and
the newest 20 per deal are kept, settings are overrides only (a version is
under 900 bytes, tested). Opening a deal or restoring a version also replaces
the account's Settings with the deal's, so its IRR comes back exactly. The
screen is Deal → Saved deals (`/deal/saved`); the rail header shows the open
deal and its save state. Deals store overrides relative to today's
`core/config.py` defaults, so changing a default changes old deals' results
until PLAN.md 3.1 records the model version.

Usage limits (PLAN.md 1.6, DEPLOY.md "Usage limits"): `api/limits.py` holds
every number and why. Per user (a dependency on `include_router`, like
sign-in, so a new route is limited too): 240 requests a minute, 30 model runs
a minute and 1,000 a day (runs = `RUN_PATHS`: simulations, backtests,
forecasts, ML, EDGAR). Per address (middleware, before sign-in): 1,200
requests and 60 refused sign-ins a minute. Monte Carlo and backtest at most
**100,000 paths** (forecast 200,000), sized from measured memory on Render
free's 512 MB; **one simulation at a time** (`simulation_slot`), 100 s
timeout (504), 1 MB bodies (413). Health checks are never limited. Counting
is in memory; only daily counts are shared, batched to **Upstash**
(`UPSTASH_REDIS_REST_URL`/`_TOKEN` on both Render services and in GitHub
secrets) every 5 minutes, with a hard daily Redis command budget per
environment and the `usage_counters` table as fallback. Keys hash the user;
nothing personal goes to Redis. `/api/health/limits` (public) reports the
store; `live.yml` and `staging.yml` require Upstash there. Tests get fresh
in-memory counters (`tests/conftest.py`); the browser tests raise limits
with `FSE_LIMITS_MULTIPLIER` (ignored in production).

Security (PLAN.md 1.7, DEPLOY.md "Security", `docs/security/threat-model.md`,
`SECURITY.md`): the web app sends security headers (`next.config.ts`) and a
**nonce-based content security policy** built per request in `src/proxy.ts`
from `src/lib/security/headers.ts`, so **every page renders per request**
(`await connection()` in the root layout, `<ClerkProvider dynamic>`). A new
third-party script, frame, image or API host must be added to that policy or
the browser refuses it; `e2e/security.spec.ts` fails on any violation. The
API (`api/security.py`) sends `default-src 'none'` and `no-store` on every
answer (a hash-based policy for `/api/docs`) and allows CORS only from exact
HTTPS app origins (`FSE_CORS_ORIGINS` can't widen it). Deployed copies force
TLS to the database. Migration 0005 creates `fse_app`, a role with row rights
only; the API connects as login role `fse_api` in it (both environments), with the
owner in `DATABASE_MIGRATION_URL` for migrations. `/api/health/database`
reports `role` restricted or privileged. `ops/check_headers.py` scans the live
headers (`live.yml`, `staging.yml`). CI: `security.yml` (gitleaks over all
history with a planted-key self-test, pip-audit, npm audit), `codeql.yml`,
Dependabot (`.github/dependabot.yml`). Every workflow's token is read-only.

Backups (PLAN.md 1.8, DEPLOY.md "Backups and recovery"): `backup.yml` dumps
the production database nightly (`pg_dump -Fc`), encrypts it with AES-256-GCM
(`ops/encryption.py`, key in the GitHub secret `FSE_BACKUP_KEY`), uploads it
to a **private** Supabase Storage bucket (`ops/backup_store.py`), rotates old
ones (7 daily, 8 weekly, 12 monthly, and a 700 MB budget of the free 1 GB),
then downloads and decrypts what it just stored. On the first of the month a
**restore drill** restores the newest backup into a throwaway database on the
Neon staging branch, re-runs the deal model on every restored deal and checks
the results against **what the manifest recorded when the dump was taken**
(read inside the same `pg_export_snapshot` `pg_dump` reads, so it describes
exactly what is in the file — comparing with today's live data would cry wolf
as soon as anyone edits a deal), then drops the database; either job failing
raises a Better Stack incident. **Recent mistakes use Neon's own restore window
instead** — it is faster and loses nothing. The manifest beside each backup
holds sizes, checksum, versions, the migration revision, how many deals and
that one-way fingerprint — **never anything from a deal**;
backups are never GitHub Actions artifacts, which are public on a public
repository. `pg_dump`/`pg_restore` must be at least the server's major
version; `ops/backup.py` finds them (PATH, `/usr/lib/postgresql/*/bin`,
`pgserver`, or `FSE_PG_BIN`) and CI installs PGDG's newest client.
The secrets are set: the first backup and restore drill succeeded on
2026-09-18.

Background and scheduled jobs (PLAN.md 1.9, DEPLOY.md "Background jobs and
scheduled jobs"): long runs go through **`/api/jobs`** — submit
`{"kind": "montecarlo.run", "input": <the endpoint's body>}`, poll
`/api/jobs/{id}` for `stage`/`progress`, get `result`, which is exactly what
the direct endpoint answers (`jobs/kinds.py` calls the same function, minus
its slot wrapper). The queue is an interface (`jobs/queue.py`):
`DatabaseQueue` (table `jobs`, `FOR UPDATE SKIP LOCKED`) whenever
`DATABASE_URL` is set, else `MemoryQueue`; `FSE_JOB_QUEUE`/`FSE_JOB_RUNNER`
choose, and phase 12's workers are `python -m jobs.worker`. The runner is a
**thread in the API** that starts on a submit or a poll and stops after a
minute idle (so Neon sleeps); it takes the same one-simulation slot as the
direct endpoints. A job whose heartbeat is silent 90 s is requeued (up to 3
tries); results are kept 6 hours, rows 7 days. The Monte Carlo screen runs
through it (progress bar, Cancel, `running` tab chip). **Scheduled jobs**:
`scheduled.yml` (nightly) calls `/api/scheduled/tasks/job-maintenance` on
both environments and runs the Supabase keep-alive; every run lands in
`scheduled_runs` and `/api/health/jobs` (public). Those endpoints take **no
secret**: GitHub's OIDC token for the run (`id-token: write`), checked in
`api/github_oidc.py` for this repository's id, the workflow file, a
protected branch and an environment audience. `staging.yml` runs the **job
drill** (ten seeded simulations at once, health polled throughout, results
against `jobs/drill.py`'s pinned summary; refused in production).

Currency and money units (PLAN.md 2.2): every deal has a **currency** (any
ISO 4217 code) and a **unit** (`thousands`, `millions`, `billions`), stored
as `currency`/`unit` in its inputs (`DealInputsIn`); a new deal starts in the
account's currency. Forecast requests, backtests and sources & uses take a
`money` object; **every answer with money carries `money`**, and Excel
workbooks get an About sheet saying what the money is in. The model never
reads the currency. The deal engine rounds money to two decimals and is
pinned to the golden snapshot in millions, so **deals, simulations and
backtests run in millions whatever their unit**: the router converts inputs
(`core.deal.in_millions`, `mc_in_millions`, `backtest_in_millions`) and
hands the answer back with `core.money.rescale` and an explicit list of
money keys (`DEAL_MONEY_KEYS`, `SOURCES_USES_MONEY_KEYS`, `MC_MONEY_KEYS`,
`BACKTEST_MONEY_KEYS`). A deal in thousands therefore answers exactly as
in millions, times a thousand (`tests/test_money.py` checks every leaf
against the unconverted engine). The forecast runs in the company's own
unit; its fixed amounts use `in_unit`/`round_money`. Web: `lib/money.ts`
makes labels from the browser's CLDR data ("€k", "£M", "CHF bn"), and
`useMoney()` gives the money on screen: DealProvider sets the open deal's
for every screen, Backtest and Forecast set their own inside it. A field
spec's unit `MONEY` shows the label; a unit change converts the deal's
money so its size stays the same. **No hardcoded "$"**:
`tests/test_no_hardcoded_currency.py` fails on one in `api`, `core`,
`lbo_engine`, `ml`, `ops`, `web/src` and the rest (a real exception ends
its line with `currency-ok`). The example backtest deals and SEC EDGAR data
are US dollar millions and say so.

CI gates (PLAN.md 0.3, docs/WORKFLOW.md step 6): the `core` and `ml` jobs
measure Python coverage (`pytest --cov`, packages listed in `.coveragerc`),
put the table in the job summary and fail below their line in
**`.coverage-floor`**, which may only rise (`ops/coverage_gate.py` compares
it with the base branch's copy). The `ml` job, which runs every test, also
fails a PR when under **80% of its changed Python lines** are covered
(`diff-cover`, needs `fetch-depth: 0`). `pr.yml` fails a **PR title**
without an ECC type (`ops/pr_title.py`, the types in WORKFLOW.md step 7;
Dependabot titles are `chore(deps): …`). Floors since 2026-09-24: core 75.5, ml 76.9
(CI's measured 75.54% and 76.93%). Raise the floor when a job's
summary suggests it; never lower it. No web unit-test framework yet: the
browser tests are the web's proof.

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
sign-in, below); 1.5 is done (saved deals, below); 1.6 is done (usage limits, below); 1.7 is done (security, below); 1.8 is done (backups, below); 1.9 is done (background and scheduled jobs, below); 2.2 is done (currency and money units, below). Added 2026-09-24: the
development cycle in **docs/WORKFLOW.md** (ECC merged with these rules), and
tasks 0.2 (own domain, when the user has bought one), 0.3 (cycle gates in CI)
and 11.4 (owner's handbook). 0.3 is done (CI gates, below). **Next is 2.3**
(0.2 jumps the queue once the domain is bought). End every task session with the handoff described in PLAN.md: tell the
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
| 5. Browser tests in CI proving every control changes its output (Playwright) | ✅ `web/e2e/` (52 tests, CI job `e2e`). Mutation-checked |
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
10. **Open (found in PLAN.md 2.2, needs the user's approval): the engine
    rounds money to two decimals of a million** (`lbo_engine/*` `round(x, 2)`,
    and the backtest's predicted EBITDA to 0.1M), i.e. to 10,000 of the
    currency. Deals now always run in millions, so this is the same in every
    unit, but a small deal (EBITDA under about 10M) loses precision: revenue
    and interest are off by up to 5,000. Fixing it changes the golden
    numbers, so it waits for approval.

### Waiting on the user

- PLAN.md 1.7 follow-up: the GitHub settings in DEPLOY.md "CI and GitHub
  settings" (private vulnerability reporting, required checks, Dependabot
  secrets) and the commit email. (Both environments' `DATABASE_URL` switched
  to the restricted `fse_api` role on 2026-09-17; `live.yml` and
  `staging.yml` now fail if either goes back to a privileged role.)

- Keep a copy of `FSE_BACKUP_KEY` outside GitHub (PLAN.md 1.8): losing it
  makes every stored backup unreadable.

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
| `web/` | Next.js 16 frontend. `src/proxy.ts` sends signed-out visitors to `/sign-in`; `components/auth/` holds the session, the account screen and the sign-in pages; `lib/auth/` decides Clerk or development sign-in. `src/lib/nav.ts` lists every mode and step (tabs, step row, search). `src/app/<mode>/<step>/page.tsx` are thin route files; screens live in `src/components/<mode>/`. State per mode sits in a provider mounted in `components/shell/AppShell.tsx` (Settings → Deal → Monte Carlo → Backtest → Forecast), so it survives mode switches. Settings overrides are saved to the account (`/api/account/settings`) and go into every run; the open deal (`DealProvider`) autosaves to `/api/deals/{id}/draft`, and the last one opened is reopened on the next visit. `components/charts/` and `components/ui/` are shared; `src/lib/api/` the typed client |
| `web/openapi.json` | Snapshot of the API schema; `src/lib/api/schema.d.ts` is generated from it |
| `Dockerfile`, `render.yaml` | API image and Render blueprint. `INSTALL_ML=true` build arg adds the ML layer |
| `docs/WORKFLOW.md`, `.github/pull_request_template.md` | The development cycle every change follows, and the PR form that records it |
| `.coverage-floor`, `.coveragerc`, `ops/coverage_gate.py`, `ops/pr_title.py`, `.github/workflows/pr.yml` | CI gates (PLAN.md 0.3): coverage floors per job (only rise), what coverage measures, the floor check, the PR title check |
| `DEPLOY.md` | Vercel + Render setup steps, environments, rollback, monitoring |
| `db/` | Database layer: `engine.py` (Neon-aware connections and retries), `models.py` (tables and column rules), `migrations/` (Alembic, numbered `0001_…`), `migrate.py` (CLI and migrate-on-first-use), `health.py` (status and storage check), `local.py` (local Postgres) |
| `db/users.py` | Account profiles: validating and storing country, currency, locale and time zone (`api/routers/account.py` serves them) |
| `db/deals.py` | Saved deals, versions and account settings: ownership, autosave checkpoints, restore, compact storage (`api/routers/deals.py` serves them) |
| `api/limits.py`, `api/usage.py`, `db/usage.py` | Usage limits: rules, refusal messages, size caps, simulation slot and timeout; in-memory counters synced to Upstash (daily command budget, `python -m api.usage` prints the monthly estimate) with the database as fallback |
| `api/security.py`, `web/src/lib/security/headers.ts`, `web/src/proxy.ts` | Security headers, CSP (nonce per page request), CORS origins |
| `ops/check_headers.py` | Header scan of a deployed web app and API (`live.yml`, `staging.yml`) |
| `ops/backup.py`, `ops/encryption.py`, `ops/backup_store.py` | Backups: dump, encrypt, upload, rotate, restore and the monthly drill (`python -m ops.backup run / list / verify / restore / drill`, run by `backup.yml`); the AES-256-GCM file format; the Supabase Storage and local-directory stores |
| `SECURITY.md`, `docs/security/threat-model.md` | How to report a vulnerability; threat model, open items and the public-repo review |
| `api/observability.py`, `web/src/lib/monitoring.ts` | Request IDs, JSON logs, model-run timings, Sentry (with privacy scrubbing) |
| `ops/betterstack.py` | Better Stack uptime monitors, status page and incidents, as code (`monitoring.yml` syncs it) |
| `core/money.py`, `web/src/lib/money.ts`, `web/src/components/ui/MoneyScope.tsx` | Currency and money units (PLAN.md 2.2): conversion to and from millions, the money keys of each answer, labels from CLDR, the money on screen |
| `jobs/` | Background jobs (PLAN.md 1.9): `queue.py` (the interface and retention rules), `memory.py` and `database.py` (the two queues), `runner.py` (the in-API runner thread), `kinds.py` (what can run as a job), `config.py` (which queue and runner), `scheduled.py` (scheduled tasks and their run log), `drill.py` (the staging drill's pinned answer), `worker.py` (phase 12's dedicated worker). Served by `api/routers/jobs.py` and `api/routers/scheduled.py` |
| `api/github_oidc.py`, `ops/scheduled.py` | The scheduler's sign-in (GitHub Actions OIDC tokens, no secret) and its side of the calls: `task`, `keepalive`, `drill` (`scheduled.yml`, `staging.yml`) |
| `ops/check_database.py` | Deployed database check (reachable, migrations current, storage under 80%) for `live.yml` and `staging.yml` |
| `tests/golden/` | Snapshot of the retired Streamlit app's outputs; the parity baseline. Its generator was removed with Streamlit (see git history) |
| `tests/`, `test_*.py` | Test suite (456 tests with a database; database tests skip without `TEST_DATABASE_URL`); `tests/test_model_fixes.py` pins each finding fix, `tests/test_database.py` the database layer, `tests/test_auth.py` sign-in, `tests/test_users.py` accounts, `tests/test_deals.py` saved deals and versions, `tests/test_limits.py` usage limits, `tests/test_security.py` headers, CORS, TLS, the database role and the header scan, `tests/test_backups.py` the backup format, stores, rotation and a real dump/restore round trip, `tests/test_jobs.py` jobs on both queues, restarts, retention, the scheduler's tokens and the drill, `tests/test_money.py` currencies and units, `tests/test_no_hardcoded_currency.py` the dollar-sign check, `tests/test_cycle_gates.py` the CI gates (PR titles, coverage floor, the workflows keep them). `tests/conftest.py` signs every other test in and hands out throwaway databases |

## Commands (Windows, from the repo root)

```bash
.venv/Scripts/python.exe -m pytest                       # all tests
.venv/Scripts/python.exe -m pytest --cov --cov-report=term   # with coverage (CI's floors: .coverage-floor)
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

Backups (PLAN.md 1.8). `FSE_BACKUP_DIR` keeps them in a directory instead of
Supabase, which is how to try the whole thing without any account:

```bash
# in the shell: FSE_BACKUP_KEY (32+ characters), FSE_BACKUP_DIR (or SUPABASE_URL
# + SUPABASE_SERVICE_ROLE_KEY), BACKUP_DATABASE_URL
.venv/Scripts/python.exe -m ops.backup run --environment local      # dump, encrypt, upload, rotate
.venv/Scripts/python.exe -m ops.backup list --environment local
.venv/Scripts/python.exe -m ops.backup verify --environment local   # download and decrypt the newest
.venv/Scripts/python.exe -m ops.backup drill --environment local --target "$ADMIN_URL"
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
jobs; `security.yml` (secrets, dependency audits), `codeql.yml` and `pr.yml`
(PR title) run beside it. `core` and `ml` also enforce the coverage floor,
and `ml` 80% of changed lines (PLAN.md 0.3). `core`, `ml`, `e2e` and `docker` get a Postgres 18 service (what Neon runs);
`FSE_REQUIRE_DB=1` makes a skipped database test fail. `docker` builds the root Dockerfile, runs it with a host-assigned PORT
and checks the default deal IRR (0.2116), Monte Carlo, an Excel export and a background job.
`tests/test_openapi_snapshot.py` fails when `web/openapi.json` is stale; the
`web` job fails when `schema.d.ts` doesn't match the snapshot. `core` and `ml`
also install PGDG's newest `postgresql-client`, which `tests/test_backups.py`
needs to dump and restore that service. Scheduled workflows beside these:
`live.yml` (daily production checks), `monitoring.yml`, `backup.yml`
(nightly backup, monthly restore drill) and `scheduled.yml` (nightly job
maintenance on both environments, Supabase keep-alive).

Setup on a fresh machine: Python 3.12, then
`pip install -r requirements-ml.txt -r requirements-dev.txt` (or just
`requirements-dev.txt` without ML). Trained model files are committed.

## Working rules

**The development cycle is docs/WORKFLOW.md** (research → plan → test first →
build → review → verify → ship → remember), ECC's framework merged with the
rules below. Where they differ, the rules below win.

@docs/WORKFLOW.md

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
- **Units:** API inputs use percentages as numbers (`60.0`) and money in the
  deal's currency and unit (US dollar millions by default); engine outputs
  keep engine units (IRR `0.157` = 15.7%). Show money with `useMoney()` /
  `moneyLabel()`, never a written currency sign.
- **Optional ML** stays behind guarded, lazy imports; the app and CI's `core`
  job must work without ML packages.
- Commit messages start with an ECC type (`feat:`, `fix:`, `docs:` …) and
  explain *why* and how it was verified. PRs use
  `.github/pull_request_template.md`.

## Tooling

Installed at **user level on the original Windows machine** — on another device,
reinstall:

| Tool | Install | Notes |
|---|---|---|
| `design-taste-frontend` (+ companions) | `npx skills add https://github.com/Leonxlnx/taste-skill` | Dials: `DESIGN_VARIANCE`, `MOTION_INTENSITY`, `VISUAL_DENSITY` |
| `web-design-guidelines` | `npx skills add vercel-labs/agent-skills --skill web-design-guidelines` | Fetches Vercel's guidelines from GitHub each run |
| `image-to-code` | `npx skills add https://github.com/Leonxlnx/taste-skill --skill image-to-code` | Written for Codex; expects to generate images, which Claude Code can't |
| Playwright CLI | `npm install -g @playwright/cli@latest` then `playwright-cli install --skills --global` | No Chrome on the original machine: use `--browser=msedge`. Writes to `.playwright-cli/` |
| ECC (Everything Claude Code) | In an interactive `claude` terminal: `/plugin marketplace add https://github.com/affaan-m/ECC`, then `/plugin install ecc@ecc` at **project scope** | Optional (docs/WORKFLOW.md "Installing ECC"). **Installed 2026-09-24, ECC 2.2.2, project scope** (`enabledPlugins` in the committed `.claude/settings.json`). Plugin only: no `install.sh`, no global rules copy, attribution unchanged. ECC's hooks default to **on**; they're **off** through `.claude/settings.json` `env`: `ECC_HOOKS_ENABLED=false`, `ECC_SESSION_START_CONTEXT=off` |
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
- A restore brings the schema and rows but **not the grants**: Neon's dump
  carries `ALTER DEFAULT PRIVILEGES FOR ROLE cloud_admin … TO neon_superuser`,
  which only Neon's superuser may replay and which `pg_dump` puts in the same
  entry as ours, so `ops/backup.py restore` passes `--no-privileges` and
  DEPLOY.md "Restoring" re-applies migration 0005's grants afterwards. The
  first drill run failed exactly here.
- `pg_dump` refuses a server **newer** than itself but reads an older one
  happily. **Neon runs Postgres 18** (it was 17 when 1.3 was written; check
  with `SHOW server_version_num` rather than trusting this line). So the
  workflows install PGDG's newest `postgresql-client` instead of pinning a
  major version, and `ops/backup.py` looks for a new-enough binary on the
  PATH, in `/usr/lib/postgresql/*/bin` and in the `pgserver` package (which
  ships Postgres 16, enough for the local database); `FSE_PG_BIN` overrides
  the search. The first backup attempt failed exactly here, with a message
  naming every binary it found.
- Deal defaults: the API uses stored 60% debt / 70% senior; the retired
  Streamlit wizard derived 42% / ~81% from 3.4x + 0.8x debt multiples, which is
  what the golden `defaults` case records.
- Never put secrets in Playwright `extraHTTPHeaders`: they go to every host
  the page calls (Sentry included). Nor in `page.route` header overrides:
  Playwright keeps them across redirects, and signed-out pages redirect to
  Clerk. `web/e2e/live.spec.ts` trades the Vercel bypass secret once for
  Vercel's bypass cookie and fails if the secret reaches another origin.
- Signed out, every page answers 404 (non-browser requests) or redirects to
  Clerk, so anything that checks the web app without an account (the Better
  Stack website monitor) uses `/healthz`, which `src/proxy.ts`'s matcher
  leaves open. It must never call the API (that would keep Render awake).
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
- A new long run as a background job: register it in `jobs/kinds.py` (the
  endpoint function, its request/response models and its `model_timer`
  names for progress), add a `*Job` model to the `JobSubmit` union in
  `api/schemas.py`, and to `_SETTINGS_CHECKED` in `api/routers/jobs.py` if it
  takes settings. Anything a job or the runner does must stay pooler-safe
  (one statement per round trip; no advisory locks or session state).
- `jobs` owners are subjects; the drill uses `system:drill`, which no sign-in
  can produce. The conftest gives every test a fresh `MemoryQueue` and turns
  the runner thread off; tests run jobs with `Runner.run_next()`.
- A new money figure in a deal, simulation or backtest answer must be added
  to that answer's money-key list (`DEAL_MONEY_KEYS` and so on), or a deal
  in thousands shows it a thousand times too small; `tests/test_money.py`'s
  leaf-by-leaf tests catch it. A new money input is converted in
  `in_millions` (and a money setting in `MONEY_SETTINGS`).
- A new top-level Python package the API imports (as `jobs/` in 1.9) must be
  added in three places: a `COPY` line in the `Dockerfile` (it copies
  packages by name), `buildFilter` in `render.yaml`, and `API_PATHS` in
  `staging.yml`. CI's `docker` job fails at start-up if the first is missing.
- A new endpoint that runs a model or calls an outside source: add its path
  to `RUN_PATHS` in `api/limits.py`; if it simulates, also to
  `SIMULATION_PATHS` and put `@simulation_slot` under its route decorator.
  A new simulation-size input needs a cap (`SimulationPaths` in
  `api/schemas.py`).
- Anything polled often (`/api/health`, the uptime monitor, Render's health
  check) must not touch the database, or Neon never scales to zero.
- API in Render Oregon, database in Neon Ohio: ~50–70 ms a round trip. Batch
  queries per request.
- Writing a file through a Bash heredoc can eat backslashes (the `\\.` in
  `proxy.ts`'s matcher became `\.`, so the proxy matched no page and the CSP
  silently vanished). Use the Write/Edit tools for source with backslashes,
  and check `.next/server/functions-config-manifest.json` after a build.
- The CSP's `script-src` has no host list: `'strict-dynamic'` lets scripts
  loaded by nonce'd scripts run (Clerk's). Styles keep `'unsafe-inline'`
  because Clerk injects styles in production; local runs have no Clerk, so
  the browser tests can't see a Clerk-only violation: check the live sign-in
  page after changing the policy.
- Vercel resolves Next rewrites at build time: changing `FSE_API_URL` needs a
  redeploy. `next.config.ts` fails the Vercel build if it's unset.
