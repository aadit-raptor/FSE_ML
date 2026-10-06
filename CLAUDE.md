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

Last updated: 2026-10-06 (PLAN.md 4.1a: company filings, the data layer). Steps 1–5 and the model finding fixes are merged
(PR #5). Step 6 is on `feat/deploy`: Streamlit parity (Excel downloads,
schedules, ML panels), Streamlit removed, and deploy config for the user's
choice of **Vercel (web) + Render (API)**. The user must create the accounts
and connect the repo themselves (DEPLOY.md); Claude can't create accounts or
sign in.

### Live site

| What | URL | Host (free plan) |
|---|---|---|
| Web app | https://variater.com (`www.` and the old https://fse-ml.vercel.app redirect here) | Vercel Hobby, project `fse-ml`, root directory `web`, `FSE_API_URL` = `https://api.variater.com` |
| API | https://api.variater.com (health: `/api/health`; also still answers on fse-api.onrender.com) | Render free web service `fse-api` from `render.yaml` |
| Staging web | Vercel preview of the `staging` branch (behind Vercel login) | same Vercel project; previews always use the staging API (`web/next.config.ts`) |
| Staging API | https://fse-api-staging.onrender.com | Render free web service `fse-api-staging`, made by hand, deploys from `staging` |

Production deploys automatically from `main`, staging from `staging` (PLAN.md
1.1, DEPLOY.md "Environments"). `/api/health` reports `environment` and
`commit`, and the status bar says `· staging` on staging. After a push to
`staging`, `.github/workflows/staging.yml` waits for both staging copies to run
that commit and runs the live browser checks there (needs the GitHub secret
`VERCEL_AUTOMATION_BYPASS_SECRET`). Claude ships end to end
(docs/WORKFLOW.md step 7): `gh pr create`, `gh pr merge --auto --merge`
(GitHub merges once the 12 required checks are green), then a PR from
`main` to `staging` titled `chore: bring staging level with main`, merged
by Claude, then the `staging.yml` and `/api/health` checks. Rollback: DEPLOY.md "Rollback" (default
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

Accounts (PLAN.md 1.4, DEPLOY.md "Accounts and sign-in"): **Clerk**, email
and Google sign-in. Since 0.2 **production uses Clerk's production instance
for `variater.com`** (`pk_live_`, Vercel Production scope and `fse-api` only);
staging, previews and local runs keep the free development instance
(`pk_test_`), since a production instance serves only its own domain.
Development-instance accounts were **dropped as test data** (decided
2026-10-01), not mapped. Google sign-in on production uses the Google Cloud project
`variater`'s own OAuth client (DEPLOY.md "Own domain"); `/privacy` is a public
page Google's consent screen links to, and every claim on it mirrors the
code, so a change to what is stored or who handles it changes that page. **Every API call except
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

Locale, part one (PLAN.md 2.3a): figures and
dates follow the **account's locale and digit grouping** (`users.digit_grouping`:
`locale`, `thousands` or `lakh`, migration 0007). `web/src/lib/locale.ts` is
**the one place that turns numbers into text**; `lib/format.ts` (`fmtMoney`,
`fmtRate`, `fmtNumber`, `fmtCount`, `fmtAxis` ...) builds on it, and
`tests/test_no_hardcoded_locale.py` fails on a quoted locale tag,
`toLocaleString` or a display `toFixed` anywhere else (`Number(x.toFixed(6))`
rounding is fine; a real exception ends its line with `locale-ok`). The
style is a module value set during render by `LocaleScope` in
`AppShell.tsx`, which holds the screens until the account answers; pages
never render on the server before the session is known, so there's no
hydration mismatch. Figures use Latin digits always. Inputs read numbers
the locale's way (`parseNumber`: "11,5" in German). Currency symbols come
from the locale ("US$" for dollars in en-GB). **Fiscal years are labels
only**: a deal's `fiscal_year_end_month` (default 12) and
`first_fiscal_year` (default none: Y1, Y2 ...), a forecast company's year-end
and latest year (EDGAR reads them from the latest 10-K; a 52/53-week year
ending in a month's first week counts as the month before). Years are named
by the year they end in: "FY2025" for December, else "FY2024/25"
(`lib/fiscal.ts`). **Stored and sent only when set** (`db/deals.py`
`OMIT_WHEN_DEFAULT`, web `apiInputs()`), so old deals are unchanged and an API
from before 2.3a (a deploy rolling out, a rollback) still answers every deal
that doesn't use them; API answers always carry them. Excel: a sheet sends
`row_formats` / `column_formats` (`money`, `percent` as fractions,
`multiple`, `integer`, `number`, `text`); standard formats show with the
reader's own separators, and the lakh choice gets a pattern sized to each
cell (Excel's conditional formats would drop the minus sign).

Interface text (PLAN.md 2.3b): **every word on screen lives in
`web/messages/<language>.json`** and reaches it through **next-intl 4**
(`useTranslations`). `en.json` is the only catalogue (1,087 keys); PLAN.md 7.8
adds languages by adding files. The language and the **text direction** come
from the account's locale, the same answer as the number format, and never
from the URL, so `src/proxy.ts` and its CSP are untouched:
`lib/i18n/config.ts` maps a locale to a catalogue and to `ltr`/`rtl`, and
`components/shell/I18nScope.tsx` mounts the provider and sets `<html lang>`
and `<html dir>` during render. A locale with no catalogue keeps its own
numbers, dates and direction and falls back to English words.
**Right to left works by itself**: the screens use logical CSS
(`border-s`/`-e`, `ps`/`pe`, `ms`/`me`, `text-start`/`-end`), and
`globals.css` makes every figure its own `direction: ltr; unicode-bidi:
isolate` island, so "-38.6" keeps its sign in front. Modules that aren't
components hold **keys, not words**: `lib/nav.ts` (`labelKey`/`summaryKey`),
`lib/deal/fields.ts` and `SettingsSteps` (the field's own key doubles as the
message key), `lib/forecast.ts`, `MONEY_UNITS`, `DIGIT_GROUPINGS`,
`lib/fields.ts` (a `FieldProblem`, not a sentence), `lib/jobs.ts` (a
`JobStage`, plus a `JobMessages` the caller passes in). Two module values are
set during render beside the number style, so the plain formatters stay plain:
`setMissingText` (what stands in for a figure the model didn't produce) and
`setDownloadFailedMessage`. **Labels the API sends** -- tranche names, equity
bridge steps, Monte Carlo drivers -- are engine output and stay as they are
(model logic); `lib/i18n/useEngineText.ts` turns each into a key and shows the
API's own word when the catalogue has none. Page `metadata` is English,
because it is produced before anyone is known to be signed in
(`lib/i18n/titles.ts`); `DocumentTitle` rewrites the tab in the account's
language once the route settles. Three CI checks:
`tests/test_no_hardcoded_text.py` fails on a JSX text node or a
`title`/`label`/`aria-label`-style value holding a word of three letters or
more (a real exception ends its line with `text-ok`);
`tests/test_translations.py` fails when a key a component asks for is
missing, when a message nothing asks for is left behind, when a language file
doesn't match English's keys and placeholders, or when the engine produces a
label the catalogue can't translate. A key built at run time is declared in a
comment, `i18n-keys: nav.*` (one `i18n-keys:` per line). `e2e/text.spec.ts`
reads the expected words out of `en.json` and checks the mirrored layout with
an Arabic account.

Debt structures and interest rates, the model (PLAN.md 2.4a): a deal's debt is
a **list of facilities** (`core/debt.py`, `DealInputs.tranches`) instead of one
senior loan plus a mezzanine tranche. Nine kinds -- amortising and
institutional term loans, unitranche, second lien, senior notes, PIK notes,
vendor loan, revolver, shareholder loan -- are **presets over one dataclass**
(`PRESETS`), not nine code paths; only three mechanics are new. **PIK**: a
share of the coupon accrues to principal, stays an expense in the P&L, and
comes back as `cash_flow.non_cash_interest` beside D&A, riding the same
convergence loop as interest. **The revolver**: `amount` is the commitment and
`drawn_pct` is drawn at close, so only the drawn part is a source of funds and
only the undrawn part pays a commitment fee (a cash cost, so it joins interest
expense); `allow_redraw` funds a cash shortfall instead of inventing cash
(finding 11). **The sweep share**: a facility takes at most its agreed share of
the cash available and what it leaves flows to the next one. Pricing lives in
`core/debt.py`, never in the engine: a floating tranche's rate for a year is
`max(reference, floor) + margin` -- the floor bites on the reference **before**
the margin -- and `lbo_engine` gets a plain `rate_path`, so PLAN.md 4.2 can
fill `reference_path` from market data without touching the engine. A path
shorter than the run **repeats its last year** (the sensitivity grid reruns at
holds of 3-7). Every engine default is an exact no-op (`pik_share` 0.0,
`sweep_share` 1.0, no `commitment`, empty `rate_path`), so a deal sized by
percentages is bit-for-bit unchanged and **`tests/golden/golden.json` is
untouched**; `assert_close` takes an `extra` allow-list for the fields the
model has gained, and a test checks each adds nothing for the recorded deals,
so the allow-list can't hide a real change. `equivalent_tranches` writes
today's structure out explicitly and gives the identical answer -- the
mezzanine as a **margin over the senior rate**, which is the expression the
engine always used, and no upfront fee, so the flat `fin_fee_pct` still
applies. Per-tranche upfront fees are **their own line** beside it
(`LBOParams.tranche_fees`), never a replacement, so setting one never switches
the other off. A structure raising more than the deal costs is a **422**
(`UnfinanceableStructure`, handled in `api/main.py`, and caught by
`jobs/runner.py` for a background job, which never reaches that handler); its
message names the deal's figures, so it is returned and never logged. `db/deals.py` `OMIT_WHEN_DEFAULT` (was `LABEL_DEFAULTS`)
stores the list only when a deal has one, so existing deals keep their exact
bytes and an API from before 2.4 still opens them.

Debt structures, the simulation and the screen (PLAN.md 2.4b): the Monte
Carlo simulation's output is **pinned** in `tests/test_montecarlo_baseline.py`
(recorded before 2.4b touched it; never re-record to go green). Its
two-bucket path is untouched; a deal with tranches takes
`_run_tranche_core`, whose debt is `simulation/tranches.py` -- the deal
model's schedule on arrays, no rounding. `SimulationParams.tranches` holds
`SimulationTranche`s built by `core.debt.simulation_tranches` (market
conventions stay in `core/debt.py`). **The rate draw moves floating
facilities only**: `max(reference + (draw - interest_mean), floor) +
margin`, floored after the shock and before the margin. A scenario's rate
multiplier moves `interest_mean`, so `apply_scenario` adds the change to
every floating reference (`shift_references`), and the heatmap passes it as
`reference_shift`; the heatmap runs on the deal's own structure.
`equivalent_tranches` writes the senior loan as **floating** at the base
rate (the deal model sees the same rate; the simulation always moved it),
so a converted deal simulates exactly as before, path by path. The Monte
Carlo answer's `params` echo leaves the tranches out (they are the deal's
own input, in millions there). Web: the Debt step's rail is
`steps/TrancheList.tsx` (the list order **is** the sweep order; a move
clears `sweep_priority`), the "Use explicit tranches" button
(`lib/deal/capital.ts` `equivalentTranches`, a line-for-line mirror), a
capital-structure tile and schedule rows for PIK, fees and draws only when
non-zero. `KIND_PRESETS` mirrors `core.debt.PRESETS` (a test reads the
block between `presets-begin`/`-end`). The Inputs, Returns and Summary steps
show `TranchesOnDebtStep` instead of the percentage fields a listed deal
ignores; sources & uses, the risk score (`risk_model_inputs`) and the Live
sliders read the facilities. Monte Carlo's stale check compares the list by
content (`changedKeys`), and its rail says when the rate draw moves nothing.
A listed deal's rates centre on each facility's own reference, not the
rail's rate mean, so written-out and percentage simulations agree exactly
only while the base rate equals that mean (the defaults). `mc_n_passes` is
bounded 1-10 at the API (`api/deps.py`): each pass reruns the simulation
while holding the one slot. The tranche path computes each year's rate as it
goes; holding every facility's rates at once cost about 140 MB more at the
caps (12 facilities, 100,000 paths, 15 years).
`e2e/tranches.spec.ts` proves every edit by output. Carried-forward items
from the 2.4 reviews are listed under PLAN.md 2.4.

Tax rules (PLAN.md 2.5): a deal's tax is its rate plus optional rules,
`DealInputs.tax_*` (all off by default, stored and sent only when set, like
the fiscal labels). **One mechanic for the deal model and the simulation**:
`lbo_engine/tax.py` `tax_schedule` uses only arithmetic and
`np.minimum`/`np.maximum`, so it runs on one deal's numbers and on arrays of
paths alike. Each year: net interest is claimed with what earlier years could
not deduct, against a cap of a share of EBITDA never below an allowance
(`ebitda_share`) or a fixed amount (`fixed`), and the rest is carried forward
without expiry (net interest *income* is taxed, not capped); profit for tax is
EBIT less the deductible interest; a loss is carried forward and a profit
absorbs losses up to an allowance plus a share of the excess; tax is the rate
on what is left, raised to the minimum tax on book profit. **With no rule on
nothing calls it**: `complete_income_statement` takes `taxes=None` and keeps
its flat-rate expression, the two-bucket simulation keeps its line, so the
golden snapshot and the pinned simulation do not move. `run_lbo` puts the
schedule on `LBOResult.tax`; the deal answer's `tax` block is `null` without
rules. Country presets are `core/tax.py` `PRESETS` (source, note of what is
simplified, `as_of`), served by `GET /api/deal/tax-presets`; the web applies
one client-side (`steps/TaxRules.tsx`, mirroring `apply_preset`), and a
preset's amounts come across **only in its own currency** (no exchange rates
until 4.2). The screen says "check with a tax adviser" beside them. Rates are
headline rates as of 2026-01; each preset's note says what it leaves out.
Decisions from the 2.5 reviews: `tax_preset` is any two-letter code, not a
list of today's presets, so retiring one never makes a saved deal
unreadable; choosing "No preset" switches every rule off (the rate stays);
the simulation asks `tax_schedule(..., detail=False)` for the taxes alone,
which keeps a run at the caps with every rule on to about 200 MB traced; the
Live sliders' surrogate knows only a flat rate, so `core/surrogate.py` lists
"tax rules" (how many are on) among the terms that differ from its training
deal; a `fixed` limit left at 0 disallows all interest, as the API's field
description says.

Accounting standards, the model (PLAN.md 2.6a): a deal says which standard its
EBITDA follows (`accounting_standard`: `ifrs`, `us_gaap` or "") and may carry
leases (`lease_cost` a year, `lease_liability` at close) and a `lease_view`
(`pre_ifrs16` or `post_ifrs16`; "" = the standard's own, post for IFRS). All
off by default and stored only when set (`OMIT_WHEN_DEFAULT`).
`core/accounting.py` `lease_terms` turns them into three numbers: the
**operating EBITDA** (always after lease costs: an IFRS figure less the lease
cost; rent is cash under either standard, so the operating model and cash
flows never depend on the view), a **valuation add-back** (the lease cost,
post view only) added where EBITDA meets a multiple, at entry, exit and in
every exit-grid cell, and the **lease liability counted with net debt** at
entry and exit (post view only). `LBOParams` and `SimulationParams` carry the
last two as `lease_ebitda_addback`/`lease_liability`, 0.0 by default, which
is bit-for-bit: the golden snapshot and the pinned simulation don't move.
Both are held flat over the hold (renewals), which the screen says. Sources
and uses pay EV less the liability taken over; tranches are sized on the
operating EBITDA; IFRS lease cost at or above EBITDA is a 422
(`InvalidLeases`, a kind of `UnfinanceableStructure`, so the same handler and
job refusal). The MC `params` echo leaves the lease fields out
(`ECHO_LEFT_OUT`). **Line items** per standard live in `core/accounting.py`
(`US_GAAP_ITEMS` is the old `TAG_MAP`, `IFRS_ITEMS`, `LEASE_ITEMS`);
`ml/edgar_extractor.py` picks the source (`_Source`: taxonomy, currency,
forms) -- US GAAP from a 10-K in USD when the filer has one, else IFRS from a
20-F/40-F in the currency its statements use -- and the EDGAR answer carries
`money` in that currency, `accounting_standard`, `leases` and `deal_inputs`
(EBITDA = operating profit + D&A, as the standard reports it). Real filings
recorded in `tests/fixtures/edgar/` (trimmed to mapped concepts and annual
rows): SAP's 20-F maps figure by figure, McDonald's 10-K must equal
`mcd_expected_before_2_6.json`. A forecast request takes the company's
`accounting_standard` as a label and echoes it (the web sends it only when a
filing set one, so an older API never sees it). Decisions from the 2.6a
review: with leases **everything a percentage sizes is sized on the
valuation EBITDA**, as the engine does -- sources and uses (the screen's debt
multiples too) and `equivalent_tranches` (and its web mirror,
`lib/deal/capital.ts` `valuationEbitda`), each tested against `run_deal` in
all four standard/view pairs; a filing's lease cost comes from **one
concept for every year** (`_single_tag`), and an IFRS cost adds interest on
lease liabilities when tagged, else the answer says it is principal only;
McDonald's tags no lease cost, so its cost is next year's payments due and
the answer says so. Left as is: the capital-structure summary's multiples
and the risk score read the EBITDA as entered.

Accounting standards, the screen (PLAN.md 2.6b): Deal → Inputs has an
"Accounting and leases" rail group (`steps/LeaseRules.tsx`); the EV and debt
tiles and `debtShareOfEv` use `valuationEbitda` (`lib/deal/capital.ts`, the
mirror of `lease_terms`); Returns shows a Leases tile from the answer's
`leases` block. **Statement labels follow the standard**: call sites keep
`t("rowNetIncome")` (so the catalogue check still sees the key) and wrap it,
`std("rowNetIncome", t("rowNetIncome"))`, where `useStandardLabel(standard)`
returns the `standards.<standard>.<key>` message when there is one; a new
IFRS word is a new key there, nothing else. With leases the deal's EBITDA
rows read "EBITDA after lease costs" (that is what the model grows). The
forecast company's standard is set by an EDGAR filing and editable; "Use in
deal" converts the deal to the filing's unit (`setMoney`) and then sets its
EBITDA, standard and leases. Older-API safety: lease fields are sent only
when set (`apiInputs`, `leaseInputs`), the forecast's standard only when one
is known (`withStandard`), and an EDGAR answer without `deal_inputs` or
`accounting_standard` (e2e/locale.spec.ts replays one) still works.

Plan vs actual (PLAN.md 2.7): the Backtest mode compares **any saved deal**
(the plan: its inputs and Settings) with what happened. `core/plan_actual.py`
`compare` runs the plan through the ordinary deal model (sensitivity grid off)
and simulates it **around its own assumptions** (its growth, exit multiple,
rate and gross margin as the means, Settings' `mc_*_std` spreads) for where the
actual IRR landed. Actuals are up to the plan's hold of yearly revenue,
EBITDA, net income, FCF (after interest and tax, **before debt repayment** --
the engine's `levered_fcf`) and total debt, any figure null, plus an optional
exit (EV, net debt, equity cheque; MOIC and IRR computed when left out). A
deal still held compares its years so far; **an exit after fewer years than
planned is compared with the plan rerun at that hold**. Attribution is the
exact three-way split (finding 5's), written so the multiple part is the
remainder of the EV gap, so it adds up to the cent; with leases both sides add
the plan's lease cost to EBITDA before the multiple. Actuals live in
`deals.actuals` (migration 0008, JSONB, unknown figures left out): **not a
version, not copied by duplicate, and they don't move the deal's
`updated_at`**; `GET/PUT/DELETE /api/deals/{id}/actuals`. They carry their own
currency and unit: another unit is converted, another currency is a 422. The
four inception-era deals are now the optional **example library**
(`core/examples.py`, `GET /api/backtesting/examples`; `FSE_EXAMPLE_LIBRARY=0`
hides it, and `/api/backtesting/deals` too). The old `/api/backtesting/run`
(fixed-spread prediction) stays only as the golden parity record and job kind;
the screen no longer calls it, so the examples now show their deal-model plan
(Burger King plan IRR 13.2%, percentile 74.6, where the old screen said 12.1%
and 71.5). Web: `BacktestProvider` picks the open deal, else the newest saved
deal, else an example; the plan of the open deal is its live inputs; it reruns
**only while a Backtest screen is mounted** (`activate()` returns its
cleanup), so editing a deal never spends simulations. Steps: Plan and actuals
(editor, CSV upload/download, `lib/backtest/actuals.ts`), Plan vs actual,
Error attribution, Year by year. `/privacy` says actual results are stored.

Risk warnings (PLAN.md 2.8): the deal answer's **`risk_warnings`** replace the
anomaly detector's written-in flags, which quoted statistics nobody could
source and are deleted (the detector now answers only its score and nearest
deals). `core/risk_warnings.py` returns **figures and source ids, never
sentences**; the words are `web/messages/en.json` `warnings.<id>`, and
`tests/test_risk_warnings.py` fails if one of them holds a digit, if a message
asks for an argument the warning doesn't carry, or if a figure is not the
deal's own (recomputed) or its table's. Four warnings: leverage at close above
the ECB/US supervisory **6.0x** (with leases on the post view, liability as
debt and lease cost added back); year-one **EBIT / interest** through
Damodaran's coverage-to-rating table (large non-financial firms, January 2026)
and, when speculative grade, **S&P's 2024 study table 26** cumulative default
rate at the deal's hold (CC, C and D read the study's pooled CCC/C row); a year
whose EBITDA does not cover interest; and the **unfunded repayment** -- exactly
the cash finding 11 conjures (the test reconciles every year's cash balance
with it). The tables are `core/risk_sources.py`, each with publisher, date,
the table or passage and the sample; a new edition is a transcription and a
new `as_of`, never an estimate. Warning money figures (`unfunded`,
`unfunded_total`, `repayment_due`) are in `DEAL_MONEY_KEYS`. Shown on
Deal → Inputs (`DealWarnings.tsx`) with every source, linked.

Model version (PLAN.md 3.1): **every model result carries `model`**
(`core/model_version.py` `stamp(cfg)`): `engine_version`, `commit`,
`settings_fingerprint` (over Settings *resolved* against today's defaults,
taken before `in_millions`; null for a result that reads none, the forecast
and the risk score), `data_vintage`/`data_fingerprint`/`data_sets` (the
published tables' editions: `core/risk_sources.py` `published`, the tax
presets' `as_of`). A new result endpoint adds `model: ModelStamp` to its
response model and `"model": stamp(cfg)` to its answer. **A change that moves
any number is a new engine version**: raise `ENGINE_VERSION`, add the
`MODEL_CHANGELOG.md` entry (its newest heading must match, tested), record
the reference results with `python -m tests.test_model_version` (never edit
an older version's pins in `tests/model_version_pins.json`). Saved deals and
versions store the stamp with IRR and MOIC in column `model` (migration
0009; NULL before 3.1 = `unknown`); create and autosave run the deal model
once without its grid to make it (a stamp failure is logged by kind and the
deal saved without one); open and restore answer `model_check`, shown as
"Results changed since saved" (`DealScreen.tsx` `ModelChangeNotice`) until
dismissed or the deal is saved again. Exports: `downloadWorkbook` takes the
result's stamp (required argument) and the About sheet shows it.
Every workbook (`api/routers/export.py`) opens its About sheet with "Source:
variater.com" and when it was generated (UTC), and its file properties
(author, last modified by) say `variater.com`. Forecast → Historicals has
downloads too (the figures as entered; no model ran on them, so the About sheet shows the API's own model version).

Methodology (PLAN.md 3.2): **`docs/methodology.md`** describes every
calculation in code order, with regional differences, worked figures from the
default deal, the open findings, a constants table and the Settings defaults.
`tests/test_methodology.py` fails when any function or method in `core/`,
`lbo_engine/`, `simulation/`, `analytics/` (and the ML and display modules
listed in it) is not named in the document, when a link is broken, when a
model module is not linked, or when a quoted constant or Settings default
differs from the code. So **a new model function, or a changed default, is a
methodology edit in the same PR**; a function that affects no number goes in
its appendix.

Audit history (PLAN.md 3.3): table **`audit_events`** (migration 0010),
**append-only for the API**: `fse_app` keeps SELECT and INSERT only (0010
revokes the UPDATE and DELETE 0005's default privileges gave), and
`/api/health/database` reports `audit_log_writable` as a privilege, so a
restore that forgets the revoke fails `live.yml`/`staging.yml` (DEPLOY.md
"Restoring" lists it). **Every action that changes something writes exactly
one entry, in its own transaction** (`db/audit.py` `record`, called from
`db/deals.py`); one that changes nothing writes none (an autosave of the same
content, a rename to the same name, archiving an archived deal, saving the
same settings). Actions: created (a duplicate carries `source_deal`), edited
(one per autosave that changes the deal, `fields` = input keys and
`settings.<key>`), renamed, archived, unarchived, versioned, restored (one,
though it may write two versions), actuals_saved/cleared, exported,
deleted, settings_changed (account level, `deal_id` null) and `shared`
(reserved for 7.2). **An entry never holds a figure, a deal name or a version
label** (`record` refuses other `detail` keys), so `deal_id` has no foreign
key and a deleted deal's entries stay, without its name. Exports: both export
endpoints take an optional `?deal_id=` (the web's `downloadWorkbook` and
`downloadMonteCarloSample` take a required `dealId: string | null`; null for
unsaved deals and the forecast), recorded before the file is answered, 404
if the deal isn't the caller's. Reads: `GET /api/deals/{id}/history` and
`GET /api/account/history` (with each deal's current name), shown on Deal ->
Saved deals as "Activity" with a "Whole account" switch. **Compaction**: the
nightly job-maintenance task calls `audit_compact(cutoff)`, a `SECURITY
DEFINER` SQL function only `fse_app` may run, which merges `edited` entries
older than `COMPACT_AFTER_DAYS` (7) one per user, deal and UTC day (`count`,
`last_at`, union of `fields`). `/privacy` says what the log holds.

Reference cases (PLAN.md 3.4): **`tests/reference/`** holds 26 small deals
solved by hand from `docs/methodology.md` (`cases.py`: inputs, reasoning and
the working as one formula per line; `reference_cases.xlsx`: the same as live
Excel formulas, one sheet per case, written by `python -m
tests.reference.workbook`). `tests/test_reference_cases.py` runs each through
`POST /api/deal/run` and requires every checked figure to match to 0.01
(money in the deal's unit, IRR and rates in percentage points), the typed
headline answers to equal the formulas, and the committed workbook to be
current. Formulas are plain arithmetic plus `max`/`min`, no leading minus
(`0 - x`), so Excel reads them the same. **Never change a case to agree with
the engine**: a real difference is a model finding. Cases steer clear of the
open findings except **case 08, which pins finding 11 by hand** (engine =
hand + the unfunded repayment it reports): fixing finding 11 changes that case.
A model change that moves a case's number is a working change *and* an engine
version (3.1).

Company filings, the data layer (PLAN.md 4.1a; 4.1b is the screen):
**`companies/`** is one interface (`sources.search`, `sources.fetch`) with a
connector per source -- `sec` (EDGAR company facts), `esef` (ESEF reports as
xBRL-JSON from filings.xbrl.org: EU- and UK-listed), `companies_house`
(accounts filed as inline XBRL, `ixbrl.py`; scanned PDFs have no figures,
warning `scanned_accounts`) and `edinet` (annual securities reports'
XBRL-to-CSV). Each reads its filings into `facts.Fact`s and `facts.summarize`
makes the **summary figures** (`items.SUMMARY_FIELDS`) per fiscal year:
annual, undimensioned facts only, named by the year they end in (a
52/53-week year ending in a month's first week is the month before), each
year from the highest-priority concept and the newest filing reporting it,
**millions of the filing's own currency**, EBITDA = operating income + D&A
as the standard reports them. Concept lists per taxonomy are in
`items.py` (`us-gaap` and `ifrs-full` reuse `core.accounting`'s, so the
summary equals the EDGAR answer for McDonald's, tested; FRC `core`, EDINET
`jppfs` and `jpigp` were checked against recorded filings; a `Sum` adds
concepts, `every=True` needs all parts). Standards here are `ifrs`,
`us_gaap`, `uk_gaap`, `jgaap` (labels; a deal takes only the first two).
Search reads an ISIN or LEI by its check digits (`identifiers.py`) and asks
**GLEIF** for the company and its home-register number (UK company number,
Japanese corporate number); anything else goes to every source as name,
ticker or number; a failing or keyless source is listed in `unavailable`,
never fails the search. A company nobody covers gets the fallback
`document_upload` (6.2). **HTTP** (`companies/http.py`) paces each host
within its rules and raises `SourceError` with a reason code and the host
only: EDINET takes its key as a query parameter, so **no URL or HTTP
library error text ever reaches a message or log**; redirects are followed
by hand, only to https hosts in `REDIRECT_HOST_SUFFIXES` (Companies House's
S3) and without params or credentials. Filings are untrusted input: iXBRL
over 16 MB is refused, a `scale` outside +-12 skipped, zip members over
50 MB refused, non-finite numbers dropped, and at most two loads run at
once (`sources._loads`) on the free 512 MB. Keys:
`COMPANIES_HOUSE_API_KEY`, `EDINET_API_KEY` (Render both services, GitHub
secrets); SEC, ESEF and GLEIF need none. Storage (`db/companies.py`,
migration 0011): `companies`, `company_years` (figures JSONB, filing link
and date), `edinet_reports` (EDINET lists documents by filing day only, so
the index is built from the day lists) and `source_cursors`. **Public data,
shared**: no owner, nothing about who asked. Under 4 KB a company with five
years (tested), `MAX_COMPANIES` 8,000, budget 64 MB, reported as
`company_data` by `/api/health/database` and failed at 80% by
`ops/check_database.py`. Endpoints: `GET /api/companies/sources`, `GET
/api/companies/search?q=`, `POST /api/companies/load` (both in
`RUN_PATHS`), `GET /api/companies/{source}/{id}` (stored only). Every answer
carries `money`. The nightly **`company-refresh`** task (`companies/refresh.py`,
`scheduled.yml` with `ops.scheduled task --repeat 8`, which calls again
while the summary says `more`) scans EDINET's day lists (400-day backfill,
60 a call), reloads up to 15 companies untouched for 14 days and evicts
beyond the cap; a refused EDINET key fails the run so it alerts. **Tests
replay recorded responses** (`tests/fixtures/companies/<case>/index.json`,
`companies/record.py`): every response trimmed to what the code reads,
zips with fixed timestamps, so recording a fixture again from itself gives
the same bytes (tested). Keyless cases record locally; Companies House and
EDINET record in **`record-filings.yml`** (by hand, or on a PR changing the
recorder) with the secrets, uploading `recorded-filings` (and untrimmed
`raw-filings`, never committed) as artifacts. Fixtures: McDonald's (SEC),
Tesco (ESEF, GBP, a 53-week February year), Heineken (ESEF, EUR),
Cambridge United (Companies House, FRS 102), Toyota (EDINET, IFRS) and
Nintendo (EDINET, Japanese GAAP).

PLAN.md 7.9 (added 2026-10-05, the user's request) puts native charts in
every download; 7.3a makes the cells formulas.

CI gates (PLAN.md 0.3, docs/WORKFLOW.md step 6): the `core` and `ml` jobs
measure Python coverage (`pytest --cov`, packages listed in `.coveragerc`),
put the table in the job summary and fail below their line in
**`.coverage-floor`**, which may only rise (`ops/coverage_gate.py` compares
it with the base branch's copy). The `ml` job, which runs every test, also
fails a PR when under **80% of its changed Python lines** are covered
(`diff-cover`, needs `fetch-depth: 0`). `pr.yml` fails a **PR title**
without an ECC type (`ops/pr_title.py`, the types in WORKFLOW.md step 7;
Dependabot titles are `chore(deps): …`). Floors since 2026-10-06: core 83.5, ml 84.3
(CI's measured 83.57% and 84.33% on PLAN.md 4.1a). Raise the floor when a job's
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
and 11.4 (owner's handbook). 0.3 is done (CI gates, below). 2.3 is done: **2.3a** (locale
formats and fiscal years) and **2.3b** (interface text in translation files,
right to left), both below. 2.4 was split the same way and is done: **2.4a**
(the model — debt structures of any shape and floating rates) and **2.4b**
(the simulation and the Debt step's editor), both below. 2.5 is done (tax
rules, below). 0.2 is done (own domain, below). 2.6a is done (accounting
standards, the model, below) and 2.6b is done (the screen, below), so 2.6 is done.
2.7 is done (plan vs actual, below). 2.8 is done (risk warnings, below).
3.1 is done (model version, below). 3.2 is done (methodology, below). 3.3 is done (audit history, below). 3.4 is done (reference cases, below). 4.1 was split: **4.1a is done** (company filings, the data layer, below); **next is 4.1b** (the screen: company search and figures on Forecast and Deal).

Own domain and name (PLAN.md 0.2, DEPLOY.md "Own domain"): the product is
**Variater**; production is `https://variater.com` and
`https://api.variater.com`, DNS at Cloudflare, **every record DNS only** (grey
cloud) so Vercel and Render issue the certificates. `web/next.config.ts`
redirects every path on `fse-ml.vercel.app` and `www.` to the domain (308);
staging is untouched. **Only the visible brand was renamed** (decided
2026-10-01): `app.brand`, page titles, README, the status page's name.
Internal names stay: `FSE_*` variables and secrets, the `fse_app`/`fse_api`
roles, the `fse`-prefixed keys, service names and the repository. Renamed
2026-10-02 because people see them: Clerk's application name (sign-in heading
and the sender of its emails), the Better Stack monitors (alert emails; the
sync finds each by `_previous_names`) and the status page
(`variater.betteruptime.com`; a page at `PREVIOUS_SUBDOMAINS` is moved, not
duplicated). The web monitor's keyword is `"status":"ok"`, so the health
route's service name can change without racing its deploy. `tests/test_domain.py` fails if anything the
project runs names an old host. End every task session with the handoff described in PLAN.md: tell the
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
    numbers, so it waits for approval. The rounding also **builds up**: a
    reference case (PLAN.md 3.4) with a net income of 30.075 a year ended
    year three with cash 0.015 off the hand answer, so the cases avoid
    recurring half-cents.

11. **Open (found in PLAN.md 2.4, needs the user's approval): a cash
    shortfall is funded out of nothing.** `lbo_engine/debt_model.py` step 7
    sets `ending_cash = minimum_cash + unswept` whatever happened, so a year
    with negative levered free cash flow, or one whose mandatory repayments
    exceed the cash in hand, quietly restores the balance to the minimum. The
    default deal hits it in year 5, where the mezzanine bullet
    (`maturity_years == hold`) is repaid in full with nothing funding it.
    A deal with a **revolver** now funds the shortfall honestly (a tranche
    with `allow_redraw` draws against its commitment), but every other deal
    still invents the cash, because fixing it generally would move every
    existing deal's numbers. Related: an explicit tranche list keeps its
    maturities whatever the hold, so the exit-sensitivity grid's other
    columns differ from the percentage path's, which repays the mezzanine at
    whatever hold it reruns — see `tests/test_debt_structures.py`
    `test_an_explicit_structure_does_not_stretch_its_maturities_with_the_hold`.
    The simulation's tranche path (2.4b) follows the deal model here too.

12. **Open (found in PLAN.md 2.4b, needs the user's approval): the
    two-bucket simulation throws away cash once the debt is repaid.**
    `simulation/vectorized_simulation.py` resets the cash balance to
    `minimum_cash` every year, so on a path where the sweep has repaid every
    tranche the surplus disappears instead of reaching exit equity (the deal
    model keeps it: `ending_cash = minimum_cash + unswept`). It never happens
    on the default deal (600 of debt against about 50 a year of cash), but it
    understates IRR for low-leverage deals. The tranche path keeps the cash,
    as the deal model does; fixing the two-bucket path would move the pinned
    simulation (`tests/test_montecarlo_baseline.py`), so it waits for approval.

13. **Open (found in the PLAN.md 2.4b review, needs the user's approval): the
    simulation ignores the deal's minimum cash.** `build_sim_params` never
    sets `minimum_cash_pct`, so both simulation paths run with no minimum
    cash, while the deal model keeps `mincash` on the balance sheet and has
    sponsor equity fund it (finding 1). A deal with minimum cash therefore
    simulates a slightly smaller equity cheque, and a revolver deal draws
    less. Wiring it in moves every simulation of such a deal.

14. **Open (found in PLAN.md 4.1a, needs the user's approval): EDGAR's D&A
    can come from the wrong concept.** `core/accounting.py` `US_GAAP_ITEMS`
    lists `DepreciationDepletionAndAmortization` before
    `DepreciationAndAmortization`; McDonald's tags 457 (2025, USD m) under
    the first and its cash-flow total, 2,199, under the second, so the
    EDGAR answer's (and the company summary's) D&A, and EBITDA, are low.
    Reordering changes the EDGAR answer pinned by
    `tests/fixtures/edgar/mcd_expected_before_2_6.json`, so it waits.

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
| `simulation/vectorized_simulation.py`, `simulation/tranches.py` | Vectorized Monte Carlo engine; the tranche path's debt schedule (PLAN.md 2.4b) |
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
| `db/audit.py`, `db/migrations/versions/0010_audit_events.py` | Audit history (PLAN.md 3.3): recording one entry per action, the account's history, compaction; the append-only grants and `audit_compact` |
| `api/limits.py`, `api/usage.py`, `db/usage.py` | Usage limits: rules, refusal messages, size caps, simulation slot and timeout; in-memory counters synced to Upstash (daily command budget, `python -m api.usage` prints the monthly estimate) with the database as fallback |
| `api/security.py`, `web/src/lib/security/headers.ts`, `web/src/proxy.ts` | Security headers, CSP (nonce per page request), CORS origins |
| `ops/check_headers.py` | Header scan of a deployed web app and API (`live.yml`, `staging.yml`) |
| `ops/backup.py`, `ops/encryption.py`, `ops/backup_store.py` | Backups: dump, encrypt, upload, rotate, restore and the monthly drill (`python -m ops.backup run / list / verify / restore / drill`, run by `backup.yml`); the AES-256-GCM file format; the Supabase Storage and local-directory stores |
| `SECURITY.md`, `docs/security/threat-model.md` | How to report a vulnerability; threat model, open items and the public-repo review |
| `api/observability.py`, `web/src/lib/monitoring.ts` | Request IDs, JSON logs, model-run timings, Sentry (with privacy scrubbing) |
| `ops/betterstack.py` | Better Stack uptime monitors, status page and incidents, as code (`monitoring.yml` syncs it) |
| `web/src/lib/locale.ts`, `web/src/lib/format.ts`, `web/src/lib/fiscal.ts`, `web/src/components/ui/FiscalSelects.tsx` | Locale (PLAN.md 2.3a): number, date and input formats from the account's locale and grouping; fiscal year labels and their controls |
| `web/messages/en.json`, `web/src/lib/i18n/`, `web/src/components/shell/I18nScope.tsx`, `web/src/components/shell/DocumentTitle.tsx` | Interface text (PLAN.md 2.3b): the catalogue; which language and direction a locale gets, the message loader, the page titles, and the hooks for fiscal labels, field text, engine labels and the honest labels |
| `core/debt.py` | Debt structures (PLAN.md 2.4): the tranche spec, the nine kinds as presets, the floating-rate rule, and the builder that hands `lbo_engine` a plain rate path |
| `lbo_engine/tax.py`, `core/tax.py`, `web/src/components/deal/steps/TaxRules.tsx` | Tax rules (PLAN.md 2.5): the mechanics (one function for deal and simulation), the country presets with sources and dates and the deal-to-rules conversion, the Tax rail group |
| `core/plan_actual.py`, `core/examples.py`, `web/src/lib/backtest/actuals.ts` | Plan vs actual (PLAN.md 2.7): the comparison and attribution, the optional example library, the actuals editor shape and its CSV |
| `companies/`, `db/companies.py`, `api/routers/companies.py` | Company filings (PLAN.md 4.1): the interface and its connectors (SEC, ESEF, Companies House, EDINET, GLEIF), the summary figures and concept maps, paced HTTP, the recorder; storage, budget and the EDINET index; the endpoints |
| `tests/reference/`, `tests/test_reference_cases.py` | Reference cases (PLAN.md 3.4): 26 deals solved by hand, their spreadsheet and the builder that writes it; the test that the engine matches each to 0.01 |
| `docs/methodology.md`, `tests/test_methodology.py` | Methodology (PLAN.md 3.2): every calculation written down, and the test that every model function, link, constant and Settings default in it matches the code |
| `core/model_version.py`, `MODEL_CHANGELOG.md`, `tests/model_version_pins.json` | Model version (PLAN.md 3.1): the stamp on every result, the saved stamp and the reopening check; what each engine version changed; reference results per version |
| `core/risk_warnings.py`, `core/risk_sources.py`, `web/src/components/deal/DealWarnings.tsx` | Risk warnings (PLAN.md 2.8): the four computed warnings, the published tables they read (with sources and samples), the tile |
| `core/accounting.py`, `ml/edgar_extractor.py`, `tests/fixtures/edgar/` | Accounting standards (PLAN.md 2.6): each standard's line items and the lease rule; the filing reader for 10-K (US GAAP) and 20-F/40-F (IFRS); real recorded filings (SAP, McDonald's) |
| `core/money.py`, `web/src/lib/money.ts`, `web/src/components/ui/MoneyScope.tsx` | Currency and money units (PLAN.md 2.2): conversion to and from millions, the money keys of each answer, labels from CLDR, the money on screen |
| `jobs/` | Background jobs (PLAN.md 1.9): `queue.py` (the interface and retention rules), `memory.py` and `database.py` (the two queues), `runner.py` (the in-API runner thread), `kinds.py` (what can run as a job), `config.py` (which queue and runner), `scheduled.py` (scheduled tasks and their run log), `drill.py` (the staging drill's pinned answer), `worker.py` (phase 12's dedicated worker). Served by `api/routers/jobs.py` and `api/routers/scheduled.py` |
| `api/github_oidc.py`, `ops/scheduled.py` | The scheduler's sign-in (GitHub Actions OIDC tokens, no secret) and its side of the calls: `task`, `keepalive`, `drill` (`scheduled.yml`, `staging.yml`) |
| `ops/check_database.py` | Deployed database check (reachable, migrations current, storage under 80%) for `live.yml` and `staging.yml` |
| `tests/golden/` | Snapshot of the retired Streamlit app's outputs; the parity baseline. Its generator was removed with Streamlit (see git history) |
| `tests/`, `test_*.py` | Test suite (974 tests with a database; database tests skip without `TEST_DATABASE_URL`); `tests/test_model_fixes.py` pins each finding fix, `tests/test_database.py` the database layer, `tests/test_auth.py` sign-in, `tests/test_users.py` accounts, `tests/test_deals.py` saved deals and versions, `tests/test_limits.py` usage limits, `tests/test_security.py` headers, CORS, TLS, the database role and the header scan, `tests/test_backups.py` the backup format, stores, rotation and a real dump/restore round trip, `tests/test_jobs.py` jobs on both queues, restarts, retention, the scheduler's tokens and the drill, `tests/test_money.py` currencies and units, `tests/test_debt_structures.py` the tranche kinds, the floating-rate rule, PIK, the revolver and the sweep share, each hand-checked, `tests/test_montecarlo_baseline.py` the simulation's pinned output, `tests/test_montecarlo_tranches.py` the simulation of tranches (the written-out structure path by path, each path at the mean against the deal model, floating only, scenarios, the heatmap, jobs), `tests/test_tax_rules.py` the tax rules, each hand-checked, the presets, the simulation and the heatmap, `tests/test_risk_warnings.py` the computed risk warnings (hand-checked default deal, sources, the cash reconciliation, the catalogue holds no number), `tests/test_plan_actual.py` plan vs actual (the plan is the deal model's answer, the plan fed back as actuals attributes nothing, early exits, leases, tranches, units, saved actuals, the library switch), `tests/test_accounting.py` accounting standards: real IFRS and US GAAP filings mapped, the IFRS 16 lease views by hand in the deal model, the grid, sources and uses and the simulation, `tests/test_no_hardcoded_currency.py` the dollar-sign check, `tests/test_locale.py` digit grouping and fiscal years, `tests/test_no_hardcoded_locale.py` the locale check, `tests/test_no_hardcoded_text.py` the interface-text check, `tests/test_translations.py` the catalogue (keys asked for, keys used, languages in step, the engine's own labels), `tests/test_cycle_gates.py` the CI gates (PR titles, coverage floor, the workflows keep them), `tests/test_audit.py` the audit history (one entry per action, none for a no-op or a failure, nothing writable by the API or its role, compaction, no figures in an entry). `tests/conftest.py` signs every other test in and hands out throwaway databases |

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

- **`main` is protected.** Every change: branch → PR → the 12 required
  checks green → merge commit. Direct pushes to `main` fail. `staging` has no
  protection; it only ever receives `main`. **Claude merges** (decided
  2026-10-01, the user wants one prompt per task): the GitHub CLI is signed
  in by the user with a token limited to this repository, the repository
  allows auto-merge, and the user's `.claude/settings.local.json` allows the
  `gh pr` / `gh run` commands (Claude may not grant itself permissions). The
  stop-and-ask cases are in docs/WORKFLOW.md step 7. If `gh auth status`
  says signed out, fall back to giving the user browser steps.
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
  `nospc` errors mean this, not a code problem. **At exactly 0 bytes Claude
  can't run any command at all**: the desktop app writes each command's output
  to `%LOCALAPPDATA%\Temp\claude\...`, so every Bash and PowerShell call fails
  with `ENOSPC` before it starts, and the sandbox refuses
  `Remove-Item` on `%LOCALAPPDATA%\npm-cache`. Only the Read, Write, Edit,
  Grep and Glob tools still work (the repo is on D:). The user has to free the
  space; `%LOCALAPPDATA%\Temp\claude` holds the old sessions' working files and
  is the usual culprit. It happened again on 2026-09-27, during PLAN.md 2.3b.
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
- In this repo's PowerShell, `Get-Content -Raw` then `WriteAllText` turns
  a non-ASCII character into mojibake (an NBSP became "Â "). Edit source with
  the Edit/Write tools or a Python script that writes UTF-8; keep test
  strings with special spaces as escapes (`"21,2\u00a0%"`).
- A new field on `DealInputsIn` must also go on `core.deal.DealInputs`
  (every router builds it with `DealInputs(**model_dump())`), and a field
  that is only a label, or that old deals should not gain, belongs in
  `db/deals.py` `OMIT_WHEN_DEFAULT` and web
  `apiInputs()`, or old deals gain a version and a rollback breaks them.
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
- A new tranche mechanic in `lbo_engine/` must default to an **exact no-op**
  (`x * 1.0`, `+ 0.0`, an empty path), or every deal sized by percentages
  moves. A new *field* on `TrancheYearRecord`, `DebtScheduleResult`,
  `CashFlowResult` or `SimulationParams` can't be a no-op at all:
  `tests/golden_compare.py` compares key sets, so it must be named in
  `TRANCHE_FIELDS` in `tests/test_core_parity.py` **and** shown to add nothing
  for the recorded deals. Never edit `tests/golden/golden.json`.
- A change to the simulation (`simulation/`) must leave
  `tests/test_montecarlo_baseline.py` green; a new tranche mechanic goes in
  **both** `lbo_engine/debt_model.py` and `simulation/tranches.py`, and a
  structure exercising it joins `STRUCTURES` in
  `tests/test_montecarlo_tranches.py`, which holds each path at the mean to
  the deal model. Pick a structure where the mechanic reaches exit: a
  revolver's fee washed out when the revolver ended fully drawn, and the
  first version of that test missed a deleted fee.
- A saving the model makes in a year whose cash is below its mandatory
  repayments vanishes into the cash finding 11 invents, so the IRR does not
  move. A test that wants a tax saving (or any extra cash) to reach the
  returns needs a year with cash left to sweep: `tests/test_tax_rules.py`
  uses a fast-growing deal whose senior loan does not amortise.
- The local database (`python -m db.local`) picks a new port when it restarts
  after an unclean stop, and its log shows recovery first; wait for
  `pg_isready` and use the port in `.localdb/postmaster.pid`.
- A `<fieldset>` (every `RailGroup`) is as wide as its widest unbreakable
  content, and a grid's `1fr` column as wide as its widest item: a long name
  in the rail widens the whole rail. Use `grid-cols-[minmax(0,1fr)]` and
  `min-w-0` with `truncate` (`TrancheList.tsx`); check with
  `aside.scrollWidth` via `javascript_tool`, since screenshots time out.
- Playwright's `getByLabel(x, { exact: true })` never finds a `<select>`
  wrapped in its `<label>` (the label's text includes every option); use
  `getByRole("combobox", { name: x, exact: true })`. The full browser suite
  times out in places when something else is using the machine (a pytest
  run, review agents); rerun the failed specs alone before suspecting code.
- A new money figure in a deal, simulation or backtest answer must be added
  to that answer's money-key list (`DEAL_MONEY_KEYS` and so on), or a deal
  in thousands shows it a thousand times too small; `tests/test_money.py`'s
  leaf-by-leaf tests catch it. A new money input is converted in
  `in_millions` (and a money setting in `MONEY_SETTINGS`).
- A new action on a deal or the account's settings writes its audit entry
  with `db/audit.py` `record` **inside the same transaction** and only when
  something changed; a new kind of action is a new value in
  `AUDIT_ACTIONS` **and** a migration changing the `ck_audit_events_action`
  check (and `AuditAction` in `api/schemas.py`, and `deal.activity_<action>`
  in `en.json`). Never put a figure, name or label in `detail`.
- A new top-level Python package the API imports (as `jobs/` in 1.9) must be
  added in three places: a `COPY` line in the `Dockerfile` (it copies
  packages by name), `buildFilter` in `render.yaml`, and `API_PATHS` in
  `staging.yml`. CI's `docker` job fails at start-up if the first is missing.
- A connector's figures must be checked against a **recorded real filing**,
  not a guessed concept list: Toyota's IFRS report and Tesco's capex use
  concepts nobody would guess. Record with `record-filings.yml` (keyed
  sources) or `python -m companies.record <case>` (keyless), read the
  untrimmed `raw-filings` artifact for concept names, then fix the map. A
  replayed case resets `sec.reset_cache()` and `edinet.reset_cache()` first:
  each fixture's name lists are trimmed to that case.
- A new endpoint that runs a model or calls an outside source: add its path
  to `RUN_PATHS` in `api/limits.py`; if it simulates, also to
  `SIMULATION_PATHS` and put `@simulation_slot` under its route decorator.
  A new simulation-size input needs a cap (`SimulationPaths` in
  `api/schemas.py`).
- Anything polled often (`/api/health`, the uptime monitor, Render's health
  check) must not touch the database, or Neon never scales to zero.
- API in Render Oregon, database in Neon Ohio: ~50–70 ms a round trip. Batch
  queries per request.
- A provider that reads the translations itself must sit **below**
  `I18nScope`, not above it. `ProfileProvider` reports "the account service is
  unreachable", so it needs words before the account can say which language:
  `AppShell` wraps it in an outer scope on the browser's language and
  `ProfileI18nScope` overrides that once the profile arrives. Get it the wrong
  way round and every screen dies in `global-error` with next-intl's "context
  from `NextIntlClientProvider` was not found" -- which the browser tests
  catch, but only `next dev` names the component.
- One `useTranslations` name per namespace per file. `tests/test_translations.py`
  reads `const x = useTranslations("ns")` and the `x("key")` calls that follow,
  so two `const t =` in one file for two namespaces makes it check the wrong
  catalogue (it reported keys as missing until `DealScreen.tsx` and
  `AppShell.tsx` used `fields` and `app` for their second scope).
- In a right-to-left locale `Intl` wraps a per-cent sign and a leading sign in
  **left-to-right marks**: an `ar-EG` account's 21.2% is "21.2", U+200E, "%",
  U+200E. They are invisible, and they are exactly what keeps the sign in front
  of the number, so a test that compares a figure exactly has to strip U+200E
  and U+200F first (`e2e/text.spec.ts`'s `plain()`).
- An ICU argument that is a **number** is formatted with the account's digit
  grouping, so a year would read "FY2,025". Pass years, indexes and ids as
  `String(...)`; pass a real quantity as a number (a plural's `{count}`, a
  duration in milliseconds) where grouping is what you want.
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
