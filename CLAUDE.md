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

Last updated: 2026-10-12 (PLAN.md R4 part b, first slice: the model format, evaluator, Excel writer and model 1; the same day R4 part a: model 1 by hand, and 6.1: file uploads). Before that 2026-10-11 (PLAN.md 5.9: personalized defaults; the same day 5.8: live sliders for any deal, and 5.7: economic regime by region). Before that 2026-10-10 (PLAN.md 5.6: driver explanations; the same day development moved to a Mac: Commands, Tooling and Gotchas below). Steps 1–5 and the model finding fixes are merged
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
`E2E_LIVE=1 npm --prefix web run test:live`,
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
2026-09-18. **The key was replaced on 2026-10-12** (the first one was kept
only in the GitHub secret and could not be read back): the backup and restore
drill run by hand that day passed with the new key, and **backups stored
before 2026-10-12 cannot be decrypted**; rotation removes them in time.

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
Live sliders run the simulation itself since 5.8, so they see every rule; a `fixed` limit left at 0 disallows all interest, as the API's field
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

Company filings, the screen (PLAN.md 4.1b): **the EDGAR ticker box is
gone**; a company search (`components/companies/`: `CompanyProvider` in
the shell, inside `ForecastProvider`, so a company found on the deal is
there on the forecast) sits at the top of Deal -> Inputs' rail and in
Forecast -> Historicals' rail, with a figures-by-year tile linking each
year to its filing. Results from every source, sources not searched (and
why), a source this server has no key for shown but not loadable, and the
`document_upload` fallback named when nothing matches. The load answer
carries **`deal_inputs`** and **`forecast_history`** (`companies/use.py`):
the deal takes the latest EBITDA as the standard reports it, the currency,
and the standard and leases only for IFRS and US GAAP (`notes`:
`standard_not_in_deal`, `lease_cost_missing`); the forecast rows put every
operating cost in cost of sales, unnamed assets in PP&E and unnamed
liabilities in other non-current liabilities, so EBIT, EBITDA and net
income are the filing's and the balance sheet balances (tested on every
recorded company). "Use in forecast" on a SEC company with a ticker takes
`/api/edgar/{ticker}`'s full statements and falls back to the summary; a
forecast not yet loaded applies the company once its defaults arrive (a
ref, not state set in an effect). Browser tests replay
**`web/e2e/fixtures/companies.json`** (`replayCompanies` in
`e2e/helpers.ts`), written by `python -m tests.e2e_companies` from the
recorded filings through the real API; `tests/test_company_use.py` fails
when it is stale, so **an API change to the company answer means
rerunning it**. `/privacy` says a search goes to the public registers. A
company answered with **no years** (an EDINET company before the nightly
`company-refresh` has indexed its filings, scanned UK accounts) shows no
standard, says why once (its warning) and offers neither button: the
connector's standard is a default until a filing is read. EDINET companies
load on an environment only after that refresh has run there (first run
after 4.1a: the 2026-10-07 nightly).

Economic data (PLAN.md 4.2, DEPLOY.md "Economic data"): **`economy/`** reads
six free sources into `Series` keyed `<indicator>.<area>.<source>`
(`connectors.py`, one or two calls each for every economy, paced through
`companies/http.py`): the **IMF WEO** (DataMapper; growth and inflation,
history and projections), the **World Bank** (latest actual years), the
**BIS** (policy rates, daily), the **OECD** (10-year yields, 3-month
interbank, monthly), the **ECB** (€STR, 3-month EURIBOR, euro reference
rates) and **FRED** (SOFR, SONIA as `IUDSOIA`, US 10-year `DGS10`;
`FRED_API_KEY` on both Render services and in GitHub secrets). 24 economies
plus the euro area `XM` (`catalogue.py`); a euro member's policy rate is
`XM`'s. `views.py` decides what is **current** by age (daily 45 days,
monthly 100, policy rate 120, the IMF's figure for this calendar year, a
World Bank year at most two back) and picks each benchmark's level from
`REFERENCE_SOURCES`, best first: SOFR and SONIA as published, else the
Fed's/BoE's policy rate; €STR and EURIBOR from the ECB; **TONA, SARON and
MIBOR stand in with their central bank's policy rate and BBSY with
Australia's 3-month bank bill rate** (no free source; the answer's `basis`
says so). **Corporate credit spreads are not shown** (decided 2026-10-07):
FRED's ICE BofA and Moody's series forbid reproduction; the app shows each
euro member's sovereign spread over Germany from the OECD's CC BY yields, and
PLAN.md 12.9 holds the licensed source. Storage (migration 0012,
`db/economy.py`): `economic_series` (newest 24 daily/monthly or 12 annual
observations per series) and `exchange_rates` (one JSONB row a day, units
per euro, 400 days), an 8 MB budget reported as `economic_data` by
`/api/health/database`. Endpoints (signed in, reading storage only, so not
runs): `GET /api/economy/countries`, `/reference-rates` (with
`currency_benchmarks`), `/exchange-rates?base=&on=` (`published_on`, the last
ECB day on or before). The nightly **`economy-refresh`** (`scheduled.yml`,
both environments, and in `staging.yml` after every staging deploy, whose
health read requires 20 economies current and rates under seven days old)
**fails when the FRED key is refused or fewer than 20 economies are
current**, after storing what it read; a source that fails, or answers a
shape its connector doesn't know (`unreadable`), keeps its stored series. **Market data never
enters a run by itself**: the web (`lib/economy.ts`) starts a new floating
facility on its currency's benchmark at today's level and sets the level
when a benchmark is chosen; the tranche editor shows its date and source
and offers "Use today's level". The level is then a stored input, so a
saved deal's answer never moves with the market and no engine version
changes. Tests replay `tests/fixtures/economy/` (`economy_open` recorded
locally with `python -m economy.record`, `economy_fred` by
**`record-economy.yml`** with the secret) judged on the day recorded
(`_recorded_on` in its index); the browser tests replay
`web/e2e/fixtures/economy.json` from `python -m tests.e2e_economy`, checked
stale by `tests/test_economy.py`.

Sourced starting figures (PLAN.md 4.3, DEPLOY.md "Industry averages"):
**`benchmarks/`** reads Aswath Damodaran's industry averages (NYU Stern,
published each January; 43,056 listed companies; five data sets --
margins, EV/EBITDA, capex, working capital, debt -- for eight regions: the
US, Japan, China and India each a file of their own, developed Europe,
Australia/NZ/Canada, emerging markets, global) and his copy of the Tax
Foundation's country tax rates, from `.xls` workbooks (`xlrd`; sheet and
header row found by the "Industry Name" row, since they vary). The scheduled
**`benchmarks-refresh`** (nightly in `scheduled.yml`, and after each staging
deploy, which requires 41 tables and 80+ industries) reads all 41 only once
the stored ones are a week old; table `benchmark_tables` (migration 0013),
4 MB budget, `benchmark_data` in `/api/health/database`.
`benchmarks/starting.py` makes a deal's starting inputs, each with source,
group, sample and date: margins, D&A and opex from one group (opex = gross
- operating margin, D&A = EBITDA - operating margin, so the engine's EBITDA
margin is the industry's); entry = exit multiple; capex = D&A x capex/D&A
(**not** Damodaran's "net cap ex/sales", which adds acquisitions and R&D);
days on revenue and cost of sales; `nwc` = w x g/(1+g). A group under
`MIN_FIRMS` (20) or with an unusable figure hands over along the country's
`chain` (own file, region, global) and the answer says why. **Decided with
the user (2026-10-07): growth is the country's nominal GDP growth (IMF
projection, real x inflation), never Damodaran's growth averages (they run
14-24%); debt starts at the industry's listed-company debt/EBITDA, all
senior, priced at the currency's benchmark plus Damodaran's default spread
for the industry's coverage rating.** Size is asked (EBITDA) but nothing free
splits by size, which the screen says. These are inputs, stored in the deal
like typed figures, so nothing in the engine or the golden snapshot moved;
the deal carries `country` and `industry` labels (`OMIT_WHEN_DEFAULT`,
`START_LABEL_DEFAULTS`). Endpoints (signed in, storage only, not runs): `GET
/api/benchmarks/industries`, `GET /api/benchmarks/starting?country=&industry=&currency=`
(a retired code such as `DD` or `UK` is read as its current country). Web:
Deal -> Inputs opens with **Starting point** (country, industry, currency,
EBITDA, "Use sourced figures"; the old Money group is merged in), an
"Illustrative figures" notice until a country is chosen, and a **Starting
figures** tile beside the deal's own values with "Use" per figure and "Use
all". **The defaults registry** `benchmarks/registry.py` gives every deal
input and every Setting a basis (`sourced`, `deal`, `choice`, `template`,
`pending`); `tests/test_defaults_registry.py` is the CI check, and its
`PENDING_PINNED` (4.4's ranges, correlations and scenarios; 4.5's fees and
amortisation) may only shrink. Tests replay trimmed real workbooks
(`tests/fixtures/benchmarks/damodaran`, `python -m benchmarks.record`, needs
`xlwt`); the browser tests replay `web/e2e/fixtures/benchmarks.json` from
`python -m tests.e2e_benchmarks`, checked stale by `tests/test_benchmarks.py`.

Risk ranges, correlations and scenarios (PLAN.md 4.4): **every Monte Carlo
Setting is sourced** for the deal's country and industry by
`benchmarks/risk.py` (`GET /api/benchmarks/risk`), each with source, group,
sample and years, on history from `benchmarks/history.py`: Damodaran's
**archive** of past January editions (file `marginEurope16.xls` = the
January 2017 edition = **year 2016**; the current edition is last year) per
group, as tables `history.<group>` plus a read log `history.read`, and
`history.macro` (IMF growth and inflation, BIS monthly policy rates averaged
over full years, from `WINDOW_START` 2000). Both live in `benchmark_tables`
(budget now 8 MB; `all_tables()` leaves `history.*` out unless asked).
`benchmarks-refresh` reads the current tables, then (in a later call, to
stay inside the scheduler's two minutes) **a dozen archive files a call**,
answering `more` until done, so both workflows run it with `--repeat 24`;
staging requires `history_groups == 8`, `history_left == 0` and
`macro_economies >= 20`. A file read once is final, except a missing one of
the newest archived year. Means = the 4.3 starting figures; spreads = the
industry's year-to-year spread of EV/EBITDA and gross margin (closest group
with `MIN_YEARS` 6), the country's nominal GDP growth, the policy rate's
yearly change; correlations per group = Spearman of yearly changes pooled
over industries, turned normal (2 sin(pi rho / 6)), shrunk toward identity
only when needed (`valid_matrix`, eigenvalue floor 0.05 so 2-decimal
rounding stays valid); presets = the country's weakest-growth,
highest-inflation and strongest-growth **fifth of years** since 2000, listed
as periods, moving each mean by what happened then (a rate change or a bull
growth change is written as a multiplier of the sourced mean). **These are
Settings values applied by the user** (Monte Carlo rail group "Sources",
"Use sourced figures"; the Scenarios step's tile lists sources and preset
years), stored with the deal; the engine and the pinned simulation are
untouched. **Decided 2026-10-07: the simulation's means stay Settings**
(applied with the rest), not tied to the deal, so the screen's stale
behaviour and pinned Monte Carlo numbers are unchanged. The registry's only
`pending` Settings left are 4.5's fees and amortisation. Tests replay
`tests/fixtures/benchmarks/history` (`python -m benchmarks.record --history`,
about 15 minutes paced; workbooks trimmed to ten industries, an unreadable
edition kept as its first 4 KB); browser tests replay the `risk` answers in
`web/e2e/fixtures/benchmarks.json`.

Reference library, part one (PLAN.md 4.5a; 4.5b adds the sourced
transactions and two-person review): a **Library** tab (`nav.ts` mode
`library`, `optional: true`, last so hiding it moves no Alt shortcut; the
shell lists modes through `components/shell/useModes.ts`) with Base rates,
Examples and Coverage. **Base rates** (`library/base_rates.py`,
`GET /api/library/base-rates?country=`) are transcribed tables, never
estimates: S&P's 2024 global default study (the Maalot copy, as 2.8 uses;
Tables 1, 3, 5, 24, 25) and Global Credit Data's 2020 LGD report on bank
loans to large corporates (Tables 2-4), parsed from the PDFs with
`pdftotext -table` (not `-layout`, which scrambles rows) and checked in
`tests/test_base_rates.py` against the sources' own summary rows (S&P's
Table 4 minimum, maximum and median per rating; GCD's totals). **Decided
with the user (2026-10-07): fixed editions, not live data** -- each source
has `checked_on`; a year later the screen says "due a check" and
`ops/check_base_rates.py` fails `live.yml`; then transcribe a newer edition
or confirm none is free and move `checked_on`. Nothing in the deal model,
simulation or risk warnings reads the library. **Coverage**
(`library/coverage.py`, `GET /api/library/coverage`) counts deals by S&P
region, size (entry EV in US dollars), sector, era and outcome, empty
buckets listed; the four inception deals are the `examples` collection
(unsourced, tagged US in `EXAMPLE_COUNTRY`) and `reference_deals` is
the approved reference transactions (4.5b, below). **The admin switch** (`library/switch.py`): on unless an
administrator hides it (`PUT /api/library/switch`, table `app_flags`,
migration 0014, one `library_switched` audit entry per change, none for a
repeat) or `FSE_EXAMPLE_LIBRARY=0` forces it off (409 to switching then).
**Administrators are `FSE_ADMINS`** (comma-separated Clerk user ids on the
Render service; none set means nobody can switch). The stored choice is
cached a minute per process (`CACHE_S`; `tests/conftest.py` resets it).
Off: the tab disappears except for administrators, `/library/*` says it is
off, examples and library endpoints answer `enabled: false`;
`e2e/library.spec.ts` opens every other step with it off. `core/examples.py`
no longer reads the switch; the routers do.

Reference library, part two (PLAN.md 4.5b): **reference transactions**,
real buyouts with **every figure read from a filing**
(`library/reference_deals.json`, `library/references.py`). A figure is one
place in a filing (`source`, `where`, a few verbatim words in `quote`) or
the sum of cited `parts` (EBITDA as operating income plus D&A, debt as its
facilities, fees less financing costs); money in millions of the deal's
currency, plus a filed US dollar value for any other currency (sizes are
US dollars). The repository proposes ten (SEC filings only: HCA, Toys "R"
Us, Dollar General, Domino's, Gymboree, Dun & Bradstreet, NXP, Masonite,
Avago, Focus Media; 1998-2019; seven exits, three distressed). **Inclusion
rules** (`problems`, codes the screen words) refuse what can't be checked:
a missing required figure (transaction value, EBITDA, debt), a source that
is not on `FILING_HOSTS` over https, an unused source, pieces off by more
than 0.05, a multiple outside 2-40x, debt above the value, a fee outside
0-10%, dates out of order, an event that doesn't fit the outcome.
**Balance rules** (`balance`) are advice: a bucket over half the library
once it holds six, and the empty buckets a proposal fills. **Two-person
review** (`library/review.py`, `db/references.py`, migration 0015: tables
`reference_deals`, `reference_reviews`): only administrators (`FSE_ADMINS`)
see Library -> Review or propose (from a JSON file in the repository's
format); a proposal joins on the **second approval by an administrator
other than its proposer** (the repository's have no proposer), one
rejection with a reason code decides it, a rules finding blocks approval
(409 with the codes), each administrator reviews once, and a corrected
version of an approved deal supersedes it. The review runs in one
transaction with the proposal row locked (`FOR UPDATE`), so two
administrators can't both be the deciding approval. The repository's
proposals are queued (`sync_repository`, keyed by `key` and
`content_hash`) whenever an administrator opens the queue; **changing a
transaction in the JSON makes a new proposal**, approved again. Audit
actions `reference_proposed` and `reference_reviewed` (detail `reference`,
`verdict`). Library -> Reference deals lists the approved ones, every
figure linked to its filing; Coverage counts them (GICS sectors) with how
many await review. **Fees and amortisation** (`library/fees.py`): the
median of each across approved deals that give it (at least
`MIN_DEALS` 3), offered on Settings -> Fees beside the values in force with
"Use sourced figures"; decided in the session: **offered Settings, never
factory defaults**, like 4.4's ranges, so no saved deal and no engine
version moves. With the ten approved: transaction fees 1.24% of value (six
deals), financing fees 3.07% of debt (seven), senior amortisation 1% a year
(three). The defaults registry has no `pending` entry left. Browser tests
replay `web/e2e/fixtures/references.json` from `python -m
tests.e2e_references`, checked stale by `tests/test_references.py`.

Model validation (PLAN.md 4.6): **`validation/`** checks whether the
model's risk read comes true as often as it claims, nightly
(`validation-report` in `scheduled.yml`; `staging.yml` runs it after every
staging deploy and requires `checks == 3` and `splits == 4`), into table
`validation_reports` (migration 0016, newest 30, about 20 KB each), read by
`GET /api/validation/report` (storage only, not a run) and shown on
**Backtest -> Validation**. Three checks: `default` (the summary's default
risk, `credit_view`, for each approved reference transaction entered with
only its filed EBITDA, multiple and debt share, everything else at
defaults, against distress within five years of closing); `irr_range`
(where the actual IRR fell among the plan's 2,000 simulated paths: share
inside the central 50/80/90% with Wilson intervals) and `loss` (the plan's
share of paths below zero IRR against a loss), both from
`core.plan_actual.compare` on **opted-in** users' deals with an exit.
Statistics: Brier, z, bias in points, `MIN_CASES` 5 per group; splits by
region, GICS sector (`tags.INDUSTRY_SECTOR` maps Damodaran's industries;
a test keeps it complete), size in US dollars (ECB rate on the plan's day)
and era. **Decided (2026-10-08): "always tested on newer data" is strict**:
a case is `out_of_time` (the headline) only when its outcome became known
after everything its prediction read -- for default risk after
`cases.fit_until()` (2025: S&P's sample ends 2024, Damodaran's January 2026
table reads 2025), for a user's deal when the plan is the newest version
saved before `deals.actuals_first_saved_at` (kept through a clear; NULL for
actuals entered before 4.6, so those are in-sample). The ten repository
transactions are therefore all in-sample today (predicted 16.73%, observed
10%: Masonite). **Anonymity by aggregation**: nothing per deal is stored;
a group with 1-4 users' deals is hidden (`MIN_CONTRIBUTED`) and the
next-smallest groups with it until the hidden ones hold five, so
subtracting shown groups from the overall figures can't single one out
(a random-property test). Opt-in is per deal (`deals.validation_opt_in`,
`GET/PUT /api/deals/{id}/validation`, audit `validation_opted_in`/`_out`,
not a version, no `updated_at` change), on Backtest -> Plan and actuals;
`/privacy` describes it. `db/validation.py` `contributions` is the one
read of other owners' deals, inside the task only, at most
`MAX_CONTRIBUTIONS` (400) a night. **The deal summary shows the full metric
set**: IRR, MOIC, probability of loss and downside IRR (the simulation's
`p_loss`, new in `risk_summary`, and P5; Summary runs the Monte Carlo
itself once per visit when no current result exists), interest cover and
default risk (the deal answer's new `credit` block, always present; the
`implied_rating` warning reads the same function). New outputs only: no
engine version moved. Browser tests replay `web/e2e/fixtures/validation.json`
from `python -m tests.e2e_validation`, checked stale by `tests/test_validation.py`.

ML evaluation and model cards (PLAN.md 5.1): **`ml/evaluation/`** tests every
model the app loads the same way. `harness.py`: `walk_forward` time splits
(fit before each cutoff year, test up to the next), S&P's four regions (as
the validation report), each statistic for the model **and a simple
baseline** on the same cases, verdict `beats_baseline` only when strictly
better on the headline statistic, `not_enough_data` under `MIN_CASES` 5.
`card.py` is the one template: `ml/cards/<id>.json` (compared by CI) and
`docs/model-cards/<id>.md` (rendered from it), both **generated by `python -m
ml.evaluation evaluate`, never edited**. **`ml/registry.json`** lists every
model and its trained files; the card holds their SHA-256, and
`tests/test_model_cards.py` (ml job) re-evaluates every model and fails when
a card is not what its files give or a `.pkl`/`.pt`/`.onnx` in `ml/` is
unregistered. **`ops/model_gate.py`** (tests.yml ml job, required): a
statistic worse than the base branch's card beyond the base card's
tolerance, overall or in any region, or a region losing its results, fails
the PR. **`ml.yml`**: evaluates on PRs touching models or `simulation/`;
trains by hand (`workflow_dispatch`, which Claude's token can't trigger)
into a scratch directory and uploads files and card as an artifact for a
person to review and commit -- never commits itself. Cards: the **deal
risk score** (5.2, below); the **distress predictor** (5.3, below); the **multiple predictor** (5.4, below); the **growth calibrator** (5.5, below); the **economic regime** (5.7, below; a card may name its own regions with `regions_note`, as it does); the **live sliders** (5.8, below; id `surrogate`) on 230 varied regional deals
(`ml/evaluation/data/regional_deals.json`, from the recorded sources by
`python -m tests.ml_regional_deals`, checked stale). A
model with nothing to train has `"train": null` and `"artifacts": []` in the
registry and `training.command` null in its card; **no registered model has
anything to train since 5.8**, and `python -m ml.evaluation train <id> --out
DIR` then writes the card as it stands into DIR (ml.yml's train job, which
only the user can edit, still runs it on every PR touching `ml/`). `card.write` renders the
Markdown from the stored (key-sorted) JSON, as the staleness check does. Nothing the app shows changed. **Retraining by hand is parked** (the user,
2026-10-08: PLAN.md 5.10, not scheduled): don't offer the Run workflow
button as a step for the user; tasks 5.2-5.9 retrain in their own PR.

Deal risk score (PLAN.md 5.2): **`ml/anomaly_detector.py` compares the deal
with its industry's listed companies in its own region** and trains nothing
(its pickles and the 30 unsourced deals are gone; capabilities'
`anomaly_detector` is always true). Leverage, entry multiple and EBITDA
margin against Damodaran's stored averages (4.3) for the deal's industry,
the peer group being the country's own file or its region with
`MIN_FIRMS` 20 companies, **never the global group** (a thin region says
"not enough data", names the group and uses nobody else's companies); each
as z = (deal - industry) / the robust spread across the group's industries
(1.4826 x MAD, `MIN_INDUSTRIES` 10), `risk_z` signed so positive is riskier.
**Score** = sum of the positive `risk_z` of leverage and price (the margin
is shown, not scored: the test cases lack revenue); unusual at `risk_z` >= 2.
**The score's value is sent only where its card (`ml/cards/deal_risk.json`,
read at run time) says "beats the baseline" for the deal's S&P region**:
today the US only (5 sourced transactions, AUC 0.67 vs leverage's 0.33);
elsewhere the screen shows the comparison and "Not enough data" for the
score. The card (`ml/evaluation/deal_risk.py`) scores the repository's ten
reference transactions (each given a Damodaran industry in `INDUSTRY`)
against `ml/evaluation/data/peer_tables.json` (`python -m
tests.ml_peer_tables`, checked stale); no time split (nothing is fitted).
**Library on**: approved reference transactions in the same S&P region and
GICS sector (`validation.tags.INDUSTRY_SECTOR`), same size bucket first
(entry value in US dollars at the ECB's newest rate); **off**: the same
answer without deals (`deals.enabled` false). `POST /api/ml/deal-risk`
(`DealRiskResponse`) reads storage only; without a database it answers
`no_peers`, without a country `no_country`. The deal's tile is
`components/deal/DealRisk.tsx` (namespace `dealRisk`); browser tests replay
`web/e2e/fixtures/deal-risk.json` (`python -m tests.e2e_deal_risk`, checked
stale by `tests/test_deal_risk.py`), except the first, which checks the
real endpoint gets the deal's leverage.

Distress predictor (PLAN.md 5.3): **`ml/distress_model.py` trains nothing**
(its 39 hand-entered cases are gone). Each year's **rating band** is the
weaker of EBIT / interest through Damodaran's coverage table
(`core.risk_sources.COVERAGE_BANDS`) and debt at the start of the year over
the year's EBITDA through **S&P's Corporate Methodology** (January 2024,
`sp_corporate_methodology_2024`: Table 17 standard volatility bounds, then
Table 3 with the deal's `business_risk`, the weaker anchor where it prints
two), folded to letter grades; the year's default rate is the band's
**forward rate at that age** from S&P's 2024 study, **Table 25 for us,
europe and emerging, Table 24 (global) otherwise**
(`library/base_rates.py` `CUMULATIVE`, read whatever the library switch
says). `business_risk` is a deal input, 1-6, **default 4 "fair"** (decided
with the user 2026-10-08), `OMIT_WHEN_DEFAULT`, registry basis `choice`.
The deal answer's `distress` and the Monte Carlo answer's `distress` (share
of paths per band each year; the simulation returns `credit_paths` only when
`run_vectorized_simulation_full(..., credit=True)`, dropped after use by
`core.montecarlo.simulated_distress`; the pinned simulation is untouched;
about 20 MB more traced at the caps, 100,000 paths and 15 years).
**Probabilities are sent only where `ml/cards/distress.json` says "beats the
baseline"** for the deal's S&P region; **today nowhere** (decided with the
user: the phase rule holds): on the ten reference transactions AUC 0.43
against the year-one default risk's 0.52 (US 0.25 vs 0.375). The card's
second set, `calibration`, holds a one-band deal to S&P's table for every
region, band and horizon (gap 0.00, tolerance 0.01 points). Screens: Deal ->
Debt "Distress by year" and the Credit rail group (business risk), Monte
Carlo -> Distribution (`components/deal/Distress.tsx`, namespace
`distress`); `country` and `business_risk` mark Monte Carlo stale.

Multiple predictor (PLAN.md 5.4): **`ml/multiple_predictor.py` trains
nothing** (its 25 typed-in rows are gone). The deal industry's latest
EV/EBITDA among its region's listed companies (Damodaran, 4.3; peer group as
5.2: own file or region, `MIN_FIRMS` 20, **never global**) is moved by the
group's own record of moves over the same horizon in his archive (4.4's
`history.<group>`, every edition since 2011): the moves are fitted by least
squares on the industry's gap from the group's median industry that year
(**mean reversion**, `Fit`: `b` is negative in every group), and the range is
the 10th/50th/90th percentile of what the line missed (`MIN_PAIRS` 30).
Entry is the year after the latest edition, exit entry + hold. **Each range
is shown only where `ml/cards/multiples.json` says it beats the region's
whole market** (sets `entry` and `exit`, headline the interval score), read
for the **peer group's** S&P region (`GROUP_REGION`; a Korean deal's peers
are emerging markets), and **only at horizons the card tested**
(`TESTED_HORIZONS`: entry 1, exit 4-8, so a hold above 7 gets no exit
range, `hidden: untested_horizon`); today every region beats it in both,
about 73% of held-out multiples inside
the 80% range. The card is **walk-forward, a cutoff a year from 2014, on
every industry** (`ml/evaluation/data/multiple_history.json`, 13,544
industry-years), written by `python -m tests.ml_multiple_history`, which
calls Damodaran's archive (about 120 workbooks, a few minutes); the browser
fixture `web/e2e/fixtures/multiples.json` comes from `python -m
tests.e2e_multiples` (answers given on a fixed day, `DAY`).
`POST /api/ml/multiples` (in `RUN_PATHS`, reads storage only, with only
the deal's own groups' history: `all_tables(history_groups=...)`) answers the ranges, the industry in every group, its sector
in the region and, with the library on, the reference transactions like it.
Screen: Deal -> Returns, Multiples tile (`components/deal/Multiples.tsx`,
namespace `multiples`), "Use suggestion" sets `entry_mult` and `exit_mult`.

Growth calibrator (PLAN.md 5.5): **`ml/growth_calibrator.py` trains
nothing** (its fixed sector table is gone). A deal's revenue growth range
comes from how **listed companies' revenue really grew**: every SEC filer's
yearly revenue from the SEC's **XBRL frames** API (one call answers one
concept and year for every filer; US GAAP and IFRS revenue concepts from
`core.accounting`, every currency filed), placed by business address in
S&P's region and by **SIC code in a GICS sector** (`SIC_SECTORS`;
financials and 9100+ left out), with a **floor of 50 million US dollars**
of start-year revenue (ECB yearly averages). A case is one company's
yearly growth over a hold of 3-7 years (`HORIZONS`) as
**ln((1 + company) / (1 + its country's nominal GDP growth))** over the same
years (IMF, the 24 economies; elsewhere its region's median, the starting
figures' rule), so regions pool currencies and a deal's range is centred on
its own country: the IMF projection the starting figures use
(`benchmarks.starting.growth_figure`). The range = the excesses' 10th/50th/
90th percentiles in region x sector x hold (the region's every sector under
`MIN_COMPANIES` 30), and the Settings are the normal draw whose central 80%
it is (`mc_growth_std` = width / 2.563). **Shown only where
`ml/cards/growth.json` beats the baseline** (the economy-wide range in force
since 4.4) in the deal's S&P region, and only for tested holds
(`untested_horizon` otherwise). The card is **strictly out of time** (each
span predicted at its start from spans that had ended by then): 77% of
67,933 held-out spans inside the 80% range against the baseline's 27%;
every region beats it (Europe, emerging and other developed rest on
US-listed foreign filers only, which the card and tile say). Data:
`ml/evaluation/data/firm_growth.json` (2.4 MB, 8,451 companies) and the
served percentiles `ml/growth_ranges.json`, both written by `python -m
tests.ml_firm_growth --cache DIR` (~30 minutes, six calls in flight within
the SEC's pace; rerun yearly once the new 10-Ks are in, then `python -m
ml.evaluation evaluate` and `python -m tests.e2e_growth`); a test fails when
the ranges disagree with the data. `POST /api/ml/growth` (in `RUN_PATHS`,
reads only stored economic data; without it `no_growth`). Screen: Monte
Carlo's **Revenue growth** rail group has "Calibrate from sector and region"
(`components/montecarlo/GrowthCalibration.tsx`, namespace `growth`; sets the
two Settings and clears the rail's growth edits) and Scenarios a tile with
the range, its sample, the anchor and the card; while calibrated growth is
in use the Sources group's "Use sourced figures" leaves it alone. Browser
tests replay `web/e2e/fixtures/growth.json` (`python -m tests.e2e_growth`,
checked stale).

Driver explanations (PLAN.md 5.6, the written comparison is
**`docs/driver-explanations.md`**): **`ml/shap_attribution.py` is gone** (an
XGBoost model of one fixed deal shape explained with SHAP; never wired in,
its packages in no requirements file, 375 MB at peak when measured) and
nothing is trained in its place. **`analytics/driver_attribution.py`**
explains the simulation with the simulation: a path's IRR is a function of
its five draws (growth, exit multiple, rate, gross margin, the one-year
EBITDA shock), so `explain_tails` reruns the simulation's core
(`_run_vectorized_core`, with chosen draws; no model logic changed) on the
**worst and the best 5% of paths** for all 32 choices of which drivers keep
their draws and which sit at the mean, and takes each driver's **Shapley
value**. So **the simulation's IRR at the mean assumptions + the five
contributions = the tail's mean IRR, exactly** (tested to 1e-12 on a
percentage deal, five tranche structures, tax rules and leases). A driver
that cannot move the deal contributes exactly zero (the rate with no
floating facility), where its rank correlation is still about -0.4:
contributions are a driver's own effect, not what it shares with correlated
drivers. Cost: slices of half the run's size and at most `MAX_TAIL_PATHS`
(2,500) paths a tail, evenly spaced by rank above a 50,000-path run (the
answer carries `paths` and `tail_paths`); at the caps the explanation took
1.5x the run's time with a lower memory peak; the worst ratio is 3.2x (a
50,000-path run), so **a run that itself took over `EXPLAIN_RUN_LIMIT_S`
(20 s, `api/limits.py`) answers `explanations: null`** and the tiles are
absent. The Monte Carlo answer's
**`explanations`** (`DriverExplanations`; the job gives the same) is a new
output: no engine version, the pinned simulation and the golden snapshot
untouched. Screen: Monte Carlo -> Drivers, two tiles between the rank
correlations and the correlation table
(`components/montecarlo/DriverExplanations.tsx`, namespace `explanations`),
which render nothing for an API from before 5.6. The old Spearman view
stays: co-movement is information too.

Economic regime by region (PLAN.md 5.7): **`ml/macro_regime.py` trains
nothing** (the four-state hidden Markov model on US-only FRED series is
gone, with `hmmlearn` and `fredapi`; so is `ml/correlation_updater.py`,
never wired in). Seven regions (`REGIONS`: us, eurozone, uk, japan, china,
india, global), each year named **recession, stagflation, bull or base**
(the scenario presets' ids) from the IMF World Economic Outlook's real
growth and inflation since 1980 (the DataMapper 4.2 reads; the euro area and
the world are the IMF's aggregates `EURO` and `WEOWORLD`). The measure is a
year's **gap from the median of the region's own previous five years**
(levels would call every recent Chinese year a recession; a mean lets 2020
and 2021 set what later years are held to), on cut-offs from **earlier years
only**: growth gap in the weakest fifth = recession, else inflation gap in
the highest fifth = stagflation, else growth gap in the strongest fifth =
bull, else base. **Which year** (`regime_year`): read from May, the current
year (the IMF's April projection for it); before May, last year (its
October estimate), because until April the current year's figure is a
year-ahead forecast. **Refit monthly**: `refresh` runs **inside the
scheduled `economy-refresh` task** (`jobs/scheduled.py`: after the economic
data, whatever that did; either failing fails the run; its summary carries
`regime_*`), so it is nightly on both environments and after each staging
deploy with **no workflow change** (the Mac's GitHub token has no `workflow`
scope: a push that touches `.github/workflows/` is refused; the task
`regime-refresh` runs the refit alone if a step of its own is ever wanted).
It reads the IMF (two
calls) only on the **first run of a calendar month** and stores figures and
fit as `regime.fit` in `benchmark_tables` (`db/benchmarks.py`
`SIDE_PREFIX`: never part of `all_tables`, read with `table(name)`); a
failed read keeps the stored fit and is tried again the next night.
`POST /api/ml/macro-regime` `{country}` (in `RUN_PATHS`, reads storage
only; `no_data` without a database or a fit) answers the deal region's
regime and every region's with its year: a country's own region, a euro
member (`EURO_COUNTRIES`, 21) the euro area's, **any other country the
world's, said on screen**. **Shown only where `ml/cards/regime.json` beats
the baseline** (headline `accuracy`; `GATE_SETS`: a projection for the year
under way is gated by the card's `spring` set, last year's estimate by its
`autumn` set) and the fit is at most `MAX_AGE_DAYS` (100) old. A read that
leaves any region without a regime is **never stored** (the run fails and
the next night reads again), so a partial IMF answer can't replace a whole
fit. The card replays what the IMF projected for each
year in its April and October reports, 2000-2022 (its **historical
forecasts database**, `WEOhistorical.xlsx`, which ends with 2022's
reports), against today's final figures; baseline "the same as last year":
spring 75% of 153 region-years right against 52%, autumn 84% against 50%,
every region beats it in both. Data and fixture are written together by
`python -m tests.ml_regime_history --cache DIR`
(`ml/evaluation/data/regime_history.json`, `tests/fixtures/regime/imf`);
then `python -m ml.evaluation evaluate` and `python -m tests.e2e_regime`
(`web/e2e/fixtures/regime.json`, checked stale). Screen: Monte Carlo ->
Scenarios, "Economic regime" tile among the sources
(`components/montecarlo/EconomicRegime.tsx`, namespace `regime`), with
"Use ... preset" when the regime is not the preset in use. `/api/capabilities`
no longer has `macro_regime_*`. **Correlations** stay 4.4's
(`benchmarks.risk.correlations`), measured on the same IMF series each
time they are asked for. On 2026-10-11 every region's 2026 is a base year.

Live sliders for any deal (PLAN.md 5.8): **the network is gone and nothing
is trained** (decided with the user 2026-10-11, on measurements: the
simulation itself takes about 1.3 ms for 5,000 paths and was three times
closer than the network). **`core/live.py`** `live_estimate` reruns the
deal through the ordinary road (`build_sim_params`,
`run_vectorized_simulation_full`; no model logic) on `live_paths` paths:
`LIVE_PATHS` 8,000 while hold x facilities x interest passes is within
`WORK_BUDGET` 200 (percentages count as two facilities), fewer in
proportion beyond it, never under `MIN_LIVE_PATHS` 2,000. **The same draws
every time** (`LIVE_SEED`), so a slider moves the answer smoothly and the
same request gets the same answer. **A slider is an input**
(`with_sliders`): growth, exit multiple and its spread, gross margin and
the rate mean replace the rail's; debt / EV is the percentages deal's own
`debt_pct`, and a listed structure is scaled to it, every facility and its
repayment schedule alike (left untouched within 1e-9 of its own share, or
with nothing drawn); a listed structure's floating references move by the
rate slider's distance from the rail's rate mean, as a scenario moves them.
So there is **no trained range and no warning**: facilities, tax rules,
leases, fees and Settings are all the deal's own. What is left is sampling
error, which the answer states: `median_interval`, the sign-test 95%
interval for the median (about half a point wide at 8,000 paths).
`POST /api/montecarlo/live` (`LiveRequest`, `sliders` optional;
`LiveResponse`: `paths`, `hurdle`, `prediction`, `median_interval`); the
old `/api/ml/surrogate` is removed, and `/api/capabilities` has `live`
(always true) instead of `surrogate`, so a web copy and an API copy from
either side of the deploy just show "not available" to each other.
**Limits**: not a model run (`RUN_PATHS`) and no simulation slot, since its
size is its own bound (under 24 MB traced at the caps, tested); its own
rate, `USER_LIVE` 180 a minute per user (`LIVE_PATH`; the screen waits 300 ms for
a slider to rest, so it cannot send more), and at most `LIVE_SLOTS` 2 run at
once across all users (`live_slot`: a 503 after waiting 2 s). **Packages**:
`requirements-ml.txt` is now only `-r requirements.txt`, the Dockerfile has
no `INSTALL_ML`, and `ml/surrogate/`, `core/surrogate.py`,
`ml/train_all.py` and `test_ml.py` are deleted; the workflows still install
torch first (harmless; only the user can edit them: the cleanup steps were
handed over in the 5.8 session). **Card** (`ml/cards/surrogate.json`; the
id is kept so `ops/model_gate.py` compared it with the network's):
headline set `varied_deals`, 230 whole deals as the API takes them (sourced
figures and correlations, the country's tax preset, own entry multiple,
hold, fees, amortisation, leases, five debt structures in turn, written by
`tests/ml_regional_deals.py` `vary`, seeded by the deal's id), truth on
100,000 paths: **median IRR 0.13 points off (every region under 0.18;
`MEDIAN_ERROR_LIMIT_PP` 0.5 is the task's bar, tested)**, 5th percentile
0.36; the network's two sets are kept (`all_regional_deals` 0.18 where the
network was 0.54, `in_training_range`). The error is one sample's, in the
same direction for similar deals, because every estimate shares its draws.
**Staging check** (no workflow edit): `ops.scheduled drill`, which
staging.yml already runs, now also calls `POST /api/scheduled/drill/live`
(OIDC, refused in production) after the ten simulations: `jobs/drill.py`
`live_drill` runs the default deal (pinned `EXPECTED_LIVE`), 40 slider
moves and the largest deal the API takes, and answers timings and the
process's memory (`/proc/self/status` VmRSS and VmHWM, the cgroup limit
capped at Render's 512 MB); `ops/scheduled.py` `live_problems` fails the
run when the peak reaches the limit, the moves add over 32 MB, the middle
estimate takes over 500 ms, or the pinned answer moved. The figures land in
the `job-drill` run's summary. Web: `MonteCarloML.tsx` (a note with the
API's path count, the median's interval under the KPI, P(IRR > the deal's
hurdle); slider tracks stretch to hold a deal that starts outside them; the
debt slider is disabled when nothing is drawn; the sliders restart from the
deal whenever the deal or the rail changes). Browser tests run the real
endpoint now (`e2e/live-sliders.spec.ts`; `fixtures/ml-responses.json` is
gone). New outputs only: no engine version.

Personalized defaults (PLAN.md 5.9): **a person's usual figures, from their
own saved deals and nobody else's, with nothing learned stored.**
`ml/personalization.py` (SQLite, never wired in) is gone.
`benchmarks/usual.py` `usual_figures`: for each field the sourced starting
figures fill (`FIELDS`; rates, multiples and days, never money), the
**median** across the person's deals in the same group, with the count and
the lowest and highest; `MIN_DEALS` 3 a figure. A group (`group_of`) is
S&P's region of the deal's country and the GICS sector of its industry
(`validation.tags.INDUSTRY_SECTOR`; an industry with no sector is its own
group); a deal counts only once it has a country, and one that lists its
facilities says nothing about `debt_pct`, `senior_pct` and `base_rate`.
**The read is one statement** (`db/deals.py` `own_deal_inputs`): deals
outer-joined to the caller's own `users` row with the switch in the join, so
no other owner's deal can be in the answer and, switched off, no deal is
read; the newest `MAX_DEALS` 200, the deal on screen (`deal_id`) left out.
**The switch** is `users.personalized_defaults` (migration 0017, on by
default; `set_suggestions`, audit action `suggestions_switched` with
`enabled`, none for a repeat). `GET /api/account/suggestions?country=&industry=&deal_id=`
(`status`: `off`, `no_country`, `not_enough`, `ok`) and `PUT` `{enabled}`;
storage only, so not a run. **Decided in the session: not a model, so no
card and no gate** -- it describes the person's own deals and predicts
nothing, and no data exists (or may be committed) to evaluate it on; the
screen says what it is and from how many deals. Archived deals count (still
what the person assumed). Because a saved deal holds
every input, a figure the person never touched counts as what they used.
Web: `lib/usual.ts` (one answer shared through `useSyncExternalStore`, read
by the Starting point rail whenever the deal's country, industry or id
changes and each time Inputs opens), the "Suggest my usual figures" switch
under Starting point, and a "Your usual" column in the Starting figures tile
(`StartingPoint.tsx`: value, range and count, "Use" per figure, "Use all
yours"; with one or two deals in the group a line says how many are needed).
`/privacy` says the choice is kept and the suggestions are not.
`web/e2e/usual.spec.ts` gives each test an account of its own.

File uploads (PLAN.md 6.1, DEPLOY.md "File uploads"): PDF, Excel and Word
files (`.pdf`, `.xlsx`, `.docx`, `.xls`, `.doc`; macro-enabled formats never)
on **Deal -> Documents** (`/deal/documents`), private to their owner.
**`uploads/`**: `rules.py` (every number), `scan.py`, `store.py`,
`service.py`; rows in `db/files.py` (table `files`, migration 0018);
endpoints `api/routers/files.py`. **The road**: `POST /api/files {name,
size}` reserves room; `PUT /api/files/{id}/content` takes the bytes as the
body, **streamed to a spool file on disk** (`service.spool()`, never held in
memory; the one request `LimitsMiddleware` does not buffer, `streams_body`),
counted and cut off past the declared size (which is what the quota
counted); the API puts the file in the store and queues job kind
**`upload.scan`** under owner `system:uploads` (not in `JobSubmit`, so
`/api/jobs` refuses it from a caller; it takes the one simulation slot, so a
scan's memory never adds to a simulation's, and it can wait behind one);
`POST /api/files/{id}/link` answers a signed download link (60 s) for a
**clean** file and its owner only; `DELETE` removes row then bytes. Another
account's file is a 404 like a missing one (owner matched in every
statement). **Only the server writes to the store, once per name**: no
upload link is ever handed to a browser. The first design did (Supabase
signed upload links, valid two hours) and the security review showed it let
more be stored than was counted and a file be swapped after its scan; don't
bring it back. The browser sends the bytes to **the API's own address**
(`NEXT_PUBLIC_FSE_API_ORIGIN`, set in `next.config.ts` from the same value as
the `/api` rewrite), past the website's proxy, because Vercel doesn't say
what a proxied request may weigh: so it is a cross-origin call (CORS:
production allows `https://variater.com`, local `http://localhost:3000`;
staging needs `FSE_CORS_ORIGINS` to try it from the staging site) and the
CSP's `connect-src` carries that one origin (`lib/security/headers.ts`
`apiOrigin`), never a wildcard. **The scanner is structural, not an
antivirus** (decided on measurements, 2026-10-11: ClamAV 1.5.4 peaked at
1,687 MB on a 60-byte PDF; the free server has 512 MB): it refuses what can
act and what it can't read. **Parsers, not patterns** (the first version's
regexes were bypassed twice in review, and one took minutes on a 30 KB
file): PDFs with **pypdf** (`_check_pdf_objects`: every object the
cross-reference table leads to, object streams, filters and predictors
resolved, walked for `/JS`, `/JavaScript`, `/Launch`, `/EmbeddedFile`,
`/RichMedia`, `/XFA`, `/SubmitForm`, attachments (`/EF`, `/FileAttachment`,
`/Collection`), `/GoToR`, media and 3D ... and `/Encrypt`; each object
stream's contents decoded and searched; a packed object pypdf returns as
null, an `/ObjStm` in the bytes that the table does not lead to, or a PDF
it can't follow is `unreadable`) **beside** a search of the raw bytes for the long
names, hex-escaped spellings too, and of anything that inflates to an object
stream wherever a `stream` keyword stands (`_check_pdf_bytes`; the short
names `/JS` and `/XFA` are not searched in raw bytes: 20 MB of compressed
data spells them by chance); Open XML with **expat** (`_parse_part`:
relationships, content types, Word field instructions and worksheet
formulas with character references, CDATA, comments, UTF-16 and odd
attributes read as Office reads them; **an attribute is checked under every
spelling of its name** (`_values`: `Type` and `type` and `x:Type` are three
attributes, and a check holds for all or the file is refused); a document
type declaration or a part not well-formed is `unreadable`; all the XML
parsed in one file has a budget, `XML_BUDGET_BYTES` 96 MB, because every
element is a Python callback) plus
the package itself (OLE or zip parts anywhere, other than a nested
`.xlsx`/`.docx` under `embeddings`, which is checked in turn; ActiveX; macro
content types and relationship types; `oleObject`, `control`, `aFChunk`,
web extensions, templates, data connections, query tables, custom ribbons,
Power Query (`DataMashup`); `TargetMode="External"` other than hyperlinks
and workbook links; Word fields that fetch or run (`DDE`, `INCLUDETEXT`,
`INCLUDEPICTURE`, `LINK`, `IMPORT`, `DATABASE`) joined across runs in every
part, and VML pictures with an address of their own; worksheet formulas
calling `WEBSERVICE`, `FILTERXML`, `RTD`, `CALL`, `REGISTER`, `EXEC` or
holding a `|` outside a text value (DDE); two parts under one name; archive
caps); **oletools 0.60.2** for macros everywhere and for `.doc` /
`.xls` (VBA, Excel 4, DDE, encryption) plus BIFF records (`FilePass`, DDE
`SupBook`) and `.doc` field marks in one- or two-byte text. **`scan_path` runs it in a child process** (`python -m
uploads.scan <kind> <path>`) with `RLIMIT_CPU` 45 s, `RLIMIT_AS` 700 MB on
Linux, a 60 s deadline and an environment holding no secret (short on
purpose: a scan holds the slot simulations share, so a huge old-format file
or a workbook with hundreds of megabytes of sheets can run out of time on
the free server and is refused); a child that
dies, overruns or answers anything but `clean`/a reason is `scan_failed`.
`scan` (in-process, what the tests mostly call) never raises. The scan
reads the spool copy the upload left (so no download from Supabase, whose
free plan allows 5 GB of egress a month) and fetches from the store only
when that is gone (a restart, a separate worker). Reasons and refusals are
**codes** (`rules.REASONS`, `REFUSALS`); the words are
`web/messages/en.json` `files.reason_*` / `files.problem_*`. Measured peak
on a 20 MB file: under 100 MB (tested with tracemalloc). **A scan is never
lost and an outage never refuses a file**: `files.queued_at` says when the
latest scan was queued; a full queue, a store that can't be reached
mid-scan or a lost job leaves the file `scanning`, and the owner's next
look at it (`GET /api/files`, which the screen repeats while a file is
being checked) queues it again once `RESCAN_AFTER_S` (60 s) has passed, one
scan between simultaneous requests; a second job for a file already scanned
finds nothing to do. **Limits**: 20 MB a file (also set on the bucket),
100 MB and 40 files a person (`near_limit` from 80% of either: the screen's
"Nearly full"), 800 MB in all (`FSE_UPLOADS_BUDGET_MB`); `MAX_IN_FLIGHT` 3
of a person's files pending or scanning at once (429 `in_flight`),
`USER_UPLOADS` 20 an hour (`api/limits.py`), `MAX_ARRIVING` 2 uploads
streaming at once (503 `busy`, the screen sends again), each within
`ARRIVE_TIMEOUT_S` 300 s and never quiet for `ARRIVE_IDLE_S` 20 s (408
`too_slow`), `PENDING_BUDGET_BYTES` 200 MB reserved by everyone for uploads
not yet arrived, `SPOOL_MAX_BYTES` 200 MB of uploads kept on the server's
disk for their scans (past it a scan fetches from the store). A reservation
counts, and **a row counts until it is removed, because its bytes are in
the store until then**; `begin` first removes the caller's own rows past
their time, with their bytes. **Deletion rules**: by the owner any time;
clean files 90 days after upload; refused bytes at once and the row a week
later; room for an upload that never arrived after 30 minutes; a scan
unfinished after 15 minutes is refused by the sweep. `service.sweep()` runs
**inside the nightly `job-maintenance` task** (no workflow change; its
summary gains flat `files_*` keys; a store that can't be reached fails the
run), lists objects **before** rows so a new upload is never taken for an
orphan, clears old spool files, and is the activity that keeps the free
Supabase project awake. **Stores** (`uploads/store.py` `get_store`):
`SupabaseFiles` when `SUPABASE_UPLOADS_URL` and `SUPABASE_UPLOADS_KEY` are
set (bucket `uploads` in production, `uploads-<environment>` elsewhere, made
and kept private with limits by `ensure_bucket`; key in headers only;
redirects never followed; `x-upsert: false`); `LocalFiles` when
`FSE_UPLOADS_DIR` is set and the environment is not production (local runs,
pytest, the browser tests: its download links are HMAC-signed and point at
`/api/files/local/{token}`, one GET route outside sign-in and the schema
that answers 404 unless that store is in use); neither = `enabled: false`
and the screen says uploads are off. Object names are `u<user id>/<file
id>`. **Never log a file's name, a link or scanner output** (tested); rows
hold the name the person gave and nothing read from the file. Audit actions
`file_uploaded` and `file_deleted` (no detail), one per upload whatever the
queue did. **Staging check** (no workflow edit): the drill's
`/api/scheduled/drill/live` answer also carries `jobs/drill.py`
`files_drill()`: one small PDF stored, read back, scanned in the child
process, downloaded through a signed link and deleted in the staging
server's own store, and a scripted PDF refused there; `ops/scheduled.py`
`live_problems` fails the run on `failed:<step>` and allows `none` (no
store set up). Web: `lib/files.ts` (`uploadFile`: reserve, send, poll),
`DocumentsStep.tsx` (reads the list again every 2.5 s while a file is being
checked). `/privacy` says files are stored. Not done here: team sharing
(7.2), reading a document (6.2), account export and deletion (10.1), a
signature scan (12.1).

Workspaces (PLAN.md 7.10, the user's request, done out of turn after 4.6):
`web/src/lib/nav.ts` `WORKSPACES` groups the modes into **LBO** (Deal,
Monte Carlo, Backtest, Settings, Library) and **Equity research**
(Forecast). A mode belongs to exactly one workspace, so **addresses have no
workspace prefix** (`/deal/inputs` as before); a new mode goes in `MODES`
and in one workspace's `modes`. **Every sign-in lands on the launcher**
(`/start`, `AFTER_SIGN_IN`), except a visitor sent to sign in from a
particular screen, who goes back there; **the root (`/`) carries on at the
last screen this browser showed** (`lib/lastScreen.ts`, localStorage), else
the launcher, and `proxy.ts` never sends the root through sign-in as a way
back (it would skip the launcher). Tabs and Alt 1-9 list the current
workspace's modes (`useWorkspaceModes`; the account screen keeps the last
workspace's); Ctrl K searches all of them. The user's parked market-event
model ideas (NIFTY reconstitution, F&O positioning, IPO lock-ins) are in
PLAN.md 7.10, not scheduled.

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

**Priority since 2026-10-12 (the user's decision): PLAN.md phase R, the
equity research workspace for Indian analysts**: results PDF to checked
statements, a vetted sector model as live Excel formulas, valuation, a
report draft, next-quarter roll forward, in the app and in an Excel add-in.
After 6.1 merges the next task is **R0**, then R1 to R9 in order; 6.2
onwards and phases 7 to 12 wait, except what a phase R task pulls forward
and upkeep that can't wait. Phase R's own rules are at its top in PLAN.md
(a tool that never picks a rating; one model definition evaluated by the
server and written as Excel formulas; new code in `research/` only). The
decisions still the user's are listed there.

**R0 is under way (started 2026-10-12)**: its progress, first results and
lessons are under R0 in PLAN.md. The 30 results PDFs, the free Gemini key
(`research-samples/.env`) and the test scripts (`research-samples/r0/`) are
on the Mac only, in a folder git ignores through `.git/info/exclude`.
Nearly every Indian results filing is a scan. The free Gemini tier gives 20
requests a day per model; the user chose to stay free and slow for the test.
After the second session (2026-10-12) all 30 files are read twice: with two
full flash models 1.04% of figures are amber and no wrong figure was found
unflagged, so the recommendation recorded there is **go, with Gemini, two
reads by two different full models**. Third session (same day): the free
allowance was spent, so no read was made; added without a request were the
check of column periods (`score.py` `periods`), the full-over-lite
preference, the high-resolution option, `xbrl_score.py` and `finish.sh`,
which runs every read still owed. **The exchanges' XBRL may not be fetched
by a program** (NSE and BSE terms), here or in the product; the user saves
ten files by hand or keys them. The user saved BSE's results-page table for
ten documents (it is a small HTML table of the quarter's headline figures,
not the XBRL): of 232 BSE amounts read A fails to give one, and that one was
already amber (`bse_score.py`); **the user decided that is truth file
enough**. Still open: `finish.sh` (once the allowance is back, after about
06:00 IST on 2026-10-13) and the final table; then R0 closes.
**R4 part a's research and decision are done (2026-10-12; the tables are
under R4 in PLAN.md).** A sector model is a **mechanism, not a sector**: the
analyst picks a sector from a list that can grow, and it chooses one of
**eleven models** and its words. The user decided all eleven are built:
1 Occupancy (hospitals, hotels), 2 Per tonne (cement, steel, chemicals),
3 Stores (retail, restaurants), 4 Order book, 5 Subscribers, 6 People,
7 General (growth by segment), 8 Balance x rate, 9 Real estate, 10 Regulated
utilities, 11 Margin businesses. **Order: model 1 in R4, models 2 and 3 in
R7, the rest after**; models with fewer than two sample companies need
filings from the user first. What the blocks lack is listed there for part
b (a running count, a cost row that picks its driver, a segment in another
currency, the days in a period, a balance x rate block).
**Part a is done (2026-10-12): model 1 was built by hand in Excel for
Yatharth Hospital and Lemon Tree Hotels** (`research-samples/r4a/`, on the
Mac only: two workbooks, `build_model1.py`, `companies.py`,
`check_model1.py`). The same 118 rows and the same 243 forecast formulas
serve both; three of the eight revenue blocks were enough. What it found is
under "Model 1 by hand" in PLAN.md R4; the points part b must build in:
**the occupancy block runs backwards for the past** (Yatharth's printed beds
x occupancy x price is a quarter above its revenue, so beds in use are
worked out), **one results PDF does not start coverage** (a Q1 or Q3 filing
has no balance sheet; the latest filing with one is needed too), the year in
progress needs last year cut the same way (hotels' second half is the strong
one), and the lower part needs leases, non-controlling interests and days.
Yatharth's red figures wait for the user to save its FY26 results filing and
Q1 FY27 presentation (its website turns programs away).
**R4 part b's first slice is done (2026-10-12, out of turn: it needs no AI
key): `research/`**, the first code of phase R. Nothing in the API imports
it yet. **A model is a table**: rows down, periods across, one rule per cell
(`research/definition.py`): `FIGURE` (typed from the documents, may be
blank), `ASSUME` (the analyst's, must be given), a formula, a `Pick` (a
choice the company's file settles: how a block runs backwards for the
past), or empty; rules are keyed by period name, kind of period or group
(`past`, `year_end`, `all`), narrowest first. **Formulas**
(`research/formula.py`) are `+ - * /`, brackets, `max`, `min`, row names,
`x[prev]` / `x[ytd]`, and **`div(a, b)`** (a over b, nil when b is nil;
decided with the user), read by a hand-written reader, never `eval`.
**One parsed tree, two renderings**: `research/evaluate.py` works a model
out for a `Company` (words, unit, choices, numbers, segments off, marks) and
`research/workbook.py` writes the same trees as live Excel formulas
(ordinary addresses; a defined name per assumption cell, `<row>_<period>`,
and per constant; `div` is `IF(b=0,0,a/b)`; every label goes in through
`_put`, so words starting with `=` stay words). `research/pieces.py` holds
the shared pieces as functions that return ordinary rows (running count,
cost = share of revenue + amount per place per day, balance x rate, ratio
row, the standard periods `ly, ytd, roy, cy, y1...`).
**Model 1 is `research/models/occupancy.py`** (106 rows; sectors
`hospitals` and `hotels` bring their words; choices `line1_past` and
`line1_end_to_date`; segments `s2`, `s3` can be switched off), marked not
vetted. **Proof**: `tests/fixtures/research/model1_*.json` hold the two
companies' typed figures (public filings) and the hand workbooks' numbers;
the model gives every one of them (103 rows x 6 periods each) to 0.01.
**Never change those fixtures to agree with the model.** The workbook is
worked out in the tests by `tests/research_support.py`'s own reader of Excel
formulas (shares no code with `research/`). **LibreOffice, the real engine,
has not run yet**: its test exists only where `soffice` is installed or
`FSE_REQUIRE_SOFFICE=1`, which needs the workflow edit under "Waiting on
the user". What is left of part b is listed under R4 in PLAN.md (the other
blocks, quarterly actuals, screens, the add-in, an endpoint; with the
endpoint come the Dockerfile, `render.yaml` and `staging.yml` lines for
`research/`). The rest of part b comes in its turn, after R1 to R3;
**next is closing R0** (`finish.sh` and the final table, once the Gemini
allowance is back), then R1.
**Analysts come after the product** (the user, 2026-10-12: nobody can be
approached with a sample page): the first showing is after R2, the second
after R4, no task waits for one, and a sector model is marked "not yet
vetted" until an analyst has read it. Don't ask the user for analysts
before R2 is live. The hand-made sample
(`research-samples/r0/sample/variater-sample.pdf`) is kept as the picture
of what R2 to R8 build.

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
3.1 is done (model version, below). 3.2 is done (methodology, below). 3.3 is done (audit history, below). 3.4 is done (reference cases, below). 4.1 is done: **4.1a** (company filings, the data layer) and **4.1b** (the screen), both below. 4.2 is done (economic data and exchange rates, below). 4.3 is done (sourced starting figures, below). 4.4 is done (risk ranges, below). 4.5 is done: **4.5a** (the library's switch, base rates and coverage) and **4.5b** (reference transactions, two-person review, fees and amortisation), both below. 4.6 is done (model validation, below). 5.1 is done (ML evaluation and model cards, below). 5.2 is done (deal risk score, below). 5.3 is done (distress predictor, below). 5.4 is done (multiple predictor, below). 5.5 is done (growth calibrator, below). 5.6 is done (driver explanations, below). 5.7 is done (economic regime by region, below). 5.8 is done (live sliders for any deal, below). 5.9 is done (personalized defaults, below), so phase 5 is done but for the parked 5.10. 6.1 is done (file uploads, below). **Next is R0** (phase R, the priority above; 6.2 waits).

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
  Shortcuts: Ctrl K, Alt 1–5, `[` `]`. Since PLAN.md 7.10 the modes sit in
  **workspaces** (LBO, Equity research): a launcher after sign-in, a
  switcher beside the brand, and tabs and Alt numbers per workspace.
- Screens show open model findings as visible markers rather than hiding them
  (none are open in the web app now).
- **Sign-in, sign-up and the account screen** use the same tokens: Clerk's
  components are themed through `appearance` in
  `components/auth/AuthScreens.tsx` (square corners, Tape colours), and the
  account screen is an ordinary form in the `type-*` classes. The top bar's
  right-hand link shows who is signed in.
- **Logo (decided with the user 2026-10-08, design canvas
  https://claude.ai/artifact/8CeJbXsbqhwnNHcP1JNDo1, boards 11-14):** the
  mark is a **waterfall that dips once and then climbs** (start bar, a grey
  dip, two cyan rises, the end bar), bars as thick as Orbitron 900's stroke,
  set at the name's capital height beside "VARIATER" in Orbitron 900 caps,
  ink, letter-spacing 0.12em (`BrandLockup`, `.type-brand`). Colours are
  theme C1 Tape; the user may still pick another of the canvas's themes,
  which is a change to `mark.json`'s `tones` only. **One source:**
  `web/src/components/brand/mark.json`; `BrandMark.tsx` draws it in the app
  and `node web/scripts/brand.mjs` (from `web/`) draws every
  file -- `src/app/icon.svg` (follows the browser's light/dark theme),
  `favicon.ico`, `apple-icon.png`, `opengraph-image.png` (+ alt text),
  and `public/brand/` (app and maskable icons, the 120 px logo for Google's
  consent screen and Clerk, avatars, light/dark/one-colour logos, the
  email logo). Rerun it after any change to `mark.json` and commit what it
  writes; `tests/test_brand.py` fails on a stale or missing file. The
  wordmark PNGs are rendered with Orbitron from Google Fonts; there are no
  SVG files with the name in them (they'd need the font), only marks.
- **Honest labels (PLAN.md 2.1):** unsourced inception-era numbers carry a
  visible label until sourced data replaces them. Wording lives in
  `web/src/lib/provenance.ts`: Settings (every step) and the Monte Carlo rail say
  "Illustrative defaults — not market data"; Backtest counts its example deals
  from the deal list; the risk score says how many companies it compares
  the deal with (PLAN.md 5.2, `web/e2e/deal-risk.spec.ts`); Live says its
  estimate is the simulation of this deal on the API's own count of paths
  (PLAN.md 5.8). Keep these when editing those screens;
  `web/e2e/provenance.spec.ts` checks them.

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

- PLAN.md R4 part b: **LibreOffice in CI** (one workflow edit Claude's token
  can't push). In `.github/workflows/tests.yml`, job `ml`: before the step
  "Run all tests, failing if any ML test skips" add a step that runs
  `sudo apt-get install -y -qq libreoffice-calc` and then
  `soffice --version`, and give the test step `env: FSE_REQUIRE_SOFFICE: "1"`.
  Until then the model workbooks are checked only by the tests' own reader of
  Excel formulas, not by a real spreadsheet program. If the first run with
  LibreOffice fails, the fix is Claude's (`tests/research_support.py`
  `recalculated_by_libreoffice`).

- PLAN.md 1.7 follow-up: the GitHub settings in DEPLOY.md "CI and GitHub
  settings" (private vulnerability reporting, required checks, Dependabot
  secrets) and the commit email. (Both environments' `DATABASE_URL` switched
  to the restricted `fse_api` role on 2026-09-17; `live.yml` and
  `staging.yml` now fail if either goes back to a privileged role.)

- Keep the copy of `FSE_BACKUP_KEY` in the password manager (PLAN.md 1.8;
  replaced 2026-10-12 after the first was lost): GitHub never shows a secret
  again, and losing it makes every stored backup unreadable.

- PLAN.md 4.5b: approving the repository's ten reference transactions needs
  **two administrators**: add a second Clerk user id to `FSE_ADMINS` on
  `fse-api` (and on `fse-api-staging`, whose Clerk instance has its own ids),
  then each opens Library -> Review and approves (three ids for a proposal an
  administrator makes in the app). Until then the library holds none and
  Settings -> Fees offers no sourced fees.

- PLAN.md 6.1: file uploads are **off until the store is set up** (DEPLOY.md
  "File uploads", "You need to set this up first"): a Supabase project for
  uploads and `SUPABASE_UPLOADS_URL` / `SUPABASE_UPLOADS_KEY` on both Render
  services (and, to try it from the staging site, `FSE_CORS_ORIGINS` on
  `fse-api-staging`). Also one workflow edit Claude's token can't push: add
  `uploads` to `API_PATHS` in `.github/workflows/staging.yml` (line 35), so
  a change to `uploads/` alone makes the staging check wait for the API.

- Whether to wire up the unused `ml/nlp_extractor.py`; PLAN.md 6.2 rebuilds it.

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
| `ml/` | The deal risk score, the distress, multiple and growth models and the economic regime, their evaluation and cards, the EDGAR extractor, plus unused modules. Nothing in it needs ML packages or a trained file (the live sliders are `core/live.py`) |
| `web/` | Next.js 16 frontend. `src/proxy.ts` sends signed-out visitors to `/sign-in`; `components/auth/` holds the session, the account screen and the sign-in pages; `lib/auth/` decides Clerk or development sign-in. `src/lib/nav.ts` lists every mode and step (tabs, step row, search). `src/app/<mode>/<step>/page.tsx` are thin route files; screens live in `src/components/<mode>/`. State per mode sits in a provider mounted in `components/shell/AppShell.tsx` (Settings → Deal → Monte Carlo → Backtest → Forecast), so it survives mode switches. Settings overrides are saved to the account (`/api/account/settings`) and go into every run; the open deal (`DealProvider`) autosaves to `/api/deals/{id}/draft`, and the last one opened is reopened on the next visit. `components/charts/` and `components/ui/` are shared; `src/lib/api/` the typed client |
| `web/openapi.json` | Snapshot of the API schema; `src/lib/api/schema.d.ts` is generated from it |
| `Dockerfile`, `render.yaml` | API image and Render blueprint. One image runs everything (no ML build argument since PLAN.md 5.8), the upload scanner's child process included |
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
| `companies/`, `db/companies.py`, `api/routers/companies.py` | Company filings (PLAN.md 4.1): the interface and its connectors (SEC, ESEF, Companies House, EDINET, GLEIF), the summary figures and concept maps, paced HTTP, the recorder, the summary as deal inputs and forecast rows (`use.py`); storage, budget and the EDINET index; the endpoints |
| `benchmarks/`, `db/benchmarks.py`, `api/routers/benchmarks.py`, `web/src/lib/benchmarks.ts`, `web/src/components/deal/StartingPoint.tsx`, `web/src/components/montecarlo/SourcedRisk.tsx` | Sourced starting figures (PLAN.md 4.3) and risk ranges (4.4): regions and data sets read, the workbook reader, a deal's starting figures, the archive and economic history (`history.py`), the Monte Carlo Settings from it (`risk.py`), the refresh, the recorder and the defaults registry; storage; the endpoints; the Starting point rail group and tile, the Monte Carlo "Sources" rail group and tile |
| `library/`, `db/flags.py`, `db/references.py`, `api/routers/library.py`, `web/src/components/library/`, `web/src/components/settings/SourcedFees.tsx`, `ops/check_base_rates.py` | Reference library (PLAN.md 4.5): the cited base-rate tables, the coverage counts, the admin switch, the reference transactions (`reference_deals.json`, `references.py` with the inclusion and balance rules, `review.py`) and their fees (`fees.py`); site-wide switches; proposals and the two-person review stored; the endpoints; the Library screens (Reference deals, Review) and their provider; the sourced fees tile on Settings -> Fees; the yearly editions check |
| `validation/`, `db/validation.py`, `api/routers/validation.py`, `web/src/components/backtest/Validation.tsx`, `tests/e2e_validation.py` | Model validation (PLAN.md 4.6): the cases (newer-data rule), the report (calibration, bias, anonymity rules), the groups, the nightly run; opted-in deals and stored reports; the endpoint; Backtest -> Validation and the per-deal opt-in switch; the browser tests' recorded report |
| `ml/distress_model.py`, `ml/evaluation/distress.py`, `components/deal/Distress.tsx`, `tests/test_distress.py` | Distress predictor (PLAN.md 5.3): each year's band and default rate from published tables, gated by its card; the card (reference transactions and calibration); the Debt and Monte Carlo tiles and the business risk choice; the tests |
| `ml/multiple_predictor.py`, `ml/evaluation/multiples.py`, `components/deal/Multiples.tsx`, `tests/ml_multiple_history.py`, `tests/e2e_multiples.py` | Multiple predictor (PLAN.md 5.4): entry and exit ranges from the region's industry multiples and their archive, gated by its card; the walk-forward card; the Returns tile; the writers of the card's data and the browser tests' answers |
| `analytics/driver_attribution.py`, `docs/driver-explanations.md`, `components/montecarlo/DriverExplanations.tsx`, `tests/test_driver_explanations.py` | Driver explanations (PLAN.md 5.6): each tail's IRR split exactly between the simulation's five drivers; the comparison with the old SHAP module and the rank correlations; the Drivers step's tiles; the tests (adds up, hand-solved games, cost) |
| `ml/growth_calibrator.py`, `ml/growth_ranges.json`, `ml/evaluation/growth.py`, `components/montecarlo/GrowthCalibration.tsx`, `tests/ml_firm_growth.py`, `tests/e2e_growth.py` | Growth calibrator (PLAN.md 5.5): ranges from SEC filers' revenue growth over their economy's, by region, sector and hold, gated by its card; the served percentiles; the out-of-time card; the Monte Carlo button and tile; the writers of the data (SEC frames, SIC codes, ECB rates, IMF growth) and the browser tests' answers |
| `core/live.py`, `ml/evaluation/surrogate.py`, `components/montecarlo/MonteCarloML.tsx`, `jobs/drill.py`, `tests/test_live.py`, `web/e2e/live-sliders.spec.ts` | Live sliders (PLAN.md 5.8): the simulation on a small, fixed set of paths with the sliders as inputs; its card on varied regional deals; the Live step; the staging drill's live session and memory read; the tests (the estimate is the run with that input changed) |
| `ml/macro_regime.py`, `ml/evaluation/regime.py`, `components/montecarlo/EconomicRegime.tsx`, `tests/ml_regime_history.py`, `tests/e2e_regime.py` | Economic regime by region (PLAN.md 5.7): each region's year named from IMF growth and inflation against its own previous five years, the monthly refit, gated by its card; the card on the IMF's past projections; the Scenarios tile; the writers of the card's data, the recorded IMF answers and the browser tests' answers |
| `research/`, `tests/research_support.py`, `tests/fixtures/research/`, `tests/test_research_*.py` | Equity research, sector models (PLAN.md R4 part b): the formula reader (`formula.py`), the model and company as data (`definition.py`), the evaluator, the Excel writer (`workbook.py`), the shared pieces, model 1 (`models/occupancy.py`); the tests' fixture loader, independent Excel-formula reader and LibreOffice runner; the two hand-solved companies; the tests (formulas, the format by hand, workbook against evaluator, model 1 against the hand workbooks) |
| `uploads/`, `db/files.py`, `api/routers/files.py`, `web/src/lib/files.ts`, `web/src/components/deal/steps/DocumentsStep.tsx`, `tests/test_uploads.py`, `tests/upload_samples.py`, `web/e2e/documents.spec.ts` | File uploads (PLAN.md 6.1): the rules, the scanner, the stores and the steps of an upload; the rows; the endpoints; the browser's side and the Documents step; the tests and the files they build in memory |
| `benchmarks/usual.py`, `web/src/lib/usual.ts`, `tests/test_usual.py`, `web/e2e/usual.spec.ts` | Personalized defaults (PLAN.md 5.9): a person's usual figures from their own saved deals; the answer shared by the switch and the tile; the tests (only the caller's deals, off stops them) |
| `ml/anomaly_detector.py`, `components/deal/DealRisk.tsx`, `tests/ml_peer_tables.py`, `tests/e2e_deal_risk.py` | Deal risk score (PLAN.md 5.2): the deal against its region's industry averages and the library's deals like it; the tile; the writers of the card's peer tables and the browser tests' answers |
| `ml/evaluation/`, `ml/registry.json`, `ml/cards/`, `docs/model-cards/`, `ops/model_gate.py`, `.github/workflows/ml.yml`, `tests/ml_regional_deals.py` | ML evaluation (PLAN.md 5.1): the harness (time splits, regions, baselines), the card template, each model's evaluation, the train/evaluate CLI and the surrogate's regional deals; the registry of trained files; the cards (JSON and Markdown); the gate that fails a PR whose card got worse; the training and evaluation workflow; the writer of the regional deals |
| `economy/`, `db/economy.py`, `api/routers/economy.py`, `web/src/lib/economy.ts` | Economic data (PLAN.md 4.2): the catalogue of economies, sources and benchmark candidates, the six connectors, what is current, the nightly refresh and the recorder; storage and budget; the endpoints; a facility's benchmark at today's level |
| `web/src/components/companies/`, `tests/e2e_companies.py` | Company search on Deal and Forecast (PLAN.md 4.1b): the shared provider, the search panel with "Use in deal"/"Use in forecast", the figures tile; the writer of the browser tests' recorded company answers |
| `tests/reference/`, `tests/test_reference_cases.py` | Reference cases (PLAN.md 3.4): 26 deals solved by hand, their spreadsheet and the builder that writes it; the test that the engine matches each to 0.01 |
| `docs/methodology.md`, `tests/test_methodology.py` | Methodology (PLAN.md 3.2): every calculation written down, and the test that every model function, link, constant and Settings default in it matches the code |
| `core/model_version.py`, `MODEL_CHANGELOG.md`, `tests/model_version_pins.json` | Model version (PLAN.md 3.1): the stamp on every result, the saved stamp and the reopening check; what each engine version changed; reference results per version |
| `core/risk_warnings.py`, `core/risk_sources.py`, `web/src/components/deal/DealWarnings.tsx` | Risk warnings (PLAN.md 2.8): the four computed warnings, the published tables they read (with sources and samples), the tile |
| `core/accounting.py`, `ml/edgar_extractor.py`, `tests/fixtures/edgar/` | Accounting standards (PLAN.md 2.6): each standard's line items and the lease rule; the filing reader for 10-K (US GAAP) and 20-F/40-F (IFRS); real recorded filings (SAP, McDonald's) |
| `web/src/components/brand/`, `web/scripts/brand.mjs`, `web/public/brand/`, `tests/test_brand.py` | The logo: the mark's one source (`mark.json`) and its React component, the script that draws every icon and image, the drawn package, the check that they match |
| `core/money.py`, `web/src/lib/money.ts`, `web/src/components/ui/MoneyScope.tsx` | Currency and money units (PLAN.md 2.2): conversion to and from millions, the money keys of each answer, labels from CLDR, the money on screen |
| `jobs/` | Background jobs (PLAN.md 1.9): `queue.py` (the interface and retention rules), `memory.py` and `database.py` (the two queues), `runner.py` (the in-API runner thread), `kinds.py` (what can run as a job), `config.py` (which queue and runner), `scheduled.py` (scheduled tasks and their run log), `drill.py` (the staging drill's pinned answer), `worker.py` (phase 12's dedicated worker). Served by `api/routers/jobs.py` and `api/routers/scheduled.py` |
| `api/github_oidc.py`, `ops/scheduled.py` | The scheduler's sign-in (GitHub Actions OIDC tokens, no secret) and its side of the calls: `task`, `keepalive`, `drill` (`scheduled.yml`, `staging.yml`) |
| `ops/check_database.py` | Deployed database check (reachable, migrations current, storage under 80%) for `live.yml` and `staging.yml` |
| `tests/golden/` | Snapshot of the retired Streamlit app's outputs; the parity baseline. Its generator was removed with Streamlit (see git history) |
| `tests/`, `test_*.py` | Test suite (2,075 tests with a database; database tests skip without `TEST_DATABASE_URL`); `tests/test_model_fixes.py` pins each finding fix, `tests/test_database.py` the database layer, `tests/test_auth.py` sign-in, `tests/test_users.py` accounts, `tests/test_deals.py` saved deals and versions, `tests/test_limits.py` usage limits, `tests/test_security.py` headers, CORS, TLS, the database role and the header scan, `tests/test_backups.py` the backup format, stores, rotation and a real dump/restore round trip, `tests/test_jobs.py` jobs on both queues, restarts, retention, the scheduler's tokens and the drill, `tests/test_money.py` currencies and units, `tests/test_debt_structures.py` the tranche kinds, the floating-rate rule, PIK, the revolver and the sweep share, each hand-checked, `tests/test_montecarlo_baseline.py` the simulation's pinned output, `tests/test_montecarlo_tranches.py` the simulation of tranches (the written-out structure path by path, each path at the mean against the deal model, floating only, scenarios, the heatmap, jobs), `tests/test_tax_rules.py` the tax rules, each hand-checked, the presets, the simulation and the heatmap, `tests/test_risk_warnings.py` the computed risk warnings (hand-checked default deal, sources, the cash reconciliation, the catalogue holds no number), `tests/test_plan_actual.py` plan vs actual (the plan is the deal model's answer, the plan fed back as actuals attributes nothing, early exits, leases, tranches, units, saved actuals, the library switch), `tests/test_accounting.py` accounting standards: real IFRS and US GAAP filings mapped, the IFRS 16 lease views by hand in the deal model, the grid, sources and uses and the simulation, `tests/test_no_hardcoded_currency.py` the dollar-sign check, `tests/test_locale.py` digit grouping and fiscal years, `tests/test_no_hardcoded_locale.py` the locale check, `tests/test_no_hardcoded_text.py` the interface-text check, `tests/test_translations.py` the catalogue (keys asked for, keys used, languages in step, the engine's own labels), `tests/test_cycle_gates.py` the CI gates (PR titles, coverage floor, the workflows keep them), `tests/test_audit.py` the audit history (one entry per action, none for a no-op or a failure, nothing writable by the API or its role, compaction, no figures in an entry), `tests/test_benchmarks.py` the starting figures (real workbooks, a German machinery deal by hand, fallbacks, the refresh, the API), `tests/test_defaults_registry.py` the defaults registry, `tests/test_uploads.py` file uploads (the scanner on built files, the security review's attack files and real workbooks; the child process; the stores; the whole road on a real database: another account gets 404, quotas, deletion rules, lost scans, nothing logged). `tests/conftest.py` signs every other test in and hands out throwaway databases |

## Commands (macOS, from the repo root)

The project moved from Windows to a Mac (Apple Silicon) on 2026-10-10. On
Windows the interpreter is `.venv\Scripts\python.exe` and a variable is set
with `$env:NAME="value"` in PowerShell; everything else is the same.
`.claude/launch.json` (git-ignored, per machine) holds the preview servers
`api`, `api-db` (starts the local database first), `web` (`next dev`) and
`web-start` (the built app), all with the development sign-in.

```bash
.venv/bin/python -m pytest                       # all tests
.venv/bin/python -m pytest --cov --cov-report=term   # with coverage (CI's floors: .coverage-floor)
.venv/bin/python -m uvicorn api.main:app --reload --port 8000   # API; docs at /api/docs

# web/ (run the API too; Next proxies /api to FSE_API_URL, default 127.0.0.1:8000)
npm --prefix web run dev                                 # http://localhost:3000
npm --prefix web run lint
npm --prefix web run typecheck
npm --prefix web run build

# After changing API schemas: refresh the snapshot, then the TS types
.venv/bin/python -m api.export_openapi web/openapi.json
npm --prefix web run api:types
```

Database (optional locally; the API runs without one):

```bash
.venv/bin/python -m pip install pgserver         # once: Postgres in a wheel, no Docker needed
.venv/bin/python -m db.local                     # starts it (data in .localdb/), prints DATABASE_URL and TEST_DATABASE_URL
# export TEST_DATABASE_URL for the tests (and DATABASE_URL only to run the API or the browser tests), then:
.venv/bin/python -m pytest tests/test_database.py
.venv/bin/python -m db.migrate upgrade | downgrade -1 | current
.venv/bin/python -m db.local stop
```

Backups (PLAN.md 1.8). `FSE_BACKUP_DIR` keeps them in a directory instead of
Supabase, which is how to try the whole thing without any account:

```bash
# in the shell: FSE_BACKUP_KEY (32+ characters), FSE_BACKUP_DIR (or SUPABASE_URL
# + SUPABASE_SERVICE_ROLE_KEY), BACKUP_DATABASE_URL
.venv/bin/python -m ops.backup run --environment local      # dump, encrypt, upload, rotate
.venv/bin/python -m ops.backup list --environment local
.venv/bin/python -m ops.backup verify --environment local   # download and decrypt the newest
.venv/bin/python -m ops.backup drill --environment local --target "$ADMIN_URL"
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
and `FSE_AUTH_DEV=1` in the shell. They run in Playwright's own Chromium
(`npx --prefix web playwright install chromium`, once per Playwright
version); `PW_CHANNEL=chrome` or `msedge` uses an installed browser instead.

```bash
npm --prefix web run build
FSE_AUTH_DEV=1 npm --prefix web run test:e2e
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

Installed at **user level**, so each machine needs them again (the Mac since
2026-10-10: Homebrew's `git`, `gh`, `python@3.12` and `node@24`; the skills
and the ECC plugin are the user's to install, with the commands below):

| Tool | Install | Notes |
|---|---|---|
| `design-taste-frontend` (+ companions) | `npx skills add https://github.com/Leonxlnx/taste-skill` | Dials: `DESIGN_VARIANCE`, `MOTION_INTENSITY`, `VISUAL_DENSITY` |
| `web-design-guidelines` | `npx skills add vercel-labs/agent-skills --skill web-design-guidelines` | Fetches Vercel's guidelines from GitHub each run |
| `image-to-code` | `npx skills add https://github.com/Leonxlnx/taste-skill --skill image-to-code` | Written for Codex; expects to generate images, which Claude Code can't |
| Playwright CLI | `npm install -g @playwright/cli@latest` then `playwright-cli install --skills --global` | Writes to `.playwright-cli/`. Pick the browser with `--browser=` (the Windows machine had only Edge: `msedge`) |
| ECC (Everything Claude Code) | In an interactive `claude` terminal: `/plugin marketplace add https://github.com/affaan-m/ECC`, then `/plugin install ecc@ecc` (the Mac: user scope) | Optional (docs/WORKFLOW.md "Installing ECC"). On the Mac (since 2026-10-10): **ECC 2.2.3, installed by the user at user scope**; the Windows machine had 2.2.2 at project scope from 2026-09-24 (`enabledPlugins` in the committed `.claude/settings.json`). Plugin only: no `install.sh`, no global rules copy, attribution unchanged. ECC's hooks default to **on**; they're **off in this project** at either scope through the committed `.claude/settings.json` `env`: `ECC_HOOKS_ENABLED=false`, `ECC_SESSION_START_CONTEXT=off` |
| awesome-design-md | `git clone https://github.com/VoltAgent/awesome-design-md` | 74 brand `DESIGN.md` files — inspiration only, don't clone a real brand's identity |

Skills load when a session starts: install first, then open a new session.

## Gotchas

- **macOS: `python -m db.local` listens on a Unix socket, not TCP**, so its
  addresses carry the socket's directory in the query
  (`postgresql://postgres:@/fse?host=...`) and no port. Build another
  database's address with `db.local.database_url` or `make_url(...).set`,
  never by cutting the string at its last slash (the Mac's first run printed
  a `DATABASE_URL` pointing nowhere). Quote the addresses in zsh: they hold
  `?` and `%`.
- **Run pytest with `TEST_DATABASE_URL` only.** With `DATABASE_URL` also
  exported, the tests of what the API answers *without* a database fail
  (seven of them on the Mac's first run). `DATABASE_URL` is for running the
  API and the browser tests.
- Driving Excel from a script (`osascript ... tell application "Microsoft
  Excel"`) opened the workbook and then timed out, leaving it open: don't
  count on it to recalculate a file. openpyxl writes formulas without their
  values, so until Excel has opened and saved a workbook, Quick Look and
  `data_only=True` show blanks. Part b's "real spreadsheet engine in CI" is
  LibreOffice, which the Mac does not have.
- A company's "investor relations" page may sit behind a bot check
  (yatharthhospitals.com): don't get round it. Newspapers keep copies of
  exchange announcements (`bsmedia.business-standard.com/_media/bs/data/announcements/...`),
  found by search; otherwise the user saves the file.
- Gemini's free tier counts **20 requests a day per model** (the error names
  the quota and when to retry) and its newest model is often "high demand";
  the pro models are not on it at all. Save every answer to disk and never
  spend a call on a probe that a saved answer could replace. A request with
  a PDF inline must stay under about 20 MB; a bigger file needs the Files
  API. The day is a rolling 24 hours; a refusal for the day must not be
  retried; model names can share one allowance (`gemini-3.5-flash` counts
  as 3.6's), and listing the models is free and says which exist
  (`gemini-2.5-flash` is listed and then refuses new users). A session
  started within a day of the last one has nothing to spend: read the
  newest `out/run-*.log` for when each model was first used before planning
  any request.
- Reading figures from a filing with an AI model (PLAN.md R0): one read is
  never enough, **a model will change a total so that its misread digit
  still adds up**, and it will write a year it finds likely (2024 for 2026)
  in a column heading. Read twice with different models and different
  inputs, check the sums from the numbers and not from an assumed layout
  (`research-samples/r0/score.py`), and check periods apart from figures.
  Lite models leave out rows and whole statements.
- A check that finds sums in a table raises false alarms unless a line that
  is only copied (every other part nil) is not counted as a sum, a formula
  is chosen once for every column, and a zero total proves nothing. The
  three false alarms R0's first checks raised were each one of these.
- zsh's `echo` rewrites backslashes, so `echo "$json" | python` breaks any
  JSON holding `\n`. Parse API answers inside Python.
- A key pasted into a `read -s` prompt can arrive more than once with a
  stray escape character; check its length and shape (never its value)
  before using it.
- zsh does not split an unquoted variable into words: `T="a.py b.py";
  pytest $T` looks for one file called "a.py b.py". Use a glob
  (`tests/test_research_*.py`) or an array.
- CI's `ml` job fails when any test is **skipped**, so a test that needs a
  program the runner may lack (LibreOffice) is defined only where the
  program is found or a `FSE_REQUIRE_*` variable demands it
  (`tests/test_research_workbook.py`), never `pytest.skip`ped.
- openpyxl turns any string that starts with `=` into a formula. Text that
  a person typed (a company's name, a label, a reason) goes into a workbook
  with its `data_type` forced to `"s"` (`research/workbook.py` `_put`).
- A sector model's row may not be named after a formula word or a kind of
  period (`div`, `max`, `min`, `prev`, `rest`, `all` ...): the dividends row
  is `dividends`. An Excel defined name may not read as a cell (`rev1`,
  `r1c1`); the writer refuses one.
- macOS has no `python` on the PATH: use `.venv/bin/python` (the browser
  tests find it themselves; `PYTHON=` overrides). Homebrew's tools live in
  `/opt/homebrew/bin`.
- zsh reads a bare `==` at the start of a word and an unmatched `*` as
  patterns: quote `echo "===="` and `--include="*.ts"`.
- **Windows only** (the first machine; kept for a return to it): Node may be
  missing from PATH in an older session (`C:\Program Files\nodejs`); the
  console is cp1252, so printing `≥`, `→` etc. from Python fails and an edit
  script needs `PYTHONUTF8=1`; in PowerShell, `Get-Content -Raw` then
  `WriteAllText` turns a non-ASCII character into mojibake (an NBSP became
  "Â "); Playwright needs `PW_CHANNEL=msedge` (no Chrome, no bundled
  browser); a full drive C: (it hit 0 bytes twice) stops every command,
  because the desktop app writes each command's output to
  `%LOCALAPPDATA%\Temp\claude`, and only the user can free it.
- Edit source with the Edit/Write tools or a Python script that writes
  UTF-8, and keep test strings with special spaces as escapes
  (`"21,2\u00a0%"`).
- The golden snapshot can't be regenerated any more (Streamlit is gone). Pin
  deliberate model changes with explicit tests instead.
- **Next.js 16 differs from older versions.** Read `web/node_modules/next/dist/docs/`
  before using an API (e.g. `params` is a Promise; `PageProps`/`LayoutProps`
  are global types generated by `next typegen`). `web/AGENTS.md` is
  re-created by `next dev`; keep it committed.
- Components that use context or Motion (`MotionConfig`, `motion.*`) must be
  client components; the shell keeps them in `workspace.tsx` and the bars.
- Playwright: open the app at `localhost`, not `127.0.0.1` (Next's dev server
  blocks its client scripts for other hosts; the page never hydrates). Next.js
  renders a hidden `role="alert"` route announcer, so scope alert locators to
  `#content`. `.next/dev` grows to ~400 MB; delete it when disk is tight.
  Playwright reuses servers already on ports 3000/8000 locally: stop any
  preview or stray uvicorn first, or tests hit stale code (a 404 on a new
  endpoint means this). A `next start` left over from an interrupted run keeps
  serving the old build after a rebuild: the page renders but never hydrates,
  so clicks do nothing and the console shows `ChunkLoadError`. Kill whatever
  holds the port (`lsof -iTCP:3000 -sTCP:LISTEN`; on Windows
  `netstat -ano | grep LISTENING | grep :3000`).
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
- A reference transaction's figure must be read, not worked out: cite the
  filing, where in it, and a **verbatim** `quote` (or none; a table cell is
  not a quote). Find filings with EDGAR full-text search
  (`efts.sec.gov/LATEST/search-index?q=...`, 2001 on, User-Agent with a
  contact) or a company's `data.sec.gov/submissions/CIK##########.json`
  (only its newest 1,000 filings: older ones need the company browse page).
  A merger proxy (DEFM14A, SC 13E3) gives price, financing and the EBITDA in
  management's projections; the first post-buyout 10-K, S-4 or F-4 (filed
  for the buyout bonds) gives sources and uses, fees and amortisation; an
  IPO 424B4 or an 8-K gives the outcome.
- gitleaks' `generic-api-key` rule reads a slug after the word `key`
  (`"key": "masonite-2005"`) as a secret, and the `secrets` check scans all
  history, so a commit can't be undone, only allowed. `.gitleaks.toml` keeps
  every default rule and allows that slug shape in the reference files only;
  widen its `paths` rather than loosen the regex.
- The Bash tool sometimes fails a heredoc whose body holds apostrophes
  ("unexpected EOF while looking for matching `''"). Write the script with
  the Write tool and run the file instead.
- `Intl.DisplayNames` names retired region codes too ("DD" is "Germany",
  "UK" is "United Kingdom"), so a country list built from it offers two
  Germanys. `lib/locale.ts` `countryOptions` drops any code that
  `Intl.getCanonicalLocales` maps elsewhere; accounts saved earlier may hold
  one, so the API reads them through `benchmarks.catalogue.canonical`.
- A refresh must store the `now` it judges freshness by (`db/benchmarks.py`
  `save(tables, now)`): storing the wall clock made "a week old" depend on the
  day the tests ran (a 4.3 test went red the day after its fixture was
  recorded).
- Damodaran's archive names a file by the year it describes, one less than
  its edition: `vebitdaEurope24.xls` is January 2025's. Older editions print
  fewer columns and some are missing or not workbooks at all; read them with
  `parse_industries(..., every_column=False)`.
- Damodaran's workbooks name the same figure differently in each data set,
  and a column name can mislead: "Net Cap Ex/Sales" includes acquisitions
  and R&D. Read the "Variables & FAQ" sheet before mapping a column, and
  check one industry's figures against its dollar columns.
- A connector's figures must be checked against a **recorded real filing**,
  not a guessed concept list: Toyota's IFRS report and Tesco's capex use
  concepts nobody would guess. Record with `record-filings.yml` (keyed
  sources) or `python -m companies.record <case>` (keyless), read the
  untrimmed `raw-filings` artifact for concept names, then fix the map. A
  replayed case resets `sec.reset_cache()` and `edinet.reset_cache()` first:
  each fixture's name lists are trimmed to that case.
- The IMF's DataMapper ignores a country list in the path and answers every
  country since 1980 (about 120 KB an indicator): read it whole, and
  `economy/record.py` trims the fixture. The OECD's SDMX API allows about 60
  calls an hour per address, so `sdmx.oecd.org` is paced at 5 s and a
  refresh makes one call. Before showing any market series, read its licence
  on the source page: FRED hosts series it may not redistribute (ICE BofA,
  Moody's).
- A scheduled task's summary must be **flat** (`int`, `float`, `bool`, `str`
  or null: `TaskRun` in `api/routers/scheduled.py`): a list or a map is stored
  fine, then the endpoint answers 500, so the workflow fails a run that
  succeeded (staging's first `economy-refresh`). `/api/health/jobs` shows
  each task's status but not its summary; a workflow that checks a figure
  reads the task's own printed answer (`staging.yml`'s economic data step).
- Economic data tests judge "current" on the day the fixture was recorded
  (`_recorded_on`), never today, or they go stale on their own; a refresh's
  exchange rate request depends on what is stored, so the test transport
  answers the recorded rates for any start day.
- A table transcribed from a PDF: read it with `pdftotext -table` (from
  `brew install poppler` on the Mac, Git's mingw64 on Windows; `-layout`
  interleaves rows), convert it
  with a script rather than by hand, and test it against the source's own
  summary rows. Some free copies only download (maalot.co.il); the user
  agreed (2026-10-07) to downloading public source documents into the
  scratchpad to read them.
- Supabase shows the **publishable** key first and the **secret** key below
  it; the first staging drill of 6.1 failed three times on the publishable
  one pasted as `SUPABASE_UPLOADS_KEY` ("new row violates row-level security
  policy" on creating the bucket: the answer a key without the service role
  gets). The drill's answer says which sort the server holds
  (`files_drill_key`: `sb_secret` or `jwt:service_role` is right) and the
  store's own reason (`files_drill_detail`), never the key. Read those
  before suspecting the code, and ask the user to reveal the value **on
  Render**, not on Supabase. **`/api/health/files`** (public, cached five
  minutes, `uploads/service.py` `status`) answers the same on staging and
  production without waiting for a staging run: check it first.
- `gh run rerun` is refused for this token ("Resource not accessible by
  personal access token"), like `workflow_dispatch`: `staging.yml` runs
  again only on a push to `staging`, so a recheck after the user changes a
  variable needs a (real) change merged through `main`.
- **Uploaded files are hostile input** (PLAN.md 6.1). Read one with a real
  parser, in the child process (`uploads/scan.py` `scan_path`), never with a
  regular expression over its text: `<tag\b[^>]*` on 100 KB of a repeated
  opening tag took seconds and a megabyte minutes (the GIL held, so the
  whole API froze), and `TargetMode="Ext&#101;rnal"`, an attribute holding
  `>`, CDATA and UTF-16 each slipped past a pattern that Office reads
  straight through. A new check gets a case in `REFUSED` in
  `tests/test_uploads.py`, written the awkward ways too. A parser is not
  enough by itself: the second review got past expat by writing an
  attribute twice (`Type` and `type`), which a dict keyed on lower-cased
  local names collapsed to the harmless one. Check every spelling
  (`_values`), or refuse.
- Never hand `zlib.decompressobj().decompress` a view of "the rest of the
  file": CPython copies what it did not consume into `unconsumed_tail` on
  every call, so trying it at each of n places is n copies of the file (18
  seconds for 4 MB). Feed it pieces (`uploads/scan.py` `_inflate`).
- A check of a PDF that depends on *where* something stands in the bytes can
  be told a different story from the reader's: a `%stream` comment made the
  first scanner treat a real object stream as some other stream's body.
  pypdf follows the cross-reference table as a reader does; the raw-bytes
  search beside it tries every `stream` keyword and trusts no position.
- Never hand a browser a link to **upload** with. A signed upload link can't
  be revoked, counts nothing, and (depending on the store's version) can
  overwrite: more is stored than the quota saw, and a file can change after
  its scan. Bytes come through the API, streamed to disk
  (`api/routers/files.py` `put_content`).
- A test PDF must be well-formed (cross-reference table or stream and
  `startxref`): pypdf refuses anything less as `unreadable`.
  `tests/upload_samples.py` `pdf()` and `jobs/drill.py` `drill_pdf()` build
  them; `web/e2e/documents.spec.ts` has its own in TypeScript.
- A local database that ran a migration **while it was still being written**
  keeps the old shape: the tests pass (each makes a fresh database) and the
  browser tests fail with a 500 (`ProgrammingError`: no such column).
  `python -m db.migrate downgrade -1` then `upgrade` on `DATABASE_URL`.
- `monkeypatch.undo()` in a test undoes the fixtures' patches too (the
  database address, the store's directory). Use `with monkeypatch.context()
  as m:` for a patch that should end mid-test.
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
- A browser test that needs a Monte Carlo result across screens must move
  in-app (`modeTab`, `stepLink`), not `page.goto`: the simulation lives in
  the page's providers, so a reload drops it (the deal itself autosaves).
  Deal -> Summary starts a simulation of its own when none is current,
  so a spec visiting Summary leaves one running or finished behind it.
  `simulationSettled` waits for a KPI, and the Drivers and Heatmap steps
  have none: open Distribution or Scenarios first, then `stepLink`.
- Anything computed after a simulation from its paths (as the driver
  explanations are) must bound its own work: rerunning the core on k paths
  for m choices costs k x m paths, so slice it below the run's size and cap
  k, or the 100,000-path caps in `api/limits.py` stop meaning what they
  were measured to mean. Measure it at the caps (12 facilities, 15 years,
  10 passes) with `tracemalloc`, not on the default deal.
- A new anonymised aggregate over users' data needs **secondary
  suppression**, not just a minimum per group: hiding one small group lets
  the overall figures minus the shown groups give it back.
  `validation/report.py` `split` hides the next-smallest groups too, and
  `tests/test_validation.py` checks it on random data.
- A change that moves a simulation number (`simulation/`, `core/live.py`'s
  paths or seed) or the recorded benchmark/economy fixtures makes a model
  card stale (and a moved default-deal simulation also `jobs/drill.py`'s
  `EXPECTED_SUMMARY` and `EXPECTED_LIVE`): rerun `python -m ml.evaluation evaluate`
  (and `python -m tests.ml_regional_deals` for fixture changes) and commit the
  cards. If a card got worse, `ops/model_gate.py` fails the PR on purpose.
  The live sliders' evaluation runs 460 simulations of 100,000 paths: about
  fifteen seconds on the Mac.
- `ml/evaluation/data/multiple_history.json` holds every industry's
  multiples from Damodaran's archive, which CI can't download:
  `tests/test_multiples.py` checks it only against the ten industries the
  recorded fixtures keep. Rewrite it (`python -m tests.ml_multiple_history`,
  then `python -m ml.evaluation evaluate` and `python -m tests.e2e_multiples`)
  when a new January edition is recorded. No group's 2014 archive edition can
  be read (missing or not a workbook), so the history skips that year.
- A deal gets its country only from **"Use sourced figures"** on Deal ->
  Inputs (the country select alone is a draft), so a browser test that needs
  a country calls `replayBenchmarks` and applies one of the recorded
  starting cases (`IN|machinery|INR`, `DE|machinery|EUR`,
  `GB|machinery|GBP`, `CA|shipbuilding_marine|CAD`). The regime spec's first
  run sent an empty country for this reason.
- Past IMF projections are free in one file, the **WEO historical forecasts
  database** (`https://www.imf.org/external/pubs/ft/weo/data/WEOhistorical.xlsx`,
  9 MB): every spring (`S`) and autumn (`F`) report's growth and inflation
  for two years back and six ahead, since 1990, countries by ISO code and
  the aggregates by name ("Euro area", "World"). It ends with the reports it
  was last rebuilt with (2022's on 2026-10-11). The DataMapper API gives
  only today's vintage and never says which report it is.
- A model whose regions are not S&P's four gives its card a `regions_note`
  and passes `regions=` to `evaluate_set`; its region names go in
  `ml/evaluation/card.py` `REGION_NAMES`.
- The Mac's GitHub token (fine-grained, this repository) has **no
  `workflow` scope**: a push containing any commit that changes
  `.github/workflows/` is refused whole. Don't work around it: put scheduled
  work in an existing task (`jobs/scheduled.py`), or give the user the
  workflow edit as steps. A branch that already holds such a commit has to
  be recommitted without it (`git reset --soft main`) before it will push.
- SEC data at scale: the **XBRL frames** API
  (`data.sec.gov/api/xbrl/frames/<taxonomy>/<concept>/<unit>/CY<year>.json`)
  gives one concept for every filer in one call, with the business address
  (`loc`) but no SIC code (that is one `submissions` call per company). A
  call takes about 0.6 s, so the 0.15 s pace alone isn't reached serially:
  `tests/ml_firm_growth.py` keeps six in flight and caches answers on disk.
  Hold one concept's years at a time: a parsed frame is several megabytes.
- Vercel resolves Next rewrites at build time: changing `FSE_API_URL` needs a
  redeploy. `next.config.ts` fails the Vercel build if it's unset.
