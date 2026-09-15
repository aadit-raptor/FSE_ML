# PLAN.md — making the software full-scale

Scope (agreed with the user, 2026-09-15): **the software only**, meaning
everything needed to turn the current app into a scalable, secure,
production-grade product with the ML and feature ideas built in.

Deliberately **out of scope for now**: market research, customer interviews,
company name, legal work, pricing decisions, marketing and sales. Those can
be added later as their own phases.

---

## How to use this plan with Claude

**One task per session, one pull request per task.** Each task is sized so a
session can finish it, including tests.

Start each session with this prompt (swap in the task number):

> Read CLAUDE.md and PLAN.md. Do task **1.3** only. Check its "Needs" are done
> and its "You first" items are in place; if not, stop and tell me what's
> missing. Follow the working rules in CLAUDE.md (branch, tests that check
> real output, mutation-check new tests, PR, CI green, merge). When done, tick
> the task in PLAN.md's status table in the same PR and tell me what to check.

**Rules**
- Do the lowest-numbered open task whose "Needs" are done.
- **"You first"** is only what Claude can't do: creating accounts on outside
  services, entering passwords or payment details, and copying secret keys
  into Render and Vercel. Do it, then tell Claude the task is ready.
- **Secrets never go in chat or in code.** Put keys in the Render and Vercel
  dashboards (and a local `.env` that git ignores). Tell Claude the variable
  *name* only.
- **"Done when"** is the acceptance test. Claude shows evidence for each line.
- If a task is bigger than one session, Claude splits it into lettered parts
  (3.1a, 3.1b), adds them to the table, and finishes 'a' first.

**Services the plan uses** (defaults; swap any before its task starts)

| Need | Service | Used from task |
|---|---|---|
| Website hosting | Vercel | 0.1 |
| Engine, workers, scheduled jobs | Render | 0.1 |
| Database | Render Postgres | 1.3 |
| Queue, cache, rate limits | Render Key Value (Redis-compatible) | 1.6 |
| Accounts and login | Clerk | 1.4 |
| Error tracking | Sentry | 1.2 |
| Uptime alerts and status page | Better Stack | 1.2 |
| Emails | Resend | 7.2 |
| File storage | Cloudflare R2 | 6.1 |
| AI | Anthropic API | 6.2 |
| Economic data | FRED (free key) | 5.7 |
| Company filings | SEC EDGAR (no key) | 4.1 |
| Usage analytics | PostHog | 8.3 |
| Subscriptions | Stripe | 8.1 |

---

## Status

Tick the box in the same PR that finishes the task.

| # | Task | Needs | Status |
|---|---|---|---|
| **0** | **Online** | | |
| 0.1 | Put the current app online | — | ☐ |
| **1** | **Foundations** | | |
| 1.1 | Test copy (staging) and production | 0.1 | ☐ |
| 1.2 | Monitoring, error tracking, logs | 1.1 | ☐ |
| 1.3 | Database | 1.1 | ☐ |
| 1.4 | Accounts and login | 1.3 | ☐ |
| 1.5 | Saved deals, versions and settings | 1.4 | ☐ |
| 1.6 | Usage limits and abuse protection | 1.4 | ☐ |
| 1.7 | Security hardening | 1.4 | ☐ |
| 1.8 | Backups and recovery | 1.3 | ☐ |
| **2** | **Trust in the numbers** | | |
| 2.1 | Model version on every result | 1.5 | ☐ |
| 2.2 | Written methodology | — | ☐ |
| 2.3 | Audit history | 1.5 | ☐ |
| 2.4 | Hand-checked reference cases | 2.2 | ☐ |
| **3** | **Handling many users** | | |
| 3.1 | Job queue and workers for heavy runs | 1.3, 1.6 | ☐ |
| 3.2 | Result caching | 3.1 | ☐ |
| 3.3 | Speed budgets (web and API) | 1.2 | ☐ |
| 3.4 | Load testing and autoscaling | 3.1, 3.2 | ☐ |
| **4** | **Data platform** | | |
| 4.1 | Company data store (SEC filings) | 1.3, 3.1 | ☐ |
| 4.2 | Historical deal database and review screen | 4.1 | ☐ |
| 4.3 | Backtests over the deal database | 4.2 | ☐ |
| **5** | **ML done properly** | | |
| 5.1 | ML evaluation harness and model cards | 4.2 | ☐ |
| 5.2 | Deal risk score on real data | 5.1 | ☐ |
| 5.3 | Distress predictor | 5.1 | ☐ |
| 5.4 | Multiple predictor | 5.1 | ☐ |
| 5.5 | Growth calibrator | 5.1, 4.1 | ☐ |
| 5.6 | Driver explanations | 5.1 | ☐ |
| 5.7 | Macro regime and market correlations, scheduled | 5.1, 3.1 | ☐ |
| 5.8 | Live sliders for any deal shape | 5.1, 3.1 | ☐ |
| 5.9 | Personalized defaults | 1.5, 5.1 | ☐ |
| **6** | **AI features** | | |
| 6.1 | File uploads | 1.4, 3.1 | ☐ |
| 6.2 | Upload a document, get a deal | 6.1, 2.1 | ☐ |
| 6.3 | Investment memo writer | 2.1, 1.5 | ☐ |
| 6.4 | Plain-English explanations on screens | 2.1 | ☐ |
| **7** | **Product features** | | |
| 7.1 | Onboarding and in-app help | 1.5 | ☐ |
| 7.2 | Teams: sharing, permissions, comments | 1.5, 2.3 | ☐ |
| 7.3 | Excel: live-formula workbooks, then add-in | 1.5 | ☐ |
| 7.4 | Portfolio tracking (actuals vs plan) | 1.5, 3.1 | ☐ |
| 7.5 | Lender view (loan terms, default risk) | 5.3 | ☐ |
| 7.6 | Phones, tablets and accessibility | — | ☐ |
| 7.7 | Public API and webhooks | 1.4, 1.6, 2.1 | ☐ |
| **8** | **Subscriptions and usage** | | |
| 8.1 | Subscription billing | 1.4 | ☐ |
| 8.2 | Plan limits and usage metering | 8.1, 1.6 | ☐ |
| 8.3 | Usage analytics and feature flags | 1.4 | ☐ |
| 8.4 | In-app support and feedback | 1.2, 1.4 | ☐ |
| **9** | **Enterprise-grade** | | |
| 9.1 | Company logins (SSO) and admin console | 7.2 | ☐ |
| 9.2 | Data export and account deletion | 1.5, 6.1 | ☐ |
| 9.3 | Automated security testing | 1.7, 1.1 | ☐ |
| 9.4 | Disaster recovery | 1.8, 3.4 | ☐ |
| **10** | **Running it day to day** | | |
| 10.1 | Release routine and rollback | 1.1, 2.1 | ☐ |
| 10.2 | Incident routine and runbooks | 1.2 | ☐ |
| 10.3 | Dependency and cost upkeep | 1.7 | ☐ |

**Order in short:** 0 → 1 (all of it) → 2 and 3 → 4 → 5 → 6 and 7 (any
order) → 8 → 9 → 10 (10.1 and 10.2 can start any time after phase 1).
Phase 2.2 can start on day one.

---

## Phase 0 — Online

### 0.1 Put the current app online
- **Why:** it only runs on this laptop today.
- **You first:** follow DEPLOY.md: create the Render blueprint and the Vercel
  project, set `FSE_API_URL`, send Claude both URLs.
- **Claude does:** verifies both sites; adds a small read-only browser test
  run against the live URLs; records the URLs in CLAUDE.md.
- **Done when:** the live status bar shows "API ok"; the default deal shows
  21.2% IRR; Monte Carlo seed 42 shows 18.0% mean; URLs are in CLAUDE.md.

---

## Phase 1 — Foundations

### 1.1 Test copy (staging) and production
- **Why:** try every change on a copy before real users see it; undo a bad
  release in one step.
- **You first:** create a second Render service and Vercel project (or
  environment) for staging.
- **Claude does:** a `staging` branch auto-deploys to staging; `main` to
  production; environment variables documented per environment; browser tests
  run against staging after each deploy; a written rollback procedure.
- **Done when:** a change merged to `staging` appears only on staging; a
  production rollback is carried out once on purpose and documented.

### 1.2 Monitoring, error tracking, logs
- **Why:** know about problems before users do.
- **You first:** create Sentry and Better Stack accounts; set `SENTRY_DSN` for
  web and API.
- **Claude does:**
  - Sentry in the web app and API: errors with context, no deal contents or
    personal data;
  - structured logs with one request ID shared by web and API;
  - timing of every model run;
  - an uptime check on `/api/health` with alerts;
  - a status page.
- **Done when:** a deliberate test error appears in Sentry with its request ID;
  stopping staging triggers an alert within 5 minutes.

### 1.3 Database
- **Why:** everything the app must remember needs a permanent home.
- **You first:** create Render Postgres (staging and production); set
  `DATABASE_URL`.
- **Claude does:** a database layer in `api/` (SQLAlchemy and Alembic
  migrations); a local development database; a CI job with a real Postgres
  running migrations and tests; database status in `/api/health`.
- **Done when:** migrations run forwards and backwards in CI; CLAUDE.md
  explains how to add a table.

### 1.4 Accounts and login
- **Why:** each user's work must be private, and later limited by plan.
- **You first:** create a Clerk application (staging and production), enable
  email and Google sign-in, set keys in Vercel and Render.
- **Claude does:** sign-up, sign-in, sign-out and password reset in the web
  app; every API call except health requires a valid login; a `users` table;
  an account page; e2e tests run as a test user.
- **Done when:** a logged-out API call gets 401 (test); a new user signs up and
  reaches Deal; the full e2e suite passes logged in.

### 1.5 Saved deals, versions and settings
- **Why:** today a refresh loses everything.
- **Claude does:**
  - tables for deals, deal versions (inputs, settings, author, time) and user
    settings;
  - API endpoints to create, list, open, rename, duplicate, archive and
    delete deals, and save and restore versions;
  - web: a deal list, save and autosave, version history, restore;
  - settings move from browser storage to the account (existing browser
    settings imported once).
- **Done when:** a saved deal reopens with identical results in another
  browser; restoring a version brings back its exact IRR; one user can't open
  another's deal (test).

### 1.6 Usage limits and abuse protection
- **Why:** one person must never slow the service for everyone.
- **You first:** create Render Key Value (Redis); set `REDIS_URL`.
- **Claude does:** per-user and per-address request limits; maximum simulation
  size per request with a safe default; request size limits; timeouts on
  model runs; clear "limit reached" messages.
- **Done when:** a test hits the limit and gets 429; an oversized simulation is
  refused with a clear message; normal e2e use never hits a limit.

### 1.7 Security hardening
- **Why:** deals are confidential; a leak would be serious.
- **Claude does:**
  - security headers and a content security policy;
  - CORS locked to the app's domains;
  - CI blocks commits containing secrets;
  - dependency vulnerability scans on every PR;
  - automatic dependency update PRs;
  - least-privilege database user;
  - encrypted connections confirmed everywhere;
  - `SECURITY.md`;
  - a threat model: what could go wrong and how each risk is handled.
- **Done when:** a security-header scan passes; CI fails on a planted fake key;
  scans run on every PR; the threat model is in `docs/security/`.

### 1.8 Backups and recovery
- **Why:** databases fail; saved work must survive.
- **You first:** confirm the Render Postgres plan has daily backups and
  point-in-time recovery.
- **Claude does:** documents the schedule; writes and rehearses a restore into
  staging; adds a monthly restore drill to the runbook.
- **Done when:** a real restore into staging brings back a saved deal with
  identical results.

---

## Phase 2 — Trust in the numbers

### 2.1 Model version on every result
- **Claude does:** every API result carries the model version (engine version,
  git commit, settings fingerprint); saved versions and exports store it;
  reopening a deal saved under an older model shows "results changed since
  saved"; `MODEL_CHANGELOG.md` starting from the nine fixed findings.
- **Done when:** exports show the version; the change notice appears for an
  old version (test); the changelog exists.

### 2.2 Written methodology
- **Claude does:** `docs/methodology.md` explaining every calculation:
  - deal model: sources and uses, operating model, cash sweep, interest loop,
    returns, bridge, sensitivity;
  - simulation, scenarios, backtest attribution, forecast, each ML model's
    limits.

  Each section links to the code and the test that pins it.
- **Done when:** every function in `core/` and `lbo_engine/` that affects a
  number is covered.

### 2.3 Audit history
- **Claude does:** an append-only log of deal created, edited, version saved,
  shared, exported, deleted, and settings changed; a history view per deal.
- **Done when:** each action creates exactly one entry (tests); entries can't be
  changed through the API.

### 2.4 Hand-checked reference cases
- **Why:** a second, independent check that the maths is right, beyond
  consistency tests.
- **Claude does:** 10–15 small deals simple enough to solve by hand or in a
  plain spreadsheet (no leverage, single tranche, zero growth, fees only, and
  so on). The spreadsheet workbook is committed, and the tests require the
  engine to match it.
- **Done when:** every reference case matches its hand calculation to 0.01; the
  workbook and the reasoning are in `tests/reference/`.

---

## Phase 3 — Handling many users

### 3.1 Job queue and workers for heavy runs
- **Why:** Monte Carlo, scenarios, backtests, exports and later AI all run
  while the user waits on one web server.
- **You first:** create a Render background worker service.
- **Claude does:** heavy work becomes jobs (submit, get an ID, watch progress,
  fetch the result); workers use the same Docker image; the web app shows
  progress and stays usable; failed jobs retry once then report; quick deal
  runs stay instant.
- **Done when:** 20 simultaneous simulations in a test all finish while the
  health check stays under 200 ms; seeded results are unchanged; e2e passes.

### 3.2 Result caching
- **Claude does:** results cached by exact inputs, settings, seed and model
  version; unseeded runs skip the cache; a model version change clears it.
- **Done when:** a repeated seeded run returns in under 100 ms with identical
  numbers; a version bump misses the cache (test).

### 3.3 Speed budgets (web and API)
- **Why:** a big app gets slow gradually unless speed is measured.
- **Claude does:** page-load and interaction budgets for the web app (checked in
  CI with Lighthouse); response-time budgets per API endpoint (checked in CI);
  fixes for anything over budget (large bundles, slow queries, chart
  rendering with 2,000 points).
- **Done when:** CI fails when a budget is exceeded; all current pages and
  endpoints pass.

### 3.4 Load testing and autoscaling
- **You first:** move the API and workers to paid plans that allow scaling.
- **Claude does:** load test scripts (k6 or Locust) with a realistic mix of
  users; runs against staging; autoscaling rules for workers; a capacity note
  (users per machine and cost per 100 active users).
- **Done when:** staging holds 200 simultaneous users with 95% of deal runs
  under 1 s and no failed simulations; the report is in `docs/capacity.md`.

---

## Phase 4 — Data platform

### 4.1 Company data store (SEC filings)
- **Claude does:** tables for companies and yearly figures; a scheduled worker
  refreshes followed companies from SEC EDGAR within its rate limits; forecast
  autofill reads the store first; each number shows its filing and date.
- **Done when:** a second DELL autofill is served from the store; the refresh
  runs on schedule in staging; numbers link to their filing.

### 4.2 Historical deal database and review screen
- **Why:** ML and backtests are only as good as the data. Today: 30 risk-score
  deals, 39 distress cases, 25 multiples rows, 4 backtest deals.
- **Claude does:**
  - a deal database (terms, sector, year, leverage, outcome, source link);
  - an import pipeline from public filings;
  - an admin screen to review, correct and approve entries;
  - quality checks (ranges, gaps, duplicates);
  - all existing seed data migrated with sources;
  - a coverage page (deals per sector and year).
- **Done when:** existing data is migrated and visible with sources; imports run
  quality checks; the review screen approves entries (test).

### 4.3 Backtests over the deal database
- **Claude does:** Backtest reads approved deals from the database; a summary
  across all deals (how often actual IRR fell inside predicted P5–P95, and the
  average error); filters by sector and year.
- **Done when:** the four current deals give identical numbers from the
  database; the across-deals summary shows on screen and exports.

---

## Phase 5 — ML done properly

**Rule:** a model is shown to users only if it beats a simple baseline on
data it hasn't seen and has a model card. Otherwise it stays switched off.
Screens always label estimates as estimates.

### 5.1 ML evaluation harness and model cards
- **Claude does:**
  - one harness that trains on older deals and tests on newer ones, compares
    each model with a simple baseline, and reports accuracy;
  - a model card template (purpose, data, size, accuracy, known failures,
    trained date);
  - a model registry table;
  - a CI check that fails if a registered model gets worse.
- **Done when:** the anomaly detector and surrogate have cards; the surrogate's
  card reproduces its 0.19pp median error.

### 5.2 Deal risk score on real data
- **Claude does:** retrain `ml/anomaly_detector.py` on the deal database instead
  of 30 deals plus synthetic ones; correct its "~100 deals" note; show "based
  on N comparable deals"; similar deals link to database entries.
- **Done when:** the card shows real training size and held-out results; each
  risk flag's statistic matches a database query (test).

### 5.3 Distress predictor
- **Claude does:** retrain `ml/distress_model.py` on database outcomes; show
  year-by-year distress probability on Deal → Debt, and the share of simulated
  paths that struggle to cover interest; model card.
- **Done when:** it beats a leverage-only baseline on held-out deals; higher
  leverage raises it (test); it shows on screen with its card.

### 5.4 Multiple predictor
- **Claude does:** retrain `ml/multiple_predictor.py` on the database; suggest
  entry and exit multiple ranges by sector and size on Deal inputs, with
  comparable deals and "use suggestion".
- **Done when:** the range contains the actual multiple for most held-out deals
  (target in the card); suggestions work on screen.

### 5.5 Growth calibrator
- **Claude does:** calibrate growth ranges in `ml/growth_calibrator.py` from the
  company data store and sector data; "calibrate from sector" on Monte Carlo,
  with the source shown.
- **Done when:** calibrated ranges match observed growth on held-out years
  (card); using them changes the simulation results.

### 5.6 Driver explanations
- **Claude does:** compare `ml/shap_attribution.py` with the current drivers
  view; if it adds value, show per-deal explanations ("exit multiple adds +4.1
  points") and add its packages to the ML requirements; otherwise remove it
  and document why.
- **Done when:** a written comparison exists; if shipped, the explanations add
  up to the prediction (test).

### 5.7 Macro regime and market correlations, scheduled
- **You first:** get a free FRED key; set `FRED_API_KEY` on Render.
- **Claude does:** monthly scheduled jobs retrain `ml/macro_regime.py` and
  recompute correlations (`ml/correlation_updater.py`); results stored with
  dates; Scenarios shows the current regime; Settings offers "use current
  market correlations" with a before and after view.
- **Done when:** the jobs run on schedule in staging; accepted correlations pass
  the validity check and change simulation results (test).

### 5.8 Live sliders for any deal shape
- **Claude does:** regenerate surrogate training data across entry multiples,
  holds, fees and leverage as a worker job; retrain; update the card; warn only
  outside the trained range.
- **Done when:** median-IRR error stays under 0.5pp on varied held-out deals.

### 5.9 Personalized defaults
- **Claude does:** move `ml/personalization.py` from a local file to the
  database; learn each user's usual assumptions from their saved deals;
  suggest defaults on new deals; an off switch.
- **Done when:** suggestions use only that user's deals (test with two users);
  switching off stops them.

---

## Phase 6 — AI features

### 6.1 File uploads
- **You first:** create a Cloudflare R2 bucket; set its keys on Render.
- **Claude does:** upload PDF, Excel and Word with size and type checks; virus
  scanning; private storage per user or team; signed download links; deletion
  rules.
- **Done when:** a user can upload and download a file; another user can't
  (test); wrong or oversized files are refused.

### 6.2 Upload a document, get a deal
- **You first:** create an Anthropic API key; set `ANTHROPIC_API_KEY` on Render;
  choose a monthly AI spending cap.
- **Claude does:**
  - rebuild `ml/nlp_extractor.py` with a current model as a worker job;
  - read the document and fill in deal inputs, citing the page and quote for
    each number;
  - a review screen where the user accepts or edits each value;
  - per-user AI cost tracking and the spending cap;
  - an evaluation set of documents with known answers.
- **Done when:** most values match on the evaluation set (target in its card);
  every value has a citation; nothing enters a deal without acceptance; cost
  per document is logged.

### 6.3 Investment memo writer
- **Claude does:** a memo draft from a saved deal version (summary, returns and
  ranges, risks, drivers, scenarios, backtest context), exported to Word and
  PDF. Every number comes from the model run; the AI writes only the words
  around them. Model version on the cover.
- **Done when:** a test checks every figure in a memo against its run; exports
  open correctly.

### 6.4 Plain-English explanations on screens
- **Claude does:** "explain" on key tiles (IRR, bridge, distribution,
  scenarios, attribution) describing this deal's numbers in plain words; a
  glossary on hover for terms like MOIC or cash sweep.
- **Done when:** explanations quote the on-screen numbers exactly (test); the
  glossary covers every label term.

---

## Phase 7 — Product features

### 7.1 Onboarding and in-app help
- **Claude does:** a guided first deal; sample deals; empty-screen guides; hover
  tips on every input; help articles written from the methodology; a "what's
  new" panel from the changelog.
- **Done when:** a new test user completes the guided deal in e2e; every input
  has a tip.

### 7.2 Teams: sharing, permissions, comments
- **You first:** enable Clerk organizations; create a Resend account and set
  its key.
- **Claude does:** teams with owner, editor and viewer roles; share deals with a
  team or person; comments on deals and specific results; @mentions with
  email; team activity from the audit history.
- **Done when:** tests prove viewers can't edit, outsiders can't see, and edits
  appear in history; mentions send email.

### 7.3 Excel: live-formula workbooks, then add-in
- **Claude does:**
  - **Part a:** exports with working formulas, so the operating model, debt
    schedule and returns recalculate in Excel;
  - **Part b:** an Excel add-in that signs in, opens a saved deal, runs the
    model on the server and writes results into sheets.
- **Done when:**
  - (a) a test recalculates the workbook and matches the app's IRR to 0.01pp;
  - (b) the add-in refreshes a saved deal in Excel (screenshots plus API tests).

### 7.4 Portfolio tracking (actuals vs plan)
- **Claude does:** mark deals as owned; enter or upload quarterly actuals;
  compare with plan (reusing backtest logic); alerts when results drift past
  thresholds; a portfolio dashboard; reforecast of exit returns from actuals.
- **Done when:** actuals below plan trigger an alert (test); dashboard totals
  match the deals; reforecast changes expected IRR.

### 7.5 Lender view (loan terms, default risk)
- **Claude does:** loan conditions per tranche (maximum leverage, minimum
  interest cover); yearly covenant checks in the deal model; share of simulated
  paths breaching each; distress probability (5.3); expected loss and recovery;
  a lender summary and export.
- **Done when:** a deal built to breach in year 2 is flagged in year 2 (test);
  breach odds come from the simulation.

### 7.6 Phones, tablets and accessibility
- **Claude does:** layouts for smaller screens (inputs in a drawer, tiles
  stack); a full accessibility audit (keyboard, screen reader, contrast,
  reduced motion) and fixes; automated accessibility checks in e2e.
- **Done when:** key journeys pass e2e at phone size; automated checks report no
  serious issues; the agreed look is kept.

### 7.7 Public API and webhooks
- **Why:** lets other software (spreadsheets, internal tools) use the engine;
  a scalable product needs a stable way in.
- **Claude does:** personal API keys per user; versioned public endpoints (v1)
  for deal runs, simulations and saved deals; public API docs; webhooks when
  long jobs finish; per-key limits.
- **Done when:** a script with an API key runs a deal and receives a webhook when
  a simulation finishes (test); revoked keys stop working immediately.

---

## Phase 8 — Subscriptions and usage

(Software only: the machinery for plans and limits. Which plans and prices to
offer is a business decision for later.)

### 8.1 Subscription billing
- **You first:** create a Stripe account (test mode is enough to build).
- **Claude does:** placeholder plans in Stripe; checkout, trial, upgrade,
  downgrade, cancel; billing portal; invoices; webhooks keeping plan status in
  the database; failed-payment handling.
- **Done when:** in test mode a user can trial, upgrade, cancel and lose paid
  features at period end (e2e); webhooks are verified and safe to replay.

### 8.2 Plan limits and usage metering
- **Claude does:** one configuration file defining what each plan includes
  (features, simulation size, AI credits, seats); enforcement in the API, not
  just hidden buttons; monthly usage counters; upgrade prompts at limits.
- **Done when:** API tests prove a lower plan can't use a higher plan's feature
  by calling the API directly; counters reset monthly.

### 8.3 Usage analytics and feature flags
- **You first:** create a PostHog project; set its key.
- **Claude does:** events for key actions (sign-up, first deal, first
  simulation, save, export, upgrade); funnels; feature flags for gradual
  rollouts; no deal contents sent.
- **Done when:** funnels show staging traffic; a test proves no deal inputs are
  in events; one feature is behind a flag.

### 8.4 In-app support and feedback
- **Claude does:** "report a problem" that attaches the request ID (linked to
  Sentry) but not deal contents unless the user agrees; a feedback board with
  votes; admin view of reports.
- **Done when:** a test report appears with its Sentry request ID; votes are
  recorded.

---

## Phase 9 — Enterprise-grade

### 9.1 Company logins (SSO) and admin console
- **You first:** enable Clerk's enterprise SSO (paid feature).
- **Claude does:** company sign-in per team; automatic team joining by email
  domain; admin console (users, roles, seats, audit export); enforced login
  rules.
- **Done when:** a test identity provider signs a user into the right team;
  removing access takes effect immediately (test).

### 9.2 Data export and account deletion
- **Claude does:** export everything a user or team owns (deals, versions,
  files) as an archive; deletion that removes rows and files, with backups aged
  out on schedule; retention settings per team.
- **Done when:** a deleted test account leaves no rows or files (test); the
  export archive is complete.

### 9.3 Automated security testing
- **Claude does:** automated security scans of staging on every release (OWASP
  ZAP); code security analysis in CI (CodeQL); container image scanning;
  permission tests for every endpoint (each role against each resource);
  findings fixed with regression tests.
- **Done when:** scans run in CI and on each staging deploy; the permission
  matrix test covers every endpoint; no high findings are open.

### 9.4 Disaster recovery
- **Claude does:** a written recovery plan (how long recovery may take, how much
  data may be lost); a standby database replica; infrastructure described as
  code so the whole system can be recreated; one full recovery rehearsal into
  a fresh environment.
- **Done when:** the rehearsal rebuilds a working environment from scratch
  within the target time, and saved deals give identical results.

---

## Phase 10 — Running it day to day

### 10.1 Release routine and rollback
- **Claude does:** a release checklist (staging checks, changelog, model version
  bump, rollback plan); release notes generated from merged PRs; tagged
  versions.
- **Done when:** one release follows the checklist end to end with notes filed.

### 10.2 Incident routine and runbooks
- **Claude does:** severity levels; what to do for each alert from 1.2;
  user-facing status updates; a short review template after incidents.
- **Done when:** a simulated incident on staging is handled with the runbook and
  reviewed.

### 10.3 Dependency and cost upkeep
- **Claude does:** a monthly routine: merge dependency updates, rerun the
  scans, check hosting and AI spending against budgets, and archive unused
  data. A dashboard of monthly costs per service.
- **Done when:** the routine is documented and run once; the cost dashboard
  shows every paid service.

---

## Where each idea lives

| Idea | Tasks |
|---|---|
| Put it online | 0.1 |
| Test copy and rollback | 1.1, 10.1 |
| Monitoring | 1.2, 10.2 |
| Database, accounts, saved work | 1.3, 1.4, 1.5 |
| Security | 1.6, 1.7, 9.3 |
| Backups and recovery | 1.8, 9.4 |
| Trust in the numbers | 2.1–2.4, 4.3 |
| Many users | 3.1–3.4 |
| Data | 4.1, 4.2 |
| ML modules | 5.1–5.9 |
| Upload a document, get a deal | 6.1, 6.2 |
| Investment memo writer | 6.3 |
| Plain-English explanations | 6.4 |
| Help and onboarding | 7.1, 8.4 |
| Teams and collaboration | 7.2, 9.1 |
| Excel connection | 7.3 |
| Portfolio tracking | 7.4 |
| Lender view | 7.5 |
| Devices and accessibility | 7.6 |
| Integrations | 7.7 |
| Plans and billing machinery | 8.1, 8.2 |
| Usage insight | 8.3 |
| Privacy tools | 9.2 |
