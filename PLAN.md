# PLAN.md — from calculator to product

Everything recommended so far, turned into tasks that Claude can do one at a
time. It covers:
- the product gaps: accounts, saving, security, scale, monitoring, trust, data,
  legal, teams, billing, help, devices;
- the ML layer;
- the new features;
- the revenue and go-to-market steps.

Written 2026-09-15. The PDF explainer is deliberately left out for now.

---

## How to use this plan with Claude

**One task per session, one pull request per task.** Tasks are sized so one
session can finish one, including tests.

Start each session with this prompt (swap in the task number):

> Read CLAUDE.md and PLAN.md. Do task **1.3** only. Check its "Needs" are done
> and its "You first" items are in place; if not, stop and tell me what's
> missing. Follow the working rules in CLAUDE.md (branch, tests that check
> real output, mutation-check new tests, PR, CI green, merge). When done, tick
> the task in PLAN.md's status table in the same PR and tell me what to check.

**Rules for every task**
- Check the table below; do the lowest-numbered open task whose "Needs" are
  done. Tasks in the same phase with no link between them can run in any
  order.
- **"You first"** lists what only you can do. Claude can't do these: creating
  accounts, entering passwords or card details, accepting terms, signing
  contracts, choosing vendors, legal sign-off. Do them before starting the
  task and hand Claude the result, e.g. "the Clerk project exists; keys are
  set in Render and Vercel".
- **Secrets never go in chat or in the code.** Put API keys in the hosting
  dashboards (Render, Vercel) and in a local `.env` that git ignores. Tell
  Claude the variable *name*, not the value.
- **"Done when"** is the acceptance test. Claude must show evidence for each
  line: test output, a screenshot, a number.
- If a task turns out bigger than one session, Claude splits it into lettered
  parts (e.g. 3.1a, 3.1b), adds them to this table, and finishes 'a' first.

**Recommended vendors.** These are defaults so work can start; override any in
task 0.2.

| Need | Default | Why this one | Alternatives |
|---|---|---|---|
| Web hosting | Vercel | Already set up | Netlify |
| API and workers | Render | Already set up; runs Docker, background workers, Redis, cron | Fly.io, Railway |
| Database | Render Postgres | Same dashboard as the API | Neon, Supabase |
| Queue / cache | Render Key Value (Redis-compatible) | Same dashboard | Upstash |
| Accounts and login | Clerk | Next.js and FastAPI support, organizations (teams) and company logins built in | Auth0, Supabase Auth |
| Payments | Stripe | Subscriptions, invoices, trials, tax | Paddle (handles sales tax for you) |
| Error tracking | Sentry | Web and API in one place | Honeybadger |
| Uptime alerts | Better Stack | Free tier, status page | UptimeRobot |
| Product analytics | PostHog | Usage funnels, feature flags, self-host option | Mixpanel |
| Email | Resend | Simple API for sign-up and alert emails | Postmark |
| File storage (uploads) | Cloudflare R2 | S3-compatible, no download fees | AWS S3 |
| AI (documents, memos) | Anthropic API (`claude-sonnet-5`) | Already used by `ml/nlp_extractor.py` | — |
| Economic data | FRED (free key) | Already used | — |

---

## Status

Tick a box in the same PR that finishes the task.

| # | Task | Needs | Status |
|---|---|---|---|
| **0** | **Decisions and launch** | | |
| 0.1 | Put the current app online (Render + Vercel) | — | ☐ |
| 0.2 | Decisions: niche, vendors, budget | — | ☐ |
| 0.3 | Customer interviews kit | 0.1 | ☐ |
| **1** | **Foundations** | | |
| 1.1 | Environments: staging and production | 0.1, 0.2 | ☐ |
| 1.2 | Monitoring, error tracking, logs | 1.1 | ☐ |
| 1.3 | Database | 1.1 | ☐ |
| 1.4 | Accounts and login | 1.3 | ☐ |
| 1.5 | Saved deals, versions and settings | 1.4 | ☐ |
| 1.6 | Usage limits and abuse protection | 1.4 | ☐ |
| 1.7 | Security hardening | 1.4 | ☐ |
| 1.8 | Backups and recovery | 1.3 | ☐ |
| 1.9 | Legal pages and disclaimers | 0.2 | ☐ |
| **2** | **Trust in the numbers** | | |
| 2.1 | Model version on every result | 1.5 | ☐ |
| 2.2 | Written methodology | — | ☐ |
| 2.3 | Audit history | 1.5 | ☐ |
| 2.4 | Independent expert review | 2.2 | ☐ |
| **3** | **Handling many users** | | |
| 3.1 | Job queue and workers for heavy runs | 1.3, 1.6 | ☐ |
| 3.2 | Result caching | 3.1 | ☐ |
| 3.3 | Load testing and autoscaling | 3.1, 1.2 | ☐ |
| **4** | **Data platform** | | |
| 4.1 | Company data store (SEC filings cached) | 1.3, 3.1 | ☐ |
| 4.2 | Historical deal database | 4.1 | ☐ |
| 4.3 | Backtests over the deal database | 4.2 | ☐ |
| 4.4 | Data sources and licences | 0.2 | ☐ |
| **5** | **ML done properly** | | |
| 5.1 | ML evaluation harness and model cards | 4.2 | ☐ |
| 5.2 | Deal risk score retrained on real data | 5.1 | ☐ |
| 5.3 | Distress predictor | 5.1 | ☐ |
| 5.4 | Multiple predictor | 5.1 | ☐ |
| 5.5 | Growth calibrator | 5.1, 4.1 | ☐ |
| 5.6 | Driver explanations | 5.1 | ☐ |
| 5.7 | Macro regime and correlation updater, scheduled | 5.1, 3.1 | ☐ |
| 5.8 | Live sliders for any deal shape | 5.1 | ☐ |
| 5.9 | Personalization | 1.5, 5.1 | ☐ |
| **6** | **AI features** | | |
| 6.1 | File uploads | 1.4, 3.1 | ☐ |
| 6.2 | Upload a document, get a deal | 6.1, 2.1 | ☐ |
| 6.3 | Investment memo writer | 2.1, 1.5 | ☐ |
| 6.4 | Plain-English explanations on screens | 2.1 | ☐ |
| **7** | **Product features** | | |
| 7.1 | Onboarding and help | 1.5 | ☐ |
| 7.2 | Teams: sharing, permissions, comments | 1.5, 2.3 | ☐ |
| 7.3 | Excel: live-linked workbooks, then add-in | 1.5 | ☐ |
| 7.4 | Portfolio tracking (actuals vs plan) | 1.5, 3.1 | ☐ |
| 7.5 | Lender view (loan terms, default risk) | 5.3 | ☐ |
| 7.6 | Smaller screens and accessibility | — | ☐ |
| **8** | **Revenue** | | |
| 8.1 | Plans and billing | 1.4, 1.9 | ☐ |
| 8.2 | Plan limits and usage metering | 8.1, 1.6 | ☐ |
| 8.3 | Product analytics | 1.4 | ☐ |
| 8.4 | Marketing site and public docs | 1.9 | ☐ |
| 8.5 | Support and feedback | 1.4 | ☐ |
| 8.6 | Education licences | 8.1, 7.2 | ☐ |
| **9** | **Go to market** | | |
| 9.1 | Private beta | 1.5, 1.2, 1.9 | ☐ |
| 9.2 | Pricing test and paid launch | 9.1, 8.1 | ☐ |
| 9.3 | Content and partnerships | 8.4 | ☐ |
| **10** | **Enterprise** | | |
| 10.1 | Company logins (SSO) and admin controls | 7.2 | ☐ |
| 10.2 | Security certification readiness (SOC 2) | 1.7, 1.8, 2.3 | ☐ |
| 10.3 | Penetration test and fixes | 1.7 | ☐ |
| 10.4 | Data retention, deletion and privacy tools | 1.5 | ☐ |
| **11** | **People and process** | | |
| 11.1 | Human code owner and review | — | ☐ |
| 11.2 | Release, incident and support routines | 1.2, 1.1 | ☐ |

### The order, in short
1. **Get it online** (0.1) and **decide who it's for** (0.2). Start **talking to
   customers** (0.3) while building; don't wait.
2. **Foundations** (phase 1): nothing else is safe to launch without them.
3. **Trust** (phase 2) runs alongside; 2.2 can start on day one.
4. **Private beta** (9.1) as soon as 1.5, 1.2 and 1.9 are done: real users
   early.
5. **Scale** (3) and **data** (4), then **ML** (5), which needs the data.
6. **AI and product features** (6, 7): build what beta users ask for first;
   reorder within these phases freely.
7. **Revenue** (8) before the paid launch (9.2).
8. **Enterprise** (10) when a larger firm asks for it.

---

## Phase 0 — Decisions and launch

### 0.1 Put the current app online
- **Why:** real users can't try it on your laptop.
- **You first:** follow DEPLOY.md. Create the Render blueprint and the Vercel
  project and set `FSE_API_URL`. Send Claude both URLs.
- **Claude does:** checks both sites respond, runs the browser tests against
  the live URLs (a Playwright config pointed at production, read-only
  checks), records the URLs in CLAUDE.md.
- **Done when:** the live status bar shows "API ok"; the default deal shows
  21.2% IRR; Monte Carlo seed 42 shows 18.0% mean; URLs are in CLAUDE.md.

### 0.2 Decisions: niche, vendors, budget
- **Why:** later tasks depend on these choices.
- **You first:** decide, and write the answers into this section:
  1. First customer group: search funds / independent sponsors, small PE
     funds, private-credit lenders, M&A boutiques, or business schools.
  2. Vendors: accept the table above or change it.
  3. Monthly budget for hosting and tools.
  4. Company name and domain.
  5. Whether you'll hire or contract a human engineer (see 11.1).
- **Claude does:** prepares a one-page comparison for any vendor you're unsure
  about (features, limits, lock-in), and updates this plan with your choices.
- **Done when:** the five answers are written here and CLAUDE.md links to them.

### 0.3 Customer interviews kit
- **Why:** build what people will pay for, not what we guess.
- **You first:** find 20–30 people in the chosen group (LinkedIn, alumni
  networks, search-fund communities) and book calls.
- **Claude does:** writes
  - an interview script (problems, current tools, what they pay, deal-breakers);
  - a demo walkthrough of the live app;
  - a notes template.

  After each batch of notes you paste in, Claude summarises patterns and
  proposes reordering phases 6–7.
- **Done when:** the kit is in `docs/research/`; after interviews, a summary
  with the top 5 requests and price signals, and the plan reordered.

---

## Phase 1 — Foundations

### 1.1 Environments: staging and production
- **Why:** try every change on a copy before customers see it; roll back in
  one click.
- **You first:** in Render and Vercel, allow a second (staging) service or
  project if your plan needs it.
- **Claude does:**
  - a `staging` branch that auto-deploys to a staging API and web app;
  - `main` deploys to production;
  - environment variables documented per environment;
  - a written rollback procedure;
  - CI runs the browser tests against staging after each deploy.
- **Done when:** a PR merged to `staging` appears on the staging URL, not
  production; rolling back production to the previous version is documented
  and tried once.

### 1.2 Monitoring, error tracking, logs
- **Why:** find out about problems before customers do.
- **You first:** create Sentry and Better Stack accounts; set `SENTRY_DSN`
  (web and API) in Render and Vercel.
- **Claude does:**
  - Sentry in the web app and API: errors with context, no personal data;
  - structured request logs with a request ID shared by web and API;
  - a timing log for model runs;
  - an uptime check on `/api/health` with alerts;
  - a public status page;
  - a runbook: what each alert means and what to do.
- **Done when:** a deliberate test error shows in Sentry with its request ID;
  stopping staging triggers an alert within 5 minutes; the runbook exists.

### 1.3 Database
- **Why:** everything that must be remembered needs somewhere to live.
- **You first:** create a Render Postgres database (staging and production);
  set `DATABASE_URL`.
- **Claude does:**
  - database layer in `api/` (SQLAlchemy and Alembic migrations);
  - a tested migration workflow;
  - a local development database;
  - a CI job that runs migrations and the tests against a real Postgres.

  No features yet, just the foundation.
- **Done when:** migrations run up and down cleanly in CI; the API reports
  database health in `/api/health`; CLAUDE.md documents how to add a table.

### 1.4 Accounts and login
- **Why:** know who users are, keep their work private, and charge them.
- **You first:** create a Clerk application (staging and production); enable
  email and Google login; set keys in Vercel and Render.
- **Claude does:**
  - sign-up, sign-in, sign-out and password reset in the web app;
  - every API request except health and public docs requires a valid login
    token;
  - a `users` table linked to Clerk IDs;
  - an account page;
  - browser tests with a test user.
- **Done when:** logged-out visitors can't call the API (a test proves 401);
  a new user can sign up and reach Deal; the e2e suite runs as a test user.

### 1.5 Saved deals, versions and settings
- **Why:** today a refresh loses everything, and settings live only in one
  browser.
- **Claude does:**
  - tables for deals, deal versions (inputs and settings snapshot, created by,
    time) and user settings;
  - API endpoints to create, list, open, rename, duplicate, archive and delete
    deals, and to save and restore versions;
  - web: a deal list page, a Save button with autosave, a version history
    panel, and "restore this version";
  - settings move from browser storage to the account; the existing browser
    settings are imported once.
- **Done when:** a saved deal reopens with identical results on another
  browser; restoring an old version brings back its IRR exactly; one user
  can't open another's deal (a test proves 404).

### 1.6 Usage limits and abuse protection
- **Why:** one person must not be able to slow the service for everyone.
- **Claude does:**
  - per-user and per-IP request limits (Redis-backed);
  - maximum simulation size per request, set by plan later (8.2), with a
    safe default;
  - request body size limits;
  - timeouts on model runs;
  - clear "limit reached" messages in the web app.
- **Done when:** tests show the 429 response after the limit; a 1,000,000-path
  request from a basic user is refused with a clear message; normal use in
  the e2e suite never hits a limit.

### 1.7 Security hardening
- **Why:** real deals are confidential; a leak ends the business.
- **Claude does:**
  - security headers and a content security policy on the web app;
  - CORS locked to the web domains;
  - secrets only from environment variables, with a check that fails CI if a
    key appears in code;
  - dependency vulnerability scanning in CI (npm audit, pip-audit) and
    automatic update PRs;
  - database encryption and TLS confirmed;
  - least-privilege database user;
  - a `SECURITY.md` for reporting issues;
  - a threat model: what could go wrong and how each risk is handled.
- **Done when:** a security-header scan passes; CI fails on a planted fake
  secret and passes after removal; the threat model is written; dependency
  scans run on every PR.

### 1.8 Backups and recovery
- **Why:** databases fail; customers expect their work to survive.
- **You first:** confirm the Render Postgres plan includes daily backups and
  point-in-time recovery.
- **Claude does:** documents the backup schedule; writes and tests a restore
  procedure into staging; adds a monthly restore drill to the runbook.
- **Done when:** a real restore into staging succeeds and a saved deal reappears
  with identical results; the procedure is written.

### 1.9 Legal pages and disclaimers
- **Why:** liability, and larger clients won't sign up without them.
- **You first:** a lawyer reviews and approves the drafts. Claude drafts;
  it doesn't give legal sign-off.
- **Claude does:** drafts terms of service, a privacy policy, a cookie notice
  and an acceptable use policy; adds "not investment advice" disclaimers on
  results and exports; a consent step at sign-up; the pages linked in the
  footer.
- **Done when:** the pages are live on staging; sign-up records acceptance
  with date and version; exports carry the disclaimer; lawyer approval noted
  here.

---

## Phase 2 — Trust in the numbers

### 2.1 Model version on every result
- **Why:** when numbers change, you must be able to say why and which version
  produced a result.
- **Claude does:**
  - a model version (engine version, git commit, settings hash) returned with
    every API result;
  - the version is stored with saved deal versions and printed in Excel
    exports;
  - a "results changed since this was saved" notice when reopening an old
    deal on a newer model;
  - a changelog of model changes, starting from the nine fixed findings.
- **Done when:** every export shows the model version; reopening a deal saved
  under an older version shows the notice; `MODEL_CHANGELOG.md` exists.

### 2.2 Written methodology
- **Why:** professionals won't rely on numbers they can't inspect.
- **Claude does:** writes `docs/methodology.md` covering every calculation:
  - deal model: sources and uses, operating model, cash sweep, interest
    loop, returns, equity bridge, sensitivity;
  - simulation: distributions, correlations, clipping;
  - scenarios, backtest attribution, the forecast's three statements, and each
    ML model's limits.

  Every formula links to the code and to the test that pins it.
- **Done when:** every public function in `core/` and `lbo_engine/` that
  affects a number is covered; a reviewer can trace any screen number to a
  section.

### 2.3 Audit history
- **Why:** teams, lenders and auditors need "who changed what, when".
- **Claude does:** an append-only audit log covering deal created, edited,
  version saved, shared, exported, deleted, and settings changes; a history
  view per deal; retention rules.
- **Done when:** each action above creates exactly one audit entry (tests);
  entries can't be edited through the API; the per-deal history view shows
  them.

### 2.4 Independent expert review
- **Why:** a finance professional's sign-off is what customers trust.
- **You first:** hire a reviewer, such as a former PE or credit analyst or a
  valuation professional, and give them the methodology and the live app.
- **Claude does:** prepares a review pack: the methodology, test deals with
  hand-checkable answers, and a findings template. It then fixes each accepted
  finding as its own task, with a test, as findings 1–9 were done.
- **Done when:** the reviewer's report is in `docs/review/`; every finding is
  fixed or explained; a "reviewed by" note is on the methodology.

---

## Phase 3 — Handling many users

### 3.1 Job queue and workers for heavy runs
- **Why:** Monte Carlo, scenarios, backtests and exports currently run while
  the user waits on the web server; many users at once would queue and stall.
- **You first:** create a Render Key Value (Redis) instance and a background
  worker service; set `REDIS_URL`.
- **Claude does:**
  - heavy endpoints become jobs: submit, get a job ID, watch progress, fetch
    the result;
  - worker service in the same Docker image;
  - the web app shows progress and lets the user keep working;
  - failed jobs retry once, then report the error;
  - quick deal runs stay instant (no queue).
- **Done when:**
  - 20 simultaneous Monte Carlo requests in a test all finish;
  - the web server stays responsive during them (health check under 200 ms);
  - seeded results are identical to before;
  - e2e tests pass.

### 3.2 Result caching
- **Why:** the same question shouldn't be computed twice.
- **Claude does:** cache results keyed by the exact inputs, settings, seed and
  model version; skip the cache for unseeded runs; invalidate on model
  version change; show "from cache" timing in dev.
- **Done when:** a repeated seeded run returns from cache in under 100 ms with
  identical numbers; a model version bump misses the cache (test).

### 3.3 Load testing and autoscaling
- **Why:** know the breaking point before customers find it.
- **You first:** move the API and workers to paid plans that allow scaling.
- **Claude does:** load test scripts (k6 or Locust) for realistic mixes of deal
  runs, simulations and exports; run against staging; worker autoscaling
  rules; a capacity note (users per worker) and a cost estimate per 100
  active users.
- **Done when:** staging handles the target load (set in 0.2, e.g. 200
  concurrent users) with 95% of deal runs under 1 s and simulations queued
  without errors; the report is in `docs/capacity.md`.

---

## Phase 4 — Data platform

### 4.1 Company data store
- **Why:** fetching SEC data live on every request is slow and fragile.
- **Claude does:** tables for companies and their yearly filings data; a
  scheduled worker job refreshes followed companies from SEC EDGAR, respecting
  its rate limits and User-Agent rule; the forecast's autofill reads the store
  first and falls back to live; each number shows its source and filing date.
- **Done when:** a second DELL autofill is served from the store; the refresh
  job runs on a schedule in staging; each autofilled number links to its
  filing.

### 4.2 Historical deal database
- **Why:** the risk score, multiple suggestions and backtests are only as good
  as their data. Today there are 30, 25 and 4 deals.
- **You first:** approve data sources (4.4). Budget time for someone to check
  entries by hand. A finance-literate contractor is ideal.
- **Claude does:**
  - schema for deals: entry and exit terms, sector, year, leverage, outcome,
    sources;
  - an import pipeline from public filings (S-4, 8-K, 10-K for public
    targets, bond prospectuses);
  - an admin screen to review, correct and approve entries;
  - data-quality checks (ranges, missing fields, duplicates);
  - migrate the existing 30 anomaly deals, 39 distress cases, 25 multiples
    rows and 4 backtest deals, all labelled with their source.
- **Done when:** the database holds the migrated data with sources; the review
  screen works; quality checks run on import; the count and coverage per
  sector are shown on an admin page. The target size is set with the
  reviewer; aim for hundreds.

### 4.3 Backtests over the deal database
- **Why:** four hard-coded deals can't prove the model works.
- **Claude does:** backtest mode reads approved deals from the database; a
  summary across all deals (how often the actual IRR fell inside the predicted
  P5–P95, and the average error); filters by sector and year; the four
  existing deals keep their current results.
- **Done when:** the four current deals give identical numbers from the
  database; the across-deals summary is on screen and in an export.

### 4.4 Data sources and licences
- **Why:** selling a product built on data you're not allowed to use is a
  legal risk.
- **You first:** decide on paid data (e.g. PitchBook, Capital IQ, Preqin) versus
  public-only; sign any licences.
- **Claude does:** a register in `docs/data-sources.md` with each source, what
  it's used for, licence terms, and whether its data can be shown to
  customers or used for training; license checks in the import pipeline (4.2).
- **Done when:** every dataset in the app appears in the register; nothing
  from a source marked "internal only" is visible to customers (test).

---

## Phase 5 — ML done properly

**Rule for this whole phase:** a model reaches customers only if it beats a
simple baseline on held-out data and has a model card. Otherwise it stays
behind a flag. The screens say clearly when something is an estimate.

### 5.1 ML evaluation harness and model cards
- **Claude does:**
  - one harness that splits data by time (train on older deals, test on
    newer), compares each model against a simple baseline, and reports
    accuracy and calibration;
  - a model card template: what it does, training data and size, accuracy,
    known failures, last trained;
  - a model registry table with the version, metrics and approval of each
    trained model;
  - a CI job that fails if a registered model's metrics drop.
- **Done when:** the harness runs on the current anomaly detector and surrogate
  and writes their cards; the surrogate's card reproduces its 0.19pp median
  error.

### 5.2 Deal risk score retrained on real data
- **Claude does:** retrain `ml/anomaly_detector.py` on the deal database
  instead of 30 deals padded with synthetic ones; fix its "~100 deals" claim;
  show "based on N comparable deals" on screen; nearest deals come from the
  database with links.
- **Done when:** the card shows real training size and held-out results; the
  risk flags are backed by counts from the database (tests check a flagged
  statistic against a query).

### 5.3 Distress predictor
- **Claude does:** retrain `ml/distress_model.py` (currently 39 examples) on
  database outcomes; expose year-by-year distress probability on Deal → Debt
  and in Monte Carlo (share of paths breaching coverage); model card.
- **Done when:** it beats a leverage-only baseline on held-out deals; the
  probability appears on screen with its card link; a test checks higher
  leverage raises it.

### 5.4 Multiple predictor
- **Claude does:** retrain `ml/multiple_predictor.py` (currently 25 rows) on
  the deal database; suggest an entry and exit multiple range for a sector and
  size on Deal inputs, with comparable deals listed; "use suggestion" buttons.
- **Done when:** its range contains the actual multiple for most held-out deals
  (target set in the card); the suggestion and comparables show on screen.

### 5.5 Growth calibrator
- **Claude does:** calibrate growth distributions in `ml/growth_calibrator.py`
  from the company data store (4.1) and sector data instead of fixed Damodaran
  averages; offer "calibrate from sector" on Monte Carlo; show the source.
- **Done when:** calibrated ranges match observed sector growth on held-out
  years (card); the button changes the simulation inputs and the results.

### 5.6 Driver explanations
- **Claude does:**
  - decide whether `ml/shap_attribution.py` (XGBoost and SHAP) adds anything
    beyond the existing rank-correlation drivers;
  - if it does, per-deal explanations on Drivers ("exit multiple adds +4.1
    points versus average");
  - add `xgboost`, `shap` and matplotlib to the ML requirements or remove the
    plotting;
  - model card.
- **Done when:** a written comparison with the current drivers view exists; if
  shipped, explanations sum to the prediction (test) and appear on screen.

### 5.7 Macro regime and correlation updater, scheduled
- **You first:** get a free FRED API key and set `FRED_API_KEY` on Render.
- **Claude does:**
  - scheduled worker jobs that retrain `ml/macro_regime.py` and recompute
    correlations with `ml/correlation_updater.py` monthly;
  - results stored with dates;
  - Scenarios shows the current regime;
  - Settings offers "use current market correlations" with a before and after
    comparison;
  - model cards.
- **Done when:** the jobs run in staging on schedule; the regime shows on
  screen with its data date; accepting market correlations passes the validity
  check and changes simulation results (test).

### 5.8 Live sliders for any deal shape
- **Why:** the surrogate only learned one fixed deal (10x entry, 5-year hold),
  so the screen warns on most real deals.
- **Claude does:** regenerate training data across entry multiples, holds,
  fees and leverage (`ml/surrogate/generate_data.py`), run as a worker job
  since it takes a long time; retrain; update the card; remove the "fixed deal"
  warning where accuracy holds.
- **Done when:** median-IRR error stays under 0.5pp across a held-out set of
  varied deals (card); the warning appears only outside the trained range.

### 5.9 Personalization
- **Claude does:** move `ml/personalization.py` from a local SQLite file to the
  database; learn each user's typical assumptions from their saved deals;
  suggest defaults for a new deal ("your usual 5.5x leverage"); an off switch
  in account settings.
- **Done when:** suggestions come from the user's own deals only (test with two
  users); turning it off stops suggestions.

---

## Phase 6 — AI features

### 6.1 File uploads
- **You first:** create a Cloudflare R2 bucket; set its keys on Render.
- **Claude does:**
  - uploads for PDFs, Excel and Word, with size and type checks;
  - virus scanning;
  - files stored privately per user or team, with signed download links;
  - automatic deletion rules;
  - uploads listed on the deal.
- **Done when:** a user can upload and re-download a PDF; another user can't
  (test); an oversized or wrong-type file is refused.

### 6.2 Upload a document, get a deal
- **You first:** create an Anthropic API key; set `ANTHROPIC_API_KEY` on
  Render; agree a monthly AI spend limit.
- **Claude does:**
  - rebuild `ml/nlp_extractor.py` with the current model ID (it uses an
    outdated one), running as a worker job;
  - read an uploaded information pack or annual report, fill in the deal
    inputs, and cite the page and quote for every number;
  - a review screen where the user accepts or edits each value before it
    goes in;
  - per-user AI cost tracking;
  - an evaluation set of real documents with known answers.
- **Done when:** on the evaluation set, most extracted values match (target in
  its card) and every value has a citation; nothing enters a deal without user
  acceptance; cost per document is logged.

### 6.3 Investment memo writer
- **Claude does:**
  - generate an investment committee memo draft from a saved deal version:
    summary, returns and ranges, key risks from the simulation and risk
    score, drivers, scenarios, and backtest context;
  - export to Word and PDF;
  - every number in the memo pulled from the model run, not written by the AI
    (the AI writes the words around fixed values);
  - model version and disclaimer on the cover.
- **Done when:** a test checks every figure in a generated memo against the run
  it came from; export works; the memo is stored as part of the deal version.

### 6.4 Plain-English explanations on screens
- **Claude does:** an "explain" button on key tiles (IRR, bridge, distribution,
  scenarios, attribution) that describes what the chart says for *this* deal,
  in plain words, using the numbers on screen; a glossary on hover for terms
  like MOIC or cash sweep.
- **Done when:** explanations quote the displayed numbers exactly (test); the
  glossary covers every term used in labels.

---

## Phase 7 — Product features

### 7.1 Onboarding and help
- **Claude does:**
  - a guided first deal for new users;
  - sample deals to open;
  - an empty-state guide on each screen;
  - a help centre (articles from the methodology in plain language);
  - hover tips on inputs;
  - a "what's new" panel tied to the changelog.
- **Done when:** a new test user completes the guided deal in the e2e suite;
  every input has a tip; help articles are linked from each screen.

### 7.2 Teams: sharing, permissions, comments
- **You first:** enable Clerk organizations.
- **Claude does:**
  - teams (organizations) with roles: owner, editor, viewer;
  - share a deal with a team or person;
  - comments on deals and on specific results;
  - @mentions with email notifications (Resend);
  - activity from the audit log (2.3).
- **Done when:** tests prove a viewer can't edit, a non-member can't see, and
  an editor's change appears in history; comments notify mentioned users.

### 7.3 Excel: live-linked workbooks, then add-in
- **Why:** deal teams live in Excel.
- **Claude does:**
  - **Part a:** exports that keep formulas, so the operating model, debt
    schedule and returns recalculate in Excel when inputs change, and match
    the app's numbers.
  - **Part b:** an Excel add-in (Office.js) that signs in, pulls a saved deal,
    runs the model or a simulation on the server, and writes results into
    sheets.
- **Done when:**
  - (a) a test recalculates the exported workbook and matches the app's IRR
    to 0.01pp;
  - (b) the add-in loads a saved deal and refreshes results in Excel (manual
    check with screenshots, plus API tests).

### 7.4 Portfolio tracking
- **Why:** firms pay monthly to watch deals after they buy, not just once per
  deal.
- **Claude does:**
  - mark a deal as "owned";
  - enter or upload quarterly actuals;
  - compare against the plan (reusing backtest logic);
  - early-warning alerts when results drift beyond thresholds;
  - a portfolio dashboard;
  - scheduled reforecast of the exit return with updated actuals.
- **Done when:** entering actuals below plan triggers the alert (test); the
  dashboard totals match individual deals; reforecast changes the expected IRR.

### 7.5 Lender view
- **Why:** private-credit lenders are a strong customer group and care about
  downside, not upside.
- **Claude does:**
  - loan terms per tranche (covenants: maximum leverage, minimum interest
    coverage);
  - covenant tests by year in the deal model;
  - share of simulated paths that breach each covenant;
  - distress probability (5.3);
  - expected loss and recovery view;
  - a lender-focused summary and export.
- **Done when:** a deal built to breach a covenant in year 2 is flagged in year
  2 (test); breach probabilities come from the simulation; the lender export
  works.

### 7.6 Smaller screens and accessibility
- **Claude does:**
  - a readable layout on tablets and phones (inputs collapse into a drawer,
    tiles stack);
  - a full accessibility audit (keyboard use, screen readers, contrast,
    reduced motion);
  - fixes, with automated accessibility checks (axe) in the e2e suite.
- **Done when:** the e2e suite runs on a phone-sized viewport for key journeys;
  axe reports no serious issues; the design system's look is preserved.

---

## Phase 8 — Revenue

### 8.1 Plans and billing
- **You first:** create a Stripe account (business details, bank, tax
  settings); decide initial plan names and prices from the interviews (0.3);
  approve the terms (1.9).
- **Claude does:**
  - plans in Stripe: Free, Pro, Team, Enterprise contact;
  - checkout, trials, upgrade, downgrade, cancel;
  - customer billing portal;
  - invoices;
  - webhooks that keep plan status in the database;
  - a pricing page;
  - failed-payment handling.
- **Done when:** in Stripe test mode, a user can start a trial, upgrade, cancel
  and lose Pro features at period end (e2e test); webhooks are verified and
  replay-safe.

### 8.2 Plan limits and usage metering
- **Claude does:**
  - one place that defines what each plan gets, for example:
    - **Free:** single unsaved deal, deterministic model only;
    - **Pro:** saving, Monte Carlo, exports, AI memo credits;
    - **Team:** collaboration and portfolio tracking;
    - **add-ons:** document credits, deal database access;
  - enforcement in the API, not just hidden buttons;
  - usage counters (simulations, AI documents) with monthly reset;
  - upgrade prompts at limits.
- **Done when:** API tests prove a Free user can't run Monte Carlo even by
  calling the API directly; counters reset monthly; limits match the pricing
  page.

### 8.3 Product analytics
- **You first:** create a PostHog project; set its key.
- **Claude does:**
  - events for key actions: sign-up, first deal, first simulation, save,
    export, upgrade;
  - funnels: sign-up to first result, trial to paid;
  - feature flags to roll features out gradually;
  - no deal contents or personal financial data sent;
  - cookie consent respected.
- **Done when:** the funnels show in PostHog from staging traffic; a test
  confirms no deal inputs appear in event payloads; a feature flag gates one
  feature.

### 8.4 Marketing site and public docs
- **You first:** buy the domain (0.2); approve the messaging.
- **Claude does:** a marketing site (home, product tour, pricing, customer
  groups, security page, blog) on the main domain with the app on a
  subdomain; SEO basics; public methodology summary; screenshots from the real
  app.
- **Done when:** the site is live on the domain; pricing matches Stripe; the
  security page matches what 1.7, 1.8 and 10.x actually deliver.

### 8.5 Support and feedback
- **You first:** choose a support tool (e.g. Crisp, Intercom, or plain email).
- **Claude does:** in-app help widget; "report a problem" that attaches the
  deal ID and request ID (no deal contents without consent); feedback voting
  on features; a support runbook with common answers.
- **Done when:** a test report arrives with the request ID linked to Sentry; the
  feedback board collects votes.

### 8.6 Education licences
- **You first:** approach one or two business schools or training firms.
- **Claude does:** a class workspace (a team of students plus instructors);
  assignments with a set deal and scenario; instructor view of submissions;
  bulk seat billing; case library from the deal database.
- **Done when:** an instructor creates an assignment, students submit, the
  instructor sees results (e2e test); seat billing works in Stripe test mode.

---

## Phase 9 — Go to market

### 9.1 Private beta
- **You first:** invite 10–20 people from the interviews.
- **Claude does:** invite-only sign-up, a beta feedback prompt after key
  actions, a weekly usage and feedback summary for you, and fixes prioritised
  from what beta users hit.
- **Done when:** the beta group is active; weekly summaries exist; the top
  issues are logged as tasks in this plan.

### 9.2 Pricing test and paid launch
- **You first:** decide prices after the beta; go live in Stripe.
- **Claude does:** pricing page variants behind a feature flag; conversion
  tracking; switch Stripe to live mode with a checklist (webhooks, tax,
  emails, refunds); launch announcement assets.
- **Done when:** the first paying customer completes checkout in live mode; the
  launch checklist is fully ticked.

### 9.3 Content and partnerships
- **Claude does:** a content plan and drafts (plain-English LBO guides, deal
  breakdowns from the backtest set, "what the simulation says" posts); partner
  pitch materials for search-fund communities, business schools and lender
  networks; a referral programme.
- **Done when:** a content calendar and first 5 pieces exist; the referral
  programme works end to end.

---

## Phase 10 — Enterprise

### 10.1 Company logins (SSO) and admin controls
- **You first:** enable Clerk's enterprise SSO (a paid feature).
- **Claude does:** SAML and OIDC login per company; automatic team membership
  by email domain; admin console (users, roles, seats, audit export); enforced
  login policies.
- **Done when:** a test identity provider logs a user into the right team;
  admins can remove access and it takes effect immediately (test).

### 10.2 Security certification readiness (SOC 2)
- **You first:** choose a compliance platform (e.g. Vanta, Drata) and an
  auditor; budget for the audit.
- **Claude does:** written policies (access, change management, incident
  response, vendor management, data classification); evidence collection
  wired up (CI logs, access reviews, backups); gap list closed task by task.
- **Done when:** the compliance platform shows readiness; the auditor's
  engagement starts. The certificate itself is issued by the auditor.

### 10.3 Penetration test and fixes
- **You first:** hire a penetration-testing firm.
- **Claude does:** prepares the scope and test accounts on staging; fixes each
  finding with a regression test.
- **Done when:** every high and medium finding is fixed and retested; the
  report is summarised on the security page.

### 10.4 Data retention, deletion and privacy tools
- **Claude does:** user data export (all their deals and files); account and
  team deletion that really removes data, backups aged out on schedule;
  retention settings per team; a data processing agreement template for
  customers.
- **Done when:** a deleted test account leaves no rows or files (test); export
  produces a complete archive.

---

## Phase 11 — People and process

### 11.1 Human code owner and review
- **Why:** a product that holds client data needs a person who understands and
  answers for the code, not only AI.
- **You first:** hire or contract an engineer, part-time at first.
- **Claude does:**
  - onboarding pack: architecture tour, CLAUDE.md, this plan;
  - make pull requests need the engineer's approval (a CODEOWNERS file plus
    branch protection);
  - a review checklist;
  - continued work under their review.
- **Done when:** branch protection requires the owner's review; the engineer
  has merged a change independently.

### 11.2 Release, incident and support routines
- **Claude does:**
  - a release checklist (staging checks, changelog, model version bump,
    rollback plan);
  - an incident process (severity levels, who's on call, customer
    communication templates, post-incident review);
  - support response targets per plan.
- **Done when:** one staged release and one simulated incident follow the
  documents, with notes filed.

---

## Where each recommendation lives

| Recommendation | Tasks |
|---|---|
| Accounts and login | 1.4, 10.1 |
| Saving work, versions | 1.5 |
| Security | 1.6, 1.7, 10.2, 10.3 |
| Many users | 3.1, 3.2, 3.3 |
| Monitoring | 1.2, 11.2 |
| Staging copy, rollback | 1.1 |
| Trust in numbers | 2.1–2.4, 4.3 |
| Data | 4.1–4.4 |
| Legal | 1.9, 10.4 |
| Teams | 7.2 |
| Billing | 8.1, 8.2 |
| Help and onboarding | 7.1, 6.4, 8.5 |
| Devices and accessibility | 7.6 |
| People and process | 11.1, 11.2 |
| Backups | 1.8 |
| ML: risk score, live sliders, macro regime | 5.2, 5.8, 5.7 |
| ML: distress, multiples, growth, drivers, documents, correlations, personalization | 5.3, 5.4, 5.5, 5.6, 6.2, 5.7, 5.9 |
| Idea: upload a document, get a model | 6.1, 6.2 |
| Idea: investment memo writer | 6.3 |
| Idea: Excel connection | 7.3 |
| Idea: deal database | 4.2 |
| Idea: portfolio tracking | 7.4 |
| Idea: lender view | 7.5 |
| Idea: collaboration | 7.2 |
| Revenue: plans, add-ons, education | 8.1, 8.2, 8.6 |
| Go to market: niche, interviews, beta, charge, content, data advantage | 0.2, 0.3, 9.1, 9.2, 9.3, 4.2 |
