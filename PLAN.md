# PLAN.md — making the software full-scale, global and universal (free first)

Scope (agreed with the user, 2026-09-15): **the software only**. That means
everything needed to turn the current app into a scalable, secure,
production-grade tool that works for **any country, currency and deal**, with
the ML and feature ideas built in.

Out of scope for now: market research, customer interviews, company name,
legal work, pricing decisions, marketing and sales.

## Guiding principles

These apply to every task; a task isn't done if it breaks one.

1. **Free until the product is functional.** Phases 0–11 use only free plans
   and free data, with no card required. Anything that needs a paid plan is
   built so it can be switched on later, and listed in phase 12. If part of a
   task can't be done free, Claude splits it and moves that part to phase 12.
2. **Global, not US by default.** Every money figure has a currency. Rates,
   tax rules, accounting conventions and data sources are chosen by region.
   English first, built for other languages.
3. **A real-world tool, not a demo built on a few famous deals.** The app must
   be fully useful with only the user's own deal. The four inception deals
   (Burger King, Hilton, Dell, Freescale) become optional examples, never an
   input to calculations, defaults or models.
4. **Every default has a source**, showing where it came from, how many
   companies it's based on, and when it was updated. Users can always override.
   Until then, defaults are labelled "illustrative".
5. **Estimates are labelled as estimates.** ML and AI outputs appear only if
   they pass accuracy tests.
6. **The calculations are proven correct by hand-checked cases**, not by
   history.
7. **Built to scale without rewrites.** Free-tier shortcuts, such as running
   long jobs inside the API, sit behind interfaces, so moving to paid
   infrastructure in phase 12 is configuration, not new code.

---

## How to use this plan with Claude

**One task per session, one pull request per task.**

Start each session with this prompt (swap in the task number):

> Read CLAUDE.md, PLAN.md and docs/WORKFLOW.md. Do task **1.3** only, following
> the cycle in docs/WORKFLOW.md. Check its "Needs" are done
> and its "You first" items are in place; if not, stop and tell me what's
> missing. Follow PLAN.md's guiding principles (free plans only) and
> CLAUDE.md's working rules (branch, tests that check real output,
> mutation-check new tests, PR, CI green, merge). When done, tick the task in
> PLAN.md's status table in the same PR and tell me what to check. Then tell
> me to start a new session, and give me the ready-to-paste prompt for the
> next task (this same prompt with the next task number), plus anything I must
> do first for it.

**Rules**
- Do the lowest-numbered open task whose "Needs" are done.
- **"You first"** is only what Claude can't do: creating accounts on outside
  services, entering passwords, and copying secret keys into the hosting
  dashboards and GitHub secrets.
- **Secrets never go in chat or code.** Keys go in Render, Vercel and GitHub
  Actions secrets (and a git-ignored `.env` locally). Tell Claude the variable
  *name*.
- **"Done when"** is the acceptance test; Claude shows evidence for each line.
- A task too big for one session is split into lettered parts (3.1a, 3.1b).
- **Every session ends with a handoff:** once the PR is merged and PLAN.md is
  ticked, Claude tells the user to start a new session and gives the exact
  prompt for the next open task (lowest number whose "Needs" are done),
  including that task's "You first" items so the user can prepare them.
- **Stay inside free limits.** Each task notes the limits that matter. Claude
  adds a check or alert when usage approaches a limit.

## Free services this plan uses (no card required)

| Need | Free service | Free limits that shape the design |
|---|---|---|
| Website | **Vercel Hobby** | Non-commercial use only: fine while building (phase 12 moves to Pro before charging anyone) |
| Engine (API) | **Render free web service** | Sleeps after 15 min idle (about a minute to wake); 750 instance hours a month shared across services; small memory; no background workers or scheduled jobs |
| Database | **Neon free** | 0.5 GB per project; 100 compute-hours a month; scales to zero when idle; never expires |
| Cache and rate limits | **Upstash Redis free** | 500,000 commands a month; 256 MB |
| Scheduled jobs (data refresh, retraining, backups) | **GitHub Actions** | Free and unlimited on this public repository; schedules pause after 60 days without a commit (task 11.3 handles this) |
| Long user jobs (simulations, AI, imports) | **Inside the API**, with a job table in Neon | Moves to dedicated workers in phase 12 without code changes |
| Accounts and login | **Clerk free** | Limited monthly active users |
| Errors and uptime | **Sentry free**, **Better Stack free** | Limited events and monitors |
| Emails | **Resend free** | Limited sends per day and month |
| File storage | **Supabase Storage free** | 1 GB; project pauses after a week without activity (a daily health check keeps it awake) |
| Usage analytics | **PostHog free** | Generous monthly event allowance |
| Payments | **Stripe test mode** | No real charges until phase 12 |
| AI | **Recorded responses** for development and tests; **Gemini API free tier** for trials on **public documents only** | Google may use free-tier inputs to improve its products, and humans may review them, so never send confidential deals. Confidential documents wait for a paid provider (phase 12) |

**Free data sources**

| Data | Sources |
|---|---|
| Company filings | SEC EDGAR (US), UK Companies House (free key), ESEF filings for EU and UK listed companies (filings.xbrl.org), Japan EDINET (free key), plus document upload for anywhere else |
| Economic data by country | IMF, World Bank, OECD, BIS, ECB Data Portal, FRED (free key) |
| Exchange rates | ECB euro reference rates |
| Sector benchmarks by region | Damodaran Online datasets, plus figures computed from the filings above |
| Default and recovery base rates | Published annual rating-agency default studies, cited by year |

---

## Status

Phases run in number order. The earlier "Handling many users" phase is now
phase 8, after the data, ML, AI and feature work, except background jobs,
which moved into Foundations (1.9) because later phases need them.

| # | Task | Needs | Status |
|---|---|---|---|
| **0** | **Online** | | |
| 0.1 | Put the current app online (free) | — | ☑ |
| 0.2 | Your own domain | 0.1, 1.4 | ☐ |
| 0.3 | Development cycle gates (coverage, PR checks) | — | ☐ |
| **1** | **Foundations** | | |
| 1.1 | Test copy (staging) and production | 0.1 | ☑ |
| 1.2 | Monitoring, error tracking, logs | 1.1 | ☑ |
| 1.3 | Database (Neon) | 1.1 | ☑ |
| 1.4 | Accounts and login | 1.3 | ☑ |
| 1.5 | Saved deals, versions and settings | 1.4 | ☑ |
| 1.6 | Usage limits and abuse protection | 1.4 | ☑ |
| 1.7 | Security hardening | 1.4 | ☑ |
| 1.8 | Backups and recovery | 1.3 | ☑ |
| 1.9 | Background jobs and scheduled jobs | 1.3, 1.6 | ☑ |
| **2** | **Universal by design** | | |
| 2.1 | Honest labels on inception-era parts (do early) | — | ☑ |
| 2.2 | Currency and money units everywhere | 1.5 | ☑ |
| 2.3 | Locale: numbers, dates, fiscal years, languages | 2.2 | ☐ |
| 2.4 | Global debt structures and interest rates | 2.2 | ☐ |
| 2.5 | Global tax rules | 2.2 | ☐ |
| 2.6 | Accounting standards (IFRS and US GAAP) | 2.2 | ☐ |
| 2.7 | Backtest becomes "plan vs actual" for any deal | 1.5, 2.2 | ☐ |
| 2.8 | Risk warnings computed, not written in | 2.1 | ☐ |
| **3** | **Trust in the numbers** | | |
| 3.1 | Model version on every result | 1.5 | ☐ |
| 3.2 | Written methodology | — | ☐ |
| 3.3 | Audit history | 1.5 | ☐ |
| 3.4 | Hand-checked reference cases (incl. non-US) | 3.2, 2.4, 2.5 | ☐ |
| **4** | **Global market data platform** | | |
| 4.1 | Company filings from many countries | 1.9, 2.6 | ☐ |
| 4.2 | Economic data by country, and exchange rates | 1.9 | ☐ |
| 4.3 | Sourced defaults by region, sector and size | 4.1, 4.2 | ☐ |
| 4.4 | Risk ranges, correlations and scenarios by region | 4.2, 4.3 | ☐ |
| 4.5 | Optional reference library (deals and base rates) | 4.1 | ☐ |
| 4.6 | Model validation framework | 4.3, 4.5, 2.7 | ☐ |
| **5** | **ML done properly** | | |
| 5.1 | ML evaluation harness and model cards | 4.6 | ☐ |
| 5.2 | Deal risk score from market data | 5.1, 2.8 | ☐ |
| 5.3 | Distress predictor | 5.1 | ☐ |
| 5.4 | Multiple predictor by region | 5.1, 4.3 | ☐ |
| 5.5 | Growth calibrator by region | 5.1, 4.3 | ☐ |
| 5.6 | Driver explanations | 5.1 | ☐ |
| 5.7 | Economic regime by region, scheduled | 5.1, 4.2 | ☐ |
| 5.8 | Live sliders for any deal, lightweight enough for free hosting | 5.1, 1.9 | ☐ |
| 5.9 | Personalized defaults | 1.5, 5.1 | ☐ |
| **6** | **AI features** | | |
| 6.1 | File uploads | 1.4, 1.9 | ☐ |
| 6.2 | Upload a document (any language), get a deal | 6.1, 3.1, 2.6 | ☐ |
| 6.3 | Investment memo writer | 3.1, 1.5 | ☐ |
| 6.4 | Plain-English explanations on screens | 3.1 | ☐ |
| **7** | **Product features** | | |
| 7.1 | Onboarding and in-app help | 1.5 | ☐ |
| 7.2 | Teams: sharing, permissions, comments | 1.5, 3.3 | ☐ |
| 7.3 | Excel: live-formula workbooks, then add-in | 1.5, 2.3 | ☐ |
| 7.4 | Portfolio tracking in any currency | 1.5, 1.9, 4.2, 2.7 | ☐ |
| 7.5 | Lender view with regional conventions | 2.4, 5.3 | ☐ |
| 7.6 | Phones, tablets and accessibility | — | ☐ |
| 7.7 | Public API and webhooks | 1.4, 1.6, 3.1 | ☐ |
| 7.8 | More languages | 2.3 | ☐ |
| **8** | **Ready to scale** | | |
| 8.1 | Result caching | 1.9 | ☐ |
| 8.2 | Speed budgets (web and API) | 1.2 | ☐ |
| 8.3 | Load testing and a scaling plan | 8.1, 8.2 | ☐ |
| **9** | **Subscriptions machinery (test mode)** | | |
| 9.1 | Subscription billing (multi-currency, test mode) | 1.4 | ☐ |
| 9.2 | Plan limits and usage metering | 9.1, 1.6 | ☐ |
| 9.3 | Usage analytics and feature flags | 1.4 | ☐ |
| 9.4 | In-app support and feedback | 1.2, 1.4 | ☐ |
| **10** | **Enterprise-grade (free parts)** | | |
| 10.1 | Data export and account deletion | 1.5, 6.1 | ☐ |
| 10.2 | Automated security testing | 1.7, 1.1 | ☐ |
| 10.3 | Region-ready data model and recovery rehearsal | 1.8, 8.3 | ☐ |
| **11** | **Running it day to day** | | |
| 11.1 | Release routine and rollback | 1.1, 3.1 | ☐ |
| 11.2 | Incident routine and runbooks | 1.2 | ☐ |
| 11.3 | Free-limit, data-freshness and dependency upkeep | 1.9, 4.3 | ☐ |
| 11.4 | Owner's handbook: running it without Claude | 1.9 | ☐ |
| **12** | **Paid tiers (only once the product is functional)** | | |
| 12.1 | Always-on hosting and dedicated workers with autoscaling | 8.3 | ☐ |
| 12.2 | Paid database with longer backups | 12.1 | ☐ |
| 12.3 | Commercial web hosting (Vercel Pro) | — | ☐ |
| 12.4 | Paid AI provider for confidential documents | 6.2 | ☐ |
| 12.5 | Live payments | 9.2, 12.3 | ☐ |
| 12.6 | Company logins (SSO) and admin console | 7.2 | ☐ |
| 12.7 | Regional hosting and data residency | 10.3, 12.1 | ☐ |
| 12.8 | Higher service limits as usage grows | 11.3 | ☐ |

**Order:** 0.1 → 2.1 (small, do early) → 1 → 2 → 3 → 4 → 5 → 6 and 7 (any
order) → 8 → 9 → 10 → 11 → 12. Task 3.2 can start any time; 11.1 and 11.2
any time after phase 1. Added 2026-09-24: **0.3 next** (before 2.3), then
back to phase 2; **0.2 as soon as the domain is bought** (it jumps the queue
then); **11.4 any time**, refreshed whenever a later task changes how the app
is run.

---

## Phase 0 — Online

### 0.1 Put the current app online (free)
- **You first:** follow DEPLOY.md: Render blueprint on the free plan, and a
  Vercel Hobby project with `FSE_API_URL`. Send Claude both URLs.
- **Claude does:** verifies both sites; adds a read-only browser test against
  the live URLs; records the URLs in CLAUDE.md; notes the free-plan sleep on
  the status page.
- **Done when:** the live status bar shows "API ok" (after wake-up); the default
  deal shows 21.2% IRR; URLs are in CLAUDE.md.

### 0.2 Your own domain
- **You first:**
  - buy the domain. Cloudflare Registrar is recommended: it charges the
    registry's price with no markup, renewals don't jump and DNS is free.
    Porkbun is a good alternative. Prefer `.com`; `.app` is fine (HTTPS only);
  - keep its DNS at the registrar (Cloudflare) and tell Claude the name;
  - in Clerk, create the **production instance** for that domain (a Clerk
    production instance can't use `*.vercel.app`), and put its keys in
    Vercel and Render;
  - add the domain in Vercel (project `fse-ml`) and `api.<domain>` in Render
    (`fse-api`), then add the DNS records both dashboards show. Claude lists
    them exactly first.
- **Claude does:**
  - production at `https://<domain>` (and `www.` redirecting to it), the API at
    `https://api.<domain>`, both on the free plans' automatic certificates;
    `fse-ml.vercel.app` redirects to the domain;
  - staging stays where it is (Vercel preview and `fse-api-staging`), so
    nothing about the test copy changes;
  - moves every place that names the production URL: CORS origins
    (`api/security.py`), the CSP (`web/src/lib/security/headers.ts`, Clerk's
    new hosts), `FSE_API_URL`, Better Stack monitors and status page
    (`ops/betterstack.py`), header scan (`ops/check_headers.py`), live tests
    and `live.yml`, Sentry's allowed origins, DEPLOY.md and CLAUDE.md;
  - **accounts:** a new Clerk instance gives every person a new user id, so
    saved deals and settings keyed to development-instance ids would be
    orphaned. Claude writes a one-off, tested mapping (old subject → new
    subject, run with the user's approval) or documents that the dev-instance
    accounts are test data to drop;
  - Cloudflare DNS records set to "DNS only" (grey cloud), so Vercel and
    Render issue their own certificates and the CSP and headers stay theirs;
  - Resend's domain records noted for later (PLAN.md emails) without enabling
    anything paid.
- **Done when:** `live.yml` passes against the domain; the old URL redirects;
  sign-in works on the Clerk production instance; `ops/check_headers.py`
  passes on both hosts; a deal saved before the switch opens after it (or the
  drop is documented and approved).
- **Free limits:** Vercel Hobby allows custom domains; Render free allows
  custom domains; the domain itself is the only cost (about 10–15 US dollars
  a year for `.com`).

### 0.3 Development cycle gates (coverage, PR checks)
The ECC framework merged in docs/WORKFLOW.md, enforced by CI where a machine
can check it.
- **You first:** optionally install the ECC plugin (docs/WORKFLOW.md
  "Installing ECC"); nothing else.
- **Claude does:**
  - Python coverage (`pytest-cov`) in the `core` and `ml` jobs, with a report
    in the job summary; the floor is today's figure and only rises
    (a `.coverage-floor` file); **80% of changed lines** (`diff-cover`)
    for new code, ECC's number;
  - a PR title check for the ECC types (`feat:`, `fix:` …), read-only token;
  - `docs/WORKFLOW.md` step 6 and CLAUDE.md updated with the new gates;
  - no web unit-test framework yet: the browser tests stay the web's proof,
    and one is added only when a task has logic worth unit-testing.
- **Done when:** CI shows coverage; a PR that lowers the floor or leaves
  changed lines under 80% fails (demonstrated on a throwaway branch); a PR
  title without a type fails.

---

## Phase 1 — Foundations

### 1.1 Test copy (staging) and production
- **You first:** create a second free Render web service for staging.
- **Claude does:**
  - Vercel preview deployments serve as the staging web app;
  - the `staging` branch deploys the staging API, `main` deploys production;
  - variables documented per environment;
  - browser tests run against staging after each deploy;
  - a written rollback procedure;
  - both environments fit inside Render's shared free hours (staging sleeps
    when unused).
- **Done when:** a staging merge appears only on staging; one production
  rollback is done on purpose and documented.

### 1.2 Monitoring, error tracking, logs
- **You first:** create free Sentry and Better Stack accounts; set `SENTRY_DSN`.
- **Claude does:**
  - error tracking in web and API (no deal contents or personal data);
  - structured logs with a shared request ID;
  - model-run timings;
  - uptime checks, with alerts that ignore expected free-plan wake-ups but
    catch real failures;
  - a status page;
  - UTC storage, shown in the viewer's time zone.
- **Done when:** a test error appears with its request ID; a broken deploy on
  staging alerts within 5 minutes.

### 1.3 Database (Neon)
- **You first:** create a free Neon project, with a `staging` branch; set
  `DATABASE_URL` for each environment.
- **Claude does:**
  - database layer (SQLAlchemy and Alembic);
  - handling for Neon waking from idle (connection retries);
  - local dev database;
  - CI with a real Postgres;
  - database status in `/api/health`;
  - money columns always next to a currency code, times in UTC;
  - a storage-usage check that warns at 80% of the 0.5 GB free limit.
- **Done when:** migrations run forwards and backwards in CI; the API recovers
  cleanly after Neon scales to zero (test); CLAUDE.md explains how to add a
  table.

### 1.4 Accounts and login
- **You first:** create a free Clerk application; enable email and Google
  sign-in; set keys.
- **Claude does:** sign-up, sign-in, reset; every API call except health needs a
  login; `users` table with country, preferred currency, locale and time zone
  (asked at sign-up); account page; e2e as a test user.
- **Done when:** a logged-out call gets 401 (test); sign-up reaches Deal; e2e
  passes logged in.

### 1.5 Saved deals, versions and settings
- **Claude does:** deals, versions and settings in the database; create, list,
  open, rename, duplicate, archive, delete, save and restore; deal list,
  autosave, version history; settings move from the browser to the account;
  versions stored compactly so the free 0.5 GB lasts.
- **Done when:** a saved deal reopens identically elsewhere; restoring a version
  brings back its exact IRR; users can't open each other's deals (test).

### 1.6 Usage limits and abuse protection
- **You first:** create a free Upstash Redis database; set its URL and token.
- **Claude does:**
  - per-user and per-address limits;
  - maximum simulation size, sized so a run fits the free server's memory;
  - request size limits and run timeouts;
  - clear messages;
  - counters batched so Redis stays well under 500,000 commands a month
    (falling back to database counters if Redis is unavailable).
- **Done when:** a test gets 429 after the limit; an oversized simulation is
  refused clearly; a usage estimate shows the monthly Redis command budget
  holds at the target traffic.

### 1.7 Security hardening
- **Claude does:**
  - security headers and a content security policy;
  - CORS locked to app domains;
  - CI blocks committed secrets;
  - free dependency scanning (Dependabot, pip-audit, npm audit);
  - free code scanning (CodeQL, free for public repositories);
  - least-privilege database role;
  - encrypted connections;
  - `SECURITY.md` and a threat model;
  - a review of what's visible in this **public** repository (no data, keys or
    internal notes).
- **Done when:** a header scan passes; CI fails on a planted fake key; CodeQL runs
  on every PR; the threat model is in `docs/security/`.

### 1.8 Backups and recovery
- **You first:** create a free Supabase project (brought forward from 1.9,
  which uses the same one for file uploads) with a **private** `backups`
  bucket, and add the GitHub Actions secrets `FSE_BACKUP_KEY`,
  `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `BACKUP_DATABASE_URL` and
  `BACKUP_STAGING_DATABASE_URL` (DEPLOY.md "Backups and recovery"). Nothing is
  backed up until they exist; `backup.yml` warns and does nothing meanwhile.
- **Claude does:**
  - a nightly scheduled GitHub Actions job that exports the database,
    **encrypts it** with a key kept in GitHub secrets, and stores it in
    Supabase Storage (1.9 sets up storage access) with rotation to stay under
    1 GB;
  - Neon's built-in restore window used for recent mistakes;
  - a documented restore procedure;
  - a monthly restore drill.
- **Done when:** a real restore from an encrypted backup into the Neon staging
  branch brings back a saved deal with identical results; old backups rotate
  out.

### 1.9 Background jobs and scheduled jobs
- **Why:** data refreshes, model retraining, AI document reading and big
  simulations take too long to run while the user waits, and the free engine
  has no separate worker machines.
- **You first:** the free Supabase project already exists (1.8 brought it
  forward for backups); add its keys for file storage and any other secrets
  the scheduled workflows need to GitHub Actions.
- **Claude does:**
  - **User-started jobs:** a job table in Neon and a small job runner inside
    the API. Submit, get a job ID, watch progress, fetch the result. Jobs
    survive a restart (resumed or marked failed with a clear message). The web
    app shows progress and stays usable.
  - **Scheduled jobs:** GitHub Actions scheduled workflows call protected job
    endpoints or run scripts directly (data refresh, retraining, backups,
    keep-alive for Supabase).
  - A **job queue interface**, so phase 12 can swap in dedicated workers by
    configuration.
  - Quick deal runs stay instant.
- **Done when:**
  - 10 simultaneous simulations on staging all finish, and the API still
    answers health checks during them;
  - seeded results are unchanged;
  - a scheduled workflow runs on staging and records its run;
  - swapping the queue implementation in a test needs no endpoint changes.

---

## Phase 2 — Universal by design

### 2.1 Honest labels on inception-era parts (do early)
- **Claude does:** label Settings defaults, Monte Carlo ranges, correlations and
  scenario presets "illustrative defaults — not market data"; Backtest says "4
  example deals from the 2006–2013 US market; not a validation of the model";
  the risk score says "early estimate based on 30 historical deals"; Live
  repeats its fixed training deal; correct the "~100 deals" claim in
  `ml/anomaly_detector.py`; note `macro_regime.py`'s discontinued ISM PMI
  series (ended 2022).
- **Done when:** every label is visible (e2e checks the text); no screen presents
  inception-era numbers as market facts.

### 2.2 Currency and money units everywhere
- **Why:** everything is in US dollars; "$M" is written into the code in 177
  places.
- **Claude does:** each deal has a currency (ISO code) and a display unit
  (thousands, millions or billions); the model stays currency-neutral while
  every response and export carries the currency; web inputs, tiles, charts
  and tables show the deal's symbol and unit; forecast companies carry their
  reporting currency; remove every hardcoded "$M", with a CI check against new
  ones.
- **Done when:** a EUR deal in thousands shows € and "k" everywhere (e2e); the
  same inputs give identical results in any currency (test); CI fails on a new
  hardcoded "$".

### 2.3 Locale: numbers, dates, fiscal years, languages
- **Why:** formats are fixed to `en-US` in 11 places; fiscal years are assumed
  to follow US filings.
- **Claude does:** number and date formats follow the user's locale (including
  lakh and crore grouping as an option); fiscal year-end per deal and company;
  all interface text in translation files (`next-intl`), English complete,
  right-to-left layout checked; Excel exports with per-cell formats.
- **Done when:** e2e passes in `en-US`, `de-DE` and `en-IN`; a March year-end
  company labels correctly; no visible text outside translation files (CI
  check).

### 2.4 Global debt structures and interest rates
- **Why:** today there's one fixed-rate senior loan plus one mezzanine tranche.
- **Claude does:**
  - any number of tranches of these types: amortising term loan, institutional
    term loan, unitranche, second lien, senior notes / high-yield bond, PIK
    notes, vendor loan, revolving credit facility, and shareholder loan;
  - per tranche: currency, size, fixed or floating, reference rate (SOFR,
    SONIA, €STR/EURIBOR, TONA, SARON, BBSY, MIBOR, or custom), margin, floor,
    yearly rate path, upfront and commitment fees, amortisation, sweep share,
    PIK and maturity;
  - rate paths default from market data (4.2) once available;
  - simulated rate uncertainty applies to floating tranches.
- **Done when:** hand-checked cases (3.4) pass for a floating SONIA loan with a
  floor, a PIK note and a unitranche; the current two-tranche deal reproduces
  today's results exactly; the UI adds and removes tranches (e2e).

### 2.5 Global tax rules
- **Claude does:** per-deal corporate rate; interest deductibility limit (none,
  a share of EBITDA, or a fixed amount) with carry-forward; tax losses carried
  forward with an optional yearly cap; optional minimum tax; editable country
  presets with source and date, marked "check with a tax adviser".
- **Done when:** hand-checked cases pass for a 30%-of-EBITDA interest cap and for
  losses carried forward; changing preset changes results as predicted (test).

### 2.6 Accounting standards (IFRS and US GAAP)
- **Claude does:** accounting standard per company and deal; mapping from each
  standard's line items to model inputs; IFRS 16 lease setting (EBITDA before
  or after leases; leases as debt or not); adaptable labels.
- **Done when:** a real IFRS filing maps to model inputs (test); the lease
  setting changes EBITDA and net debt as a hand-checked case predicts.

### 2.7 Backtest becomes "plan vs actual" for any deal
- **Claude does:** pick any saved deal as the plan; enter or upload actual yearly
  results and exit; compare with the exact attribution (EBITDA, multiple and
  net debt) in the deal's currency; the 4 historical deals move to the optional
  example library (4.5); the screen works with no library.
- **Done when:** a user's own saved deal is backtested end to end (e2e); with the
  library off the screen still works (test); attribution still adds up.

### 2.8 Risk warnings computed, not written in
- **Claude does:** replace fixed statistics ("38% historical distress rate",
  "Only 2 of 9…") with figures computed from sourced data: cited rating-agency
  base rates and the deal's own ratios. Each warning shows its source and
  sample; warnings with no data behind them are removed.
- **Done when:** a test fails if a warning contains a number not computed from
  data; each warning shows its source.

---

## Phase 3 — Trust in the numbers

### 3.1 Model version on every result
- **Claude does:** every result carries engine version, git commit, settings
  fingerprint and data vintage; saved versions and exports store them;
  reopening an old deal shows "results changed since saved";
  `MODEL_CHANGELOG.md`.
- **Done when:** exports show version and data vintage; the change notice
  appears for an old version (test).

### 3.2 Written methodology
- **Claude does:** `docs/methodology.md` covering every calculation, with
  regional differences called out and links to code and tests.
- **Done when:** every function affecting a number is covered.

### 3.3 Audit history
- **Claude does:** append-only log of deal created, edited, versioned, shared,
  exported, deleted, and settings changes; history view per deal; old entries
  compacted to respect free storage.
- **Done when:** each action creates exactly one entry (tests); entries can't be
  changed through the API.

### 3.4 Hand-checked reference cases (incl. non-US)
- **Claude does:** 20+ small deals solved by hand in a committed spreadsheet (no
  leverage; single tranche; fees only; zero growth; floating with floor; PIK;
  unitranche; interest cap; tax losses; IFRS 16; non-USD; March year-end),
  with tests requiring the engine to match.
- **Done when:** all match to 0.01; workbook and reasoning in `tests/reference/`.

---

## Phase 4 — Global market data platform

### 4.1 Company filings from many countries
- **You first:** get free API keys for UK Companies House and Japan EDINET; set
  them as secrets.
- **Claude does:**
  - one company-data interface with connectors for SEC EDGAR, Companies House,
    ESEF filings (filings.xbrl.org) and EDINET;
  - document-upload fallback (6.2) for everywhere else;
  - figures stored with currency, accounting standard, fiscal year-end, source
    link and filing date;
  - scheduled refresh via GitHub Actions within each source's rules;
  - search by name, LEI, ISIN, ticker or company number;
  - storage kept compact (summary figures, not whole filings) to fit the free
    database.
- **Done when:** a US, UK, EU and Japanese company each load with correct
  currency and standard (tests with recorded responses); figures link to
  filings; storage use stays within the free budget set in the task.

### 4.2 Economic data by country, and exchange rates
- **You first:** get a free FRED key; set `FRED_API_KEY`.
- **Claude does:** connectors for IMF, World Bank, OECD, BIS, ECB and FRED; per
  country GDP growth, inflation, policy rate, bond yields and credit spreads
  where available; current reference rates for 2.4; ECB exchange rates; stored
  with dates and sources; scheduled refresh.
- **Done when:** at least 20 major economies have current, sourced data; a SONIA
  tranche picks up the current rate by default (test); exchange rates refresh
  daily on staging.

### 4.3 Sourced defaults by region, sector and size
- **Claude does:**
  - starting assumptions from broad data (thousands of companies, never a few
    deals) per region, sector and size band: growth, margins, capex, working
    capital, tax, EV/EBITDA and typical leverage where public;
  - calculated by scheduled jobs; results stored as compact summary tables;
  - each shows source, sample size and date; users can override;
  - thin data falls back from country to region to global, with a message;
  - new deals start by asking region, sector, size and currency.
- **Done when:** a German industrials deal gets sourced defaults with sample sizes;
  a thin-data case shows the fallback; no unsourced default remains (CI check
  on the defaults registry).

### 4.4 Risk ranges, correlations and scenarios by region
- **Claude does:** uncertainty ranges per region, sector and size from real
  year-to-year variation; correlations per region; scenario presets built from
  each region's own past recessions and inflation spells (dates listed); all
  sourced.
- **Done when:** UK and India deals get different sourced ranges; each scenario
  lists its historical periods; all correlation matrices pass validity (test).

### 4.5 Optional reference library (deals and base rates)
- **Claude does:** reference transactions with inclusion rules for balanced
  coverage (regions, sizes, sectors, eras, successes and failures), every
  figure sourced, review screen with two-person approval; the 4 inception deals
  as examples; cited rating-agency default and recovery base rates by region,
  rating band and year; coverage page; an admin switch that hides the library
  without breaking anything.
- **Done when:** with the library off, every screen and model still works (e2e);
  base rates show citations; the coverage page reports counts.

### 4.6 Model validation framework
- **Claude does:** reports on calibration (outcomes inside predicted ranges as
  often as claimed) and bias, split by region, sector, size and era; uses the
  library plus opt-in anonymised plan-vs-actual results, always tested on newer
  data; published in the methodology; deal results always show IRR, MOIC,
  probability of loss, downside case, coverage and default risk together.
  Report generation runs as a scheduled job.
- **Done when:** a report is generated on staging with all splits; anonymisation
  is tested; the deal summary shows the full metric set.

---

## Phase 5 — ML done properly

**Rule:** models learn from global market data (4.1–4.4), are tested on newer
data split by region, work without the reference library, and appear only
where they beat a simple baseline. Elsewhere the app shows "not enough data".
**Training runs in GitHub Actions** (free); what the API loads must be small
enough for the free server.

### 5.1 ML evaluation harness and model cards
- **Claude does:** one harness (time-based splits, per-region results, baseline
  comparison); model card template; model registry; CI fails if a model gets
  worse; training and evaluation run as GitHub Actions jobs.
- **Done when:** the surrogate and anomaly detector have cards with per-region
  results.

### 5.2 Deal risk score from market data
- **Claude does:** rebuild `ml/anomaly_detector.py` to compare a deal with
  companies and deals like it in the same region, sector and size (4.3, 4.4,
  plus the library when on); "based on N companies in [region, sector]"; off
  where N is too small.
- **Done when:** it works with the library off; each region has its own
  comparison group (test); thin regions show "not enough data".

### 5.3 Distress predictor
- **Claude does:** rebuild `ml/distress_model.py` from coverage and leverage
  paths, calibrated to cited default base rates by region and rating band;
  yearly distress probability on Debt and in Monte Carlo; model card.
- **Done when:** implied default rates match base rates for comparable bands
  within the card's tolerance; higher leverage raises risk (test).

### 5.4 Multiple predictor by region
- **Claude does:** rebuild `ml/multiple_predictor.py` on regional listed-company
  multiples (4.3); ranges with comparables and "use suggestion".
- **Done when:** ranges contain actual multiples for most held-out companies per
  region (card); it works with the library off.

### 5.5 Growth calibrator by region
- **Claude does:** rebuild `ml/growth_calibrator.py` on the filings store and
  regional data (replacing US-only SimFin and fixed averages); "calibrate from
  sector and region" on Monte Carlo.
- **Done when:** ranges match observed growth on held-out years per region
  (card); using them changes the simulation.

### 5.6 Driver explanations
- **Claude does:** compare `ml/shap_attribution.py` with the current drivers
  view; ship per-deal explanations if they add value, precomputed or light
  enough for free hosting; otherwise remove with a note.
- **Done when:** written comparison; if shipped, explanations add up to the
  prediction (test).

### 5.7 Economic regime by region, scheduled
- **Claude does:** rebuild `ml/macro_regime.py` on 4.2's country data (replacing
  US-only series, including one discontinued in 2022) for the US, eurozone, UK,
  Japan, China, India and global; monthly retraining in GitHub Actions;
  Scenarios shows the deal region's current regime; correlations refresh from
  the same data.
- **Done when:** each region shows a dated regime; an India deal shows India's
  regime (test).

### 5.8 Live sliders for any deal, lightweight enough for free hosting
- **Why:** the surrogate learned one fixed deal, and its runtime (PyTorch) is
  too large for the free server's memory.
- **Claude does:** regenerate training data across multiples, holds, fees, debt
  types, rates and tax settings in GitHub Actions; retrain; export to a
  lightweight format (ONNX) that runs without PyTorch; warn only outside the
  trained range.
- **Done when:** median-IRR error stays under 0.5pp on varied held-out deals from
  several regions (card); the live sliders run on the free Render service
  within its memory (staging check).

### 5.9 Personalized defaults
- **Claude does:** move `ml/personalization.py` into the database; learn each
  user's usual assumptions per region and sector; suggest them alongside
  sourced defaults; off switch.
- **Done when:** suggestions use only that user's deals (test); off stops them.

---

## Phase 6 — AI features

### 6.1 File uploads
- **Claude does:** PDF, Excel and Word uploads to Supabase Storage with size and
  type checks; malware scanning using a free open-source scanner in the job
  runner; private per user or team; signed links; deletion rules; per-user
  quotas to stay within 1 GB.
- **Done when:** upload and download work; others can't access (test); bad or
  oversized files are refused; a quota message appears near the limit.

### 6.2 Upload a document (any language), get a deal
- **You first:** create a free Gemini API key for trials with public documents;
  set `AI_PROVIDER` and its key.
- **Claude does:**
  - rebuild `ml/nlp_extractor.py` behind a provider switch (Gemini free tier
    now; a paid provider in phase 12);
  - run as a background job;
  - read annual reports and information packs in major languages, and detect
    currency, units, accounting standard and fiscal year;
  - cite page and quote for each number;
  - a review screen where the user accepts each value;
  - **a clear warning, blocking upload on the free provider unless the user
    confirms the document is public**;
  - development and CI use recorded responses (no AI cost, no data sent);
  - an evaluation set of public reports from several countries and languages.
- **Done when:** extraction accuracy meets its card target on the public
  evaluation set; nothing enters a deal without acceptance; CI runs on recorded
  responses; the confidentiality warning blocks non-public uploads on the free
  provider (e2e).

### 6.3 Investment memo writer
- **Claude does:** memo drafts from a saved version (summary, full metric set,
  risks, drivers, scenarios, plan vs actual), exported to Word and PDF, in the
  user's language. Every number comes from the model run. The same free-provider
  confidentiality rule applies: on the free tier the AI receives only
  anonymised, non-identifying text, and every figure is filled in afterwards
  by the app.
- **Done when:** a test checks every figure against its run; a test proves no
  company names or identifiers are sent on the free provider.

### 6.4 Plain-English explanations on screens
- **Claude does:** "explain" on key tiles built mainly from **templates that use
  the on-screen numbers** (free, instant, no data leaves the app), with optional
  AI wording later; glossary on hover including regional terms; translated
  where available.
- **Done when:** explanations quote displayed numbers exactly (test); the
  glossary covers every label term.

---

## Phase 7 — Product features

### 7.1 Onboarding and in-app help
- **Claude does:** guided first deal (region, sector, size, currency, then
  sourced defaults); sample deals from several regions; empty-screen guides;
  input tips; help articles from the methodology; "what's new".
- **Done when:** a new test user completes the guided deal (e2e); every input has
  a tip.

### 7.2 Teams: sharing, permissions, comments
- **You first:** enable Clerk organizations (free tier); create a free Resend
  account; set its key.
- **Claude does:** teams with owner, editor and viewer roles; sharing; comments
  and @mentions emailed in the recipient's time zone and language (batched to
  stay inside free send limits); activity from audit history.
- **Done when:** permission tests pass; mentions email correctly.

### 7.3 Excel: live-formula workbooks, then add-in
- **Claude does:**
  - **Part a:** exports with working formulas (all debt types, tax rules,
    currency formats);
  - **Part b:** an Excel add-in (free to build and sideload) that signs in,
    opens a saved deal, runs the model and writes results.
- **Done when:**
  - (a) the recalculated workbook matches the app's IRR to 0.01pp for a
    non-USD floating-rate deal;
  - (b) the sideloaded add-in refreshes a saved deal.

### 7.4 Portfolio tracking in any currency
- **Claude does:** mark deals as owned; quarterly actuals (reusing 2.7); drift
  alerts; portfolio dashboard converted to a reporting currency at dated ECB
  rates, with local figures alongside; reforecast from actuals as a job.
- **Done when:** a GBP and EUR portfolio totals correctly in USD at stated rates
  (test); below-plan actuals trigger an alert.

### 7.5 Lender view with regional conventions
- **Claude does:** covenants per tranche (leverage, interest cover, fixed-charge
  cover, minimum liquidity) with definitions selectable by market convention;
  yearly tests; breach odds from simulation; distress probability (5.3);
  expected loss and recovery with regional base rates; lender export.
- **Done when:** a deal built to breach in year 2 is flagged in year 2 (test);
  breach odds come from the simulation.

### 7.6 Phones, tablets and accessibility
- **Claude does:** smaller-screen layouts; accessibility audit and fixes;
  automated accessibility checks in e2e; right-to-left check.
- **Done when:** key journeys pass at phone size; no serious accessibility issues.

### 7.7 Public API and webhooks
- **Claude does:** API keys per user; versioned v1 endpoints (deal runs,
  simulations, saved deals, market defaults by region); docs; webhooks for
  finished jobs; per-key limits sized for free hosting.
- **Done when:** a script with a key runs a deal and receives a webhook (test);
  revoked keys stop immediately.

### 7.8 More languages
- **Claude does:** full translations for a first set chosen when the task
  starts (e.g. Spanish, French, German, Japanese, Hindi, Arabic); translated
  help and glossary; CI fails on missing translation keys.
- **Done when:** e2e key journeys pass in each added language.

---

## Phase 8 — Ready to scale

### 8.1 Result caching
- **Claude does:** cache by inputs, settings, seed, model version and data
  vintage (Upstash, with a database fallback); unseeded runs skip it; version or
  data changes clear it; cache writes budgeted against the free command limit.
- **Done when:** a repeated seeded run returns in under 100 ms identically; a
  version bump misses (test); command usage stays in budget in the load test.

### 8.2 Speed budgets (web and API)
- **Claude does:** page-load and interaction budgets (Lighthouse in CI);
  response-time budgets per endpoint (CI); fixes for anything over budget.
- **Done when:** CI fails over budget; all pages and endpoints pass.

### 8.3 Load testing and a scaling plan
- **Why:** free hosting can't autoscale, but we must know the limits and have the
  paid switch-over ready.
- **Claude does:**
  - load tests (k6, free) against staging to find how many simultaneous users
    the free setup handles;
  - fixes for anything that fails early;
  - `docs/capacity.md` with the measured limits, the first thing to break, and
    for each paid step in phase 12, the expected capacity and monthly cost;
  - autoscaling rules written and tested locally, ready for 12.1.
- **Done when:** the free setup's limits are measured and documented; nothing
  fails with errors below that limit (it slows down, clearly); the phase 12
  configuration is ready.

---

## Phase 9 — Subscriptions machinery (test mode)

(Built and tested free in Stripe test mode. Real charges start in 12.5.)

### 9.1 Subscription billing (multi-currency, test mode)
- **You first:** create a Stripe account (test mode only, no card).
- **Claude does:** placeholder plans in several currencies; checkout, trial,
  upgrade, downgrade, cancel; billing portal; invoices; webhooks keeping plan
  status; failed-payment handling. All in test mode.
- **Done when:** in test mode a user can trial, upgrade and cancel in EUR and USD
  (e2e); webhooks are verified and safe to replay.

### 9.2 Plan limits and usage metering
- **Claude does:** one configuration defining plan contents; enforcement in the
  API; monthly counters; upgrade prompts.
- **Done when:** a lower plan can't use a higher plan's feature via the API
  (test); counters reset monthly.

### 9.3 Usage analytics and feature flags
- **You first:** create a free PostHog project (EU hosting if preferred); set
  its key.
- **Claude does:** key-action events and funnels; feature flags; no deal contents
  sent; consent respected.
- **Done when:** funnels show staging traffic; a test proves no deal inputs in
  events.

### 9.4 In-app support and feedback
- **Claude does:** "report a problem" with request ID (no deal contents without
  consent); feedback board with votes; admin view.
- **Done when:** a test report links to its Sentry request; votes recorded.

---

## Phase 10 — Enterprise-grade (free parts)

### 10.1 Data export and account deletion
- **Claude does:** full export of a user's or team's deals, versions and files;
  real deletion of rows and files, including from backups on rotation;
  retention settings per team.
- **Done when:** a deleted account leaves nothing (test); exports are complete.

### 10.2 Automated security testing
- **Claude does:** OWASP ZAP scans of staging on each release (free); CodeQL and
  dependency scans; container image scanning (free tools); a permission test for
  every endpoint and role; fixes with regression tests.
- **Done when:** scans run automatically; the permission matrix covers every
  endpoint; no high findings open.

### 10.3 Region-ready data model and recovery rehearsal
- **Why:** regional hosting costs money (12.7), but the software can be made
  ready for it now.
- **Claude does:**
  - every team has a data region field;
  - storage, backups and requests go through a region lookup, which is a
    single region for now;
  - infrastructure described as code (Render blueprint, Neon and Supabase
    setup scripts, Vercel config);
  - a full rebuild rehearsal of staging from code plus an encrypted backup.
- **Done when:** a test proves all data access goes through the region lookup;
  staging is rebuilt from scratch from code and backup, and saved deals give
  identical results.

---

## Phase 11 — Running it day to day

### 11.1 Release routine and rollback
- **Claude does:** release checklist (staging checks, changelog, model version,
  data vintage, rollback plan); release notes from merged PRs; tagged versions.
- **Done when:** one release follows the checklist end to end.

### 11.2 Incident routine and runbooks
- **Claude does:** severity levels; runbook per alert (including free-limit
  alerts); status updates; post-incident review template.
- **Done when:** a simulated incident on staging is handled with the runbook.

### 11.3 Free-limit, data-freshness and dependency upkeep
- **Claude does:**
  - a dashboard of usage against every free limit: Render hours, Neon storage
    and compute, Upstash commands, Supabase storage, Clerk users, Sentry
    events, Resend sends, PostHog events, AI free quota;
  - alerts at 80%;
  - a check that scheduled GitHub workflows are still enabled, since they
    pause after 60 days without a commit, with a monthly maintenance commit
    that also merges dependency updates;
  - alerts when a data source hasn't refreshed on time.
- **Done when:** the dashboard covers every free service; a simulated 80% usage
  alert fires; a deliberately broken data connector raises an alert.

### 11.4 Owner's handbook: running it without Claude
- **You first:** nothing.
- **Claude does:**
  - `docs/MAINTAINING.md`, written for you, not for Claude: set up a new
    computer; run everything locally; make and ship a small change through
    docs/WORKFLOW.md by hand; update dependencies (Dependabot PRs, Python and
    Node upgrades, Next.js majors); what each alert and failing workflow
    means and what to do; rotate each secret; restore a backup;
  - an **accounts table**: every outside service, what it's for, the variables
    it sets, its free limit, and when and how to upgrade it (linking PLAN.md
    phase 12);
  - a **monthly 15-minute checklist**: workflows still enabled, free-limit
    usage, backups verified, Dependabot merged, domain and certificates
    renewing;
  - `docs/ARCHITECTURE.md`: how the pieces fit, the design decisions and why
    (the "why" now spread across CLAUDE.md), with a diagram;
  - refreshed whenever a later task changes how the app is run.
- **Done when:** following only the handbook, you set up a clean checkout,
  ship a one-line change to staging and handle a simulated staging alert.

---

## Phase 12 — Paid tiers (only once the product is functional)

Each task is a switch-over prepared by the free phases. Do them one at a time,
when a limit is actually reached or before charging customers.

### 12.1 Always-on hosting and dedicated workers with autoscaling
- **You first:** upgrade the Render API to an always-on plan; add background
  worker services.
- **Claude does:** switch the job queue from in-API to dedicated workers
  (configuration from 1.9); apply the autoscaling rules from 8.3; move scheduled
  jobs from GitHub Actions to Render cron where better; allow the ML runtime to
  use full PyTorch if needed.
- **Done when:** the 8.3 load test passes at the new target with no sleep
  delays.

### 12.2 Paid database with longer backups
- **You first:** upgrade Neon (or move to another managed Postgres).
- **Claude does:** longer point-in-time recovery, higher storage, connection
  pooling, read replica if needed; update the backup job.
- **Done when:** restore to any point within the new window is rehearsed.

### 12.3 Commercial web hosting (Vercel Pro)
- **You first:** upgrade to Vercel Pro before any commercial use (charging users
  or paid development).
- **Claude does:** confirm settings, analytics and limits on Pro.
- **Done when:** the production project runs on Pro.

### 12.4 Paid AI provider for confidential documents
- **You first:** create a paid AI account (e.g. Anthropic) whose terms don't use
  your data for training; set a spending cap.
- **Claude does:** switch `AI_PROVIDER`; lift the "public documents only"
  restriction for that provider; per-user cost tracking.
- **Done when:** a confidential test document is processed on the paid provider;
  costs are logged; the free-provider restriction still applies if switched
  back.

### 12.5 Live payments
- **You first:** activate Stripe live mode (business details, bank).
- **Claude does:** live keys, webhooks, tax settings and a launch checklist.
- **Done when:** a real test purchase and refund succeed.

### 12.6 Company logins (SSO) and admin console
- **You first:** enable Clerk's enterprise SSO (paid).
- **Claude does:** company sign-in per team; joining by email domain; admin
  console (users, roles, seats, audit export); enforced login rules.
- **Done when:** a test identity provider signs into the right team; removing
  access is immediate (test).

### 12.7 Regional hosting and data residency
- **You first:** create API, worker, database and storage in the added regions
  (start with the US and Frankfurt, then Singapore).
- **Claude does:** turn on multiple regions in the region lookup from 10.3;
  teams choose their region; data, files and backups stay in it; shared
  market data replicated; residency test.
- **Done when:** an EU team's data exists only in the EU region (test).

### 12.8 Higher service limits as usage grows
- **You first:** upgrade whichever service the 11.3 dashboard shows nearing its
  limit (Clerk, Sentry, Upstash, Supabase, Resend, PostHog, GitHub).
- **Claude does:** adjust configuration and alerts for the new limits; consider
  making the repository private once paid GitHub Actions minutes are available.
- **Done when:** the dashboard shows healthy headroom on every service.

---

## Appendix — US and inception-deal assumptions found in the code (2026-09-15)

| Where | Assumption | Fixed by |
|---|---|---|
| 177 places across `core/`, `api/`, `lbo_engine/`, `ml/`, `web/src` | Money shown as "$M" | 2.2 (done: deal currency and unit; CI check against new dollar signs) |
| 11 places in `web/src` | Number formats fixed to `en-US` | 2.3 |
| `ml/edgar_extractor.py` | US SEC only, `us-gaap` tags, USD, US fiscal years | 2.6, 4.1 |
| `lbo_engine/capital_structure.py` `build_simple_two_tranche_structure` | One fixed-rate senior loan (5% amortisation) plus one mezzanine bullet | 2.4 |
| `lbo_engine/operating_model.py` and returns | Flat tax, interest always fully deductible | 2.5 |
| `core/config.py` `DEFAULTS` | Growth, margins, multiples, rates, leverage, fees, ranges, correlations, scenario multipliers and the 20% hurdle typed in with no source | 2.1, 4.3, 4.4 |
| `simulation/vectorized_simulation.py` `DEFAULT_CORR` | Correlation matrix typed in | 4.4 |
| `core/backtesting.py` `PRELOADED_DEALS` | Backtest limited to 4 US mega-deals (2006–2013), unsourced actuals, fixed ranges | 2.7, 4.5 |
| `ml/anomaly_detector.py` | 30 US deals plus synthetic; claims "~100"; fixed warning statistics | 2.1, 2.8, 5.2 |
| `ml/distress_model.py` | 39 hand-entered cases | 5.3 |
| `ml/multiple_predictor.py` | 25 rows | 5.4 |
| `ml/growth_calibrator.py` | US SimFin, fixed Damodaran averages | 5.5 |
| `ml/macro_regime.py`, `ml/correlation_updater.py` | US-only FRED series (incl. ISM PMI, discontinued 2022) | 4.2, 5.7 |
| `ml/surrogate/` | One fixed training deal; PyTorch runtime too large for free hosting | 5.8 |
| `web/src` Backtest screens | Only the preloaded deals can be tested | 2.7 |
| `render.yaml` and DEPLOY.md | Render free database would expire after 30 days; not used (Neon instead) | 1.3 |
