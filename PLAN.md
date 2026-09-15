# PLAN.md — making the software full-scale, global and universal

Scope (agreed with the user, 2026-09-15): **the software only**. That means
everything needed to turn the current app into a scalable, secure,
production-grade tool that works for **any country, currency and deal**, with
the ML and feature ideas built in.

Out of scope for now: market research, customer interviews, company name,
legal work, pricing decisions, marketing and sales.

## Guiding principles

These apply to every task; a task isn't done if it breaks one.

1. **Global, not US by default.** Every money figure has a currency. Every
   rate, tax rule, accounting convention and data source is chosen by region,
   never assumed to be American. English first, but built so other languages
   can be added.
2. **A real-world tool, not a demo built on a few famous deals.** The app must
   be fully useful with *only the user's own deal*. The four historical deals
   used during inception (Burger King, Hilton, Dell, Freescale) become an
   optional example library, never an input to calculations, defaults or
   models.
3. **Every default has a source.** Starting assumptions come from broad market
   data for the chosen region, sector and company size, each showing where it
   came from, how many companies it's based on, and when it was updated. Users
   can always override. Until that exists, defaults are labelled
   "illustrative".
4. **Estimates are labelled as estimates.** ML and AI outputs appear only if
   they pass accuracy tests, and always show what they're based on.
5. **The calculations are proven correct by hand-checked cases, not by
   history.** Historical data is for calibrating assumptions and testing
   predictions, not for the maths.

---

## How to use this plan with Claude

**One task per session, one pull request per task.**

Start each session with this prompt (swap in the task number):

> Read CLAUDE.md and PLAN.md. Do task **1.3** only. Check its "Needs" are done
> and its "You first" items are in place; if not, stop and tell me what's
> missing. Follow PLAN.md's guiding principles and CLAUDE.md's working rules
> (branch, tests that check real output, mutation-check new tests, PR, CI
> green, merge). When done, tick the task in PLAN.md's status table in the
> same PR and tell me what to check.

**Rules**
- Do the lowest-numbered open task whose "Needs" are done.
- **"You first"** is only what Claude can't do: creating accounts on outside
  services, entering passwords or payment details, and copying secret keys
  into Render and Vercel.
- **Secrets never go in chat or code.** Keys go in the Render and Vercel
  dashboards (and a git-ignored `.env`). Tell Claude the variable *name*.
- **"Done when"** is the acceptance test; Claude shows evidence for each line.
- A task too big for one session is split into lettered parts (3.1a, 3.1b).
- **Free, open data first.** Paid data sources are noted as options, never
  required.

**Services** (defaults; swap before a task starts)

| Need | Service | Global notes | From task |
|---|---|---|---|
| Website | Vercel | Serves worldwide from edge locations | 0.1 |
| Engine, workers, jobs, database, Redis | Render | Pick the region nearest users (US, Frankfurt, Singapore); more regions in 10.4 | 0.1 |
| Accounts and login | Clerk | Many sign-in languages; company logins later | 1.4 |
| Errors and uptime | Sentry, Better Stack | Uptime checks from several continents | 1.2 |
| Emails | Resend | — | 8.2 |
| File storage | Cloudflare R2 | Region can be pinned (e.g. EU) | 7.1 |
| AI | Anthropic API | Reads documents in many languages | 7.2 |
| Subscriptions | Stripe | Multi-currency prices, local payment methods and tax | 9.1 |
| Usage analytics | PostHog | EU hosting available | 9.3 |

**Open data sources this plan uses** (all free or free-tier; each has a
dedicated task)

| Data | Sources |
|---|---|
| Company filings | SEC EDGAR (US), UK Companies House, EU/UK ESEF filings (filings.xbrl.org), Japan EDINET, and document upload for everywhere else |
| Economic data by country | IMF, World Bank, OECD, BIS (policy rates), ECB Data Portal, FRED (which also carries many non-US series) |
| Exchange rates | ECB euro reference rates |
| Sector benchmarks by region | Damodaran Online datasets (US, Europe, Japan, emerging markets, global), plus figures computed from the filings above |
| Default and recovery base rates | Published annual default studies from rating agencies, cited by year |

---

## Status

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
| **2** | **Universal by design** | | |
| 2.1 | Honest labels on inception-era parts (do first) | — | ☐ |
| 2.2 | Currency and money units everywhere | 1.5 | ☐ |
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
| **4** | **Handling many users** | | |
| 4.1 | Job queue and workers for heavy runs | 1.3, 1.6 | ☐ |
| 4.2 | Result caching | 4.1 | ☐ |
| 4.3 | Speed budgets (web and API) | 1.2 | ☐ |
| 4.4 | Load testing and autoscaling | 4.1, 4.2 | ☐ |
| **5** | **Global market data platform** | | |
| 5.1 | Company filings from many countries | 1.3, 4.1, 2.6 | ☐ |
| 5.2 | Economic data by country, and exchange rates | 1.3, 4.1 | ☐ |
| 5.3 | Sourced defaults by region, sector and size | 5.1, 5.2 | ☐ |
| 5.4 | Risk ranges, correlations and scenarios by region | 5.2, 5.3 | ☐ |
| 5.5 | Optional reference library (deals and base rates) | 5.1 | ☐ |
| 5.6 | Model validation framework | 5.3, 5.5, 2.7 | ☐ |
| **6** | **ML done properly** | | |
| 6.1 | ML evaluation harness and model cards | 5.6 | ☐ |
| 6.2 | Deal risk score from market data | 6.1, 2.8 | ☐ |
| 6.3 | Distress predictor | 6.1 | ☐ |
| 6.4 | Multiple predictor by region | 6.1, 5.3 | ☐ |
| 6.5 | Growth calibrator by region | 6.1, 5.3 | ☐ |
| 6.6 | Driver explanations | 6.1 | ☐ |
| 6.7 | Economic regime by region, scheduled | 6.1, 5.2 | ☐ |
| 6.8 | Live sliders for any deal, currency and region | 6.1, 4.1 | ☐ |
| 6.9 | Personalized defaults | 1.5, 6.1 | ☐ |
| **7** | **AI features** | | |
| 7.1 | File uploads | 1.4, 4.1 | ☐ |
| 7.2 | Upload a document (any language), get a deal | 7.1, 3.1, 2.6 | ☐ |
| 7.3 | Investment memo writer | 3.1, 1.5 | ☐ |
| 7.4 | Plain-English explanations on screens | 3.1 | ☐ |
| **8** | **Product features** | | |
| 8.1 | Onboarding and in-app help | 1.5 | ☐ |
| 8.2 | Teams: sharing, permissions, comments | 1.5, 3.3 | ☐ |
| 8.3 | Excel: live-formula workbooks, then add-in | 1.5, 2.3 | ☐ |
| 8.4 | Portfolio tracking in any currency | 1.5, 4.1, 5.2, 2.7 | ☐ |
| 8.5 | Lender view with regional conventions | 2.4, 6.3 | ☐ |
| 8.6 | Phones, tablets and accessibility | — | ☐ |
| 8.7 | Public API and webhooks | 1.4, 1.6, 3.1 | ☐ |
| 8.8 | More languages | 2.3 | ☐ |
| **9** | **Subscriptions and usage** | | |
| 9.1 | Subscription billing (multi-currency) | 1.4 | ☐ |
| 9.2 | Plan limits and usage metering | 9.1, 1.6 | ☐ |
| 9.3 | Usage analytics and feature flags | 1.4 | ☐ |
| 9.4 | In-app support and feedback | 1.2, 1.4 | ☐ |
| **10** | **Enterprise-grade** | | |
| 10.1 | Company logins (SSO) and admin console | 8.2 | ☐ |
| 10.2 | Data export and account deletion | 1.5, 7.1 | ☐ |
| 10.3 | Automated security testing | 1.7, 1.1 | ☐ |
| 10.4 | Regional hosting and data residency | 1.8, 4.4 | ☐ |
| 10.5 | Disaster recovery | 1.8, 10.4 | ☐ |
| **11** | **Running it day to day** | | |
| 11.1 | Release routine and rollback | 1.1, 3.1 | ☐ |
| 11.2 | Incident routine and runbooks | 1.2 | ☐ |
| 11.3 | Dependency, data-freshness and cost upkeep | 1.7, 5.3 | ☐ |

**Order in short:** 0.1 → 2.1 (small, do early) → 1 → 2 → 3 and 4 → 5 →
6 → 7 and 8 → 9 → 10 → 11 (11.1 and 11.2 can start after phase 1). 3.2 can
start any time.

---

## Phase 0 — Online

### 0.1 Put the current app online
- **You first:** follow DEPLOY.md: Render blueprint, Vercel project with
  `FSE_API_URL`; send Claude both URLs.
- **Claude does:** verifies both sites, adds a read-only browser test against
  the live URLs, records the URLs in CLAUDE.md.
- **Done when:** the live status bar shows "API ok"; the default deal shows
  21.2% IRR; URLs are in CLAUDE.md.

---

## Phase 1 — Foundations

### 1.1 Test copy (staging) and production
- **You first:** create a staging Render service and Vercel environment.
- **Claude does:** `staging` branch deploys to staging and `main` to
  production; variables documented; browser tests run against staging after
  each deploy; a written rollback procedure.
- **Done when:** a staging merge appears only on staging; one production
  rollback is done on purpose and documented.

### 1.2 Monitoring, error tracking, logs
- **You first:** create Sentry and Better Stack accounts; set `SENTRY_DSN`.
- **Claude does:** error tracking in web and API (no deal contents or personal
  data); structured logs with a shared request ID; model-run timings; uptime
  checks from several continents; status page. All timestamps are in UTC and
  shown in the viewer's time zone.
- **Done when:** a test error appears with its request ID; stopping staging
  alerts within 5 minutes.

### 1.3 Database
- **You first:** create Render Postgres (staging and production); set
  `DATABASE_URL`.
- **Claude does:** database layer (SQLAlchemy and Alembic); local dev database;
  CI with a real Postgres; database status in `/api/health`. Money columns
  always sit next to a currency code; times are stored in UTC.
- **Done when:** migrations run forwards and backwards in CI; CLAUDE.md
  explains how to add a table.

### 1.4 Accounts and login
- **You first:** create a Clerk application; enable email and Google sign-in;
  set keys.
- **Claude does:** sign-up, sign-in, reset; every API call except health needs
  a login; `users` table with the user's country, preferred currency, locale
  and time zone (asked at sign-up, editable); account page; e2e as a test
  user.
- **Done when:** a logged-out call gets 401 (test); sign-up reaches Deal; e2e
  passes logged in.

### 1.5 Saved deals, versions and settings
- **Claude does:** deals, versions and user settings in the database; create,
  list, open, rename, duplicate, archive, delete, save and restore; deal list,
  autosave, version history in the web app; settings move from the browser to
  the account.
- **Done when:** a saved deal reopens identically in another browser; restoring
  a version brings back its exact IRR; users can't open each other's deals
  (test).

### 1.6 Usage limits and abuse protection
- **You first:** create Render Key Value (Redis); set `REDIS_URL`.
- **Claude does:** per-user and per-address limits; maximum simulation size;
  request size limits; model-run timeouts; clear messages.
- **Done when:** a test gets 429 after the limit; an oversized simulation is
  refused clearly; normal e2e use never hits a limit.

### 1.7 Security hardening
- **Claude does:** security headers and content security policy; CORS locked
  to app domains; CI blocks committed secrets; dependency scans and automatic
  update PRs; least-privilege database user; encrypted connections;
  `SECURITY.md`; a threat model.
- **Done when:** a header scan passes; CI fails on a planted fake key; the threat
  model is in `docs/security/`.

### 1.8 Backups and recovery
- **You first:** confirm daily backups and point-in-time recovery on the
  database plan.
- **Claude does:** documents the schedule; rehearses a restore into staging;
  adds a monthly drill to the runbook.
- **Done when:** a real restore brings back a saved deal with identical results.

---

## Phase 2 — Universal by design

### 2.1 Honest labels on inception-era parts (do first)
- **Why:** until real data replaces them, users must know what's illustrative.
- **Claude does:**
  - Settings defaults, Monte Carlo ranges, correlations and scenario presets
    are labelled "illustrative defaults — not market data";
  - the Backtest screen says "4 example deals from the 2006–2013 US market;
    not a validation of the model";
  - the deal risk score says "early estimate based on 30 historical deals";
  - the Live screen repeats its fixed training deal;
  - the stale "~100 deals" claim in `ml/anomaly_detector.py` is corrected;
  - `macro_regime.py`'s discontinued data series (ISM PMI, ended 2022) is
    noted.
- **Done when:** each label is visible (e2e checks the text); no screen
  presents inception-era numbers as market facts.

### 2.2 Currency and money units everywhere
- **Why:** every figure is in US dollars today; "$M" is written into the code
  in 177 places.
- **Claude does:**
  - each deal has a currency (ISO code: USD, EUR, GBP, INR, JPY and others)
    and a display unit (thousands, millions or billions);
  - the model stays currency-neutral, but every API response and export
    carries the currency;
  - web inputs, tiles, charts and tables show the deal's symbol and unit;
  - forecast companies carry their reporting currency;
  - remove every hardcoded "$M" (a CI check rejects new ones).
- **Done when:**
  - a EUR deal in thousands shows € and "k" on every screen and export (e2e);
  - the same inputs give identical model results in any currency (test);
  - the CI check fails on a new hardcoded "$".

### 2.3 Locale: numbers, dates, fiscal years, languages
- **Why:** formatting is fixed to US style (`en-US` in 11 places), and fiscal
  years are assumed to follow US filings.
- **Claude does:**
  - number and date formatting follow the user's locale (1,234.5 vs 1.234,5;
    lakh and crore grouping for India as an option);
  - fiscal year-end is chosen per deal and company (March, June, December and
    so on), and labels use it;
  - all interface text moves into translation files (`next-intl`), with
    English complete; right-to-left layout support is checked;
  - Excel exports use locale-neutral numbers with formats set per cell.
- **Done when:** e2e passes in `en-US`, `de-DE` and `en-IN` locales with
  correctly formatted numbers; a March year-end company shows FY labels
  correctly; no user-visible text remains outside translation files (CI check).

### 2.4 Global debt structures and interest rates
- **Why:** today there's one fixed-rate senior loan (5% amortisation) plus one
  mezzanine tranche: a US mid-2000s structure.
- **Claude does:**
  - **Tranches:** any number of them, of these types: amortising term loan,
    institutional term loan, unitranche, second lien, senior notes / high-yield
    bond, PIK notes, vendor loan, revolving credit facility, and shareholder
    loan.
  - **Per tranche:** currency, amount (as a multiple of EBITDA or an amount),
    fixed or floating rate, reference rate (SOFR, SONIA, €STR/EURIBOR, TONA,
    SARON, BBSY, MIBOR, or custom), margin, floor, rate path by year, upfront
    fee, commitment fee, amortisation profile, cash-sweep share, PIK toggle and
    maturity.
  - **Rate paths:** by default from the country's current market rates (5.2);
    until then, entered by the user.
  - The Monte Carlo interest-rate uncertainty applies to floating tranches
    only.
- **Done when:**
  - hand-checked cases (3.4) pass for a floating SONIA term loan with a floor,
    a PIK note and a unitranche;
  - the existing two-tranche deal reproduces today's results exactly;
  - the UI adds and removes tranches (e2e).

### 2.5 Global tax rules
- **Why:** tax is one flat rate on profit; interest is always fully deductible.
  Many countries cap it (e.g. rules limiting net interest deductions to a share
  of EBITDA).
- **Claude does:**
  - tax settings per deal: corporate rate; interest deductibility limit (none,
    a share of EBITDA, or a fixed amount); carry-forward of disallowed
    interest; tax losses carried forward (with an optional yearly cap);
    optional minimum tax;
  - country presets with the source and date of each figure, always editable;
  - presets clearly marked "check with a tax adviser", since rules change.
- **Done when:** hand-checked cases pass for a deal hitting a 30%-of-EBITDA
  interest cap, and for losses carried forward; changing the preset changes
  results as the case predicts (test).

### 2.6 Accounting standards (IFRS and US GAAP)
- **Why:** the forecast and filings import only understand US GAAP tags; most
  of the world reports under IFRS, where leases, for example, change EBITDA.
- **Claude does:**
  - an accounting standard per company and deal (IFRS, US GAAP, or other
    local);
  - a mapping from each standard's line items to the model's inputs;
  - lease handling (IFRS 16: EBITDA before or after lease costs, with leases
    as debt or not) as a clear setting;
  - labels adapt ("turnover" vs "revenue" as an option).
- **Done when:** an IFRS company's statements map to model inputs (test with a
  real filing); the lease setting changes EBITDA and net debt as a
  hand-checked case predicts.

### 2.7 Backtest becomes "plan vs actual" for any deal
- **Why:** Backtest only offers the 4 inception deals; you can't test your own.
- **Claude does:**
  - Backtest becomes a general tool: pick any saved deal as "the plan", then
    enter or upload what actually happened (yearly revenue, EBITDA, cash flow,
    debt, exit);
  - compare the plan with actual results, using the existing exact attribution
    (EBITDA, multiple and net debt), in the deal's currency;
  - the 4 historical deals move into the optional example library (5.5),
    shown under "examples";
  - the Backtest screen works with an empty library.
- **Done when:** a user's own saved deal can be backtested end to end (e2e);
  with the example library switched off the screen still works (test);
  attribution still adds up exactly.

### 2.8 Risk warnings computed, not written in
- **Why:** the risk panel's statements are fixed sentences ("38% historical
  distress rate", "Only 2 of 9 such deals…").
- **Claude does:** replace every fixed statistic with a figure computed from
  sourced data. Until phase 5 exists, that means the rating-agency default
  base rates (cited) and the deal's own numbers (coverage ratios, leverage).
  Each warning shows its source and sample size; warnings with no data behind
  them are removed.
- **Done when:** a test fails if any warning text contains a number not
  computed from data; each warning shows its source.

---

## Phase 3 — Trust in the numbers

### 3.1 Model version on every result
- **Claude does:** every result carries the engine version, git commit, a
  settings fingerprint and the data vintage (which market data snapshot it
  used); saved versions and exports store them; reopening an old deal shows
  "results changed since saved"; `MODEL_CHANGELOG.md`.
- **Done when:** exports show version and data vintage; the change notice
  appears for an old version (test).

### 3.2 Written methodology
- **Claude does:** `docs/methodology.md` covering every calculation (deal model,
  debt types, rates, tax rules, accounting options, simulation, scenarios,
  plan vs actual, forecast, each ML model's limits), linking code and tests.
  Written so it applies to any country, with regional differences called out.
- **Done when:** every function affecting a number is covered.

### 3.3 Audit history
- **Claude does:** append-only log of deal created, edited, versioned, shared,
  exported, deleted, and settings changes; history view per deal.
- **Done when:** each action creates exactly one entry (tests); entries can't be
  changed through the API.

### 3.4 Hand-checked reference cases (incl. non-US)
- **Why:** proves the maths is right independently of any historical data.
- **Claude does:**
  - 20+ small deals solved by hand in a plain spreadsheet (committed): no
    leverage; single tranche; fees only; zero growth; floating rate with
    floor; PIK; unitranche; interest cap; tax losses; IFRS 16 leases;
    non-USD currency; March year-end;
  - tests require the engine to match each one.
- **Done when:** all cases match to 0.01; workbook and reasoning are in
  `tests/reference/`.

---

## Phase 4 — Handling many users

### 4.1 Job queue and workers for heavy runs
- **You first:** create a Render background worker.
- **Claude does:** simulations, scenarios, plan-vs-actual, exports, data
  refreshes and AI become jobs (submit, progress, result); workers use the same
  image; the web app stays usable; retries then clear errors; quick deal runs
  stay instant.
- **Done when:** 20 simultaneous simulations finish while health stays under
  200 ms; seeded results unchanged; e2e passes.

### 4.2 Result caching
- **Claude does:** cache by inputs, settings, seed, model version and data
  vintage; unseeded runs skip it; a version or data change clears it.
- **Done when:** a repeated seeded run returns in under 100 ms identically; a
  version bump misses (test).

### 4.3 Speed budgets (web and API)
- **Claude does:** page-load and interaction budgets (Lighthouse in CI);
  response-time budgets per endpoint (CI); fixes for anything over budget.
  Budgets are measured from a distant region too.
- **Done when:** CI fails over budget; all pages and endpoints pass.

### 4.4 Load testing and autoscaling
- **You first:** move the API and workers to scalable paid plans.
- **Claude does:** load tests (k6 or Locust) with users spread across regions;
  autoscaling rules; a capacity and cost note per 100 active users.
- **Done when:** staging holds 200 simultaneous users with 95% of deal runs under
  1 s; report in `docs/capacity.md`.

---

## Phase 5 — Global market data platform

### 5.1 Company filings from many countries
- **Why:** autofill today reads only US SEC filings under US GAAP.
- **Claude does:**
  - one "company data" interface with a connector per source: SEC EDGAR (US),
    UK Companies House, ESEF annual reports for EU and UK listed companies
    (filings.xbrl.org), and Japan EDINET;
  - a **document-upload fallback** (7.2) so any company anywhere can be
    loaded from its annual report PDF or Excel;
  - figures stored with currency, accounting standard, fiscal year-end,
    source link and filing date;
  - scheduled refresh within each source's usage rules;
  - search by name or identifier (LEI, ISIN, ticker, company number).
- **Done when:** a US, a UK, an EU and a Japanese company each load into the
  forecast with correct currency and standard (tests with recorded responses);
  every figure links to its filing.

### 5.2 Economic data by country, and exchange rates
- **Why:** economic data today is US only (FRED US series).
- **Claude does:**
  - connectors for IMF, World Bank, OECD, BIS policy rates, ECB Data Portal
    and FRED (including its international series);
  - per country: GDP growth, inflation, policy rate, government bond yields
    and, where available, credit spreads;
  - current reference rates (SOFR, SONIA, €STR and others) for 2.4;
  - ECB reference exchange rates for currency conversion;
  - stored with dates and sources; refreshed on schedule.
- **Done when:** at least 20 major economies (G20 plus others chosen in the
  task) have current data with sources; a SONIA-based tranche picks up the
  current rate by default (test); exchange rates update daily in staging.

### 5.3 Sourced defaults by region, sector and size
- **Why:** today's defaults (5% growth, 10x entry, 6.5% rate, 60% debt, fees
  of 2.3% and 2.6%) are numbers typed in during inception.
- **Claude does:**
  - calculate starting assumptions from broad data (thousands of companies,
    never a few deals) for each **region, sector and size band**: revenue
    growth, margins, capex, working capital, tax rate, EV/EBITDA multiples, and
    typical leverage where public data allows;
  - sources: the filings store (5.1), Damodaran's regional datasets and the
    economic data (5.2);
  - each default shows its source, sample size and date; users can always
    override;
  - where data is thin (small markets or niche sectors), fall back to wider
    groupings (country → region → global) and say so on screen;
  - the new deal screen asks for region, sector, size and currency first.
- **Done when:**
  - a new deal in, say, German industrials gets defaults with visible sources
    and sample sizes;
  - a thin-data case shows the fallback message;
  - no default in the app remains unsourced, apart from clearly labelled user
    preferences (CI check on the defaults registry).

### 5.4 Risk ranges, correlations and scenarios by region
- **Why:** Monte Carlo ranges, correlations and recession/stagflation presets
  are typed-in guesses, and scenario meanings differ by economy.
- **Claude does:**
  - uncertainty ranges (growth, margins, exit multiples, rates) per region,
    sector and size, from how those values actually varied year to year in the
    data;
  - correlations per region from the same history;
  - scenario presets per region, built from that region's own past recessions
    and inflation spells (dates listed);
  - each shows its source, period and sample.
- **Done when:** UK and India deals get different, sourced ranges and
  correlations; each scenario lists the historical periods it's based on; all
  correlation matrices pass the validity check (test).

### 5.5 Optional reference library (deals and base rates)
- **Why:** examples and base rates help, but the tool must never depend on them.
- **Claude does:**
  - a library of reference transactions with inclusion rules for balanced
    coverage (regions, sizes, sectors, eras, successes and failures), every
    figure sourced, and a review screen with two-person approval;
  - the 4 inception deals move here as examples;
  - published default and recovery base rates from rating-agency studies,
    stored by region, rating band and year with citations;
  - a coverage page showing where the library is thin;
  - an admin switch to hide the whole library, which must not break any
    screen.
- **Done when:** with the library switched off, every screen and model still
  works (e2e); base rates show citations; the coverage page reports counts by
  region, size and era.

### 5.6 Model validation framework
- **Why:** measure whether the tool's predictions are trustworthy, the right
  way, not by where one famous deal landed.
- **Claude does:**
  - validation reports that check whether real outcomes fall inside the
    predicted ranges as often as claimed (calibration), whether predictions are
    consistently too optimistic or pessimistic (bias), and how both split by
    region, sector, size and era;
  - uses the reference library plus **opt-in, anonymised** plan-vs-actual
    results from users (2.7), always tested on data newer than what was used
    for calibration;
  - reports are published in the app's methodology section;
  - deal results show IRR, MOIC, probability of loss, downside case, debt
    coverage and default risk together, never IRR alone.
- **Done when:** a validation report is generated from staging data with all
  splits; opt-in anonymisation is tested (no deal names or company identifiers
  leave the user's account); the deal summary shows the full metric set.

---

## Phase 6 — ML done properly

**Rule:** models learn from the global market data (5.1–5.4) and are tested on
data newer than they learned from, split by region. They must work without the
reference library, and appear only if they beat a simple baseline in the
region where they're used; otherwise that region shows "not enough data".

### 6.1 ML evaluation harness and model cards
- **Claude does:** one harness (time-based splits, per-region results,
  baseline comparison); model card template (purpose, data by region, accuracy
  by region, known failures, trained date); model registry; CI fails if a
  registered model gets worse.
- **Done when:** the surrogate and the current anomaly detector have cards with
  per-region results (showing where they don't apply).

### 6.2 Deal risk score from market data
- **Claude does:** rebuild `ml/anomaly_detector.py` so "unusual" means compared
  with companies and deals like this one in the same region, sector and size
  (from 5.3 and 5.4), plus the reference library when available. Show "based
  on N companies in [region, sector]"; switch the score off where N is too
  small.
- **Done when:** the score works with the library off; each region gets its own
  comparison group (test); thin regions show "not enough data".

### 6.3 Distress predictor
- **Claude does:** rebuild `ml/distress_model.py` from coverage and leverage
  paths in the simulation, calibrated to the published default base rates by
  region and rating band (5.5); year-by-year distress probability on Debt and
  in Monte Carlo; model card.
- **Done when:** implied default rates match the published base rates for
  comparable leverage bands within the card's tolerance; higher leverage
  raises risk (test).

### 6.4 Multiple predictor by region
- **Claude does:** rebuild `ml/multiple_predictor.py` on regional valuation
  data (listed company multiples by region, sector and size from 5.3); suggest
  entry and exit ranges with comparables and "use suggestion".
- **Done when:** ranges contain actual multiples for most held-out companies per
  region (card); it works with the library off.

### 6.5 Growth calibrator by region
- **Claude does:** rebuild `ml/growth_calibrator.py` on the filings store and
  regional data (replacing its US-only SimFin and fixed averages);
  "calibrate from sector and region" on Monte Carlo, with the source shown.
- **Done when:** ranges match observed growth on held-out years per region
  (card); using them changes the simulation.

### 6.6 Driver explanations
- **Claude does:** compare `ml/shap_attribution.py` with the current drivers
  view; ship per-deal explanations if they add value (adding their packages),
  otherwise remove it with a note.
- **Done when:** written comparison; if shipped, explanations add up to the
  prediction (test).

### 6.7 Economic regime by region, scheduled
- **You first:** get a free FRED key; set `FRED_API_KEY` (other sources in 5.2
  need no key).
- **Claude does:** rebuild `ml/macro_regime.py` on the country data from 5.2
  (replacing its US-only series, one of which was discontinued in 2022) for
  the US, eurozone, UK, Japan, China, India and a global view; monthly
  retraining; Scenarios shows the current regime for the deal's region;
  correlations refresh from the same data.
- **Done when:** each listed region shows a dated regime; tests use recorded
  data; a deal in India shows India's regime, not the US's.

### 6.8 Live sliders for any deal, currency and region
- **Claude does:** retrain the surrogate across entry multiples, holds, fees,
  debt types, rate levels and tax settings covering all regions' ranges
  (currency doesn't matter, since the model is currency-neutral), as a worker
  job; warn only outside the trained range.
- **Done when:** median-IRR error stays under 0.5pp on varied held-out deals
  from several regions (card).

### 6.9 Personalized defaults
- **Claude does:** move `ml/personalization.py` into the database; learn each
  user's usual assumptions per region and sector; suggest them on new deals
  alongside the sourced market defaults; off switch.
- **Done when:** suggestions use only that user's deals (test); switching off
  stops them.

---

## Phase 7 — AI features

### 7.1 File uploads
- **You first:** create a Cloudflare R2 bucket (choose its region); set keys.
- **Claude does:** PDF, Excel and Word uploads with checks; virus scanning;
  private storage; signed links; deletion rules.
- **Done when:** upload and download work; other users can't access (test);
  bad files are refused.

### 7.2 Upload a document (any language), get a deal
- **Why:** this makes the tool usable for companies anywhere, even without
  structured filings data.
- **You first:** create an Anthropic API key; set `ANTHROPIC_API_KEY`; choose a
  monthly AI spending cap.
- **Claude does:**
  - rebuild `ml/nlp_extractor.py` (current model, worker job);
  - read annual reports and information packs in major languages, and fill in
    financials and deal inputs with the currency, units, accounting standard
    and fiscal year detected;
  - cite page and quote for each number;
  - a review screen where the user accepts each value;
  - cost tracking and the cap;
  - an evaluation set spanning several countries and languages.
- **Done when:** extraction accuracy meets its card target for each language in
  the set; currency and units are detected correctly; nothing enters a deal
  without acceptance.

### 7.3 Investment memo writer
- **Claude does:** memo draft from a saved version (summary, full metric set,
  risks, drivers, scenarios, plan vs actual) exported to Word and PDF, in the
  user's language; every number comes from the model run; model version and
  data vintage on the cover.
- **Done when:** a test checks every figure against its run; exports open.

### 7.4 Plain-English explanations on screens
- **Claude does:** "explain" on key tiles using the on-screen numbers; glossary
  on hover that includes regional terms (e.g. SONIA, IFRS 16, unitranche);
  both in the user's language when available.
- **Done when:** explanations quote displayed numbers exactly (test); the
  glossary covers every label term.

---

## Phase 8 — Product features

### 8.1 Onboarding and in-app help
- **Claude does:** guided first deal (region, sector, size, currency, then
  sourced defaults); sample deals from several regions; empty-screen guides;
  input tips; help articles from the methodology; "what's new".
- **Done when:** a new test user completes the guided deal (e2e); every input
  has a tip.

### 8.2 Teams: sharing, permissions, comments
- **You first:** enable Clerk organizations; create Resend; set its key.
- **Claude does:** teams with owner, editor and viewer roles; share deals;
  comments and @mentions with email in the recipient's time zone and language;
  activity from the audit history.
- **Done when:** permission tests pass; mentions email correctly.

### 8.3 Excel: live-formula workbooks, then add-in
- **Claude does:**
  - **Part a:** exports with working formulas (all debt types, tax rules,
    currency formats);
  - **Part b:** an Excel add-in that signs in, opens a saved deal, runs the
    model on the server and writes results.
- **Done when:**
  - (a) the recalculated workbook matches the app's IRR to 0.01pp for a
    non-USD deal with a floating tranche;
  - (b) the add-in refreshes a saved deal.

### 8.4 Portfolio tracking in any currency
- **Claude does:**
  - mark deals as owned and enter or upload quarterly actuals (reusing plan vs
    actual from 2.7);
  - drift alerts;
  - a portfolio dashboard with totals converted to a chosen reporting
    currency using dated exchange rates (5.2), alongside local-currency
    figures;
  - reforecast from actuals.
- **Done when:** a portfolio with GBP and EUR deals totals correctly in USD at
  stated rates (test); alerts fire when actuals fall below plan.

### 8.5 Lender view with regional conventions
- **Claude does:** covenants per tranche (leverage, interest cover, fixed-charge
  cover, minimum liquidity) with definitions selectable by market convention;
  yearly covenant tests; breach odds from the simulation; distress probability
  (6.3); expected loss and recovery using regional base rates; lender export.
- **Done when:** a deal built to breach in year 2 is flagged in year 2 (test);
  breach odds come from the simulation.

### 8.6 Phones, tablets and accessibility
- **Claude does:** smaller-screen layouts; accessibility audit and fixes;
  automated accessibility checks in e2e; right-to-left check.
- **Done when:** key journeys pass at phone size; no serious accessibility
  issues.

### 8.7 Public API and webhooks
- **Claude does:** API keys per user; versioned v1 endpoints (deal runs,
  simulations, saved deals, market defaults by region); docs; webhooks for
  finished jobs; per-key limits.
- **Done when:** a script with a key runs a deal and receives a webhook (test);
  revoked keys stop immediately.

### 8.8 More languages
- **Claude does:** full translations for a first set of languages chosen when
  the task starts (e.g. Spanish, French, German, Japanese, Hindi, Arabic with
  right-to-left); translated help and glossary; a translation check in CI (no
  missing keys).
- **Done when:** e2e key journeys pass in each added language; CI fails on
  missing translations.

---

## Phase 9 — Subscriptions and usage

(The machinery only; which plans and prices to offer is a business decision
for later.)

### 9.1 Subscription billing (multi-currency)
- **You first:** create a Stripe account (test mode).
- **Claude does:** placeholder plans priced in several currencies; checkout,
  trial, upgrade, downgrade, cancel; local payment methods and automatic tax
  where Stripe supports them; billing portal; invoices; webhooks keeping plan
  status in the database; failed-payment handling.
- **Done when:** in test mode a user can trial, upgrade and cancel in EUR and USD
  (e2e); webhooks are verified and safe to replay.

### 9.2 Plan limits and usage metering
- **Claude does:** one configuration defining plan contents; enforcement in the
  API; monthly counters; upgrade prompts.
- **Done when:** a lower plan can't use a higher plan's feature via the API
  (test); counters reset monthly.

### 9.3 Usage analytics and feature flags
- **You first:** create a PostHog project (EU hosting if preferred); set its key.
- **Claude does:** key-action events and funnels; feature flags; no deal
  contents sent; consent banner respected per region.
- **Done when:** funnels show staging traffic; a test proves no deal inputs in
  events.

### 9.4 In-app support and feedback
- **Claude does:** "report a problem" with request ID (no deal contents without
  consent); feedback board with votes; admin view.
- **Done when:** a test report links to its Sentry request; votes recorded.

---

## Phase 10 — Enterprise-grade

### 10.1 Company logins (SSO) and admin console
- **You first:** enable Clerk enterprise SSO (paid).
- **Claude does:** company sign-in per team; joining by email domain; admin
  console (users, roles, seats, audit export); enforced login rules.
- **Done when:** a test identity provider signs into the right team; removing
  access is immediate (test).

### 10.2 Data export and account deletion
- **Claude does:** full export of a user's or team's deals, versions and files;
  real deletion of rows and files, with backups aged out; retention settings
  per team.
- **Done when:** a deleted account leaves nothing (test); exports are complete.

### 10.3 Automated security testing
- **Claude does:** OWASP ZAP scans of staging on each release; CodeQL in CI;
  container scanning; a permission test for every endpoint and role; fixes with
  regression tests.
- **Done when:** scans run automatically; the permission matrix covers every
  endpoint; no high findings open.

### 10.4 Regional hosting and data residency
- **Why:** global customers want low latency, and many need their data kept in
  their region (e.g. EU data in the EU).
- **You first:** create API, worker, database and storage instances in the
  added regions (start with the US and Frankfurt, then Singapore).
- **Claude does:**
  - each team chooses its data region at creation;
  - deals, files and backups stay in that region;
  - requests route to the team's region;
  - shared market data is replicated to all regions;
  - infrastructure described as code so a new region can be added by
    configuration;
  - a residency test.
- **Done when:** an EU team's deals and files exist only in Frankfurt storage
  (test); a new region can be brought up from configuration in staging.

### 10.5 Disaster recovery
- **Claude does:** a recovery plan (time to recover, acceptable data loss);
  standby replicas per region; a full rebuild rehearsal of one region from
  scratch.
- **Done when:** the rehearsal rebuilds a working region within the target time,
  and saved deals give identical results.

---

## Phase 11 — Running it day to day

### 11.1 Release routine and rollback
- **Claude does:** release checklist (staging checks, changelog, model version
  and data vintage, rollback plan); release notes from merged PRs; tagged
  versions.
- **Done when:** one release follows the checklist end to end.

### 11.2 Incident routine and runbooks
- **Claude does:** severity levels; a runbook per alert; status updates; post-
  incident review template; on-call hours noted across time zones.
- **Done when:** a simulated incident on staging is handled with the runbook.

### 11.3 Dependency, data-freshness and cost upkeep
- **Claude does:** a monthly routine: merge updates, rerun scans, check every
  data source refreshed on time (alerts for stale or broken connectors), review
  hosting and AI costs per region; a cost dashboard per service.
- **Done when:** the routine runs once; a deliberately broken data connector
  raises an alert; the cost dashboard covers every paid service.

---

## Appendix — US and inception-deal assumptions found in the code (2026-09-15)

Starting points for phase 2, 5 and 6 tasks. Tick them off as they're removed.

| Where | Assumption | Fixed by |
|---|---|---|
| 177 places across `core/`, `api/`, `lbo_engine/`, `ml/`, `web/src` | Money shown as "$M" | 2.2 |
| 11 places in `web/src` | Number formats fixed to `en-US` | 2.3 |
| `ml/edgar_extractor.py` | US SEC only, `us-gaap` tags, USD units, US fiscal years | 2.6, 5.1 |
| `lbo_engine/capital_structure.py` `build_simple_two_tranche_structure` | One fixed-rate senior loan (5% amortisation) plus one mezzanine bullet | 2.4 |
| `lbo_engine/operating_model.py` and returns | Flat tax, interest always fully deductible | 2.5 |
| `core/config.py` `DEFAULTS` | Growth, margins, multiples, rates, leverage, fees, ranges, correlations and scenario multipliers typed in with no source | 2.1, 5.3, 5.4 |
| `simulation/vectorized_simulation.py` `DEFAULT_CORR` | Correlation matrix typed in | 5.4 |
| `core/backtesting.py` `PRELOADED_DEALS` | Backtest limited to 4 US mega-deals (2006–2013), unsourced actuals, fixed ranges (growth ±4%, exit ±1.5x) | 2.7, 5.5 |
| `ml/anomaly_detector.py` | 30 US deals plus synthetic ones; claims "~100"; fixed warning statistics ("38% distress rate", "2 of 9") | 2.1, 2.8, 6.2 |
| `ml/distress_model.py` | 39 hand-entered cases | 6.3 |
| `ml/multiple_predictor.py` | 25 rows | 6.4 |
| `ml/growth_calibrator.py` | US SimFin, fixed Damodaran averages | 6.5 |
| `ml/macro_regime.py`, `ml/correlation_updater.py` | US-only FRED series (Fed funds, US GDP, ISM PMI discontinued 2022) | 5.2, 6.7 |
| `ml/surrogate/generate_data.py` `TRAINING_FIXED` | Trained on one fixed deal shape with US-style defaults | 6.8 |
| `web/src` Backtest screens | Only the preloaded deals can be tested | 2.7 |
| Settings "mc_hurdle" 20% | One hurdle for all markets | 5.3 |
