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
> mutation-check new tests, PR, CI green). Ship it yourself end to end
> (docs/WORKFLOW.md step 7): open the PR, auto-merge it once the checks are
> green, bring staging level with main, and verify staging and production.
> Only stop to ask me for a decision the task can't make alone or an
> outside-account step. Tick the task in PLAN.md's status table in the same
> PR. When it is live, tell me what changed and what to check, then give me
> the ready-to-paste prompt for the next task (this same prompt with the
> next task number), plus anything I must do first for it.

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
- **Every session ends with a handoff:** once the PR is merged (by Claude,
  through auto-merge), staging is level and production is checked, Claude tells the user to start a new session and gives the exact
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
| 0.2 | Your own domain | 0.1, 1.4 | ☑ |
| 0.3 | Development cycle gates (coverage, PR checks) | — | ☑ |
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
| 2.3a | Locale: numbers, dates, fiscal years, Excel formats | 2.2 | ☑ |
| 2.3b | Locale: interface text in translation files, right-to-left | 2.3a | ☑ |
| 2.4a | Debt structures and interest rates: the model | 2.2 | ☑ |
| 2.4b | Debt structures: simulation and the screen | 2.4a | ☑ |
| 2.5 | Global tax rules | 2.2 | ☑ |
| 2.6 | Accounting standards (IFRS and US GAAP) | 2.2 | ☑ |
| 2.7 | Backtest becomes "plan vs actual" for any deal | 1.5, 2.2 | ☑ |
| 2.8 | Risk warnings computed, not written in | 2.1 | ☑ |
| **3** | **Trust in the numbers** | | |
| 3.1 | Model version on every result | 1.5 | ☑ |
| 3.2 | Written methodology | — | ☑ |
| 3.3 | Audit history | 1.5 | ☑ |
| 3.4 | Hand-checked reference cases (incl. non-US) | 3.2, 2.4b, 2.5 | ☑ |
| **4** | **Global market data platform** | | |
| 4.1a | Company filings: the data layer (connectors, storage, search, refresh) | 1.9, 2.6 | ☑ |
| 4.1b | Company filings: the screen | 4.1a | ☑ |
| 4.2 | Economic data by country, and exchange rates | 1.9 | ☑ |
| 4.3 | Sourced defaults by region, sector and size | 4.1, 4.2 | ☑ |
| 4.4 | Risk ranges, correlations and scenarios by region | 4.2, 4.3 | ☑ |
| 4.5a | Reference library: admin switch, base rates, coverage page | 4.1 | ☑ |
| 4.5b | Reference library: sourced transactions, two-person review, fees and amortisation | 4.5a | ☑ |
| 4.6 | Model validation framework | 4.3, 4.5b, 2.7 | ☑ |
| **5** | **ML done properly** | | |
| 5.1 | ML evaluation harness and model cards | 4.6 | ☑ |
| 5.2 | Deal risk score from market data | 5.1, 2.8 | ☑ |
| 5.3 | Distress predictor | 5.1 | ☑ |
| 5.4 | Multiple predictor by region | 5.1, 4.3 | ☑ |
| 5.5 | Growth calibrator by region | 5.1, 4.3 | ☑ |
| 5.6 | Driver explanations | 5.1 | ☑ |
| 5.7 | Economic regime by region, scheduled | 5.1, 4.2 | ☑ |
| 5.8 | Live sliders for any deal, lightweight enough for free hosting | 5.1, 1.9 | ☑ |
| 5.9 | Personalized defaults | 1.5, 5.1 | ☑ |
| 5.10 | Retraining models by hand (parked, not scheduled) | 5.1 | ☐ |
| **6** | **AI features** | | |
| 6.1 | File uploads | 1.4, 1.9 | ☑ |
| 6.2 | Upload a document (any language), get a deal | 6.1, 3.1, 2.6 | ☐ |
| 6.3 | Investment memo writer | 3.1, 1.5 | ☐ |
| 6.4 | Plain-English explanations on screens | 3.1 | ☐ |
| **7** | **Product features** | | |
| 7.1 | Onboarding and in-app help | 1.5 | ☐ |
| 7.2 | Teams: sharing, permissions, comments | 1.5, 3.3 | ☐ |
| 7.3 | Excel: live-formula workbooks, then add-in | 1.5, 2.3a | ☐ |
| 7.4 | Portfolio tracking in any currency | 1.5, 1.9, 4.2, 2.7 | ☐ |
| 7.5 | Lender view with regional conventions | 2.4b, 5.3 | ☐ |
| 7.6 | Phones, tablets and accessibility | — | ☐ |
| 7.7 | Public API and webhooks | 1.4, 1.6, 3.1 | ☐ |
| 7.8 | More languages | 2.3b | ☐ |
| 7.9 | Charts in every Excel download | 2.3a | ☐ |
| 7.10 | Workspaces: LBO and equity research, launcher after sign-in | 1.4 | ☑ |
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
| 12.9 | Licensed market data (corporate credit spreads) | 4.2 | ☐ |
| **R** | **Equity research workspace (the priority since 2026-10-12)** | | |
| R0 | Proof before code: reader accuracy | 6.1 | ☑ |
| R1a | Results reader, built and measured on R0's filings | R0 | ☑ |
| R1b | Results reader measured on held-out filings (recorded over days) | R1a | ☐ |
| R2 | Coverage, review screen and the checked workbook | R1a | ☑ |
| R3 | Excel add-in: sign in and import results | R2 | ☑ |
| R4 | Sector research, the model definition and the first sector model | R2 | ☐ |
| R5 | Save back and roll forward | R3, R4 | ☐ |
| R6 | Valuation | R4 | ☐ |
| R7 | Second and third sector models | R4 | ☐ |
| R8 | Report draft | R6 | ☐ |
| R9 | Ready for the paid switch | R5 | ☐ |

**Priority since 2026-10-12 (the user's decision): phase R comes before
everything still open.** 6.1 ships first (the reader stands on it), then R0
to R9 in number order. 6.2 to 6.4 and phases 7 to 12 wait, except the parts
a phase R task pulls forward (each names them) and upkeep that can't wait:
security fixes, Dependabot, a failing scheduled job or check. The order
below is the order for the rest, once phase R is done or the user says
otherwise.

**Order:** 0.1 → 2.1 (small, do early) → 1 → 2 → 3 → 4 → 5 → 6 and 7 (any
order) → 8 → 9 → 10 → 11 → 12. Task 3.2 can start any time; 11.1 and 11.2
any time after phase 1. Added 2026-09-24: **0.3 next** (before 2.3), then
back to phase 2; **0.2 as soon as the domain is bought** (it jumps the queue
then); **11.4 any time**, refreshed whenever a later task changes how the app
is run.

---

## Phase R — Equity research workspace (the priority since 2026-10-12)

**The product.** A ready-made workflow for an Indian equity research analyst.
It takes the mechanical work out of starting and keeping coverage of a
company, so the analyst supplies only judgement: the company-specific
changes, the forward assumptions and the Buy, Hold or Sell call. **The tool
never picks the rating.**

1. The analyst brings in a company's quarterly results, often a scanned PDF.
   The tool fills in the income statement, balance sheet, cash flow and
   segment tables, every figure traceable to its place on the page, checked
   (subtotals add up, the balance sheet balances), doubtful figures amber.
2. The analyst picks a **sector model**: a pre-built, formula-wired, vetted
   model shaped like that business, so the plumbing never has to be checked.
   This is what sets the product apart.
3. The analyst makes last changes for the company and types assumptions;
   projections update at once.
4. A discounted cash flow and a target multiple give a value per share
   against a share price the analyst types in.
5. The tool drafts the results summary, tables and charts, every number from
   the model; the analyst writes the argument and picks the rating.
6. Next quarter: new results in, model and assumptions kept, actuals shown
   against the estimates, the model rolled forward. **This is why the analyst
   comes back.**

It lives in the existing **Equity research** workspace (7.10) and in an
**Excel add-in** (a task pane inside Excel on Windows, Mac and the web).

**What is already in place and what is new** (checked 2026-10-12):

| Part | State |
|---|---|
| App hosts: Vercel (web), Render (API) | In place |
| Database: Neon, free and paid on the same platform | In place |
| File storage: Supabase Storage, private bucket, scanned uploads | In place once 6.1 merges |
| Accounts, saved versions, limits, background jobs, audit history, Excel downloads, plan vs actual, lakh digit grouping, fiscal year labels | In place |
| An AI key for reading documents | **In place (R1a, 2026-10-14)**: `AI_PROVIDER=gemini`, a free key of Google project `variater-reader`; billing with a cap before outside users |
| Money in lakhs and crores | **In place (R2, 2026-10-15)**: thousands, lakhs, millions, crores, billions |
| Quarter by quarter (the product follows Indian results, which are quarterly) | **To build.** Today's forecast screen works in whole years |
| Indian companies' results | **To build, by upload** (decided 2026-10-12: uploading the PDF is enough for a start; fetching from BSE or NSE comes later). 4.1's registers hold no Indian company |
| Models as live Excel formulas | **New.** 7.3a planned it for deals; R4 builds it for research |
| The Excel add-in | **New.** 7.3b planned one for deals; R3 and R5 replace it |

**Rules for this phase** (beside the guiding principles):

1. **A tool, not an adviser.** It never states a rating, a target it chose or
   a recommendation; reports carry the analyst's firm and SEBI disclosures as
   the analyst's own. A securities lawyer's view before the first outside
   user is a "You first" in R8.
2. **The reader ships only at a measured accuracy** (principle 5): a model
   card on real results, and the gate that fails a PR that makes it worse.
3. **One model definition, two renderings.** A sector model is written once
   as data (rows and formulas in plain arithmetic); the server evaluates it
   and the workbook holds it as Excel formulas. A test proves both give the
   same figures, so the app, the download and the add-in can never disagree.
4. **"Vetted" means solved by hand.** Each sector model has reference
   companies worked by hand (as 3.4) and is read by a practising analyst
   before it is offered. A model that is mostly right is worse than a blank
   sheet.
5. **New code only.** `research/` and its screens; `core/`, `lbo_engine/`
   and `simulation/` are not touched, and the golden snapshot does not move.
6. **Private, and public documents only on the free AI.** Coverage belongs
   to its owner like a deal (another account's is a 404). Listed companies'
   results are public; anything else waits for 12.4.
7. **India first, not India only.** Currency, unit, accounting standard and
   fiscal year are read from the document, never assumed.
8. **Free now, paid by a switch.** Each paid step is a setting: the AI
   provider (`AI_PROVIDER`, 12.4), always-on API (12.1; the free one takes a
   minute to wake, which an add-in user will feel), database (12.2), storage.

**What Indian results look like, to be confirmed in R0:** the income
statement and segment table come every quarter, the balance sheet and cash
flow only at the half year and the year end; standalone and consolidated
figures sit side by side; amounts are in lakhs or crores; most years end in
March; the standard is Ind AS. So the quarter-end balance check only runs
twice a year, and the reader must never mix standalone with consolidated.

**Left out at first:** banks, insurers and lenders (their statements are
shaped differently); consensus estimates and live prices (licensed data);
reading the history of drivers such as beds or stores from presentations
(the analyst types it); help forming the assumptions themselves.

**Decisions that are the user's** (asked 2026-10-12; a task that needs one
stops for it):
- which sectors, and how many models: **settled 2026-10-12 in R4 part a**
  (eleven models, the table under R4; hospitals and hotels first, then
  cement, steel and chemicals, then shops and restaurants);
- who the five test analysts are (asked for after R2, see "Analysts come
  after the product" below), and which practising analyst reads each sector
  model before it is called vetted (R4);
- a paid AI key with a spending cap, if no free one reads accurately enough
  (R0 decides; it would bring 12.4 forward): **settled 2026-10-14**: R1 is
  built on a free Gemini key (project `variater-reader`, no billing yet);
  billing with a spending cap is switched on **before the first outside
  showing after R2** (then `AI_FREE_TIER=0`, DEPLOY.md "Results reader").

**Analysts come after the product (the user's decision, 2026-10-12).** No
analyst can be approached with a sample page: they need something finished
or nearly finished to try. So no task waits for an analyst:
- R0 is about the reader's accuracy only. Its hand-made sample is kept as
  the picture of what R2 to R8 build towards, not shown to anyone yet.
- **First showing: after R2** (upload a results PDF, review the amber
  figures, download the checked workbook), on public filings only and on a
  paid key with a cap. **Second: after R4** (the first sector model).
  Each time the user asks for twenty minutes: what is wrong, what is
  missing, would you use it. The answers are written under that task and
  may reorder R3 to R9; work goes on meanwhile.
- R4's sector model is marked **not yet vetted**, on screen and in the
  workbook, until a practising analyst has read it; R4 does not wait.
- The cost of this order: R1 and R2 are built before anyone outside has
  said they want it. R2 is kept small for that reason, and what the
  analysts say after it decides whether R3 onwards goes ahead as written.

### R0 Proof before code
- **Needs:** 6.1 merged.
- **You first:** collect 30 real quarterly results PDFs
  from the exchanges (at least 10 scanned, at least 5 half-year or year-end,
  several sectors); create a free Gemini API key.
- **Claude does:** a script, outside the app, that reads each PDF twice with
  the candidate AI models and scores every figure against a truth file (the
  exchange's own machine-readable filing where one is free to use, else
  figures the user keys for ten of them); the count of wrong figures, of
  wrong figures the arithmetic checks did not catch, and of amber flags; a
  one-page sample of what the tool will produce for one company (checked
  statements, a model, a draft note), made by hand.
- **Done when:** the accuracy figures are written in this file, with every
  file read by two full models and ten documents scored against a truth
  file; the bar is **no wrong figure left unflagged across the 30, and
  under 2% of figures amber**; a go, a change of AI provider, or a stop is
  recorded here. (Showing analysts was part of this task until 2026-10-12;
  it now follows R2.)
- **Progress (2026-10-12), not yet done:**
  - **The 30 documents are collected**: `research-samples/` on the Mac
    (kept out of git: the repository is public), named
    `company_FYyy-Qn[-scanned|-mixed].pdf`. 20 June-2026 quarters, 5 March
    year-ends, 5 September half-years; 5 companies with two quarters; 14
    sectors, no banks (one bank's file is in `left-out/`). Screener's Excel
    exports for nine of the companies sit beside them as a second source.
  - **Nearly every filing is a scan**: 20 fully scanned, 8 partly, 2 clean
    text, large companies included (results are printed, signed and scanned;
    many carry a rough text layer that reads as garbage). Reading pictures is
    the normal case, not the exception.
  - **Sizes**: a year-end scan reached 19 MB and one filing is 193 pages and
    23 MB (it bundles two sets of statements), against 6.1's 20 MB limit; one
    is 89 pages. R1 needs a rule for these, and R2 decides the limit.
  - **The free Gemini key is made** (in `research-samples/.env`, never in
    chat or git). **Free tier: 20 requests a day per model**, flash models
    only (the pro models answer "quota exceeded"; the newest flash was often
    "high demand"). Decided by the user: **stay free for now and spread the
    test over models and days**; a paid key with a cap comes before outside
    analysts use the reader (the free tier is about ten documents a day for
    everyone).
  - **The test scripts** are `research-samples/r0/`: `gemini.py` (the calls,
    and the upload for a file too big to send inline), `statements.py` (one
    request reads every consolidated statement of a filing: results,
    segments and, at the half year and year end, balance sheet and cash
    flow), `read_test.py` (the first pass, results table only), `score.py`
    (the checks, no call), `report.py` (the accuracy table, no call),
    `crop.py` (cuts a strip from a page to check a figure by eye). Every
    answer is saved in `r0/out/`, so nothing is asked twice;
    `out/adjudicated.json` holds what was checked by eye. Read A sends the
    whole PDF, read B pictures (170 dpi) of the pages A found the tables on,
    to a different model.
  - **Results (second session, 2026-10-12): all 30 files read twice, 63
    tables, 7,998 figures** (results 30 tables, segments 13, balance sheet
    10, cash flow 10; the 193-page, 23 MB file went through the upload and
    read without trouble).

    | Readers (A whole PDF + B page pictures) | Files | Figures | Amber | Of which a row one read left out |
    |---|---|---|---|---|
    | two reads by full models (3.6 to 3.8 flash, 3 flash preview) | 18 | 5,298 | 55 (**1.04%**) | 32 |
    | full model + a lite model as B | 6 | 1,753 | 214 (12.2%) | 202 |
    | two lite models (3.5 and 3.1 flash lite) | 6 | 947 | 131 (13.8%) | 123 |

    - **Wrong figures found, all of them flagged**: checked by eye against
      the page, read A (the whole PDF) was wrong in 18 figures in 6 files
      (about 1 in 400), read B in 5 figures in 4 files; 2 small ones were not
      checked. Every one was amber because the reads differed; in 5 of
      the 6 files a sum the table states also failed to add up.
    - **Wrong figures left unflagged: none found.** What that rests on: 483
      figures on pages with real text all match the PDF's own text; 101 of
      111 comparisons with Screener match and the other 10 differ by a
      little in every column for three companies, the mark of another
      basis; 10 figures the text layer or Screener
      disputed were looked at and were right; 89% of figures sit inside a sum
      that holds; and a wrong digit planted in 1,780 figures was flagged
      every time (a sum alone caught 78%). **What it does not rest on: a
      full truth file.** Two readers making the same mistake in a figure
      outside every sum (about 1 in 10 figures: per-share lines, equity
      lines) would not show. Closing that needs the exchange's own XBRL
      filing where it is free to use, or ten files keyed by the user.
    - **The period in a column heading was misread twice** (2024 for 2026 in
      every column, once by each kind of read, on files whose figures were
      otherwise right). The figures check cannot see this; R1 must check
      periods against the filing date and each other.
    - **Cost on a paid key, both reads** (from the saved token counts, list
      prices of 2026-10-12: 3.8 flash 0.75 and 3.75 US dollars a million
      tokens in and out, 3 flash preview 0.50 and 3.00): a quarterly filing
      about **0.09 US dollars**, a half-year or year-end filing about
      **0.12**, the dearest 0.19; under 20 rupees. Over half of what
      is paid for is the model's thinking. Google lists 3.8 flash at double
      from 2027-01-01. The lite pair cost 0.02 and is not usable.
  - **Recommendation: go, with Gemini, on these conditions** (conditions 2
    and 3 were met on 2026-10-14; the final table is under "Result" below):
    1. two reads by **two different full models**, never one read and never
       a lite model (they leave out rows and whole statements);
    2. the amber rate that counts is the two-full-model one, 1.04% against
       the 2% bar; the 12 files a lite model read are to be read again by
       full models before the table is final (18 requests, one free day);
    3. a truth file for ten documents before R1's card is called met;
    4. a paid key with a spending cap before anyone outside uses it: the
       free tier is about ten documents a day for everyone, and Google may
       use free-tier content to improve its products (fine for public
       filings, not for anything else).
  - **Lessons for R1**: (1) **a wrong read can still add up**: twice a
    model took one digit for another **and moved the total to match** (once
    the picture read, once the whole-PDF read), so only the second read
    caught it; two reads stay, and they must differ in what they are given
    and in the model; (2) **the whole PDF did not read better than
    pictures**: it made 18 of the 23 errors, most on dense pages (standalone
    and consolidated side by side, a faint scan); the picture's detail
    matters, so try the API's high-resolution setting on both reads; (3)
    **checks cannot assume a layout**: `score.py` finds the sums from the
    numbers (a row that is the sum of the rows above, one row less the rows
    between, a heading that carries the sum of the rows under it), chooses a
    formula once for all columns so it cannot bend to one wrong cell,
    ignores a "sum" that is only a copy of a line, and believes a formula
    that fails somewhere only when it is a real sum in most columns; (4) a
    dash read as 0 against a blank is not a disagreement, and **a "less" row
    comes back positive from one read and negative from the other**: line
    rows up by size, and let a sum that holds settle the sign; (5) rows are
    lined up by their numbers, not their labels (one read keeps a heading
    the other drops); (6) Screener is not a truth file for net profit or
    revenue (other basis), and a scan's text layer is not one either; (7)
    one request can read all four statements; (8) an answer can come back
    wrapped in a list.
  - **The sample is drafted** (kept for the first showing after R2):
    `research-samples/r0/sample/variater-sample.pdf` (built by
    `build_sample.py` from the saved read of Avenue Supermarts' September
    2026 half-year filing): checked statements with the one amber figure,
    the checks run, a next-quarter model as formulas, valuation arithmetic
    and the factual part of a note, with the rating left blank.
  - **Third session (2026-10-12, an hour after the second): no request was
    possible.** All four full flash models refused for the day (their
    allowance is a rolling 24 hours from the morning's reads, back after
    about 06:00 IST on 2026-10-13), so the 12 re-reads and the
    high-resolution trial are not done and the table above stands. Done
    without a request:
    - **Column periods are checked** (`score.py` `periods`, in `report.py`'s
      output): every heading against the quarter the filing is for (the
      quarter, the one before, the same one a year ago; the year so far and a
      year ago; the last full year, and the year before it only beside it;
      balance sheets likewise), month ends only, no period twice, no column
      after the filing's quarter, the segment table's periods the results
      table's. On the 60 saved reads it flags **exactly the two known
      misreads** (2024 for 2026) and nothing else; of 1,368 wrong years
      planted one at a time in the 58 clean reads it noticed **all 1,368**
      (`plant_periods.py`). It takes a March year end and the quarter from
      the file name; R1 reads both from the filing.
    - **A full model's read is preferred to a lite one's** (`report.py`, and
      `statements.py` now reads a file again when only a lite read is saved);
      `reread_b.py` reads B again for the two first-pass files;
      `statements.py --high` sends the API's high-resolution setting and
      saves beside the ordinary reads (`.HA.`, `.HB.`).
    - **`finish.sh` runs everything still owed in one go** (27 requests at
      most, inside one free day, safe to run again).
    - **Truth file: the exchanges' XBRL is not free for a program to
      fetch.** NSE's terms allow personal, non-commercial use and forbid
      "systematic or automated data collection" without written consent;
      BSE's forbid building a database by systematically downloading pages
      and commercial use of the site's contents. (Read from search results
      and an NSE-run copy of the terms: both sites refused a direct fetch.)
      So nothing downloads them here and **the product must not fetch them
      either** without the exchange's consent, which is what phase R already
      assumes (upload only). What is left is a person saving ten files by
      hand from the public results page for a private test, which is the
      user's call; the other way is keying ten files. **`xbrl_score.py` is
      ready for the first**: it matches the XBRL to a read by period and
      value (no taxonomy to map), counts the exchange's numbers the read
      does not show and how many of those nothing flagged; tested on a
      made-up file with one figure changed (found, and counted as
      unflagged). The XBRL usually holds only the current quarter and year
      so far, so it scores those columns.
    - **The ten documents** (for XBRL or for keying; half-years, year-ends,
      every kind of mistake seen): lemon-tree-hotels FY26-Q2,
      jk-lakshmi-cement FY26-Q2, avenue-supermarts FY27-Q2, tata-steel
      FY26-Q4, eternal FY26-Q4, hindustan-aeronautics FY27-Q1,
      kirloskar-pneumatic FY27-Q1, reliance-industries FY27-Q1,
      devyani-international FY27-Q1, indoco-remedies FY27-Q1.
    - **Truth file (same day): ten documents scored against BSE's own
      figures, no wrong figure left unflagged.** The user saved,
      by hand, the table BSE's results page shows for each filing (BSE's
      "XBRL" link gives a small HTML table named `.xls`, not the XBRL
      itself: the filing's quarter only, 14 to 26 amounts in millions of
      rupees, earnings per share and other comprehensive income among them)
      for nine of the ten; Kirloskar's gave a site error, so V-Mart Retail
      FY26-Q2 stands in (BSE gave it the other shape it has: six lines over
      several periods, of which the filing prints four). `bse_score.py`
      looks for each amount, by size, in the read's column for its period:

      | | Figures |
      |---|---|
      | BSE amounts, ten documents | 232 |
      | in read A as printed | 206 (188 not amber, 18 amber and right) |
      | BSE's own grouping of lines the read holds (other expenses plus impairment, profit before associates less tax, ...), which add up exactly from read A | 25 |
      | not reproduced by read A | **1, and it is amber** |

      The one is JK Lakshmi's September 2025 quarter: BSE's other expenses
      are power and fuel plus transport plus other expenses, which add up
      from read B and not from read A, the three cells already found wrong
      in A by eye and flagged. BSE also confirms read A against read B on
      Hindustan Aeronautics' revenue (the 5 read for a 4). **What this
      does and does not cover**: about 23 figures a document, the current
      quarter of the results table (V-Mart's: three comparative columns
      too); not the other comparative columns, the
      segment table, the balance sheet or the cash flow. It is the
      exchange's check on the headline figures, per-share lines included,
      not a full truth file.
  - **Decided by the user (2026-10-12): BSE's tables for the ten documents
    are truth file enough**; no filing is keyed in full.
- **Result (fourth session, 2026-10-14): done. Go, with Gemini.**
  `finish.sh` ran with the allowance back (the last request had been
  2026-10-12 06:52 IST). The full models were "high demand" several times
  (a 503 spends nothing, so a second and a third pass picked up what was
  missed) until 3.8, 3.7 and 3.6 flash each used their day. 11 of the 12
  files a lite model had read were read again by two full models; the
  final table:

  | | Final, 2026-10-14 |
  |---|---|
  | Files read twice by **two different full models** | **29 of 30** (Yatharth Hospital FY27-Q1: read B full, read A still lite, see below) |
  | Tables, figures | 62 tables, **7,899 figures** (results 4,374, segments 1,766, cash flow 884, balance sheet 875; Yatharth's 99 apart) |
  | **Amber** | **76, 0.96%** (bar: under 2%): a figure the reads give differently 31, a row one read left out 35, a sum that fails 9, a sign 1 |
  | Inside a sum that holds | 6,999, 89% |
  | **Wrong figures found** (every disagreement looked at on the page) | **30**: read A (the whole PDF) 24 in 8 files, read B (page pictures) 6 in 5 files; 2 small ones not settled (0.02 or nil; 26.85 or 26.89) |
  | **Wrong figures left unflagged** | **none found** (below: what that rests on) |
  | Column periods misread | 4 reads (2024 for 2026 in every heading: Hindustan Aeronautics B, Kirloskar A, Bharat Electronics B, Suprajit A), **all four caught by the period check**, nothing else flagged |
  | BSE's own figures, ten documents | 232 amounts: 206 printed in read A as one line (192 not amber, 14 amber and right); 25 are BSE's grouping of lines and come out of read A's lines exactly (`bse_groups.py` finds 24 among two to four lines; Hindustan Aeronautics' other expenses, by hand, is five: other expenses + impairment + direct input + provisions - capitalised); **1 does not, and it is amber** (JK Lakshmi's other expenses, from read B; read A's three cells are the known wrong ones) |
  | Cost on a paid key, both reads | mean **0.10 US dollars** a filing, dearest 0.19 (list prices of 2026-10-12) |

  - **What "none unflagged" rests on**: every one of the 30 wrong figures
    was amber because the reads differed; BSE's figures for ten documents
    (the current quarter, per-share lines included) found no wrong figure
    the reads agreed on; of 26 figures the PDF's text layer or Screener
    disputed, 16 were looked at on the page and are right (figures under a
    company stamp, a scan's rough text such as "(23 I)" for (231)) and 10
    are Screener's other basis for HDFC AMC and Reliance (a little off in
    every column); a wrong digit planted in 1,780 figures was flagged every
    time.
    What it does not cover: two reads making the same mistake outside every
    sum, in the columns and statements BSE's table does not show.
  - **The new reads made the same kinds of mistake**: Star Cement's read A
    (3.8 flash, whole PDF, standalone and consolidated side by side) got
    three figures of one column wrong, read B a fourth; Suprajit's read A
    two, and its headings 2024 for 2026. So a newer model is not a reason
    to drop the second read, and **the 2024-for-2026 heading is a habit**
    (four reads of four different files, both kinds of read): R1 checks
    periods on every read.
  - **High resolution** (`--high`, read A by 3.7 flash, read B by 3 flash
    preview), on two of the three files (Lemon Tree's read A was refused):
    JK Lakshmi went from 8 amber to 0 with both reads right (the ordinary
    read A had six wrong figures); Tata Steel's three balance-sheet ambers
    went (read A's two wrong figures now right), but read A made one new
    mistake in the segment table (caught by a sum) and left out a few
    cash-flow figures. Read A's model differs from the ordinary pair's
    (3.7, not 3.8), so this is a hint, not a measurement: **R1's evaluation
    tries high resolution on both reads** and keeps it if its card is
    better.
  - **Left over**, because the allowance ran out (each full model refused
    for the day by 20:10 IST on 2026-10-14; back after about 19:05 IST on
    2026-10-15): Yatharth's read A by a full model (its 99 figures: the
    lite read A and the full read B agree on all 99, none amber, 90 match
    the page's text) and Lemon Tree's high-resolution read A. Neither can
    change the decision; R1's evaluation reads all 30 again on its own key
    and model card, so they are not chased here. `finish.sh` still runs
    both if wanted.
  - **The decision: go, with Gemini**, on conditions 1 and 4 above (two
    reads by two different full models; a paid key with a cap before
    anyone outside uses it), plus what R1 must build from what was seen:
    the period check, a sum check that chooses a formula once for all
    columns, rows lined up by numbers not labels, and every disagreement
    amber. No change of provider.

### R1 Results reader
- **Pulls forward:** 6.2's provider switch and review rule. 6.2's "get a
  deal" from an information pack stays where it was.
- **You first:** set `AI_PROVIDER` and its key on both Render services and
  as a GitHub secret (for recording answers).
- **Claude does:**
  - `research/reader/` behind a provider switch; the PDF goes to the provider
    from the API (never from the browser), as background job
    `research.read`, only for a file 6.1 has scanned clean;
  - first the statement pages are found, then each table read **twice**;
    figures that differ, fail a check or sit in a poor scan are amber;
  - every figure carries its page and position, its period column, and
    whether it is standalone or consolidated; the unit line is read (lakhs,
    crores) and the currency;
  - the company's labels matched to standard lines from a fixed list, the
    AI only for the rest, the company's own wording kept;
  - checks: subtotals, balance sheet, cash flow against the change in cash,
    this document's comparative columns against what is already stored;
  - a per-user daily page limit inside the free provider's allowance;
    nothing from a document in any log;
  - CI and development replay recorded answers; the evaluation set is R0's
    30 documents; a model card and the gate (5.1's harness).
- **Done when:** the card meets R0's bar on documents the prompts were not
  tuned on; a wrong digit planted in a recorded answer is flagged (test);
  CI makes no AI call; the confidentiality confirmation blocks a file the
  user does not confirm is public (e2e).
- **Decided by the user (2026-10-14):** R1 is built on a **free Gemini
  key** (Google project `variater-reader`, no billing yet); **billing with a
  spending cap comes before the first outside showing after R2**; the
  recording of answers is **spread over days within the free allowance**
  (20 requests a day per model, two a filing). The user set `AI_PROVIDER=gemini`
  and `GEMINI_API_KEY` on both Render services and `GEMINI_API_KEY` as a
  GitHub secret on 2026-10-14 (proved through staging: `/api/health/reader`
  and the staging drill list Google's models with the key, which spends no
  allowance).
- **Split (2026-10-14)**, because the card's headline needs filings nobody
  tuned the prompt on, and none existed: **R1a** builds the reader and
  measures it on R0's 30 filings (in-sample); **R1b** reads new filings
  (held-out) over days and makes the card's headline. Production keeps the
  reader closed until R1b's card meets the bar; staging reads with a "not yet
  measured" notice. R2 needs only R1a, so it goes on meanwhile; R1b must be
  done before the first outside showing.
- **R1a, done (2026-10-14):**
  - `research/reader/`: `providers.py` (the switch: `gemini`, `replay` of
    recorded answers, off; two different full models, a spent model skipped,
    "high demand" moves on, the key only in a header), `prompt.py` (R0's
    prompt word for word as `r0`; `r1` adds each row's page), `pages.py`
    (page count, each page's scan resolution, pictures for read B, all in a
    child process with limits; `pypdfium2`), `checks.py` (R0's sums,
    agreement, periods, ported; plus balance sheet, cash flow against the
    change in cash, comparatives against what is stored), `lines.py` (the
    company's labels to standard lines: a fixed list first, the AI's line
    only for the rest, its own wording kept), `assemble.py` (two reads to
    checked statements: every figure with its page, row and column, period,
    scope, line and amber reason), `read.py`, `service.py` (who may read
    what; the job), `gate.py` (production reads only once the card meets the
    bar), `record.py` (the recorder: R0's saved reads, or new reads saved as
    they arrive).
  - Job `research.read` (no simulation slot: mostly a wait on the provider),
    `POST /api/research/reads`, `GET /api/research/reads/{id}`,
    `GET /api/research/reader`, `GET /api/health/reader`. Refusals are codes.
    The free tier needs the person's confirmation that the file is public
    (`AI_FREE_TIER`); only a PDF of the caller's that 6.1 scanned clean.
  - Limits: 30 reads a day for the service, 10 per person, **250 pages a
    day per person**, charged once the job counts the pages; 300 pages a
    filing at most.
  - **Poor scans**: a page whose picture is under 100 dpi (Tata Steel's
    year-end statements are 95) is a poor scan; a figure on it is amber
    unless a sum that holds confirms it (36 figures of R0's 30 filings).
  - Equity research -> **Coverage -> Results** (`components/coverage/`):
    upload or choose a PDF, the quarter (optional), the public
    confirmation, the read's stages, then the statements with every amber
    figure and its reason, and the headings and units to check. R2 adds the
    review.
  - **Card** `ml/cards/reader.json` (`ml/evaluation/reader.py`; the bar is
    the card's: R0's): **development** set, R0's 30 filings as recorded
    answers (`tests/fixtures/research/reader/development`, R0's prompt):
    7,998 figures, **1.40% amber**, **0 of 360 planted wrong digits missed**
    (read A alone with its sums misses 75%), **0 of 232 truth-file figures
    (BSE's, ten filings) missed**, 97.8% of them confirmed without amber.
    In-sample, so it does not count for the bar; the **held_out** set
    (headline) is empty: "not enough data".
  - Tests: `tests/test_research_reader.py` (a planted wrong digit in every
    recorded filing is flagged; a misread that still adds up is caught by
    the second read; no socket opened on replay), 
    `tests/test_research_reader_api.py` (another account's file and read
    are 404s; the confirmation; the limits; nothing from a document in the
    log, mutation-checked), `web/e2e/reader.spec.ts` (the confirmation
    blocks an unconfirmed file on screen and at the API; a read shows its
    amber figure).
- **R1b, next in parallel with R2** (needs the user first):
  - **You first:** save about ten **new** results PDFs by hand into
    `research-samples/held-out/` (September 2026 quarter filings, published
    from mid-October; none of R0's companies' same filing; several sectors;
    some scanned; a half-year with a balance sheet and cash flow), named
    `company_FY27-Q2.pdf`; and, where it works, BSE's results-page table for
    each into `research-samples/xbrl/` as before (the truth file).
  - **Claude does:** `python -m research.reader.record read --set held_out
    --env-file research-samples/.env research-samples/held-out/*.pdf` once a
    day until every filing is read (20 requests a day per model; a spent
    day stops the run and the same command carries on), the truth file
    joined, `python -m ml.evaluation evaluate`; the high-resolution trial
    (`READER_HIGH_RESOLUTION=1`) on the same filings as a second set,
    kept if its card is better; then R1 is ticked.
  - **Done when:** the card's `held_out` set has at least five filings and
    meets R0's bar; production then reads (`gate.py`).

### R2 Coverage, review screen and the checked workbook
- **Claude does:**
  - migration: covered companies and their periods (owner, quarter or year,
    standalone or consolidated, figures, citations, the source file), sized
    for the free database; money units **lakhs and crores** beside the
    existing three;
  - a **Coverage** mode in the Equity research workspace: add a company,
    upload results, watch the read, then a review screen with the page
    beside the figures (drawn in the browser from the stored PDF), amber
    figures to accept or correct; **nothing is saved until accepted**;
  - a download: the checked statements as a workbook whose subtotals are
    formulas, with each figure's page in a note;
  - the source page stays viewable while the file is kept (6.1's 90 days;
    a longer life for results files is decided here against the storage
    budget); the figures themselves stay.
- **Done when:** a scanned results PDF becomes an accepted quarter and a
  workbook that recalculates to the same totals (e2e on recorded answers);
  another account cannot read it (test); a corrected figure is the one
  stored and downloaded. **This is the first version to put in front of the
  five analysts**: the user shows it, and what they say is written here
  before R3 onwards is confirmed (R4 part a can go on meanwhile).
- **Done (2026-10-15):**
  - **Units**: `lakhs` and `crores` join thousands, millions and billions
    everywhere money has a unit (`core/money.py` `UNIT_SCALE`, the API's
    `MoneyUnit`, the web's `MONEY_UNITS`: "₹L", "₹Cr"). A deal in lakhs or
    crores runs in millions as every unit does and answers exactly the
    millions answer scaled (`tests/test_money.py`'s leaf-by-leaf tests now
    cover all four other units). No engine number moved.
  - **Storage** (migration 0019, `db/coverage.py`): `coverage_companies`
    (owner, name) and `coverage_periods` (company, period end, months 3 or
    12 (6 and 9 allowed), scope, unit in `units`/thousands/lakhs/millions/
    crores/billions, currency, the source file while kept, the figures as
    JSONB, accepted at). One period per scope of a filing; accepting the same
    quarter again replaces it. Every read and write matches the owner in the
    same statement (another account's is a 404). A period is about 20 KB;
    100 companies and 400 periods a person, 64 MB in all (`coverage_data`
    in `/api/health/database`). Audit actions `coverage_added`,
    `coverage_removed`, `period_accepted`, `period_removed`, and `exported`
    with `results_workbook`.
  - **Accepting** (`research/coverage.py` `accept`, `POST
    /api/research/companies/{id}/periods`): the figures come from the
    caller's own finished read job (kept six hours), never from the browser;
    the review answers every amber figure (accepted as read, or corrected)
    and may correct any other; an unanswered amber figure is a 422
    `amber_unreviewed`. Each figure keeps its page and, when the analyst
    touched it, a mark (`accepted` with its amber reason, or `corrected`
    with what the reader read).
  - **The comparison with what is kept**: a read for a covered company
    (`company_id` on `POST /api/research/reads`) is checked against its
    stored periods in the unit the new filing prints (R1's
    `stored_differs`), so a restated comparative is amber.
  - **Screens**: Coverage now has **Companies** (add, the accepted periods,
    Show with the page, Workbook, Remove) and **Results** (choose or add the
    company, read, then the review: the figures beside the page they came
    from, drawn in the browser by **pdf.js 4.10.38** from `GET
    /api/files/{id}/content`; amber figures answered one by one with "Next
    amber figure"; "Accept quarter into ..." only when none is left; the
    workbook offered at once).
  - **The checked workbook** (`GET /api/research/periods/{id}/workbook`): an
    About sheet and a sheet per statement; the statement's own sums
    (`checks.identities` run again on the accepted figures) are **formulas**
    wherever the parts add to the printed total exactly, and a printed total
    they don't reach stays a figure with a note saying what they add to;
    every figure's note gives its page and what the analyst did. Tests work
    every figure out from the workbook's formulas with the tests' own Excel
    reader (`tests/research_support.py`) and get the stored figure.
  - **Decided here: the PDF keeps 6.1's 90 days**, not longer. A longer
    life would hold the per-person quota (40 files, 100 MB) for good; the
    figures and their page numbers stay after the file goes, and the screen
    says the page can no longer be shown.
  - Tests: `tests/test_research_coverage.py` (accepting, the workbook
    recalculates, notes, formula-looking words), `tests/test_research_coverage_api.py`
    (the whole road, a corrected figure stored and downloaded, another
    account's 404s, limits, nothing logged), `web/e2e/coverage.spec.ts` (a
    PDF read, reviewed beside its drawn page, corrected, accepted, the
    downloaded workbook recalculated by `tests/coverage_workbook_check.py`
    to every stored figure; another account sees none of it).
  - **Before the first showing to analysts** (the user's): R1b's held-out
    filings measured (production reads only then), and billing with a cap on
    `variater-reader` with `AI_FREE_TIER=0`.

### R3 Excel add-in: sign in and import results
- **Pulls forward:** 7.3b, for research instead of deals.
- **Claude does:**
  - an Office add-in whose pane is a page of the existing web app (one
    codebase, no new host); a manifest for sideloading, with staging and
    production variants;
  - sign-in inside Excel through Office's dialog, proven on Windows, Mac
    and Excel on the web; if Google refuses sign-in inside Excel's browser,
    the pane offers email sign-in and says why;
  - the security headers changed **only for the pane's route**: Office's
    script host in the content security policy, and framing allowed from
    Office's own hosts; every other page keeps today's policy (tests);
  - **Import results**: pick a covered company and quarter, or upload a PDF
    from the pane, and the accepted statements land as cells with R2's
    formulas; amber figures are reviewed in the pane first.
- **Done when:** on a sideloaded add-in, signing in and importing a recorded
  quarter writes the same cells as R2's download (a test compares them); the
  header scan passes on every other route; DEPLOY.md says how a tester
  installs it.
- **Done (2026-10-15):**
  - **The pane is `/excel`**, a page of the web app without the shell (no new
    host), and **the sign-in dialog `/excel/connect`**. Manifests (the "add-in
    only" XML, which Excel on Windows, the Mac and the web all sideload) for
    production and staging, written by `python -m ops.addin_manifest` into
    `web/public/excel/` (`tests/test_addin_manifest.py`: current, HTTPS on the
    environment's own host, icons at their sizes); a **Variater** button on the
    Home tab; icons `public/brand/addin-*.png` from `scripts/brand.mjs`.
  - **Sign-in** (decided here): the pane can't hold the site's session (on
    Excel on the web it is a frame of Microsoft's page, which keeps the site's
    cookies from it), so Office's dialog signs in on variater.com as any tab
    does and asks `POST /api/addin/sessions` for an **add-in pass**, handed to
    the pane with `messageParent`. A pass reaches only the research endpoints,
    the three upload steps of the PDF it reads and its own session, by method
    (`api.auth.ADDIN_PATHS`, else 403; not the file list, links or deleting),
    can't make another, lasts 12 hours, at most five a person, ends at Sign
    out, and is stored only as its SHA-256 (table `addin_sessions`, migration
    0020; `db/addin.py`; expired ones deleted by the nightly job maintenance).
    Google's sign-in may be refused inside Excel's own window on Windows and
    the Mac: the pane and the dialog say so and point to email sign-in, which
    never leaves the page. `/privacy` says what is kept.
  - **Headers changed for those pages only** (`lib/security/headers.ts`
    `excelPage`, `next.config.ts`, `proxy.ts`): opener policy `unsafe-none`
    (the dialog's opener) and Office.js's CDN host in `script-src` on both;
    on the pane alone `frame-ancestors` names Office's hosts and there is no
    `X-Frame-Options` (the dialog is never framed); every other page keeps
    today's policy. `ops/check_headers.py` now scans `/excel`
    against exactly Office's hosts and `/privacy` and `/sign-in` against the
    strict policy; `tests/test_security.py` keeps its list equal to the web's.
  - **Import results**: pick a covered company and accepted quarter, or read a
    PDF from the pane (the amber figures answered there, accepted, then
    imported). The cells come from `GET /api/research/periods/{id}/cells`,
    built by the same `research/coverage.py` `layout` the download is now
    written from, so the two hold the same cells: words as words (text
    format, so "=..." stays words), figures as numbers, subtotals as formulas,
    every figure's page in a note (a comment on an Excel without notes), the
    column widths (Excel's characters-to-points rule) and frozen panes; each import goes into new sheets (a name
    taken gets a number). Recorded as `exported` with `results_excel`.
  - **Proof**: `web/e2e/excel.spec.ts` with `e2e/fake-office.js` served in
    place of Office.js (the dialog in a window of its own, as on Excel on the
    web; every write recorded): signing in through the dialog (signed in on the
    site already, and signed out with the dialog's own sign-in), importing a
    recorded quarter and reading, correcting, accepting and importing one, each
    compared with the download cell by cell by `tests/addin_cells_check.py`
    (values, formulas, formats, bold, notes, widths, frozen panes;
    mutation-checked); the pass refused outside research and after Sign out;
    the headers of the pane, the dialog and five other routes.
    `tests/test_addin.py` holds the pass's rules on a real database.
  - **Not proven here, the user's checks**: the real Excel on Windows, on the
    Mac and on the web (DEPLOY.md "Excel add-in" has the installing steps);
    whether Google's sign-in works inside the dialog on each. If one of them
    fails, what it shows decides the fix.

### R4 Sector research, the model definition and the first sector model
- **Part a, sector research (a session of its own, the user and Claude
  together):** read what is public about how analysts model Indian
  companies (published research reports, company presentations, textbooks),
  note for each sector the drivers and the lines its models carry, and agree
  how many sectors and how many distinct models are needed and which comes
  first. Reports show a model's drivers and tables, not its formulas, so
  what is learned is the shape; nothing is copied. The result is written
  here as a table of sectors, their blocks and their sources. Then the
  first model is built by hand in Excel for two companies with only the
  blocks below, to see that they are enough.
- **Part a is done (2026-10-12): the research and the decision, then the
  first model by hand in Excel for two companies** (Yatharth Hospital and
  Lemon Tree Hotels, so the test also shows one model serving two sectors).
  What the hand build found is under "Model 1 by hand" below; part b is
  next.
  - **What was read.** Four broker notes in full (ICICI Direct on HDFC AMC,
    July 2026; JM Financial on Lemon Tree, August 2025; Centrum on V-Mart,
    July 2025; BOBCAPS on Tata Steel, July 2026); for the other sectors,
    broker notes, rating agencies' sector reports and company presentations
    as search summaries only; the CFA Institute reading "Company Analysis:
    Forecasting". One session's reading, not a survey: the analyst who reads
    each model corrects it. Nothing was copied; reports show drivers and
    tables, not formulas.
  - **What every note has in common:** a short "key assumptions" table of
    drivers (nine rows for Tata Steel, two for V-Mart), a standard income
    statement, balance sheet and cash flow, then a multiple on a forward
    year (EV/EBITDA for most, P/E for the asset manager, a sum of parts for
    Tata Steel).
  - **A model is a mechanism, not a sector** (decided with the user): a
    revenue block, a cost layout and a set of words. Hospitals and hotels
    are two sectors and one model, since their formulas are the same.
    **The analyst picks a sector** from a list that can grow without a new
    model (it chooses the model and its words: beds and revenue per occupied
    bed, or rooms and room rate); underneath sit the vetted models, each
    with its own two hand-solved companies, analyst and server-against-Excel
    test.
  - **The sample's sectors** (16 lines of business by this split; R0's
    count of 14 groups restaurants with retail and capital goods with
    defence):

    | Sector (sample companies) | Drivers the reports carry | Block | Where it does not fit | Model |
    |---|---|---|---|---|
    | Hospitals (Narayana, Shalby, Yatharth) | Operating beds, occupancy, revenue per occupied bed per day | Capacity x utilisation x realisation | Outpatient and pharmacy need a second line; new beds fill slowly | 1 |
    | Hotels (Lemon Tree, Royal Orchid) | Rooms, occupancy, average room rate; fees on managed rooms | Capacity x utilisation x realisation; units x revenue per unit for fees | Food and drink is a share of room revenue (free formula) | 1 |
    | Cement (JK Lakshmi, Sagar, Star) | Capacity, utilisation, volume, realisation, costs and EBITDA per tonne | Capacity x utilisation x realisation | Costs are per tonne, not a share of sales | 2 |
    | Steel (Tata Steel) | Volume, realisation and EBITDA per tonne by country; exchange rate | Price x volume per geography | Costs per tonne; a segment in another currency; valued in parts | 2 |
    | Speciality chemicals (Vinati) | Capacity, utilisation, realisation by product | Capacity x utilisation x realisation | Fits | 2 |
    | Retail (Avenue Supermarts, V-Mart) | Stores opened, retail area, sales per square foot, same-store growth | Units x revenue per unit | Same-store growth splits old stores from new (two segments) | 3 |
    | Restaurants (Devyani) | Stores per brand, average daily sales | Units x revenue per unit, per brand | Fits | 3 |
    | Defence (Hindustan Aeronautics, Bharat Electronics) | Order book, new orders, share executed | Order book x execution | Repair and overhaul is a growth line | 4 |
    | Capital goods (Kirloskar Pneumatic) | Order book, new orders, execution | Order book x execution | Fits | 4 |
    | Telecom (Bharti Airtel) | Subscribers, revenue per user, churn, by business | Subscribers x price less churn | Africa in another currency | 5 |
    | IT services (Infosys) | Dollar revenue growth, exchange rate, headcount, utilisation, margin | Headcount x utilisation x billing rate | Billing rates are not published, so reports forecast dollar growth; headcount drives cost more than revenue | 6 |
    | Pharma (Ajanta, Indoco) | Growth by geography, gross margin, R&D share | Growth on last year, per geography | Fits | 7 |
    | Auto components (Gabriel, Suprajit) | Vehicle makers' volumes, aftermarket, exports | Growth on last year, per channel | Fits | 7 |
    | Asset manager (HDFC AMC) | Assets managed by type, yield on each, cost as a share of assets | None cleanly (units x revenue per unit, loosely) | Assets grow by flows plus market movement; costs are a share of assets | 8 |
    | Internet platform (Eternal) | Order value, take rate, dark stores, margin on order value | Free formula | A chain of multipliers; margins are a share of order value | 8 |
    | Conglomerate, refining (Reliance) | Margin per barrel, throughput; Jio; retail | None for refining | A margin on volume, not a revenue; four businesses, each with its own profit; valued in parts | 11, combined |

  - **Eleven models, all to be built** (the user's decision, 2026-10-12):

    | # | Model | Revenue is | Sectors in the sample | Others it should serve (general knowledge, not this session's reading) | Sample companies |
    |---|---|---|---|---|---|
    | 1 | Occupancy | Places x share filled x price per day | Hospitals, hotels | Cinemas, airlines | 5 |
    | 2 | Per tonne (per unit) | Volume x realisation, costs per unit | Cement, steel, chemicals | Sugar, textiles, fertiliser, building materials, paper, vehicle makers, ports | 5 |
    | 3 | Stores | Stores or area x sales per store | Retail, restaurants | Jewellery, diagnostics chains | 3 |
    | 4 | Order book | Order book x share executed | Defence, capital goods | Construction, contractors, electronics makers | 3 |
    | 5 | Subscribers | Subscribers x price less churn | Telecom | Broadband, pay TV | 1 |
    | 6 | People | Dollar growth x exchange rate; headcount for costs | IT services | Staffing, engineering services | 1 |
    | 7 | General | Growth on last year, per segment | Pharma, auto components | Consumer goods, paints, durables, agrochemicals, anything else | 4 |
    | 8 | Balance x rate | A balance x a yield or take rate | Asset manager, internet platform | Exchanges, registrars | 2 |
    | 9 | Real estate | Homes sold, counted as revenue years later | none | Developers | 0 |
    | 10 | Regulated utilities | An allowed return on money invested | none | Power, gas and water networks | 0 |
    | 11 | Margin businesses | Volume x margin per unit | Reliance's refining only | City gas, fuel retailing, refiners | 1, in part |

    **Order:** model 1 in R4; models 2 and 3 in R7; the other eight after
    R7, in an order set then (what the analysts ask for after R4's showing
    counts). The General model is the shared lower part with the simplest
    revenue block, so it is built inside R4 whichever comes first. **You
    first, before each of models 9, 10 and 11 (and 5, 6 and 8, which have
    one or two sample companies): collect results filings for at least two
    companies of that kind**, so two can be solved by hand. Conglomerates
    such as Reliance wait until models can be combined, one per segment.
  - **What the blocks lack** (the list written before the hand build; to
    add in part b, with the additions under "after the hand build" above):
    1. **A running count**: opening plus additions less closures, and the
       average, for beds, rooms, stores, subscribers and dark stores. Only
       the subscriber block has it today; make it one shared piece.
    2. **A cost row that picks its driver**: a share of revenue, an amount
       per unit of volume (cement, steel, chemicals), an amount per head
       (IT services), a share of another base (order value, assets managed),
       or last year's grown.
    3. **A segment in another currency** with an exchange rate row (IT
       services, steel, telecom, pharma).
    4. **The days in the period** as a row (per-day prices; a quarter is not
       a quarter of 365).
    5. **A balance x a rate** as a ninth block, with the balance moved by
       flows and a market return (the CFA reading's "return-based" driver).
    6. **For R6:** profit by segment and value as a sum of parts (Tata
       Steel, Reliance, Airtel), and P/E beside EV/EBITDA.
    7. **Old units against new** (same-store growth, new beds filling) is
       done as two segments, not a new block.
  - **Model 1 by hand (2026-10-12).** Two workbooks on the Mac only, in
    `research-samples/r4a/` (git ignores the folder):
    `model1-yatharth-hospital.xlsx` and `model1-lemon-tree-hotels.xlsx`,
    written by `build_model1.py` from the typed figures in `companies.py`;
    `check_model1.py` works the formulas out and prints every check. Each
    has a **Reported** sheet (the figures as typed, with the page's own sums
    as formulas), a **Model** sheet and a **Check by eye** sheet. No Gemini
    request was made: the figures were read by eye from the two results
    PDFs and from the companies' investor presentations.
    - **The blocks are enough for the revenue.** Three of the eight served
      both companies: places x share filled x price per day, units x revenue
      per unit (Lemon Tree's fees on managed rooms) and a free formula (food
      and drink as a share of room revenue). **The model sheet is the same
      118 rows in both; all 243 forecast formulas are identical, cell for
      cell**, and only the words and the past differ. Yatharth leaves lines
      2 and 3 at nil. Plain arithmetic with `MAX` was enough throughout
      (`MIN` was never needed; `ROUND` once, in a check on typed figures).
    - **Checks:** every sum the filings print holds on the typed figures
      (27 for Lemon Tree, 7 for Yatharth), the model's past profit before
      tax, after tax and for the owners equals the filings' to the paisa,
      and the balance sheet balances in every forecast year. These were
      worked out by a throwaway reader, **not yet by Excel** (Excel did not
      answer the request to recalculate: open each workbook once and look
      at the check rows).
    - **The block has to run backwards for the past (the main finding).**
      Yatharth prints bed capacity, occupancy and revenue per occupied bed,
      and their product is not its revenue: FY25's 1,605 beds, 61% and
      Rs 30,829 give Rs 11,017 million against Rs 8,805 million reported,
      because occupancy is counted on fewer beds than the capacity printed
      and that count is not printed. So the past types revenue, occupancy
      and price and **works out the beds in use**; Lemon Tree prints rooms,
      occupancy and rate for its own hotels and **works out room revenue**.
      Which of the four is worked out is a choice per company, and the
      format must allow it.
    - **One results PDF does not start coverage.** A first or third quarter
      filing has no balance sheet and no cash flow, so the model starts from
      the last year-end (or half-year) balance sheet and rolls the year in
      progress forward from there with the whole year's flows. Yatharth's
      opening balance sheet had to come from a secondary source. R2 and R4
      part b must ask for the latest filing that carries a balance sheet as
      well.
    - **The year in progress needs last year cut the same way.** A hotel's
      second half is its strong half (FY25: 74% and 78% occupancy against
      67% and 68%), so "the rest of the year" cannot be the first half
      grown; it is typed, or grown on the same part of last year, which
      means holding last year's quarters and their drivers too.
    - **Drivers seldom arrive ready.** Half-year occupancy and rate are not
      printed (weighted from the two quarters by nights sold); Yatharth's
      full-year price was worked out from occupied bed days; the count at
      the end of the latest period is not printed; the company's "EBITDA"
      counts other income and the rate is called "gross". Presentations are
      in crore where the filing is in lakh. A typed driver therefore needs
      a place for its own working, its source and a mark.
    - **The sector table was wrong on one point:** Yatharth's revenue per
      occupied bed covers all hospital revenue (outpatients and pharmacy
      included), so no second line was needed. Whether a hospital needs one
      depends on how that company defines the figure.
    - **Figures the user should check by eye** (each is amber or red in the
      workbook, with its reason): all of Yatharth's 31 March 2026 balance
      sheet except equity (from stockanalysis.com's rounded table; equity
      agrees with the filing to 0.4); Yatharth's occupancy for FY26 (the
      nine months' 66% standing in) and Q1 FY27 (68%, one press report),
      and its revenue per occupied bed for Q4 FY26 and Q1 FY27 (press
      reports of the company's release); Lemon Tree's FY25 interest on
      leases (two half years added, the year's own split is not printed);
      Yatharth's Q1 FY26 earnings per share (printed 4.35, the page's own
      figures give 4.36). Yatharth's website turns programs away and the
      exchanges' copies may not be fetched by one, so **the user saves
      Yatharth's FY26 results filing and its Q1 FY27 investor presentation
      into `research-samples/`** before the red figures can be replaced.
    - **The forecast columns hold placeholder assumptions** so the formulas
      have something to work on. They are nobody's forecast.
  - **What the blocks lack, after the hand build** (the list below is right
    in what it has and too short; nothing on it was wrong):
    - Item 1 (a running count): confirmed, and used twice in one model
      (rooms, managed rooms).
    - Item 2 (a cost row that picks its driver): confirmed already in
      model 1, not only per tonne. Employees and power follow rooms and
      beds, not revenue. It needs no switch: **cost = a share of revenue +
      an amount per place per day x places x days, with one of the two left
      at nil.** "Last year's grown" was not needed.
    - Item 4 (the days in the period): confirmed, and needed well beyond
      prices: depreciation, interest and other income for a part year, and
      working capital in days.
    - Item 5 (a balance x a rate): not needed as revenue here, but the
      lower part is the same piece three times (interest on borrowings,
      interest on leases, income on cash), so build it once.
    - Item 7 (old units against new as two segments): Yatharth is the case
      (new hospitals are 27% of revenue at other prices), but the company
      prints occupancy for the group only, so the split would be the
      analyst's estimate. It stays a choice, not a default.
    - Items 3 and 6 were not tested by this model.
    - **To add** (all found in the hand build): 8. the past run backwards
      (above); 9. a constant for rupees in one unit of the statements, since
      prices per day are in rupees and statements in lakh, crore or million;
      10. in the lower part, **lease liabilities and their interest** (23%
      of Lemon Tree's finance cost, and the filing does not split it),
      **non-controlling interests** (19% of Lemon Tree's profit),
      associates, and other income as a yield on cash; 11. one kind of row
      for every ratio: worked out in the past, typed in the future, and "as
      it came out" for the year in progress; 12. a segment switched off
      must not divide by nil (the hand build used `MAX(x, 1)`); 13. a check
      on a printed ratio needs a tolerance of one in the last place (the
      earnings per share check raised a false alarm); 14. deferred tax is
      left out (tax paid = tax charged), which the model must say.
  - **Sources** (2026-10-12): hospitals: CareEdge
    (careratings.com/uploads/newsfiles/1743058970_Hospital%20Industry%20-%20CareEdge%20Report.pdf),
    CRISIL "Reading the vitals" (2023), ICRA; cement: ICRA's FY2027 outlook
    as reported by Outlook Business and Indian Cement Review, Equirus as
    reported by NBM&CW; hotels: JM Financial on Lemon Tree (12 August 2025),
    Motilal Oswal's initiation (April 2019); IT services: Motilal Oswal
    (2011) and Angel Broking (2016) on Infosys, Infosys's own quarterly
    figures; retail and restaurants: Centrum on V-Mart (27 July 2025), HDFC
    Sky on Avenue Supermarts, Devyani's results; defence: HDFC Securities
    and ICICI Direct on Bharat Electronics, JM Financial (April 2026), Angel
    One's explainer on order books; capital goods: Arihant, Prabhudas
    Lilladher and MNCL on Kirloskar Pneumatic; telecom: FundsIndia (2018)
    and Motilal Oswal (2020) on Bharti Airtel; pharma: BOB Capital and ICICI
    Direct on Ajanta, Ajanta's investor presentation (Q1 FY26); auto
    components: FundsIndia (June 2026), Gabriel's results presentations;
    steel: BOBCAPS on Tata Steel (31 July 2026), ICICI Direct (2022, 2025);
    chemicals: HDFC Securities, ICICI Direct and Sharekhan (May 2025) on
    Vinati; asset manager: ICICI Direct on HDFC AMC (16 July 2026), JM
    Financial; platform: ICICI Securities, Kotak, Nomura and JM Financial on
    Eternal as reported by Business Today and Business Standard;
    conglomerate: BOB Capital on Reliance (18 July 2026), YES Securities;
    textbook: CFA Institute, "Company Analysis: Forecasting"
    (cfainstitute.org refresher readings, 2026).
- **Part b, first slice, is done (2026-10-12; out of turn, because it needs
  no AI key): the format, the evaluator, the Excel writer and model 1.** All
  in `research/`; no screen, no endpoint, nothing the API imports yet.
  - **The format** (`research/definition.py`): a model is rows down and
    periods across with **a rule in each cell**: a figure typed from the
    documents (may be blank), an assumption (must be given), a formula, a
    **choice** between rules that the company's file settles, or empty. A
    row gives its rules by period name, by kind of period (last year, year
    to date, rest of year, current year, forecast) or for a group of kinds;
    the narrowest wins. A company is only data: words, unit, choices,
    numbers, segments switched off, and marks (amber or red with a reason).
    A model goes to JSON and back unchanged.
  - **Formulas** (`research/formula.py`): `+ - * /`, brackets, numbers,
    `max`, `min`, row names, `cash[prev]` and `rev[ytd]`, and **`div(a, b)`:
    a over b, or nil when b is nil** (decided with the user: the hand
    build's `MAX(x, 1)` is wrong for small or negative bottoms). Read by
    its own small reader, never `eval`; a formula has a size (1,000
    characters, 300 parts, 40 levels).
  - **One tree, two renderings** (rule 3): `research/evaluate.py` works
    every cell out in dependency order (a circle, an unknown row, a missing
    assumption and a plain division by nil are errors that name the row and
    period); `research/workbook.py` writes the same trees as Excel text
    with ordinary cell addresses (`div` becomes `IF(b=0,0,a/b)`), a defined
    name on every assumption cell (`s1_occ_y1`) and constant
    (`unit_rupees`), "NOT YET VETTED" at the top and a Notes sheet.
  - **Each "to add" item of the hand build is in:** the past run backwards
    (choice `line1_past`: `drivers` or `revenue`); a running count, a cost
    as a share of revenue plus an amount per place per day, a balance x a
    rate (used four times: depreciation, income on cash, interest on
    borrowings and on leases) and a ratio row, each one function in
    `research/pieces.py`; the days in the period; `unit_rupees`; a segment
    switched off reads nil and divides nothing; the model's notes say
    deferred tax is left out.
  - **Model 1** (`research/models/occupancy.py`): 106 rows, the hand
    build's 103 plus the filing's three printed profits, which the checks
    on the past compare with. Sectors `hospitals` and `hotels` bring their
    own words; a company's words override them.
  - **Proof.** The two companies' typed figures and the hand workbooks'
    numbers are committed (`tests/fixtures/research/`, public filings
    only): **model 1 gives the hand workbooks' number on every row and
    period, 103 rows x 6 periods each, to 0.01** (in fact to rounding).
    The written workbook is worked out by a reader in the tests that
    shares no code with `research/` and agrees with the evaluator on every
    cell; typing a new assumption into the workbook moves its own formulas
    to what the server gives. 53 deliberate breaks of the code were each
    caught by a test.
  - **From the review** (fixed in the slice): a formula of nested `div`s
    doubled its Excel text at every level, so a formula whose Excel text
    would pass Excel's 8,192 characters is now refused when the model is
    checked; formulas are plain ASCII (Python's digits include other
    scripts', which Excel would not read back); every part of a company is
    checked for shape and size before anything reads it; a label's
    `{word}` slots are filled by a pattern, not `str.format`. **Recorded,
    not fixed:** Excel reads a sum within rounding of nil as exactly nil
    and the evaluator keeps the tiny number, so `div(1, 0.1 + 0.2 - 0.3)`
    would differ. No row of model 1 divides by such a difference; a model
    that does needs a tolerance in both renderings.
  - **The real spreadsheet engine in CI is LibreOffice, and it has not run
    yet.** `tests/test_research_workbook.py` has the test: LibreOffice opens
    the workbook without a screen, works it out, and its values must be the
    evaluator's. It exists only where `soffice` is installed or
    `FSE_REQUIRE_SOFFICE=1` is set (the `ml` job fails on a skipped test,
    so it is left out, not skipped). **It needs one workflow edit that is
    the user's** (CLAUDE.md "Waiting on the user"); until then agreement
    with a real engine is unproved, and the first run may need a fix.
  - **Left for the rest of part b:** the other seven revenue blocks and the
    General model; quarterly actuals feeding the year in progress (today
    the year to date and the rest of the year are two typed columns);
    last changes by the analyst (add or drop a segment, swap a block);
    screens, the add-in's "Load model", any endpoint, and with the endpoint
    the three places a new package goes (Dockerfile, `render.yaml`,
    `staging.yml`); a mark and a working for each typed driver on screen.
- **Needs from the user (part b):** the analyst who reads the model. If
  none is found yet, the model ships marked "not yet vetted" and this line
  stays open.
- **Claude does (part b):**
  - the model definition format: rows, periods and formulas in plain
    arithmetic with `max`/`min` (as the reference cases use), an evaluator
    on the server and a writer that emits the same rows as Excel formulas
    with named assumption cells;
  - the revenue blocks, one per segment: price x volume; units x revenue
    per unit; subscribers x price less churn; capacity x utilisation x
    realisation; headcount x utilisation x billing rate; order book x
    execution; growth on last year; and a free-formula row;
  - the shared lower part: costs, depreciation, capex, working capital,
    debt, interest, tax, the three statements tied together;
  - quarterly actuals feeding a yearly forecast (the year in progress =
    quarters reported plus the rest assumed);
  - the first sector model, with last changes the analyst can make (add or
    drop a segment, swap a block) without breaking a formula;
  - screens: pick a model, type driver history and assumptions, see
    projections; "Load model" in the add-in writes it into the workbook;
  - two reference companies solved by hand, as 3.4.
- **Done when:** for every model the server and the recalculated workbook
  agree to 0.01 on every row (test, with a real spreadsheet engine in CI);
  the hand-solved companies match; changing one assumption cell in Excel
  moves the projection with no call to the server; the named analyst has
  read the model and their points are fixed or recorded, or the model says
  "not yet vetted" wherever it is shown.

### R5 Save back and roll forward
- **Claude does:** a research model saved per company with versions (the
  pattern of 1.5: autosave, history, restore); "Save to Variater" in the
  add-in reads the named cells back, and refuses with a clear message if
  the sheet's structure was changed rather than guess; on a new quarter,
  **Roll forward**: the model and assumptions kept, the estimate made
  before the results set beside the actuals (2.7's comparison, line by
  line), the period moved on, in the app and in the sheet; audit entries.
- **Done when:** an estimate saved before a quarter is shown against that
  quarter's actuals after it, unchanged by later edits (test); a workbook
  saved, changed in Excel and saved again returns the changed assumptions;
  a restructured sheet is refused.

### R6 Valuation
- **Claude does:** inside the model definition, so it is formulas too: a
  discounted cash flow (discount rate and terminal growth as assumptions,
  with the country's risk-free rate from 4.2 offered as a starting point)
  and a target multiple on a chosen year; net debt and share count from the
  accepted results; the share price typed in, with its date; value per
  share and the gap to the price. No rating.
- **Done when:** both methods match a hand-solved case to 0.01; the workbook
  agrees with the server; nothing on screen or in the workbook words a
  recommendation (test over the catalogue).

### R7 Second and third sector models
- **Needs from the user:** the readers. The models are R4 part a's:
  **second, Per tonne (cement, steel, chemicals); third, Stores (retail,
  restaurants)**. The other eight of the eleven follow as tasks added after
  R7.
- **Claude does:** each as R4's last three bullets; what R4's format lacked
  is added to the format, never worked around in one model.
- **Done when:** as R4, per model.

### R8 Report draft
- **Pulls forward:** 6.3's writer and its rule, and 7.9's charts.
- **You first:** a securities lawyer's view that the tool stays outside
  SEBI's research analyst rules; the disclosure wording your testers use.
- **Claude does:** a results-note template with slots: summary table,
  actual against estimate, changes to estimates, charts, valuation; code
  fills every number from the saved model; the AI writes linking sentences
  only, from the figures, and a check refuses a draft holding a number the
  model does not; the argument, the rating and the target are empty fields
  the analyst fills; firm name and disclosures from the account; Word and
  PDF.
- **Done when:** a test checks every figure in a draft against its model; a
  planted wrong number is refused; no draft contains a rating the analyst
  did not type.

### R9 Ready for the paid switch
- **Claude does:** pages read and AI cost counted per user; the paid
  provider's path proven on recorded answers; limits per plan as settings;
  what a store listing of the add-in needs, written as steps for the user.
- **Done when:** switching `AI_PROVIDER` on staging changes the provider and
  nothing else (test); DEPLOY.md lists each paid switch and what it lifts.

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
- **Done 2026-10-01** (DEPLOY.md "Own domain"): visible brand only renamed;
  development-instance accounts dropped as test data (both decided with the user).
- **Decided 2026-10-01:** the product is named **Variater** (a coined
  spelling: `variator.com` has been taken since 2000, and "variator" is a
  scooter part). **`variater.com` is bought** at Cloudflare Registrar, one
  year at cost, auto-renew on, DNS at Cloudflare, no records yet. Other
  endings (`.in`, `.app`) wait for the public launch. Trademark: an Indian
  filing (IP India, form TM-A, class 42, "proposed to be used") before the
  public launch; `Variater™` until then; no ® until registered.
- **Rename scope (proposed; confirm at the start of 0.2):** only what people
  see -- the brand `app.brand` in `web/messages/en.json` ("FSE/ML"), page
  titles, README, the Better Stack status page, Clerk's sign-in branding.
  Internal names stay: `FSE_*` settings and GitHub secrets, the `fse_app` /
  `fse_api` database roles (migration 0005), the backup file format, the
  Vercel, Render and Neon service names, the repository and its folder.
- **Still to decide at the start of 0.2:** map today's development-instance
  accounts to their new Clerk ids, or drop them as test data.
- **You first:**
  - ~~buy the domain~~ (done: `variater.com`, Cloudflare);
  - keep its DNS at the registrar (Cloudflare);
  - (done 2026-10-01: Clerk production instance for `variater.com` created
    by cloning the development one; its five CNAMEs -- `clerk`, `accounts`,
    `clkmail`, `clk._domainkey`, `clk2._domainkey` -- are in Cloudflare,
    DNS only, and resolve publicly. Its keys are **not** in Vercel or Render
    yet: they go in at the switch-over, since a production instance can't
    serve `fse-ml.vercel.app`. Google sign-in still needs custom OAuth
    credentials.)
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
- **Split (2026-09-24), too big for one session:**
  - **2.3a (done):** figures and dates follow the account's locale
    (`web/src/lib/locale.ts`, `lib/format.ts`) with a digit-grouping choice
    on the account (as the locale does, thousands, or lakh and crore;
    migration 0007); inputs read numbers the locale's way; a fiscal
    year-end per deal (`fiscal_year_end_month`, `first_fiscal_year`, labels
    only) and per forecast company (read from the latest 10-K by EDGAR);
    labels "FY2025" or "FY2024/25"; Excel exports with a number format per
    cell (lakh patterns sized per cell); CI check against hardcoded
    locales and ad-hoc `toFixed` (`tests/test_no_hardcoded_locale.py`);
    e2e `locale.spec.ts` in `en-US`, `de-DE` and `en-IN`, March year-end
    deal and company.
  - **2.3b (done):** all interface text in translation files
    (`next-intl` 4, one catalogue per language in `web/messages`, the
    language and the text direction taken from the account's locale with no
    locale in the URL, so `src/proxy.ts` and its CSP are untouched, and
    `lib/locale.ts` still does the numbers); English complete (1,090 keys);
    right-to-left checked with an Arabic account, the layout mirroring on
    logical CSS properties while figures stay left-to-right islands;
    three CI checks -- no interface text in `web/src`
    (`tests/test_no_hardcoded_text.py`), every key asked for exists and
    every message is used (`tests/test_translations.py`), and every label
    the engine produces has a key; the three-locale e2e run plus
    `e2e/text.spec.ts`. Native digits (Arabic-Indic, Devanagari) stay out:
    figures keep Latin digits so the mono columns align, and 7.8 may
    revisit it with the languages.

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
- **Split (2026-09-29), too big for one session:**
  - **2.4a (done):** the model. A deal's debt is a list of facilities
    (`core/debt.py`, `DealInputs.tranches`); the nine kinds are presets over
    one dataclass, with three new mechanics -- interest accruing to principal
    (PIK), a revolver's undrawn commitment, its fee and its draw against a
    cash shortfall, and a per-facility share of the cash sweep. A floating
    rate is `max(reference, floor) + margin` resolved in `core/debt.py`, so
    `lbo_engine` sees only a rate path and 4.2 can fill it from market data.
    Hand-checked cases in `tests/test_debt_structures.py` for the floating
    SONIA loan with a floor, the PIK note and the unitranche, each with its
    arithmetic in the docstring; `equivalent_tranches` writes today's senior +
    mezzanine out explicitly and gives the identical answer, and
    `tests/golden/golden.json` is untouched. Stored only when a deal has one
    (`OMIT_WHEN_DEFAULT`), so existing deals and a rollback are unaffected.
  - **2.4b (done):** the simulation and the screen. The Monte Carlo
    simulation's output is pinned first (`tests/test_montecarlo_baseline.py`),
    and its two-bucket path is untouched; a tranche path beside it
    (`simulation/tranches.py`) is the deal model's debt schedule vectorised.
    The rate draw is a shock to reference rates and moves **floating
    facilities only** (`max(reference + shock, floor) + margin`); a
    scenario's rate stress reaches them through their references; PIK is
    added back in the simulated cash flow; the heatmap runs on the deal's
    structure. Today's structure written out simulates exactly as the
    percentages, path by path. The Debt step lists the facilities (add,
    edit, reorder -- the list is the sweep order -- and remove), shows a
    capital-structure table and the PIK, fee and draw rows a facility has;
    the other deal steps point to it instead of offering the percentage
    fields a listed deal ignores. `e2e/tranches.spec.ts` proves each edit by
    output (convert, add PIK notes, remove, reorder, save and reopen, Monte
    Carlo).
  - **Carried forward, not 2.4's job** (recorded from the 2.4 reviews):
    1. No cap on deals per account or saved versions per deal (a deal with
       twelve full facilities is about 20 KB): wants a per-account deal cap, a
       version cap and a byte budget checked at the write (1.6's
       `api/limits.py`, or 9.2).
    2. `sens_*` settings are unbounded and `/api/deal/run` has no timeout
       (`sens_hp_max` of 200 with 50 exit steps does not finish): bound them
       at the request edge and put the deal endpoints under the 100 s cap.
    3. Exception messages still reach the logs and Sentry
       (`api/observability.py`'s `JsonFormatter` uses `formatException`;
       `scrub_event` keeps `exception.values[].value`): drop the message,
       keep the type. Nothing in 2.4 leaks today; every refusal is a
       returned 422.
    4. A shareholder loan is treated as debt at exit; whether it joins the
       sponsor's proceeds instead is a modelling decision for the user.
    5. Cross-currency facilities: `TrancheSpec.currency` exists but the API
       does not expose it; revisit after 4.2's exchange rates.
    6. Excel export of the capital structure (`api/routers/export.py`): 7.3.
    7. The backtest never sees tranches (`core/backtesting.py` builds
       `LBOParams` from a plain dict). Done in 2.7: plan vs actual runs the
       saved deal itself, tranches, tax rules and leases included.

### 2.5 Global tax rules
- **Claude does:** per-deal corporate rate; interest deductibility limit (none,
  a share of EBITDA, or a fixed amount) with carry-forward; tax losses carried
  forward with an optional yearly cap; optional minimum tax; editable country
  presets with source and date, marked "check with a tax adviser".
- **Done when:** hand-checked cases pass for a 30%-of-EBITDA interest cap and for
  losses carried forward; changing preset changes results as predicted (test).
- **Done (2026-09-30):** the rules are `lbo_engine/tax.py`, one function for the
  deal model and the simulation (numbers or arrays of paths), called only when
  a rule is on, so every existing deal and the golden snapshot are untouched.
  Interest limit (a share of EBITDA never below an allowance, or a fixed cap)
  with the refused interest carried forward; losses carried forward under an
  allowance plus a share of the excess; a minimum tax on book profit. Eleven
  country presets in `core/tax.py` (US, GB, DE, FR, NL, IE, IN, JP, AU, CA, SG),
  each with its source, what it simplifies and the date it was checked; a
  preset's amounts apply only in its own currency. Hand-checked cases in
  `tests/test_tax_rules.py`; Ireland's year-one tax is exactly half the UK's
  on the same deal (the prediction test); `e2e/tax.spec.ts` proves the screen
  by output. Not modelled, and said so in each preset's note: loss expiry,
  state and local taxes, group relief, small-profits rates.

### 2.6 Accounting standards (IFRS and US GAAP)
- **Claude does:** accounting standard per company and deal; mapping from each
  standard's line items to model inputs; IFRS 16 lease setting (EBITDA before
  or after leases; leases as debt or not); adaptable labels.
- **Done when:** a real IFRS filing maps to model inputs (test); the lease
  setting changes EBITDA and net debt as a hand-checked case predicts.
- **Split, like 2.3 and 2.4:**
  - **2.6a (done):** the model, the mapping and the API. `core/accounting.py`
    holds each standard's line items (`us-gaap` and `ifrs-full` concepts) and
    the lease rule; the EDGAR extractor reads a 20-F or 40-F in IFRS, in the
    filing's own currency, and answers the standard, the leases and the latest
    year as deal inputs. Pinned by two real filings recorded in
    `tests/fixtures/edgar/`: SAP SE's 2025 20-F (IFRS, euros) maps to model
    inputs figure by figure, and McDonald's 10-K extracts exactly as it did
    before 2.6. A deal's `accounting_standard`, `lease_view`, `lease_cost` and
    `lease_liability` (off by default, stored only when set) move EBITDA and
    net debt as `tests/test_accounting.py`'s hand-checked case predicts, in
    the deal model, the exit grid, sources and uses and the simulation.
  - **2.6b (done):** the screen. The Inputs step's "Accounting and leases"
    rail group (`steps/LeaseRules.tsx`: standard, lease cost, liability,
    view); EV and debt tiles on the valuation EBITDA; sources and uses say
    when the price is less leases taken over; a Leases tile on Returns;
    statement labels follow the standard (`lib/i18n/useStandardLabel.ts`,
    the `standards` namespace: "Finance costs", "Profit for the year" under
    IFRS); the forecast company's standard (set by a filing, editable) and
    "Use in deal" from an EDGAR filing. `e2e/accounting.spec.ts` proves each
    control by output (mutation-checked).

### 2.7 Backtest becomes "plan vs actual" for any deal
- **Claude does:** pick any saved deal as the plan; enter or upload actual yearly
  results and exit; compare with the exact attribution (EBITDA, multiple and
  net debt) in the deal's currency; the 4 historical deals move to the optional
  example library (4.5); the screen works with no library.
- **Done when:** a user's own saved deal is backtested end to end (e2e); with the
  library off the screen still works (test); attribution still adds up.
- **Done (2026-10-02):** `core/plan_actual.py` runs the saved deal through the
  deal model (and simulates it around its own assumptions) against actuals
  stored on the deal (`deals.actuals`, migration 0008, not versioned);
  `POST /api/backtesting/plan-vs-actual`, `GET/PUT/DELETE
  /api/deals/{id}/actuals`; partial years for a deal still held, an early exit
  compared with the plan rerun at that hold, CSV upload and download. The four
  examples are `core/examples.py` (`GET /api/backtesting/examples`,
  `FSE_EXAMPLE_LIBRARY=0` hides them); `/api/backtesting/run` stays as the
  golden parity record. Proof: `tests/test_plan_actual.py`,
  `web/e2e/backtest.spec.ts`.

### 2.8 Risk warnings computed, not written in
- **Claude does:** replace fixed statistics ("38% historical distress rate",
  "Only 2 of 9…") with figures computed from sourced data: cited rating-agency
  base rates and the deal's own ratios. Each warning shows its source and
  sample; warnings with no data behind them are removed.
- **Done when:** a test fails if a warning contains a number not computed from
  data; each warning shows its source.
- **Done (2026-10-04):** the anomaly detector's six written-in flags ("38%
  historical distress rate", "6 of 8", "2 of 9", "9% ... margins > 20%",
  "median 18-22%", "1.5x covenant threshold common") are gone: none had a
  source. `core/risk_warnings.py` computes four warnings on the deal answer
  (`risk_warnings`), each carrying figures only, never sentences, and its
  sources: debt at close above the **6.0x** EBITDA of the ECB's 2017 and the US
  2013 leveraged-lending guidance; year-one EBIT / interest mapped to a rating
  by **Damodaran's January 2026 coverage table**, and when speculative grade,
  **S&P's 2024 study, table 26** cumulative default rate for that rating over
  the deal's hold (23,831 issuers, 1981-2024); a year whose EBITDA does not
  cover interest; a year whose repayments exceed the cash the deal has (the
  amount finding 11 conjures). Tables in `core/risk_sources.py`, transcribed
  with source, date and sample. Words in `web/messages/en.json` `warnings.*`;
  Deal → Inputs shows them with their sources (`DealWarnings.tsx`). Proof:
  `tests/test_risk_warnings.py` (the catalogue's warning words hold no digit;
  every figure recomputed from the run or read back from its table; the
  unfunded amount reconciles the cash balance), `web/e2e/warnings.spec.ts`.

---

## Phase 3 — Trust in the numbers

### 3.1 Model version on every result
- **Claude does:** every result carries engine version, git commit, settings
  fingerprint and data vintage; saved versions and exports store them;
  reopening an old deal shows "results changed since saved";
  `MODEL_CHANGELOG.md`.
- **Done when:** exports show version and data vintage; the change notice
  appears for an old version (test).
- **Done (2026-10-04):** `core/model_version.py` stamps every model result
  (deal run, sources and uses, Monte Carlo run and scenarios, backtest, plan
  vs actual, forecast, the ML endpoints, and the same answers as background
  jobs) with `model`: `engine_version` (1.0.0), `commit` (Render's
  `RENDER_GIT_COMMIT`), `settings_fingerprint` (over the Settings resolved
  against today's defaults, so a changed default shows), `data_vintage` and
  `data_fingerprint` over the published data sets (`data_sets`: the risk
  tables' editions and the tax presets'). `/api/health` names the engine
  version and the status bar shows it. Saved deals and every version store
  the stamp with the deal's IRR and MOIC (migration 0009, column `model`,
  about 200 bytes; versions stay under 900); restore and duplicate carry it.
  Opening or restoring a deal answers `model_check` (`changed` / `unchanged`
  / `unknown` for deals saved before 3.1, with the causes `engine_version`,
  `data`, `settings`), and the deal screens show "Results changed since
  saved" with IRR and MOIC then and now. Excel About sheets show the stamp of
  the result they hold (the web sends it; the Monte Carlo sample is stamped
  by the API). `MODEL_CHANGELOG.md` says when the version changes;
  `tests/model_version_pins.json` records reference deals' results per
  version, so a change that moves a number fails until the version is
  raised. Proof: `tests/test_model_version.py` (an aged stamp gives the
  notice; a changed default is named; a 1bp rate change fails the pins),
  `web/e2e/model-version.spec.ts`.

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
- **Done (2026-10-06):** table `audit_events` (migration 0010), written by
  `db/audit.py` `record` in the same transaction as each action; `GET
  /api/deals/{id}/history` and `/api/account/history`; Deal → Saved deals
  "Activity". "Shared" is a reserved action until 7.2 adds sharing. The API's
  role may only insert and read entries; `audit_compact` (nightly) merges
  edits older than a week one per deal and day. Proof: `tests/test_audit.py`,
  `web/e2e/audit.spec.ts`.

### 3.4 Hand-checked reference cases (incl. non-US)
- **Claude does:** 20+ small deals solved by hand in a committed spreadsheet (no
  leverage; single tranche; fees only; zero growth; floating with floor; PIK;
  unitranche; interest cap; tax losses; IFRS 16; non-USD; March year-end),
  with tests requiring the engine to match.
- **Done when:** all match to 0.01; workbook and reasoning in `tests/reference/`.
- **Done (2026-10-06):** 26 cases in `tests/reference/` (`cases.py` holds the
  working as formulas and the reasoning; `reference_cases.xlsx` is the same
  working as live Excel formulas, one sheet per case; `README.md` lists them).
  `tests/test_reference_cases.py` runs each through `POST /api/deal/run` and
  requires every checked figure (at least 18 per case, 1,155 in all) to match to 0.01;
  twelve deliberate breaks of the engine were each caught. Case 08 measures
  finding 11 by hand rather than avoiding it (the engine is higher by exactly
  the unfunded repayment it reports). Writing the cases showed finding 10's
  rounding building up: half-cents recurring each year left cash 0.015 off by
  year three, so the cases pick numbers that don't. No model change.

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
- **Split, like 2.3, 2.4 and 2.6:**
  - **4.1a (done):** the data layer. `companies/` is the interface
    (`sources.search`, `sources.fetch`) with a connector per source: SEC
    EDGAR's company facts, ESEF reports as xBRL-JSON from filings.xbrl.org,
    Companies House accounts filed as inline XBRL, EDINET's XBRL-to-CSV.
    Each turns a filing into the same summary figures per fiscal year
    (`companies/items.py`: revenue, operating income, D&A, EBITDA, net
    income, tax, interest, capex, cash, receivables, inventories, payables,
    debt, total assets, equity, leases) in millions of the filing's
    currency, with the standard (IFRS, US GAAP, UK GAAP, Japanese GAAP),
    the fiscal year end and a link to the filing with its date. Search by
    name, ticker, CIK, UK company number, EDINET or securities code,
    Japanese corporate number, LEI or ISIN (GLEIF turns the last two into
    the company and its home-register number). Stored compactly (migration
    0011; under 4 KB a company, at most 8,000, a 64 MB budget reported by
    `/api/health/database`); refreshed nightly by `company-refresh`
    (EDINET's day lists into an index, stale companies reloaded), every
    call paced within the source's rules. Recorded responses for the US
    (McDonald's), the UK (Tesco's ESEF report; Cambridge United's Companies
    House accounts, FRS 102), the EU (Heineken) and Japan (Toyota, IFRS;
    Nintendo, Japanese GAAP) pin the figures (`tests/test_companies.py`),
    the storage, the API and the refresh (`tests/test_company_data.py`).
    `record-filings.yml` records the keyed sources with the repository
    secrets. A company no source covers is answered with the fallback
    `document_upload`, which 6.2 builds.
  - **4.1b (done):** the screen. A company search on Forecast -> Historicals and on
    the deal (replacing the EDGAR ticker box): results from every source,
    a company's figures by year with a link to each filing, "Use in deal"
    and "Use in forecast" from any source (the forecast's full statement
    lines come from the EDGAR answer where the source is SEC; others fill
    what the summary has), the document-upload fallback named for a company
    not found, and `/privacy` saying that a company search is sent to the
    public registers searched. Browser tests replay recorded answers.
    Built as: the company answer carries `deal_inputs` and
    `forecast_history` (`companies/use.py`, hand-checked on Tesco, every
    recorded company balancing and keeping its filed EBIT, EBITDA and net
    income); one `CompanyProvider` shared by Deal -> Inputs and Forecast ->
    Historicals, a figures-by-year tile linking each year's filing; a SEC
    company with a ticker takes EDGAR's full statements, falling back to the
    summary. The browser tests replay `web/e2e/fixtures/companies.json`,
    the API's answers for the recorded filings (`python -m
    tests.e2e_companies`; a test fails when it is stale).

### 4.2 Economic data by country, and exchange rates
- **You first:** get a free FRED key; set `FRED_API_KEY`.
- **Claude does:** connectors for IMF, World Bank, OECD, BIS, ECB and FRED; per
  country GDP growth, inflation, policy rate, bond yields and credit spreads
  where available; current reference rates for 2.4; ECB exchange rates; stored
  with dates and sources; scheduled refresh.
- **Done when:** at least 20 major economies have current, sourced data; a SONIA
  tranche picks up the current rate by default (test); exchange rates refresh
  daily on staging.
- **Built as (done):** `economy/` -- one connector per source
  (`connectors.py`: IMF WEO DataMapper, World Bank WDI, BIS policy rates,
  OECD yields and 3-month rates, ECB €STR, EURIBOR and euro reference rates,
  FRED SOFR, SONIA and the US 10-year yield), each a call or two covering
  every economy, paced through `companies/http.py`. 24 economies and the euro
  area (`catalogue.py`); each figure stored with its period, source, series id
  and a link (migration 0012, `db/economy.py`, an 8 MB budget), and judged
  *current* by its age (`views.py`); 23 are current on the recorded data
  (Singapore has no policy rate). Growth and inflation show the IMF's
  projection for this year beside the World Bank's latest actual. A new
  floating facility starts on its currency's benchmark at today's level, and
  choosing a benchmark sets it (`web/src/lib/economy.ts`, the tranche editor
  shows the level's date and source, and what stands in for TONA, SARON,
  BBSY and MIBOR, which have no free source); the level is stored in the deal
  as an input, so no model result moves on its own. The nightly
  `economy-refresh` runs on both environments and after every staging deploy
  (`staging.yml` checks 20 economies current and exchange rates under seven
  days old). Decision: FRED's corporate spread series (ICE BofA, Moody's)
  forbid reproduction, so the app shows sovereign spreads over Germany
  (OECD, CC BY) instead; corporate spreads wait for a licensed source (phase
  12). Tests replay recorded responses (`tests/test_economy.py`; FRED's
  recorded by `record-economy.yml` with the repository key).

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
- **Built as (done):** `benchmarks/` -- Aswath Damodaran's industry averages
  (NYU Stern, January 2026: margins, EV/EBITDA, capex, working capital, debt
  and interest coverage for about 85 non-financial industries, five data sets
  for eight regions, 43,056 listed companies worldwide) and his copy of the
  Tax Foundation's country tax rates, read by the weekly-gated
  `benchmarks-refresh` into 41 compact tables (migration 0013, a 4 MB
  budget). `starting.py` turns them, with 4.2's IMF projections and
  benchmark rates, into a deal's starting inputs, each with its source,
  group, sample and date; a group under 20 companies (or with an unusable
  figure) hands over from country to region to global and says why.
  Decisions with the user (2026-10-07): growth is the country's nominal GDP
  growth (industry growth averages run far too high); debt starts at the
  industry's listed-company leverage, all senior (no free source publishes
  buyout leverage). Size is asked (EBITDA) but no free source splits the
  figures by size, which the screen says. Deal -> Inputs opens with
  "Starting point" (country, industry, currency, size, "Use sourced
  figures"), labels the figures illustrative until then, and a "Starting
  figures" tile shows each sourced figure beside the deal's own with "Use".
  The defaults registry (`benchmarks/registry.py`, checked by
  `tests/test_defaults_registry.py`) gives every deal input and Setting a
  basis; the illustrative Settings left are pinned there, owned by 4.4
  (ranges, correlations, scenarios) and 4.5b (fees, amortisation).

### 4.4 Risk ranges, correlations and scenarios by region
- **Claude does:** uncertainty ranges per region, sector and size from real
  year-to-year variation; correlations per region; scenario presets built from
  each region's own past recessions and inflation spells (dates listed); all
  sourced.
- **Done when:** UK and India deals get different sourced ranges; each scenario
  lists its historical periods; all correlation matrices pass validity (test).
- **Built as (done):** `benchmarks/history.py` reads Damodaran's archive of
  past January editions (EV/EBITDA since 2011, EBITDA/sales since 2013, gross
  margin since 2017, for all eight groups; the file for year NN is the
  January NN+1 edition) and the IMF's growth and inflation and the BIS's
  monthly policy rates (averaged over full years) since 2000; the nightly
  `benchmarks-refresh` fills it a dozen archive files a call (`more`,
  `--repeat`), reading each file once. `benchmarks/risk.py` turns it into
  every Monte Carlo Setting, each with source, group, sample and years:
  means are the deal's sourced starting figures; spreads are the industry's
  own year-to-year spread of EV/EBITDA and gross margin (closest group with
  six usable years), the country's nominal GDP growth and policy-rate
  changes; correlations per group are rank correlations of yearly changes
  pooled over industries, shrunk toward none only if needed to be valid
  (none needed on the recorded data); recession, stagflation and bull are the
  country's weakest-growth, highest-inflation and strongest-growth fifth of
  years since 2000, listed as periods, moving each mean by what happened in
  them. `GET /api/benchmarks/risk`; the Monte Carlo rail's "Sources" group
  applies them as Settings ("Use sourced figures"), the Scenarios step lists
  every figure's source and each preset's years. Decided in the session:
  the simulation's means stay Settings (applied with the rest) rather than
  following the deal, which keeps the screen's agreed stale behaviour; no
  free source splits by size or gives credit spreads, which the screen says.

### 4.5 Optional reference library (deals and base rates)
- **Claude does:** reference transactions with inclusion rules for balanced
  coverage (regions, sizes, sectors, eras, successes and failures), every
  figure sourced, review screen with two-person approval; the 4 inception deals
  as examples; cited rating-agency default and recovery base rates by region,
  rating band and year; coverage page; an admin switch that hides the library
  without breaking anything.
- **Done when:** with the library off, every screen and model still works (e2e);
  base rates show citations; the coverage page reports counts.
- **Split, like 4.1** (agreed with the user, 2026-10-07):
  - **4.5a (done):** the switch, the base rates and the coverage page. A
    **Library** tab (Base rates, Examples, Coverage; `library/`,
    `/api/library*`). **Base rates** are transcribed, cited tables
    (`library/base_rates.py`): S&P Global Ratings' 2024 global default study
    -- annual default rates by rating category 1981-2024 (Table 3) and by
    grade (Table 1), the speculative-grade rate by region (Table 5: U.S.,
    Europe, emerging, other developed), cumulative rates by rating after
    1-15 years globally and for the U.S., Europe and emerging markets
    (Tables 24, 25) -- and Global Credit Data's 2020 LGD report on bank loans
    to large corporates (recovery by seniority and collateral, region and
    year of default, 11,527 borrowers). Each edition records when it was
    last confirmed as the newest free one; a year later the screen says so
    and `live.yml` fails (`ops/check_base_rates.py`). Decided with the user:
    the figures are fixed editions, not live data (no free feed exists).
    **Coverage** counts the library's deals by region, size, sector, era and
    outcome, empty buckets shown, and what each base-rate table spans; the
    four inception deals are counted apart as unsourced examples and listed
    on Library -> Examples. **The admin switch**: administrators
    (`FSE_ADMINS`, Clerk user ids) show or hide the library for everyone in
    the environment (`app_flags`, migration 0014, one audit entry per
    change); `FSE_EXAMPLE_LIBRARY=0` still forces it off. Off, the tab, the
    examples and the library endpoints go and nothing else changes
    (`tests/test_library.py`, `web/e2e/library.spec.ts`).
  - **4.5b (done):** reference transactions with inclusion rules for balanced
    coverage (regions, sizes, sectors, eras, successes and failures), every
    figure sourced from filings; the review screen with two-person approval
    (two administrators other than the proposer); fees and amortisation
    sourced from the transactions' filings (the defaults registry's last
    `pending` Settings); the coverage page counts them as the sourced
    collection already reserved for them. Built: `library/references.py`
    (the rules: a figure is one place in a filing or a sum of cited pieces;
    sources must be filings on a regulator's, register's or exchange's own
    site; required figures, multiples, fees and dates checked; balance is
    advisory: a bucket over half the library once it holds six), ten
    transactions proposed by the repository (`library/reference_deals.json`,
    every figure read from SEC filings: HCA, Toys "R" Us, Dollar General,
    Domino's, Gymboree, Dun & Bradstreet, NXP, Masonite, Avago, Focus Media;
    1998-2019; US, Europe, other developed and emerging; seven exits and
    three distressed), Library -> Reference deals and Library -> Review
    (`db/references.py`, migration 0015: a proposal joins on the second
    approval by an administrator other than its proposer, one rejection
    with a reason decides it, a finding of the rules blocks approval, a
    corrected version supersedes the approved one), and `library/fees.py`:
    the medians across approved deals, offered on Settings -> Fees as "Use
    sourced figures" (transaction fees 1.24% of value over six deals,
    financing fees 3.07% of debt over seven, senior amortisation 1% a year
    over three). Decided in the session: the sourced fees are offered
    Settings like 4.4's ranges, never new factory defaults, so no saved deal
    moves; a deal's outcome may be shown by another company's filing where
    its own filings stop (Focus Media's relisting). Gaps the coverage page
    shows: no deal under $100m and none since 2020.

### 4.6 Model validation framework
- **Claude does:** reports on calibration (outcomes inside predicted ranges as
  often as claimed) and bias, split by region, sector, size and era; uses the
  library plus opt-in anonymised plan-vs-actual results, always tested on newer
  data; published in the methodology; deal results always show IRR, MOIC,
  probability of loss, downside case, coverage and default risk together.
  Report generation runs as a scheduled job.
- **Done when:** a report is generated on staging with all splits; anonymisation
  is tested; the deal summary shows the full metric set.
- **Done (2026-10-08).** `validation/` (cases, report, groups, run), the
  nightly `validation-report` task (`scheduled.yml`, and `staging.yml` after
  every staging deploy, which requires three checks and four splits),
  `GET /api/validation/report`, Backtest -> Validation, migration 0016
  (`deals.validation_opt_in`, `deals.actuals_first_saved_at`,
  `validation_reports`, two audit actions), `docs/methodology.md` "Model
  validation". Three checks: the deal summary's **default risk** against
  distress within five years on the approved reference transactions (the
  app's answer for each deal's filed headline figures, everything else at
  defaults); the plan's **IRR range** (central 50/80/90% of paths, Wilson
  intervals) and **probability of loss** (Brier, z) on users' deals that
  opted in and have exited. Decided in the session: **"newer data" is
  strict** -- a case is out of time only when its outcome became known after
  everything its prediction read (for default risk, after 2025: S&P's study
  sample ends 2024 and Damodaran's January 2026 table reads 2025; for a
  user's deal, a plan version saved before its actuals were first entered),
  so the ten repository transactions are all in-sample and the headline
  waits for newer outcomes; in-sample results are shown apart, never as the
  headline. **Anonymised by aggregation**: no per-deal record is stored;
  the report holds counts and rates per group, a group with 1-4 users' deals
  is hidden, and the next-smallest groups are hidden with it until the
  hidden ones hold five (so subtraction can't single one out). Opt-in is
  per deal, off by default, on Backtest -> Plan and actuals; `/privacy`
  says what it does. The deal summary now shows IRR, MOIC, probability of
  loss and downside (the simulation's 5th percentile; the summary runs the
  simulation itself when none is current), interest cover and default risk
  together (the deal answer's `credit` block, `core.risk_warnings.credit_view`;
  the simulation's `p_loss`). No engine version moved: both are new outputs.

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
- **Done (2026-10-08).** `ml/evaluation/` (`harness.py`: walk-forward time
  splits, S&P's four regions, a baseline per model and the verdict "beats the
  baseline", "does not beat the baseline" or "not enough data" under five
  cases; `card.py`: the one card template; `anomaly.py`, `surrogate.py`: each
  model's cases, baseline and statistics; `python -m ml.evaluation evaluate |
  train`), the registry `ml/registry.json` (every trained file, hashed in its
  card), cards in `ml/cards/` and `docs/model-cards/`, the gate
  `ops/model_gate.py` (in tests.yml's required ml job: a statistic worse than
  the base branch's beyond its tolerance, overall or in any region, fails the
  pull request), and `ml.yml` (evaluates on pull requests; trains by hand into
  a scratch directory and uploads files and card for review). Results: the
  **risk score**, out of time (refitted on deals before 2008 and 2010, 14 US
  deals), ranks every distressed deal first (AUC 1.0) but so does leverage
  alone, so it **does not beat the baseline**, and every other region has no
  data. The **live sliders**, on 230 deals built from the sourced figures of
  23 economies and ten industries, are 0.54 points off the median IRR against
  0.79 for a live 200-path simulation and beat it in every region; only 23 of
  the 230 lie inside the network's training ranges (5.8 retired the network), and its
  95th percentile is 3.1 points off in emerging markets. Nothing the app shows
  changed; 5.2 onward use the harness and show "not enough data" where a card
  says so.

### 5.2 Deal risk score from market data
- **Claude does:** rebuild `ml/anomaly_detector.py` to compare a deal with
  companies and deals like it in the same region, sector and size (4.3, 4.4,
  plus the library when on); "based on N companies in [region, sector]"; off
  where N is too small.
- **Done when:** it works with the library off; each region has its own
  comparison group (test); thin regions show "not enough data".
- **Done (2026-10-08).** `ml/anomaly_detector.py` no longer trains anything:
  it compares the deal's leverage, entry multiple and EBITDA margin with its
  industry's listed companies **in its own region** (Damodaran's averages,
  4.3: the country's own file or its region, at least 20 companies, **never
  the global group**), each as spreads from the industry's (how much the
  region's industries differ: 1.4826 x their median absolute deviation, at
  least ten industries). "Based on N companies in [group, industry]"; a
  region with too few companies says "not enough data" and names it. The
  score (how far leverage and price sit above the industry's) is **shown only
  where its card says it beats leverage alone** (the phase rule): the new
  card `deal_risk` (ml/evaluation/deal_risk.py) scores the ten sourced
  reference transactions against the recorded averages; eight have a peer
  group, and in the US (five, two distressed) it ranks distress better than
  leverage (AUC 0.67 against 0.33), so US deals show it; every other region
  has one case and shows "not enough data". With the library on, the
  approved reference transactions in the same S&P region and sector are
  listed, the same size first; off, nothing else changes (tests). The old
  card `anomaly_detector` and its trained files are retired (the CI gate
  reports the removal); the score needs no ML packages, so it shows on every
  server with stored averages. Browser tests replay
  `web/e2e/fixtures/deal-risk.json` from `python -m tests.e2e_deal_risk`.

### 5.3 Distress predictor
- **Claude does:** rebuild `ml/distress_model.py` from coverage and leverage
  paths, calibrated to cited default base rates by region and rating band;
  yearly distress probability on Debt and in Monte Carlo; model card.
- **Done when:** implied default rates match base rates for comparable bands
  within the card's tolerance; higher leverage raises risk (test).
- **Done (2026-10-08).** `ml/distress_model.py` trains nothing (the 39
  hand-entered cases and the logistic regression are gone). Each year of the
  deal model's run, and of every simulated path, gets a **rating band**: the
  weaker of EBIT / interest through Damodaran's coverage table and debt at
  the start of the year over the year's EBITDA through S&P's Corporate
  Methodology (January 2024, the free Maalot copy: Table 17's standard
  volatility bounds, then Table 3 with the deal's **business risk profile**,
  a new deal input, 4 "fair" by default, decided with the user), folded to
  S&P's letter grades. The band's **forward default rate at that age** comes
  from S&P's 2024 study, Table 25 for the US, Europe and emerging markets,
  Table 24 elsewhere; a deal held in one band defaults exactly as the table
  says (the card's calibration set: 350 cases, every gap 0.00 points, within
  the 0.01 tolerance; the region-blind baseline is off by up to 19.39).
  Higher leverage and a weaker business both raise the chance of default
  (tests). The card `distress` tests it on the ten sourced reference
  transactions against the deal's year-one default risk: it does **not**
  beat it (AUC 0.43 against 0.52; US, six deals, 0.25 against 0.375:
  Dollar General at 10x and Avago at 13x exited well, Toys "R" Us and
  Gymboree at about 6.5x failed years later). **Decided with the user: the
  phase rule holds**, so Deal -> Debt and Monte Carlo -> Distribution show
  each year's coverage, leverage and band (the simulation: the share of
  paths in each band) everywhere, and the chances of default nowhere today
  ("not enough data"); they appear by themselves in a region once its card
  says it beats the baseline.

### 5.4 Multiple predictor by region
- **Claude does:** rebuild `ml/multiple_predictor.py` on regional listed-company
  multiples (4.3); ranges with comparables and "use suggestion".
- **Done when:** ranges contain actual multiples for most held-out companies per
  region (card); it works with the library off.
- **Done (2026-10-08).** `ml/multiple_predictor.py` trains nothing (the 25
  typed-in rows and the gradient boosting are gone). A deal's industry is read
  among its region's listed companies (Damodaran's averages, 4.3: the
  country's own file or its region, at least 20 companies, **never the
  global group**, as 5.2), and the ranges come from how that group's
  industries' EV/EBITDA moved over the same number of years in Damodaran's
  archive (4.4, every edition since 2011): each move fitted on the
  industry's gap from the region's median industry (expensive industries
  cheapen, cheap ones catch up), the range the latest multiple moved by
  that line plus the 10th to 90th percentiles of what it missed. Entry is
  the year after the latest edition, exit that plus the hold. The new card
  `multiples` (ml/evaluation/multiples.py) tests both walk-forward, a cutoff
  a year from 2014, on every industry of every group (13,544 industry-years,
  `ml/evaluation/data/multiple_history.json` from `python -m
  tests.ml_multiple_history`): about 73% of held-out multiples fall inside
  the 80% range in every region (entry 67-78%, exit 72-74%), and the range
  beats the region's whole market on the interval score in every region and
  both sets, so it shows wherever a peer group is large enough (the exit
  range for holds of three to seven years, the horizons tested). Deal ->
  Returns has a Multiples tile: the ranges, "Use suggestion" (sets the entry
  and exit multiples), the industry in every region, its sector in the
  region and, with the library on, the reference transactions like it with
  their entry multiples; off, only that list goes (tests). Browser tests
  replay `web/e2e/fixtures/multiples.json` from `python -m
  tests.e2e_multiples`.

### 5.5 Growth calibrator by region
- **Claude does:** rebuild `ml/growth_calibrator.py` on the filings store and
  regional data (replacing US-only SimFin and fixed averages); "calibrate from
  sector and region" on Monte Carlo.
- **Done when:** ranges match observed growth on held-out years per region
  (card); using them changes the simulation.
- **Done (2026-10-09).** `ml/growth_calibrator.py` is rebuilt on real
  filings: every SEC filer's yearly revenue from the SEC's XBRL frames (the
  filings the company search reads, 4.1; 8,451 companies, 2008-2025, US
  GAAP and IFRS in every currency filed), placed by business address in
  S&P's regions and by SIC code in GICS sectors, financials left out, and a
  floor of 50 million US dollars of revenue (ECB yearly rates). A case is
  a company's yearly growth over a hold of 3 to 7 years **over its own
  country's nominal GDP growth** in the same years (IMF, 4.2's economies;
  the region's median elsewhere); the range is the 10th to 90th percentile
  of those excesses in the deal's region, sector and hold (the region's
  every sector under 30 companies), centred on the deal country's IMF
  projection, and becomes the normal draw whose central 80% it is
  (`mc_growth_mean`, `mc_growth_std`). The card `growth`
  (ml/evaluation/growth.py) is strictly out of time (each span predicted
  from spans that had ended by its start, 2012-2022): 77% of 67,933 held-out
  company spans fell inside the 80% range (US 77%, Europe 83%, emerging
  78%, other developed 74%) against 27% for the economy-wide range in force
  since 4.4, and it beats that baseline in every region. Monte Carlo's
  Revenue growth rail group has "Calibrate from sector and region" and the
  Scenarios step a tile with the range, its sample and the card's result;
  a test proves the calibrated Settings change the simulation and that its
  draws' 10th and 90th percentiles are the range. Written by `python -m
  tests.ml_firm_growth` (about half an hour; `ml/growth_ranges.json` with
  it); browser tests replay `web/e2e/fixtures/growth.json` from `python -m
  tests.e2e_growth`.

### 5.6 Driver explanations
- **Claude does:** compare `ml/shap_attribution.py` with the current drivers
  view; ship per-deal explanations if they add value, precomputed or light
  enough for free hosting; otherwise remove with a note.
- **Done when:** written comparison; if shipped, explanations add up to the
  prediction (test).
- **Done (2026-10-10):** the comparison is `docs/driver-explanations.md`.
  Shipped: the Monte Carlo answer's `explanations`
  (`analytics/driver_attribution.py`), the worst and the best 5% of paths'
  mean IRR split between the simulation's five drivers by Shapley values of
  the simulation itself, so they add up to the simulated figure exactly
  (`tests/test_driver_explanations.py`, to 1e-12, every debt structure; a run slower than 20 s answers without them) with
  no trained model and no new package; Monte Carlo -> Drivers shows them
  beside the rank correlations. `ml/shap_attribution.py` is removed: fitted
  to one deal shape, 0.61 points off, explaining against an arbitrary average
  deal, and 375 MB at peak on a 512 MB server.

### 5.7 Economic regime by region, scheduled
- **Claude does:** rebuild `ml/macro_regime.py` on 4.2's country data (replacing
  US-only series, including one discontinued in 2022) for the US, eurozone, UK,
  Japan, China, India and global; monthly retraining in GitHub Actions;
  Scenarios shows the deal region's current regime; correlations refresh from
  the same data.
- **Done when:** each region shows a dated regime; an India deal shows India's
  regime (test).
- **Done (2026-10-11):** `ml/macro_regime.py` names each region's year a
  recession, stagflation, bull or base year from the IMF World Economic
  Outlook's growth and inflation (4.2's source; the euro area and the world
  are the IMF's own aggregates), each against the median of the region's
  previous five years, on cut-offs from its earlier years. Nothing is
  trained, so "monthly retraining" is a monthly refit: GitHub Actions'
  nightly `economy-refresh` task (`scheduled.yml`; `staging.yml` after each
  deploy) runs it after the economic data, and it reads the figures again
  on the first run of a calendar month and stores the fit (inside the
  existing task because this machine's GitHub token may not change workflow
  files; a step of its own is optional). Monte Carlo -> Scenarios shows the deal region's regime
  with its year and the day read, and every region's beside it
  (`tests/test_macro_regime.py`, `web/e2e/regime.spec.ts`: an India deal
  shows India's). The card (`docs/model-cards/regime.md`) replays the IMF's
  spring and autumn projections for 2000-2022 against the final figures:
  75% of years named right in spring against 52% for "the same as last
  year", and every region beats it. **Correlations**: the ones the Monte
  Carlo screen offers (4.4) are already measured on the same IMF series
  (with the BIS's rates and Damodaran's archive) each time they are asked
  for, so `ml/correlation_updater.py` (US-only FRED, one series licensed
  against reproduction, never wired in) is removed with the hidden Markov
  model and its two packages.

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
- **Done (2026-10-11), by another road, decided with the user.** Measured
  first: the simulation itself takes about 1.3 ms for 5,000 paths (63 ms at
  the size caps) and is 0.18 points off the median IRR on the card's 230
  regional deals, where the network was 0.54; a retrained network would
  have needed about 95 inputs to cover facilities, tax rules and leases and
  would still have warned beyond its range. So **nothing is trained**: the
  Live step reruns the simulation on up to 8,000 fixed, seeded paths per
  move (`core/live.py`), for the deal exactly as Run Monte Carlo treats it,
  with the six sliders as inputs (a listed structure is scaled by the debt
  slider and its floating references move with the rate slider). No trained
  range, so no warning; the sampling error is shown as a 95% interval for
  the median. The network, its files, PyTorch, scikit-learn, joblib and
  pyarrow are gone, and the API image has no ML build argument. **Card**
  (`docs/model-cards/surrogate.md`, id kept so the gate compared it with
  the network's): on 230 varied deals from every region (own entry
  multiple, hold, fees, tax preset, leases, five debt structures) the
  median IRR is 0.13 points from a 100,000-path run, under 0.18 in every
  region; the gate passed against the network's card on its own sets.
  **Staging check**: the job drill staging.yml runs after every deploy now
  also runs a session of the live sliders on the staging server (40 moves
  and the largest deal) and fails if the API's memory peaked at or above
  the service's 512 MB, the moves added over 32 MB or the middle estimate
  took over 500 ms. `POST /api/montecarlo/live` replaces
  `/api/ml/surrogate`; it has its own rate (180 a minute, two at once across all users) instead
  of counting as a model run.

### 5.9 Personalized defaults
- **Claude does:** move `ml/personalization.py` into the database; learn each
  user's usual assumptions per region and sector; suggest them alongside
  sourced defaults; off switch.
- **Done when:** suggestions use only that user's deals (test); off stops them.
- **Done (2026-10-11).** `ml/personalization.py` (a local SQLite file keyed by
  a hash of the machine, never wired in) is gone, and nothing learned is
  stored in its place: a person's **usual figures** are worked out from their
  own saved deals each time they are asked for (`benchmarks/usual.py`): for
  each of the sourced starting figures' fields, the median across their deals
  in the same S&P region and GICS sector, with how many deals that is and the
  lowest and highest; at least three deals a figure, the deal on screen left
  out. `db.deals.own_deal_inputs` joins the deals to the caller's own account
  row in one statement, so no other account's deal can be in the answer
  (`tests/test_usual.py`: two accounts' deals side by side, a new account,
  an account with none). The off switch is `users.personalized_defaults`
  (migration 0017; on by default) with one history entry per change
  (`suggestions_switched`); off, the statement reads no deal and the API
  answers nothing. `GET`/`PUT /api/account/suggestions`. Screen: Deal ->
  Inputs, a "Your usual" column in the Starting figures tile beside the
  sourced figures, with "Use" per figure and "Use all yours", and the switch
  under Starting point; `/privacy` says what is kept (the choice) and what is
  not (the suggestions). **Not a model, so no card**: it describes the
  person's own deals and predicts nothing; the screen says exactly what it
  is and from how many deals.

### 5.10 Retraining models by hand (parked, not scheduled)
- **Parked by the user (2026-10-08):** "we will add this idea later". Skip it
  when picking the lowest open task; do it only when the user asks.
- **The idea:** the user retrains a model themselves -- Actions -> ml -> Run
  workflow -> choose the model -- reads the card the run produces, and the
  new files reach the app only through a reviewed pull request (the 5.1 gate
  still blocks a worse card). 5.1 already built the `train` job; what is
  parked is making this a routine the user runs: a step-by-step guide (in
  the owner's handbook, 11.4), a way to bring the run's files into a pull
  request without copying by hand, and deciding whether retraining should
  also run on a schedule.
- **Until then:** models are retrained only inside the tasks that rebuild
  them (5.2 to 5.9), by Claude, in the same pull request as their new card.
- **Done when:** the user can retrain a model and open its pull request by
  following the guide, without Claude.

---

## Phase 6 — AI features

### 6.1 File uploads
- **Claude does:** PDF, Excel and Word uploads to Supabase Storage with size and
  type checks; malware scanning using a free open-source scanner in the job
  runner; private per user or team; signed links; deletion rules; per-user
  quotas to stay within 1 GB.
- **Done when:** upload and download work; others can't access (test); bad or
  oversized files are refused; a quota message appears near the limit.
- **Done (2026-10-12).** Deal → Documents uploads `.pdf`, `.xlsx`, `.docx`,
  `.xls` and `.doc` files (`uploads/`, table `files`). The bytes are sent to
  the API, which streams them to disk, counts them, and puts them in a
  private Supabase Storage bucket itself; a scan in the job runner must pass
  before a file can be downloaded, through a signed link that lasts a minute
  and is made only for its owner. **The scanner is structural, not an
  antivirus**, on measurements: ClamAV needed 1,687 MB for a 60-byte PDF
  against the free server's 512 MB, so `uploads/scan.py` refuses whatever can
  act (macros, scripts, embedded objects, DDE, remote templates), whatever is
  encrypted and whatever it cannot read, with pypdf, an XML parser and
  oletools, in a child process with CPU, memory and time limits. A signature
  scan is 12.1's. **The security review changed the design** before release:
  the first version handed the browser a signed upload link (which let more
  be stored than was counted, and a file be swapped after its scan) and read
  files with patterns (two bypasses, and a 30 KB file that froze the API);
  both are gone, and each of the review's files is a test. Limits: 20 MB a
  file, 100 MB and 40 files a person (the screen warns from 80%), 800 MB in
  all. Deletion: by the owner any time, after 90 days, at once when refused;
  the nightly `job-maintenance` task sweeps. "Private per team" waits for
  teams (7.2): files have one owner today. DEPLOY.md "File uploads" has the
  set-up (a Supabase project for uploads, apart from the backups').

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
  help and glossary; CI fails on missing translation keys (2.3b's
  `tests/test_translations.py` already fails when a language file's keys or
  placeholders don't match English's).
- **Three things 2.3b left safe only while English is the one catalogue**, and
  which the second one makes real:
  - `lib/format.ts`'s `setMissingText` and `lib/export.ts`'s
    `setDownloadFailedMessage` are module values, like 2.3a's number style. On
    the server they are always English today, because nothing there knows the
    account; with two catalogues, two requests rendering at once could hand
    each other's wording over. Move them into the provider, or keep them
    client-only.
  - `browserLocale()` (components/shell/I18nScope.tsx) is `en` on the server
    and the browser's own language in the browser, so the signed-out pages'
    server HTML and first client render would disagree once the words differ.
  - `setDownloadFailedMessage` fills `{status}` with one `String.replace`, not
    ICU, so a translation using the placeholder twice fills only the first.
- **Done when:** e2e key journeys pass in each added language.

### 7.9 Charts in every Excel download
Asked for by the user (2026-10-05): the charts on screen should also be in the
workbooks.
- **Claude does:**
  - native Excel charts (openpyxl's chart objects, no images, no paid
    library), drawn from the workbook's own cells so they stay editable and
    move with the numbers, in every download that has a chart on screen:
    deal (revenue and EBITDA, debt paydown, equity bridge as a waterfall),
    Monte Carlo (IRR and MOIC distributions, the CDF, scenarios, drivers),
    plan vs actual, forecast (statements, the revenue and EBITDA fans);
  - where Excel has no native form, the nearest honest one, said on the
    sheet: fans as stacked areas between percentiles, the heatmap as a
    table with a colour scale, the scatter as a sampled scatter;
  - the Tape look kept to what Excel allows (dark-friendly palette, cyan for
    the answer, green/red for gain/loss), titles and axis labels from
    `web/messages/en.json` like the screen's, money labels from the
    workbook's currency and unit;
  - the web sends which chart goes with which sheet; the API draws it
    (`api/routers/export.py`), so every download shares one chart builder;
  - with 7.3a (live formulas) the charts follow recalculated cells for free.
- **Done when:** each download with a chart on screen opens in Excel with the
  same chart over the same cells (a test reads the chart's series ranges back
  with openpyxl and checks they point at the right cells), and changing a
  cell moves the chart.

### 7.10 Workspaces: LBO and equity research, launcher after sign-in
Asked for by the user (2026-10-08), done out of turn after 4.6: the modes are
grouped into workspaces, and a sign-in offers them instead of opening Deal
inputs.
- **Claude does:**
  - `web/src/lib/nav.ts` `WORKSPACES`: **LBO** (Deal, Monte Carlo, Backtest,
    Settings, Library) and **Equity research** (Forecast; DCF, comps and more
    later). A mode belongs to exactly one workspace, so addresses stay as they
    were (`/deal/inputs`), with no prefix: bookmarks and links keep working;
  - the launcher at `/start` (`components/shell/Launcher.tsx`): a tile per
    workspace with its screens and an "Open" action, the LBO tile naming the
    open deal; the brand mark leads there;
  - **every sign-in lands on the launcher** (`AFTER_SIGN_IN`), the first one
    after the account questions; a visitor sent to sign in from a particular
    screen goes back to it; **the site's root carries on at the last screen
    this browser showed** (`lib/lastScreen.ts`, localStorage), so someone
    still signed in continues where they were, and the root never carries a
    way back to itself through sign-in (`proxy.ts`), so signing in from it
    lands on the launcher too;
  - the top bar's workspace switcher; mode tabs and Alt 1-9 list only the
    current workspace's modes (the account screen keeps the last workspace's);
    Ctrl K searches every workspace and names each result's.
- **Done when:** sign-in (first or again) lands on the launcher, an open
  session's visit to the root returns to its last screen, each workspace shows
  only its own tabs and shortcuts, and search crosses workspaces
  (`e2e/shell.spec.ts`, `e2e/auth.spec.ts`).
- **Parked ideas (the user, 2026-10-08), not scheduled:** a third workspace of
  India-first market-event models, kept as ideas until the user asks:
  (a) NIFTY 50 reconstitution predictor (NSE's published rules as dated data,
  probability of inclusion by simulating prices to the cutoff, passive-flow
  size; other rule-based indices later); (b) NSE F&O positioning screener (OI
  build-up buckets, 52-week and 3-year OI extremes, basis, PCR); (c) IPO
  lock-in expiry calendar (SEBI lock-in rules, holidays, unlock value against
  free float). The user has one of them built in Excel to share. Open
  questions before any of them: NSE data access and its redistribution terms
  (user uploads of the bhavcopy, or a fetch with uploads as the fallback), and
  SME lock-in rules.

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

### 12.9 Licensed market data (corporate credit spreads)
- **You first:** license a corporate bond spread source that allows display in
  the product (ICE, Moody's or another vendor), and set its key.
- **Claude does:** a connector beside 4.2's (`economy/connectors.py`), stored
  and refreshed the same way; corporate spreads by region and rating band next
  to the sovereign spreads 4.2 shows. (Added in 4.2: the free spread series on
  FRED forbid reproduction.)
- **Done when:** US and euro investment-grade and high-yield spreads show with
  their source and date.

---

## Appendix — US and inception-deal assumptions found in the code (2026-09-15)

| Where | Assumption | Fixed by |
|---|---|---|
| 177 places across `core/`, `api/`, `lbo_engine/`, `ml/`, `web/src` | Money shown as "$M" | 2.2 (done: deal currency and unit; CI check against new dollar signs) |
| 11 places in `web/src` | Number formats fixed to `en-US` | 2.3a (done: account locale and grouping; CI check) |
| `ml/edgar_extractor.py` | US SEC only, `us-gaap` tags, USD, US fiscal years | 2.6, 4.1 |
| `lbo_engine/capital_structure.py` `build_simple_two_tranche_structure` | One fixed-rate senior loan (5% amortisation) plus one mezzanine bullet | 2.4a — still the default when a deal lists no tranches; `core/debt.py` now takes any structure |
| `lbo_engine/operating_model.py` and returns | Flat tax, interest always fully deductible | 2.5 |
| `core/config.py` `DEFAULTS` | Growth, margins, multiples, rates, leverage, fees, ranges, correlations, scenario multipliers and the 20% hurdle typed in with no source. Since 4.3 a new deal starts from sourced figures instead and `benchmarks/registry.py` gives every default a basis; since 4.4 the Monte Carlo means, ranges, correlations and presets are sourced per country and industry (the factory values stay the illustrative fallback, labelled); since 4.5b fees and amortisation are offered from the approved reference transactions' filings (`library/fees.py`), so no default is left without a source | 2.1, 4.3, 4.4, 4.5b (done) |
| `simulation/vectorized_simulation.py` `DEFAULT_CORR` | Correlation matrix typed in; used only when no matrix is passed (every app run passes the Settings', sourced per region since 4.4) | 4.4 |
| `core/backtesting.py` `PRELOADED_DEALS` | 4 US mega-deals (2006–2013), unsourced actuals | 2.7 (done: now an optional example library, `core/examples.py`), 4.5a (done: labelled unsourced examples in the Library, counted apart), 4.5b (done: sourced reference transactions beside them, counted apart) |
| `ml/anomaly_detector.py` | 30 US deals plus synthetic; claims "~100"; fixed warning statistics | 2.1, 2.8 (done: its warnings removed; `core/risk_warnings.py` computes them), 5.2 (done: compares the deal with its region's industry averages; the 30 deals are gone) |
| `ml/distress_model.py` | 39 hand-entered cases | 5.3 |
| `ml/multiple_predictor.py` | 25 rows | 5.4 (done: regional listed-company multiples and their archive; the 25 rows are gone) |
| `ml/growth_calibrator.py` | US SimFin, fixed Damodaran averages | 5.5 (done: SEC filers' revenue growth by region and sector over their economy's; the fixed averages are gone) |
| `ml/macro_regime.py`, `ml/correlation_updater.py` | US-only FRED series (incl. ISM PMI, discontinued 2022) | 4.2, 5.7 (done: seven regions from IMF figures, refitted monthly; the FRED models are gone) |
| `ml/surrogate/` | One fixed training deal; PyTorch runtime too large for free hosting | 5.8 |
| `web/src` Backtest screens | Only the preloaded deals can be tested | 2.7 (done: any saved deal) |
| `render.yaml` and DEPLOY.md | Render free database would expire after 30 days; not used (Neon instead) | 1.3 |
