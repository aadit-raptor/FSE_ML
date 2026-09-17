# Deploying

The web app (`web/`, Next.js) runs on **Vercel**; the API (`api/`, FastAPI in
the root `Dockerfile`) runs on **Render**. The browser only talks to Vercel:
Next proxies `/api/*` to the Render URL, so no CORS setup is needed.

```
browser ──> Vercel (Next.js) ──/api/*──> Render (FastAPI + model)
```

Do the API first; the web app needs its URL at build time.

## Environments

There are two copies of the app, both on free plans.

| | Production | Staging |
|---|---|---|
| Git branch | `main` | `staging` |
| Web | Vercel production, https://fse-ml.vercel.app | Vercel preview of the `staging` branch: https://fse-ml-git-staging-aadit7.vercel.app (and a per-commit preview URL). Behind Vercel login |
| API | Render `fse-api`, https://fse-api.onrender.com | Render `fse-api-staging`, https://fse-api-staging.onrender.com |
| `/api/health` | `"environment": "production"` | `"environment": "staging"`; the status bar says `API ok · v0.1.0 · staging` |
| Browser checks | `.github/workflows/live.yml`, daily | `.github/workflows/staging.yml`, after every push to `staging` |

Pull-request previews on Vercel also use the staging API. A preview never
talks to the production API: `web/next.config.ts` picks the API by
`VERCEL_ENV`.

**Variables per environment**

| Variable | Where | Production | Staging | Local |
|---|---|---|---|---|
| `FSE_API_URL` | Vercel (Production scope) | `https://fse-api.onrender.com` (required; the build fails without it) | ignored on previews | default `http://127.0.0.1:8000` |
| `FSE_STAGING_API_URL` | Vercel (Preview scope), optional | — | default `https://fse-api-staging.onrender.com` | — |
| `FSE_CORS_ORIGINS` | Render service | optional | optional | default `http://localhost:3000` |
| `FRED_API_KEY` | Render service | optional (ML image only) | optional | `.env` |
| `FSE_ENV` | Render service, optional | derived: `production` | derived from the `-staging` service name | `local` |
| `RENDER_GIT_COMMIT`, `RENDER_SERVICE_NAME` | set by Render automatically | | | unset |
| `VERCEL_AUTOMATION_BYPASS_SECRET` | GitHub Actions secret | — | lets `staging.yml` open the protected preview | — |
| `SENTRY_DSN` | both Render services, and Vercel (Production + Preview) | set | set | unset (no Sentry) |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Vercel (Production + Preview) | `pk_…` from Clerk; the build fails without it | same | unset = development sign-in |
| `CLERK_SECRET_KEY` | Vercel (Production + Preview) | `sk_…` from Clerk | same | unset |
| `CLERK_ISSUER` *or* `CLERK_PUBLISHABLE_KEY` | both Render services | the Clerk instance (`https://<instance>.clerk.accounts.dev`) | same | unset = development sign-in |
| `FSE_AUTH_DEV` | local and CI only | **never set it** | never | `1` accepts `dev:<name>` tokens |
| `DATABASE_URL` | both Render services | Neon branch `production`, pooled connection string | Neon branch `staging`, pooled | optional: `python -m db.local` prints one; unset = no database |
| `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` | both Render services, and GitHub Actions secrets | shared usage counters, keys `fse:production:` | same database, keys `fse:staging:` | unset = counters in memory (or the database); see "Usage limits" |
| `TEST_DATABASE_URL` | CI (`tests.yml`, a Postgres 17 service) | — | — | optional: tests create and drop their own databases through it |
| `BETTERSTACK_API_TOKEN` | GitHub Actions secret (Better Stack Uptime API token) | monitors, status page, alerts from `live.yml` | alerts from `staging.yml` | — |

**Moving a change through staging**

1. Merge or push the change to `staging` (for example
   `git push origin <branch>:staging`, fast-forward only). Render redeploys
   `fse-api-staging`, Vercel builds the preview, and `staging.yml` waits for
   both to run that commit, then runs the browser checks on it.
2. When the checks are green, open the pull request into `main` as usual.
3. After a merge to `main`, bring staging level again:
   `git push origin main:staging`.

**Free hours:** Render gives 750 instance hours a month across all free
services. Both services sleep after 15 minutes idle, so staging costs only
the time it's being used. Production's uptime check runs every 30 minutes and
keeps it awake roughly half the time (about 400 h a month; see "Monitoring").
Don't add a scheduled job that keeps staging awake, and don't check the API
more often than every 16 minutes: `tests/test_betterstack.py` fails if the
monitor plan leaves less than 250 h for everything else.

Creating the staging API (done once, 2026-09-15): Render → **New → Web
Service**, this repo, branch `staging`, runtime Docker, free plan, health
check path `/api/health`, name `fse-api-staging`.

## 1. API on Render

1. Sign in at <https://render.com> with GitHub and allow access to this repo.
2. **New → Blueprint**, pick the repo. Render reads `render.yaml` and creates
   the `fse-api` web service (Docker, free plan, health check `/api/health`).
3. Leave `FSE_CORS_ORIGINS` and `FRED_API_KEY` empty for now and apply.
4. When the deploy is live, open `https://<your-service>.onrender.com/api/health`.
   It should return `{"status":"ok",...}`. Note this URL.

Every push to `main` that touches the API or model redeploys it
(`buildFilter` in `render.yaml` skips web-only and docs changes).

**Free plan:** the service sleeps after about 15 minutes idle, and the first
request then waits for it to start (often 30–60 s). The web app's status bar
says "starting up, retrying" meanwhile. Change `plan: free` to `starter` for an
always-on instance. Monte Carlo runs are CPU-bound, so a paid instance is also
noticeably faster.

**ML features** (deal risk score, Live sliders, macro regime) are off in the
default image. To enable them, add a build argument `INSTALL_ML=true` in the
service's settings (Render: *Settings → Build → Docker build arguments*). The
image grows by about 1 GB and torch needs more than the free plan's 512 MB of
RAM. Screens hide these features automatically when the API can't run them.
Macro regime also needs `FRED_API_KEY` set on the service and the model
trained (`python -m ml.macro_regime`).

## 2. Web app on Vercel

1. Sign in at <https://vercel.com> with GitHub and import this repo.
2. **Root Directory: `web`**. Vercel detects Next.js; keep the default build
   settings.
3. **Environment variable:** `FSE_API_URL` = the Render URL from step 1, with
   no trailing path (e.g. `https://fse-api.onrender.com`). Apply it to
   Production and Preview. The build fails on purpose if it's missing.
4. Deploy, then open the Vercel URL. The status bar should show
   `API ok · v0.1.0` and Deal → Returns should show the default deal at
   21.2% IRR, 2.61x MOIC.

Pushes to `main` redeploy the web app; pull requests get preview URLs.

If you later change `FSE_API_URL`, redeploy: rewrites are fixed at build time.

## 3. Accounts and sign-in (Clerk)

Every API call except the health checks needs a signed-in user (PLAN.md 1.4).
Free plan, no card.

1. Sign in at <https://clerk.com> and **create an application**. Name it
   `FSE/ML`. Under sign-in options turn on **Email** and **Google**.
2. Copy the two keys from **API keys** (the Next.js snippet shows them):
   `pk_…` (publishable, public) and `sk_…` (secret).
3. **Vercel → fse-ml → Environment Variables** (in the project's main left
   menu, not under Settings): add `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` = the
   `pk_…` key and `CLERK_SECRET_KEY` = the `sk_…` key, both to **Production
   and Preview**. Redeploy: the publishable key is baked in at build time, and
   the build fails on purpose if it is missing.
4. **Render → fse-api → Environment**: add `CLERK_PUBLISHABLE_KEY` = the same
   `pk_…` key (the API reads the instance's address out of it and fetches its
   public keys; it needs no secret). Do the same on `fse-api-staging`. Set
   `CLERK_ISSUER` instead if you ever move Clerk to a custom domain.
5. Open the site, sign up, answer the four account questions (country,
   currency, number format, time zone) and you land on Deal → Inputs.

Nothing about the person is stored in this database except Clerk's user id
(`db/models.py`), and the API never logs it.

While Clerk's keys are `pk_test_`/`sk_test_` the instance is a **development**
one: free, limited to 100 users, and it shows Clerk's development banner.
Moving to a production instance needs a custom domain (PLAN.md 12).

**The development sign-in.** Without a Clerk key the web app and the API fall
back to signing in as `dev:<name>`, which is what local runs and the browser
tests use. It is refused whenever the API is production or Clerk is
configured, and `staging.yml` checks that a `dev:` token is refused there.

## 4. Optional: limit who can reach the API

The API only answers callers with a valid Clerk token, so the model is no
longer open to anyone who finds the URL. Large Monte Carlo runs (up to
1,000,000 paths per request) are still possible for any signed-in account
until usage limits arrive (PLAN.md 1.6). `FSE_CORS_ORIGINS` on Render can
also restrict which websites' browsers may call it directly.

## Rollback

Roll back production when a deploy breaks it. Pick the fastest route that
fits; all are free.

**A. Git revert (no dashboard, always works, leaves a record).** This is the
default.

1. Find the merge commit that broke production: `git log --first-parent main`.
   `/api/health` shows the commit the API runs.
2. `git checkout -b rollback/<what> origin/main`, then
   `git revert -m 1 <merge-sha>`, push, open a pull request titled
   "Roll back …", wait for CI, merge with a merge commit.
3. Render and Vercel redeploy `main` (a few minutes; watch the commit in
   `/api/health` change, and run the `live` workflow).
4. To bring the change back once it's fixed, revert the revert the same way.

**B. Dashboard rollback (fastest, needs your login).**

- *Render:* service `fse-api` → **Events** → pick the last good deploy →
  **Rollback**. Render turns auto-deploy off after a rollback; turn it back on
  (Settings → Auto-Deploy) once `main` is fixed, or the next merge won't
  deploy.
- *Vercel:* project `fse-ml` → **Deployments** → the last good production
  deployment → **⋯ → Instant Rollback**. Hobby can roll back only to the
  previous production deployment. New pushes to `main` don't go live until
  you promote one (**⋯ → Promote**) or undo the rollback.

Afterwards, still revert the bad change on `main` (route A) so the code and
the live site agree.

**Rollback drill log**

| Date | What | How | Result |
|---|---|---|---|
| 2026-09-15 | Drill for PLAN.md 1.1: rolled production back from PR #13 (53a68f2) to the code before it | Route A: PR #14 reverted the merge; merged 17:53 UTC | Render deployed 17:54 UTC, Vercel 17:54 UTC. `/api/health` went from `{"environment":"production","commit":"53a68f2…"}` to the old `{"status":"ok","version":"0.1.0"}`; the deal still showed 21.2% IRR / 2.61x; the live environment check failed as expected (old code). Restored by reverting the revert (PR #15) |

## Monitoring

Set up in PLAN.md 1.2, all on free plans.

| What | Where | Notes |
|---|---|---|
| Errors (API and web) | Sentry organization `aadit-xc`, project `fse-api`; `SENTRY_DSN` holds that project's DSN on both Render services and in Vercel | API: `api/observability.py`; web: `web/src/lib/monitoring.ts`. Browser reports are titled `API <status> on <path>` and tagged `api_path`. Environment `production`, `staging` or `local`; release = git commit |
| Logs | Render → service → **Logs** | One JSON line per request: `ts` (UTC), `request_id`, `method`, `route` (template, e.g. `/api/edgar/{ticker}`), `status`, `duration_ms`, `model_ms`; plus a `model_run` line per model run. Health checks aren't logged |
| Uptime and status page | Better Stack, kept in `ops/betterstack.py` | Synced by `.github/workflows/monitoring.yml` when that file changes on `main` or `staging` (or run it by hand). Status page: `https://fse-ml.betteruptime.com` |
| Staging alerts | `.github/workflows/staging.yml` | Staging has no scheduled check; each deploy is checked as soon as it is live and a failure raises a Better Stack incident |
| Wrong answers in production | `.github/workflows/live.yml` (daily) | Raises a Better Stack incident when the browser checks fail |

**What is never sent.** Sentry events carry the error, the route and the
request ID. Request and response bodies, query strings, cookies, headers,
local variables, IP addresses, user details, clicks and console output are
stripped (`scrub_event` and `beforeSend`), so no deal contents leave the app.
Logs have no bodies or query strings either.

**Following one request.** The web app sends a fresh `X-Request-ID` with each
API call; the API returns it, logs it and tags its Sentry events with it, and
the browser tags its own Sentry event for a failed call with the same ID. In
Sentry search `request_id:<id>`; in Render's logs search for the ID.

**Times.** Everything is stored and logged in UTC (ISO 8601 with `Z`).
Screens show times in the viewer's time zone (`formatForViewer` in
`web/src/lib/monitoring.ts`; the status bar's tooltip shows the API's clock).

**Uptime checks and free-plan wake-ups.**

| Monitor | URL | Every | Alerts after |
|---|---|---|---|
| Website | `https://fse-ml.vercel.app/healthz` (keyword `"service":"FSE/ML web"`) | 3 min | 2 min failing |
| Model API | `https://fse-api.onrender.com/api/health` (keyword `"status":"ok"`) | 30 min | 60 s timeout + 4 min rechecking |

A check that reaches a sleeping API waits while it wakes (about a minute);
Better Stack only opens an incident if it is still failing after the
confirmation period, so a normal wake-up never alerts but an API that stays
down does. Checking every 30 minutes lets the free API sleep between checks.

**Test error.** Outside production, `GET /api/debug/error` raises a
deliberate error (at most one a minute) and returns its request ID:
`curl -H "X-Request-ID: test-error-0001" https://fse-api-staging.onrender.com/api/debug/error`.

**Alert drill log**

| Date | What | Result |
|---|---|---|
| 2026-09-15 | Test error for PLAN.md 1.2: `GET /api/debug/error` on staging with `X-Request-ID: drill-1-2-sentry-2` (20:02 UTC) | Sentry issue FSE-API-1, tags `request_id=drill-1-2-sentry-2`, `environment=staging`, `release=af4315c`; request section only method and URL |
| 2026-09-15 | Broken-deploy drill for PLAN.md 1.2: commit 33c54f4 made `/api/deal/run` raise; pushed to `staging` 20:07:45 UTC | Live on staging 20:08:30 UTC; `staging.yml` quick API check failed and raised Better Stack incident 1015977762 ("Staging API broken"); alert email arrived 20:09 UTC (**under 1 minute**). The browser checks' failed call reached Sentry from the browser (FSE-API-4) and from the API (FSE-API-2) with the same `request_id` `f46a243bbd804a0eb63deda53969c8a9`. Restored by reverting 33c54f4 |

The drill also found that the live browser checks sent the Vercel bypass
secret to every host the page called, which leaked it to Sentry and broke
the browser's error reports; `web/e2e/live.spec.ts` now sends it to the site
only and fails if it reaches any other host.

Accounts (PLAN.md 1.4) broke two checks without breaking the site, fixed on
2026-09-17. Signed-out pages answer 404 to a monitor and redirect a browser
to Clerk, so the website monitor was down for a day: it now checks
`/healthz` (`web/src/app/healthz/route.ts`), which the sign-in proxy leaves
open and which never calls the API (that would wake Render). And the bypass
secret added to the site's own requests followed the redirect to Clerk,
because Playwright keeps changed headers across redirects: the live checks
now trade the secret once for Vercel's bypass cookie (`/healthz`,
`x-vercel-set-bypass-cookie`) and send API requests with `maxRedirects: 0`.

## Database

Set up in PLAN.md 1.3, on Neon's free plan.

| | Production | Staging |
|---|---|---|
| Neon | project `fse-ml`, region AWS us-east-2 (Ohio), branch `production` (the default branch) | branch `staging` of the same project (never auto-deletes) |
| Render variable | `DATABASE_URL` on `fse-api` | `DATABASE_URL` on `fse-api-staging` |
| Checked by | `live.yml`, daily (`ops/check_database.py`) | `staging.yml`, after each deploy |

`DATABASE_URL` is the **pooled** connection string (host contains `-pooler`).
Migrations switch to the direct host by themselves (`db.engine.direct_url`).

**Schema changes deploy themselves.** Render's free plan has no pre-deploy
command, so the API applies pending migrations the first time a request needs
the database in each process (`db/migrate.py`, under a Postgres advisory
lock). The first request after a deploy (normally `staging.yml`'s or
`live.yml`'s database check) pays for it. Every migration must work with the
code before it too, because the old deploy keeps serving until the new one is
live: add columns and tables first, remove them in a later release.

**Free compute hours.** Neon's free plan gives 100 compute hours a month and
suspends the compute after 5 minutes idle; the next connection wakes it (under
a few seconds). The API is built so that only real use wakes it:

- `/api/health` (called by Render every few seconds and by the uptime monitor)
  never queries the database. Its `database` block is the last state this
  process saw: `unchecked`, `ok` or `error`, plus the migration state.
- `/api/health/database` connects, applies migrations if needed, measures the
  database size and stores the reading in `storage_checks`. It reuses its
  result for 10 minutes, so calling it repeatedly doesn't keep the compute
  awake. The workflows call it once a day (production) and per staging deploy.
- The API opens no connection at start-up, so a free Render instance waking
  up doesn't wake Neon.

**Waking from idle.** Connections cut when the compute suspends are detected
before use and replaced; opening a connection retries with backoff for about
15 seconds. If the database stays unreachable, endpoints that need it answer
`503` ("The database is unavailable; try again shortly") and the model
endpoints keep working. `tests/test_database.py` proves both with a proxy that
cuts connections the way a suspend does.

**Storage.** The free plan has 0.5 GB per project. The check warns at 80%
(410 MB): the response carries `"warning": true`, the API logs
`database_storage_warning` and sends one Sentry warning a day, and the
workflow fails and raises a Better Stack incident. The size is this branch's
`pg_database_size`; branches share unchanged data, so production's reading is
the one that matters. Readings older than 90 days are pruned.

**Latency.** The API runs in Render's Oregon region and the database in Ohio,
roughly 50–70 ms per round trip. Keep each request to a few queries (load a
deal in one query, not one per row). Moving both to one region means a new
Render service or Neon project; PLAN.md 12.7 covers regions.

**Backups and restore** come in PLAN.md 1.8. Until then, Neon's free plan can
restore a branch to a recent point in time from its console (**Branches →
branch → Restore**).

## Usage limits

Set up in PLAN.md 1.6 (`api/limits.py`, `api/usage.py`). Upstash Redis free
plan: one database, shared by both environments (keys start `fse:production:`
or `fse:staging:`).

| | Production | Staging | CI |
|---|---|---|---|
| Variables | `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` on `fse-api` | the same on `fse-api-staging` | GitHub Actions secrets of the same names (`tests.yml`, `core` job) |
| Checked by | `live.yml`, daily | `staging.yml`, after each deploy | `tests/test_limits.py` round trip |

**The limits** (a refusal always says what was limited and when to try again,
with `Retry-After`):

| Limit | Value | Answer |
|---|---|---|
| Requests per signed-in user | 240 a minute | 429 |
| Model runs per user (simulations, backtests, forecasts, ML, EDGAR) | 30 a minute, 1,000 a day | 429 |
| Requests per network address | 1,200 a minute | 429 |
| Refused sign-ins per address | 60 a minute | 429 |
| Simulation size | 100,000 paths (Monte Carlo, backtest); 200,000 (forecast) | 422 |
| Request body | 1 MB | 413 |
| Simulations at once | 1; others wait up to 30 s | 503 |
| Simulation run time | 100 s | 504 |

Health checks are never limited. Size: a 15-year Monte Carlo run peaks at
about 1.35 KB a path (100,000 paths: 134 MB; four scenarios: 160 MB) and the
process uses about 200 MB, so one run at the cap fits Render free's 512 MB
and two would not. To change a limit, edit `api/limits.py` (and
`web/src/lib/limits.ts` for the path cap).

**How counting stays inside 500,000 commands a month.** Counting happens in
the API's memory. Only the per-day run counts are shared, and they go to
Redis in one script call every 5 minutes, only when they changed; an idle API
sends nothing. Keys hold a hash of the user, never the id. Estimate at the
target traffic (`python -m api.usage`, pinned by
`test_monthly_redis_estimate_holds_at_target_traffic`):

| | Daily active users (45 active minutes each) | Commands a month |
|---|---|---|
| Production | 200 | 124,992 |
| Staging | 20 | 24,552 |
| CI | | 3,000 |
| **Total** | | **152,544 (31% of 500,000)** |

Whatever the traffic, a hard **daily budget** holds: each sync also counts
its commands in Redis (conservatively, every command inside the script), and
past 8,000 a day on production, 2,000 on staging, 1,000 anywhere else, that
process uses the database for the rest of the UTC day. The budgets add up to
341,000 a month at most. `/api/health/limits` shows the store in use, whether
Upstash answers (a PING, not billed, reused for 10 minutes) and today's
command count as last seen.

**When Redis is down** (or over budget) the counts go to the `usage_counters`
table in Neon instead; if that fails too they wait in memory for the next
sync. Limits keep working throughout: one free instance enforces them
exactly from memory. With several instances (phase 12), per-minute limits
become per instance and daily ones are shared within 5 minutes.

**Browser tests** raise every count limit with `FSE_LIMITS_MULTIPLIER=20`
(`web/playwright.config.ts`); the API ignores it in production.

**Addresses.** uvicorn takes the client address from `X-Forwarded-For`
(`--proxy-headers`). Through the Vercel proxy that is the visitor's address
if Vercel passes it on; otherwise many visitors share Vercel's, which is why
the address limits are generous and signed-in users are limited per account.

## Checking a deploy

The CI `docker` job builds this same image, starts it with a host-assigned
`PORT`, and checks the default deal IRR (21.16%), a seeded Monte Carlo run
and an Excel export. The `e2e` job runs the browser tests against the API and
a production build of the web app. It also starts the image with a Postgres
17 database and checks that `/api/health/database` migrates it and reports
storage.

`/api/health` returns `environment` (production, staging or local) and
`commit` (the git SHA Render built, `null` locally).

**Live checks and sign-in.** Since accounts arrived, the daily production
checks (`live.yml`) and the per-deploy staging checks run **signed out**:
they check the web app's `/healthz`, the API's `/api/health` through the
website's proxy (and that it is the expected environment), that the site
sends a visitor to sign-in, that the API answers 401 without a token, and
that staging refuses a `dev:` token. The status bar can't be checked signed
out, since no screen renders. Checking the
live model output again means a Clerk test account whose credentials live in
GitHub secrets; the model itself is checked on every pull request by the
`core`, `e2e` and `docker` jobs.
