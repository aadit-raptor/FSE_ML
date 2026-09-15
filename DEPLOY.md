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

## 3. Optional: lock the API to the web app

The API is public. Anything a visitor can do in the web app they can do
directly, including large Monte Carlo runs (up to 1,000,000 paths per
request). If that matters, set `FSE_CORS_ORIGINS` on Render to your Vercel
domain (this only stops other websites' browsers, not scripts), and consider
Render's paid plans or a lower path cap in `api/schemas.py` (`MCInputsIn.n`).

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
| Website | `https://fse-ml.vercel.app/deal/inputs` (keyword `FSE/ML`) | 3 min | 2 min failing |
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

## Checking a deploy

The CI `docker` job builds this same image, starts it with a host-assigned
`PORT`, and checks the default deal IRR (21.16%), a seeded Monte Carlo run
and an Excel export. The `e2e` job runs the browser tests against the API and
a production build of the web app.

`/api/health` returns `environment` (production, staging or local) and
`commit` (the git SHA Render built, `null` locally).
