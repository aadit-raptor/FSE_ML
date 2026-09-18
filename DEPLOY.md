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
| `FSE_CORS_ORIGINS` | Render service, optional | default `https://fse-ml.vercel.app`; replaces it when set | default none (previews use the proxy) | default `http://localhost:3000` |
| `FRED_API_KEY` | Render service | optional (ML image only) | optional | `.env` |
| `FSE_ENV` | Render service, optional | derived: `production` | derived from the `-staging` service name | `local` |
| `RENDER_GIT_COMMIT`, `RENDER_SERVICE_NAME` | set by Render automatically | | | unset |
| `VERCEL_AUTOMATION_BYPASS_SECRET` | GitHub Actions secret | — | lets `staging.yml` open the protected preview | — |
| `SENTRY_DSN` | both Render services, and Vercel (Production + Preview) | set | set | unset (no Sentry) |
| `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` | Vercel (Production + Preview) | `pk_…` from Clerk; the build fails without it | same | unset = development sign-in |
| `CLERK_SECRET_KEY` | Vercel (Production + Preview) | `sk_…` from Clerk | same | unset |
| `CLERK_ISSUER` *or* `CLERK_PUBLISHABLE_KEY` | both Render services | the Clerk instance (`https://<instance>.clerk.accounts.dev`) | same | unset = development sign-in |
| `FSE_AUTH_DEV` | local and CI only | **never set it** | never | `1` accepts `dev:<name>` tokens |
| `DATABASE_URL` | both Render services | Neon branch `production`, pooled connection string, as the restricted role `fse_api` (see "Security") | Neon branch `staging`, pooled, `fse_api` | optional: `python -m db.local` prints one; unset = no database |
| `DATABASE_MIGRATION_URL` | both Render services | the schema owner's pooled string (`neondb_owner`), used only for migrations | same, staging branch | unset = migrations use `DATABASE_URL` |
| `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` | both Render services, and GitHub Actions secrets | shared usage counters, keys `fse:production:` | same database, keys `fse:staging:` | unset = counters in memory (or the database); see "Usage limits" |
| `TEST_DATABASE_URL` | CI (`tests.yml`, a Postgres 18 service) | — | — | optional: tests create and drop their own databases through it |
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

**Backups and restore** are below, under "Backups and recovery": Neon's own
restore window for recent mistakes, and an encrypted nightly copy in Supabase
Storage for everything else.

## Backups and recovery

Set up in PLAN.md 1.8. Everything is in `ops/backup.py` (dump, restore,
drill), `ops/encryption.py` (the file format) and `ops/backup_store.py`
(where backups live); `.github/workflows/backup.yml` is the schedule.

**Two ways back, for two kinds of accident:**

| What happened | Use | How far back |
|---|---|---|
| A bad migration, a wrong delete, something noticed within days | **Neon's restore window** — Neon console → **Branches → `production` → Restore**, pick a time | Neon free keeps a 24-hour history |
| A backup older than that, or the Neon project itself gone | **The encrypted nightly backup** in Supabase Storage | last 7 nights, one a week for 8 weeks, one a month for 12 months |

Neon's own restore is faster and loses nothing, so it is the first thing to
reach for. The nightly backup is what covers losing Neon.

### What a backup is

`pg_dump -Fc` of the whole production database, encrypted with AES-256-GCM and
uploaded to a **private** Supabase Storage bucket as
`production/<when>.dump.enc`, with a small `<when>.json` beside it holding
sizes, the checksum, the Postgres version, the commit, and what the dump held:
the migration revision, how many deals, and a one-way fingerprint of their
model results. **Never anything from a deal**, and nothing a fingerprint could
be turned back into.

The counts and the fingerprint are read **inside the same snapshot `pg_dump`
reads** (`pg_export_snapshot`), so the manifest describes exactly what is in
the file — which is what makes the drill's comparison meaningful weeks later.

The key never leaves the GitHub Actions secret `FSE_BACKUP_KEY`; without it a
backup is bytes nobody can read, which is why it is kept outside this
repository, outside Supabase and outside Neon.

The nightly run (02:40 UTC) dumps, encrypts, uploads, rotates, and then
**downloads what it just stored and decrypts it**, so a backup is never
assumed to work. A failure raises a Better Stack incident.

Backups are not kept as GitHub Actions artifacts: on a public repository
anyone can download those.

### You need to set this up first

Nothing is backed up until these GitHub Actions secrets exist (the workflow
says so with a warning and does nothing until then). Repository → **Settings →
Secrets and variables → Actions → New repository secret**:

| Secret | Where it comes from |
|---|---|
| `FSE_BACKUP_KEY` | make one: `python -c "import base64, secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"`. **Keep a copy somewhere outside GitHub** (a password manager): losing it loses every backup |
| `SUPABASE_URL` | Supabase → the project's home page, the `https://<id>.supabase.co` line under its name (also **Project Settings → Data API → Project URL**) |
| `SUPABASE_SERVICE_ROLE_KEY` | Supabase → **Project Settings → API Keys → Secret keys → `default`** (reveal, then copy). New projects show an `sb_secret_…` key; older ones a `service_role` JWT — either works, and the variable keeps the old name. It bypasses row policies, so it lives only in GitHub secrets |
| `BACKUP_DATABASE_URL` | Neon → project `fse-ml` → branch **`production`** → **Connect** → the **owner** (`neondb_owner`) connection string, **unpooled** (untick *Connection pooling*) |
| `BACKUP_STAGING_DATABASE_URL` | the same for the **`staging`** branch — the drill's target |

The Supabase project is free and needs no card; create it once (PLAN.md 1.9
uses the same project for file uploads), then **Storage → New bucket**, name
it `backups` and leave it **private**. A free Supabase project pauses after a
week with no activity — the nightly backup is itself the activity that keeps
it awake.

The two database URLs are the **owner** role, not `fse_api`: a backup has to
read every table and a restore has to create them. They are the most valuable
secrets in the repository; `docs/security/threat-model.md` says what that
means.

### Restoring

Anything below needs `FSE_BACKUP_KEY`, `SUPABASE_URL` and
`SUPABASE_SERVICE_ROLE_KEY` in the shell (a git-ignored `.env`, never in a
command), plus a `pg_restore` at least as new as the server — Neon runs
Postgres 18 today, so install PGDG's newest client (`FSE_PG_BIN` points at it
if it isn't on the PATH). An older one is refused with a message naming every
binary it found.

```bash
python -m ops.backup list --environment production
python -m ops.backup verify --environment production
```

Into a fresh database (what to do if Neon is gone — make a project, then):

```bash
python -m ops.backup restore --name production/2026-09-18T024000Z --into "$NEW_DATABASE_URL"
```

Over a database that still has the wrong data in it, add `--clean`.

**Then put the app's grants back.** A restore brings the schema and the rows,
not the rights: Neon's dump carries its own platform grants
(`ALTER DEFAULT PRIVILEGES FOR ROLE cloud_admin … TO neon_superuser`), only
Neon's superuser may replay them, and `pg_dump` writes them in the same entry
as ours — so `pg_restore` leaves privileges out altogether (`--no-privileges`)
rather than aborting on them. Roles are cluster-wide and aren't in a dump
either, so a brand-new project needs them made first. As the **owner**
(`DATABASE_MIGRATION_URL`), against the restored database:

```bash
psql "$RESTORED_OWNER_URL" -c "GRANT USAGE ON SCHEMA public TO fse_app" -c "GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO fse_app" -c "GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO fse_app" -c "REVOKE INSERT, UPDATE, DELETE ON alembic_version FROM fse_app" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO fse_app" -c "ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO fse_app"
```

That is migration `0005_app_role`'s `upgrade()`, which stays the source of
truth — copy from there if it has changed. A new cluster also needs the roles
themselves: "Least-privilege database role" below has the `fse_app` and
`fse_api` statements.

Then point `DATABASE_URL` on the Render service at the restored database and
redeploy. Check with `python3 ops/check_database.py https://fse-api.onrender.com`
— it reports migrations current **and** `role: restricted`, which only passes
if the grants above landed — and by opening a saved deal: its IRR must be what
it was.

`--with-grants` replays the source's privileges instead, which works only when
every role in the dump is one you own outright (not a managed host's).

### The restore drill

Proof that the backups restore, run **monthly** (the first of the month) by
`backup.yml`, and on demand from the **Actions → backup → Run workflow**
button with *Run the restore drill* ticked. It restores the newest production
backup into a **new, throwaway database on the staging branch**, re-runs the
deal model on every restored deal, compares the results with **what the
manifest recorded when the dump was taken** — not with the live database,
which has moved on since — and drops the database again. It prints counts and
whether they match, never a deal. A failure raises a Better Stack incident:
the backups are not restorable, and that is an incident in itself.

By hand, against any Postgres:

```bash
python -m ops.backup drill --environment production --target "$STAGING_OWNER_URL"
```

It needs no access to the production database: everything it checks against
travels in the backup's manifest.

`tests/test_backups.py` runs the same code against a real Postgres on every
pull request: a saved deal is backed up, encrypted, restored into another
database and read back through the API, and its IRR has to come out identical.

### Free-plan limits that shape this

- Supabase Storage free is **1 GB**, and one upload may be at most 50 MB.
  Rotation keeps the total under **700 MB** (`STORAGE_BUDGET_BYTES`) — daily,
  then weekly, then monthly, then oldest-first once the budget is reached; the
  newest backup is never deleted. An upload over 45 MB is refused with a
  message rather than failing half-way (a database that big needs the
  resumable uploads of PLAN.md 12.2).
- Neon free gives no scheduled jobs, so GitHub Actions is the scheduler. It is
  free on this public repository, but **schedules pause after 60 days with no
  commit** (PLAN.md 11.3).
- A nightly `pg_dump` wakes the Neon compute for a few seconds: nothing
  against the 100 free compute hours.

## Background jobs and scheduled jobs

PLAN.md 1.9. **Nothing to set up**: no new account, key or secret.

### Jobs a user starts

Long runs go through a queue instead of holding a request open: `POST
/api/jobs` with `{"kind": "montecarlo.run", "input": <the body POST
/api/montecarlo/run takes>}` answers 202 with a job id at once; `GET
/api/jobs/{id}` gives progress (`stage`, `progress`, `ahead`) and, once it
succeeded, `result`, which is exactly what the direct endpoint answers. Kinds:
`montecarlo.run`, `montecarlo.scenarios`, `backtesting.run`,
`forecasting.run` (`jobs/kinds.py`). The Monte Carlo screen uses it: a
progress bar with Cancel, a `running` chip on its tab, and every other screen
usable meanwhile. The deal model stays a direct call (it answers instantly).

- **Where they wait:** the `jobs` table in Neon (`jobs/database.py`); without
  a database, in memory (`jobs/memory.py`).
- **Who runs them:** a thread inside the API (`jobs/runner.py`), since
  Render's free tier has no worker machines. It starts when a job is submitted
  or checked, and stops after a minute with nothing to do, so a sleeping Neon
  stays asleep. It runs one simulation at a time, sharing the slot the direct
  endpoints use: two big runs never share the 512 MB.
- **Restarts:** a running job's heartbeat stops when Render restarts or puts
  the service to sleep. The next check on the job (the screen polls) wakes a
  runner, which puts it back in the queue after 90 s of silence and runs it
  again from the start (runs are seeded, so the result is the same); after 3
  tries it fails with a message saying so.
- **Limits:** submitting counts as a model run (`api/limits.py`); at most 4
  unfinished jobs per account (429) and 40 in the whole queue (503).
- **Staying small:** a job's inputs are deleted when it ends, its result after
  6 hours (or once the account has 10 newer ones), the row after 7 days.
  A Monte Carlo result is about 0.25 MB.

**Configuration** (defaults are right for the free plan): `FSE_JOB_QUEUE`
(`database` | `memory`), `FSE_JOB_RUNNER` (`api` | `external`). Phase 12
sets `FSE_JOB_RUNNER=external` on the API and runs `python -m jobs.worker`
on worker machines instead: no endpoint changes.

### Scheduled jobs

GitHub Actions is the scheduler (`scheduled.yml`, nightly at 04:10 UTC, and
**Run workflow** by hand):

| Step | What it does | Recorded as |
|---|---|---|
| `api-tasks` (production and staging) | `POST /api/scheduled/tasks/job-maintenance`: requeues or fails jobs whose runner went silent, applies retention | `job-maintenance` |
| `supabase-keepalive` | Lists the backup bucket, so the free Supabase project never pauses for inactivity | `supabase-keepalive` |

`staging.yml` also runs the **job drill** after each staging deploy: ten
seeded simulations queued at once must all finish with the pinned result
while `/api/health` keeps answering (`ops/scheduled.py drill`, recorded as
`job-drill`). It is refused in production.

**How the workflows sign in, with no secret:** each call carries the OpenID
Connect token GitHub mints for the workflow run (`permissions: id-token:
write`, which allows nothing else). The API checks it against GitHub's
published keys (`api/github_oidc.py`): this repository (by id, not just
name), the workflow file and a protected branch (production accepts only
`scheduled.yml` on `main`; staging also `staging.yml` and `scheduled.yml` on
`staging`), and an audience naming the environment, so a staging token is
refused by production.

**Checking it:** `/api/health/jobs` (public; counts and statuses only, cached
for a minute) shows the queue, whether the runner is running, jobs per status
and the newest run of each scheduled task:

```bash
curl -s https://fse-api-staging.onrender.com/api/health/jobs
```

A failed step raises a Better Stack incident. Waking staging once a night
costs about 15 minutes of the free instance hours a day.

To add a scheduled task (data refresh, retraining): a function with
`@task("name", "what it does")` in `jobs/scheduled.py` returning a small dict
of counts, and a step in `scheduled.yml`. Anything longer than a request
should queue a job instead.

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

## Security

Set up in PLAN.md 1.7. The threat model, and what is still open, is in
`docs/security/threat-model.md`; how to report a problem is in `SECURITY.md`.

**Headers.** The web app sends HSTS, `nosniff`, `X-Frame-Options: DENY`, a
strict referrer policy and a permissions policy on every response
(`web/next.config.ts`), and a content security policy with a fresh nonce on
every page (`web/src/proxy.ts`, `web/src/lib/security/headers.ts`). Every page
is therefore rendered per request (a nonce can't be baked into a static
page); on Vercel Hobby that counts as function invocations, far inside the
free allowance at this traffic. The policy allows Clerk's Frontend API (from
the publishable key), Clerk's images and telemetry, Cloudflare Turnstile (bot
protection on sign-up) and the Sentry DSN's host. **A new third-party script,
frame or API host must be added there**, or the browser refuses it. The API
sends its own headers (`api/security.py`): `default-src 'none'` and
`Cache-Control: no-store` on every answer, and a hash-based policy for the
Swagger page at `/api/docs`.

**Header scan.** `ops/check_headers.py` checks all of that on a deployed copy
(and that CORS refuses an unknown origin). `live.yml` runs it daily on
production and `staging.yml` after each staging deploy. By hand:

```bash
python3 ops/check_headers.py --web https://fse-ml.vercel.app --api https://fse-api.onrender.com
```

**CORS.** The web app calls the API through its own proxy, so the browser
never needs CORS. The API allows only exact HTTPS origins: by default
`https://fse-ml.vercel.app` in production, none on staging and
`http://localhost:3000` locally. A wildcard, a path or plain `http://` in
`FSE_CORS_ORIGINS` is dropped (and logged as `cors_origin_refused`), never
widened.

**Encrypted connections.** Browsers reach Vercel, Render and Clerk over HTTPS
only (HSTS). A deployed API adds `sslmode=require` to a database URL without
one and refuses to connect with `sslmode=disable`, `allow` or `prefer`
(`db/engine.py`). Upstash, Clerk, Sentry and the data sources are HTTPS URLs.

### Least-privilege database role

Migration `0005_app_role` creates the group role `fse_app`, which can select,
insert, update and delete rows in the app's tables and nothing else. The API
connects as a login role in that group, `fse_api`; the owner (`neondb_owner`)
is used only for migrations, through `DATABASE_MIGRATION_URL`.
`/api/health/database` reports `"role": {"status": "restricted"}` once the
API connects as `fse_api`, and `"privileged"` with the reasons before. The
database checks (`ops/check_database.py --require-restricted-role` in
`live.yml` and `staging.yml`) fail when it is privileged. Both environments
were switched on 2026-09-17; follow the steps below again for a new Neon
branch, or to change the `fse_api` password (use `ALTER ROLE fse_api PASSWORD
'...';` in step 3).

Roles belong to a Neon branch, so do this once per branch, **staging first**,
and only after a deploy with migration 0005 has run there (the health check
shows `"migrations": {"revision": "0005", …}` or later):

1. **Keep the owner for migrations.** Render → the service
   (`fse-api-staging`, later `fse-api`) → **Environment** → copy the value of
   `DATABASE_URL` → **Add Environment Variable** → key
   `DATABASE_MIGRATION_URL`, paste the value → **Save changes**.
2. **Make a password.** In PowerShell on your computer (it goes straight to
   the clipboard, not the screen):

   ```powershell
   $b = New-Object byte[] 32; [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($b); -join ($b | % { 'ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz23456789'[$_ % 57] }) | Set-Clipboard
   ```

3. **Create the login role.** https://console.neon.tech → project `fse-ml` →
   **SQL Editor** → branch `staging` (later `production`), database `neondb`
   → run, pasting the password between the quotes:

   ```sql
   CREATE ROLE fse_api LOGIN PASSWORD 'paste-here' IN ROLE fse_app;
   ```

   "Success" means done. (Roles made with SQL get no extra rights; roles made
   in Neon's Roles page get `neon_superuser`, which is why this one is made
   with SQL.)
4. **Point the API at it.** Take the `DATABASE_URL` value from step 1 and
   replace the user and password between `://` and `@` with
   `fse_api:<the password>`, keeping the `-pooler` host and everything after
   `@`. Render → **Environment** → edit `DATABASE_URL` → paste → **Save and
   deploy**. The API reads the variable only when it starts: if Render only
   saved, click **Manual Deploy → Deploy latest commit**, and wait for
   **Deploy live** before checking. If the check then shows an error, put
   the owner string back in `DATABASE_URL` and deploy at once: the live
   app can't reach its data until you do.
5. **Check.** Open `https://fse-api-staging.onrender.com/api/health/database`
   (production: `https://fse-api.onrender.com/api/health/database`); wait
   for the free service to wake. It should show `"status": "ok"` and
   `"role": {"status": "restricted", "privileges": []}`. If it shows an
   authentication error, the password or user in step 4 is wrong: put the
   old value back from `DATABASE_MIGRATION_URL` and redo step 4.

To undo: set `DATABASE_URL` back to the owner's value.

### CI and GitHub settings

On every pull request: `security.yml` (gitleaks over the whole history after
proving it catches a planted fake key; `pip-audit`; `npm audit`) and
`codeql.yml` (CodeQL for Python, TypeScript and the workflows). Both also run
weekly. Dependabot opens grouped update PRs weekly (`.github/dependabot.yml`).

Repository settings these rely on (GitHub → the repository → **Settings**):

- **Advanced Security**: turn on *Private vulnerability reporting*,
  *Dependabot alerts* and *Dependabot security updates*. Secret scanning and
  push protection are on by default for public repositories; keep them on.
- **Branches** → the `main` rule → *Require status checks*: add `secrets`,
  `python-dependencies`, `npm-dependencies` and the three
  `analyze (…)` CodeQL checks next to the existing ones.
- **Secrets and variables → Dependabot**: add `UPSTASH_REDIS_REST_URL` and
  `UPSTASH_REDIS_REST_TOKEN` with the same values as the Actions secrets.
  Dependabot's pull requests can't read Actions secrets, and the `ml` job
  fails when the Upstash test skips.

**Commit email.** This repository is public, so commit authors' emails are
too. Use GitHub's private address for new commits: GitHub → **Settings →
Emails** → tick *Keep my email addresses private* and *Block command line
pushes that expose my email*, copy the `…@users.noreply.github.com` address,
then run `git config --global user.email "<that address>"`.

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
