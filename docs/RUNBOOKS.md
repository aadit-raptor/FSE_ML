# RUNBOOKS.md — when something alerts

PLAN.md 11.2. For whoever is on call, which is one person. Each alert the
project can raise has a runbook here: **what it means, how to confirm it, what
to do, how to roll back.** Times are UTC; the owner is in India (IST = UTC+5:30).

`tests/test_runbooks.py` checks every endpoint, workflow, module, task and
variable named in backticks below against the code, and that every Better
Stack incident a workflow can raise has a runbook, so this file cannot drift
quietly. Setting things up, rotating secrets, restoring and the accounts table
are in [MAINTAINING.md](MAINTAINING.md); how the pieces fit is in
[ARCHITECTURE.md](ARCHITECTURE.md). The setup behind each alert is in
[DEPLOY.md](../DEPLOY.md) (this file points to its sections, it does not repeat
them).

---

## 1. The first five minutes

1. **Read the alert.** Better Stack emails the incident name and, for the
   workflow alerts, a link to the failed run. Open the run and find the red
   step: the step names below match the runbooks.
2. **Is it real?** One check lands on a sleeping free API and waits about a
   minute to wake it (43 seconds on staging on 2026-11-03); the monitors
   already ride that out, so an alert is not a wake-up. Run the triage:

   ```bash
   API=https://api.variater.com        # staging: https://fse-api-staging.onrender.com
   curl -s -m 150 $API/api/health                  # up? which commit? which environment?
   curl -s -m 150 $API/api/health/database          # migrations, role, storage per budget
   curl -s -m 60  $API/api/health/limits            # Upstash: reachable, over_budget
   curl -s -m 60  $API/api/health/jobs              # queue, runner, newest run of each task
   curl -s -m 60  $API/api/health/files             # uploads store and the kind of key
   curl -s -m 60  $API/api/health/reader            # AI provider key accepted, models listed
   curl -s -m 60  https://variater.com/healthz      # the website, never touches the API
   ```

   `/api/health` names the commit the API runs. Production runs a commit of
   `main`; **staging runs the merge commit of the `staging` branch** (the
   "bring staging level" PR), so look it up with `git log origin/staging`.
   Only `/api/health/database` wakes Neon; the others are cheap. Do not poll
   them in a loop (Render hours and Neon compute hours are the free limits).
3. **Pick a severity** (section 2) and write one line in the incident note
   (section 6): what, since when (UTC), severity.
4. **Stop the bleeding first, then find the cause.** If a deploy is the
   suspect (`/api/health` shows the commit the API runs; `git log --first-parent
   main` shows what merged), roll back before debugging
   ([DEPLOY.md "Rollback"](../DEPLOY.md#rollback), summarised in section 5).
5. **Tell people** if it is SEV1 or SEV2 (section 4).

Your token cannot dispatch workflows (`gh workflow run` is refused), so
"re-run" below means **Actions → the workflow → Run workflow** in the browser,
or a re-run of the failed run there. A push to `staging` also runs `staging.yml`.

---

## 2. Severity levels

| Level | Means | Examples | Acknowledge | Aim to fix or mitigate |
|---|---|---|---|---|
| **SEV1** | Production is down or wrong for everyone, data is lost, or personal data may be exposed | Website or API monitor down and not a wake-up; the deal model gives wrong answers; saved deals unreadable; a leaked secret or a suspected breach | 1 hour | 4 hours; public status update within 1 hour |
| **SEV2** | A main feature is broken for many people, or a safety net has failed | Sign-in failing; model runs all 5xx; uploads or results reads down; the nightly backup failed twice running; the restore drill failed (the backups may not restore) | Same day | 24 hours; status update if people can see it |
| **SEV3** | One feature, a refresh or staging is degraded; a free limit is near | One scheduled task failed; staging checks failed; Neon storage at 80%; Upstash over its daily budget (the database takes over); AI day used up | 3 days | Next week's maintenance; no status update |
| **SEV4** | Nothing is broken for anyone | A check that passes when re-run; a Dependabot advisory with no exposed path; a stale doc | Next maintenance | Optional |

Raise a level when the alert repeats, when the cause is unknown, or when more
than one thing is failing. Lower it only once the check is green again and
you know why. **Security issues** (a leaked key, an exposed deal, a bypassed
sign-in) are SEV1 until shown otherwise: rotate first ([MAINTAINING.md
"Rotating secrets"](MAINTAINING.md)), then find out how; `SECURITY.md` says
how outsiders report one.

---

## 3. Every alert and its runbook

What raises it, where it lands, and where to read what to do. Better Stack
incidents are raised by `ops/betterstack.py incident` (email and push, no
call); Sentry and the free-limit warnings reach you in other ways, listed too.

| Alert (exact name) | Raised by | Level | Runbook |
|---|---|---|---|
| Monitor "Variater web (production)" down | Better Stack, `https://variater.com/healthz`, every 3 min, 2 min of rechecks | SEV1 | [RB-01](#rb-01-website-monitor-down) |
| Monitor "Variater API (production)" down | Better Stack, `https://api.variater.com/api/health`, every 30 min, 60 s timeout + 4 min of rechecks | SEV1 | [RB-02](#rb-02-api-monitor-down) |
| Production checks failed | `live.yml` (daily 06:17) | SEV1–3 by step | [RB-03](#rb-03-production-checks-failed) |
| Staging API broken | `staging.yml`, quick API check | SEV3 | [RB-04](#rb-04-staging-api-broken-and-staging-checks-failed) |
| Staging checks failed | `staging.yml`, any other step | SEV3 | [RB-04](#rb-04-staging-api-broken-and-staging-checks-failed) |
| Nightly backup failed | `backup.yml` (daily 02:40) | SEV2 | [RB-05](#rb-05-nightly-backup-failed) |
| Restore drill failed | `backup.yml` (1st of the month 03:40) | SEV2 | [RB-06](#rb-06-restore-drill-failed) |
| Scheduled task failed (production) / (staging) | `scheduled.yml` (daily 04:10) | SEV3 | [RB-07](#rb-07-scheduled-task-failed) |
| Supabase keep-alive failed | `scheduled.yml` | SEV3 | [RB-08](#rb-08-supabase-keep-alive-failed) |
| Red `security`, `codeql`, `monitoring`, `pr`, `tests` run | GitHub's own email for a failed workflow | SEV3–4 | [RB-09](#rb-09-a-github-workflow-failed-without-an-incident) |
| A new issue in Sentry | Sentry's email (project `fse-api`) | SEV2–4 | [RB-10](#rb-10-sentry-error) |
| Database storage at 80% (any budget) | `ops/check_database.py` inside `live.yml` and `staging.yml` → "Production checks failed" / "Staging checks failed" | SEV3 | [RB-11](#rb-11-neon-storage-or-compute-near-the-free-limit) |
| Render hours, memory or restarts | Render's email; the staging drill's memory check | SEV2–3 | [RB-12](#rb-12-render-hours-or-memory) |
| Upstash unreachable or over its daily budget | `/api/health/limits` in `live.yml`, `staging.yml` | SEV3 | [RB-13](#rb-13-upstash-unreachable-or-over-budget) |
| AI allowance spent, provider busy, cost, reader below its bar | Users see `allowance_spent`, `provider_busy`, `reader_not_measured`; the nightly summary shows `ai_cost_usd_yesterday` | SEV3 | [RB-14](#rb-14-ai-allowance-busy-cost-or-the-reader-gate) |
| Resend near its limit, mentions not sent | Resend's email; `mentions_*` in the nightly summary | SEV3 | [RB-15](#rb-15-resend-limits-and-mention-emails) |
| Uploads failing or Supabase storage / egress / pause | `/api/health/files`, the drill's `files_drill`, Supabase's email | SEV2–3 | [RB-16](#rb-16-uploads-and-supabase) |
| Sign-in failing | Users; Clerk's status; header scan | SEV1–2 | [RB-17](#rb-17-sign-in-failing) |
| Scheduled workflows paused (60 days without a commit) | GitHub's email; `/api/health/jobs` shows a task's old `started_at` | SEV3 | [RB-18](#rb-18-schedules-stopped) |

The "Alert drill log" in [DEPLOY.md "Monitoring"](../DEPLOY.md#monitoring)
shows the first two alerts working end to end (2026-09-15).

Each runbook has the same four parts.

### RB-01 Website monitor down

**Means.** `https://variater.com/healthz` did not answer `"status":"ok"` for
two minutes (Vercel). `/healthz` never calls the API, so this is Vercel, DNS or
the web build, not Render.

**Confirm.** `curl -si https://variater.com/healthz`. Then the Vercel project
`fse-ml` → **Deployments** (a failed production build leaves the last good one
serving, so a down site usually means DNS, a domain or a Vercel incident) and
vercel-status.com. DNS records at Cloudflare must all be **DNS only** (grey
cloud) or certificates break ([DEPLOY.md "Own domain"](../DEPLOY.md#5-own-domain-variatercom)).

**Do.** Vercel incident: nothing to fix, post the status update and wait.
Bad deploy: roll back (below). A changed environment variable (`FSE_API_URL`,
Clerk keys): Vercel resolves rewrites at build time, so fix it and redeploy.

**Roll back.** Vercel → project `fse-ml` → **Deployments** → last good
production deployment → **⋯ → Instant Rollback**, then route A (a revert PR)
so `main` and the site agree. Read DEPLOY.md "Rollback" first if the bad
change touched sign-in (the Clerk keys travel with the code).

### RB-02 API monitor down

**Means.** `https://api.variater.com/api/health` did not answer
`"status":"ok"` after the 60 s timeout and 4 minutes of rechecks, so it is not
the normal minute-long wake from sleep. Render (service `fse-api`) is down,
crashing, out of memory, out of instance hours, or the domain is broken.

**Confirm.** `curl -si -m 150 https://api.variater.com/api/health`; also the
service's own address `https://fse-api.onrender.com/api/health` (if that works
and `api.` does not, it is DNS or a certificate). Render → `fse-api` →
**Events** (a deploy that failed or rolled back) and **Logs** (one JSON line a
request; a crash loop shows tracebacks). If the log ends with "Killed" or the
service restarts under load it is memory (RB-12). If Render says the free
instance hours are used it is RB-12 too.

**Do.** Fix forward only if the cause is plain and small; otherwise roll back.
A crash on start usually names a missing variable in the log. The site stays
up without the API (sign-in works; model screens show the error), so the
status update names "Model API".

**Roll back.** Render → `fse-api` → **Events** → last good deploy → **Rollback**
(then turn **Auto-Deploy** back on once `main` is fixed), then route A. If a
migration ran that the old code cannot read, see "Rolling back past a change"
in DEPLOY.md "Rollback"; `python -m db.migrate downgrade -1` only when the
rollback is permanent. After it, `/api/health` shows the older commit and the
monitor recovers on its next check, which can be 30 minutes away: **do not
wait for it, run the triage and resolve the incident by hand.**

### RB-03 Production checks failed

**Means.** `live.yml`, the daily production check, found the site up but
wrong. One incident name for several steps: open the run and find the red one.

**Do.** Per red step:

| Red step | Means | Level | Do |
|---|---|---|---|
| Production database check | `ops/check_database.py` found the database unreachable, migrations not current, the API on a privileged role (`--require-restricted-role`), or a storage budget at 80% | SEV1 if unreachable, else SEV3 | Run `python3 ops/check_database.py https://api.variater.com`: its line says which. Unreachable → Neon status, `DATABASE_URL` on `fse-api`. Migrations behind → a deploy is half done; check Render Events. Role privileged → someone put the owner's URL in `DATABASE_URL` ([DEPLOY.md "Least-privilege database role"](../DEPLOY.md#least-privilege-database-role)). Storage → RB-11 |
| Production usage limits check | `/api/health/limits` did not answer `store == "upstash"` with Upstash reachable and not over budget | SEV3 | RB-13. Limits keep working from the database meanwhile |
| Production header scan | `ops/check_headers.py` found a missing or wrong security header, CSP or CORS rule | SEV2 | Run `python3 ops/check_headers.py --web https://variater.com --api https://api.variater.com`. Almost always the last web deploy changed `web/next.config.ts`, `web/src/proxy.ts` or `lib/security/headers.ts`; roll back |
| Base rates editions check | `python3 -m ops.check_base_rates`: an S&P or GCD edition is a year past `checked_on` | SEV4 | Not an outage. Check for a newer edition and move `checked_on` (PLAN.md 4.5a); the screen already says "due a check" |
| `npm run test:live` (browser checks) | The read-only browser checks against production failed (a page, a number, the environment) | SEV1 if a number is wrong | Open the run's Playwright report. Wrong IRR for the default deal (21.2%) is a model fault: roll back |

**Confirm.** Open the run (the incident summary links it) and re-running
the failed workflow once; a check that passes on re-run is SEV4 (note it).

**Roll back.** The last merge to `main` (route A, DEPLOY.md "Rollback").

### RB-04 Staging API broken and Staging checks failed

**Means.** Staging gets every `main` merge first (`staging.yml`, run by the
push to `staging`). "Staging API broken": the quick check failed (the API
did not refuse an anonymous call with 401, the development token was accepted,
the database or Upstash check failed). "Staging checks failed": any later
step (header scan, browser checks, the economic data, industry averages and
validation refreshes, the job drill). **Production is unaffected; the change
that caused it is not yet proven.**

**Confirm.** The incident's link opens the run; the red step names the check.
Then the same triage against `https://fse-api-staging.onrender.com`. A run that
dies at "Wait for the staging API to run this commit" or "Wait for the Vercel
preview" is a deploy that did not finish: look in Render (`fse-api-staging`)
and Vercel (preview of the `staging` branch).

**Do.** Do not merge anything else to `main`. Reproduce locally with the
failing step's command; fix in a PR to `main`; the staging bring-level PR
reruns the checks. If staging is only stuck, re-run the failed run.

**Roll back.** Nothing in production changed, so nothing to roll back; if the
merge to `main` is the cause and cannot be fixed fast, revert it (route A).
Staging itself returns to a good commit when `staging` receives the fixed or
reverted `main`.

### RB-05 Nightly backup failed

**Means.** `backup.yml` could not dump, encrypt, upload, rotate or read back
the production database. The last good backup is still there, but each missed
night is a night of data that could be lost.

**Confirm.** Open the run; the red step is one of: *Is backup storage set up
yet?* (a secret is missing or empty: `SUPABASE_URL`,
`SUPABASE_SERVICE_ROLE_KEY`, `FSE_BACKUP_KEY`, `BACKUP_DATABASE_URL`; the
run only warns, **no incident is raised for it**, so read the run's
annotations from time to time), *Newest Postgres client* (PGDG's apt
repository changed), or *Back up, rotate, then read it back* (`python -m ops.backup
run|verify|list`). Then, with the backup variables in your shell:

```bash
python -m ops.backup list --environment production
python -m ops.backup verify --environment production
```

`verify` downloads the newest backup and decrypts it. Common causes: Neon was
asleep or down (retry), the Supabase project paused (RB-16), the Supabase bucket
is past its 700 MB budget, `pg_dump` older than the server (Neon moved to a new
Postgres major; the workflow installs the newest client), or the key was
changed (backups made with the previous `FSE_BACKUP_KEY` cannot be decrypted;
it was replaced on 2026-10-12).

**Do.** Fix the cause, then **Run workflow** on `backup`; confirm `list` shows
a new one. Two missed nights in a row is SEV2: take a manual backup now.

**Roll back.** Nothing to roll back. If a bad change to the database is
discovered meanwhile, Neon's own restore window is faster than any dump
([MAINTAINING.md "Restore a backup"](MAINTAINING.md), DEPLOY.md "Restoring").

### RB-06 Restore drill failed

**Means.** The first-of-the-month drill restored the newest backup into a
throwaway database on the Neon staging branch and the deals did not come back
the way the manifest recorded. **The backups may not be restorable. Treat it as
SEV2 even though nobody can see it.**

**Confirm.** Open the run: the drill prints counts and whether they match,
never a deal. Re-run with **Run workflow** → *Run the restore drill* ticked.
By hand: `python -m ops.backup drill --environment production --target
"$STAGING_OWNER_URL"` (DEPLOY.md "The restore drill").

**Do.** A one-off (Neon staging asleep, a timeout) passes on re-run. A real
failure means the dump or the encryption is wrong or the model changed what
a deal computes: the drill re-runs the deal model, so an engine change
without a version bump (PLAN.md 3.1) shows up here. Take a fresh backup, drill
again, and do not rely on backups until it passes.

**Roll back.** Nothing live changed.

### RB-07 Scheduled task failed

**Means.** One of the nightly tasks in `scheduled.yml` failed on the named
environment. Tasks (`jobs/scheduled.py`): `job-maintenance`,
`company-refresh`, `economy-refresh`, `benchmarks-refresh`,
`validation-report` (and `regime-refresh`, which runs inside
`economy-refresh`). Failures are meant: `economy-refresh` fails when the FRED
key is refused or fewer than 20 economies are current, `company-refresh` when
EDINET refuses its key, `benchmarks-refresh` while a table is missing.
Stored data stays as it was; screens keep working on the last good data.

**Confirm.**

```bash
curl -s https://api.variater.com/api/health/jobs     # status of each task's newest run
```

Then the run's log: `ops.scheduled` prints `task <name>: failed <summary>`
(the summary is only in the log, not in `/api/health/jobs`). The failing
source is in the summary or the API's Render log.

**Do.** Source down or rate-limited: wait one night. A key refused: replace
it on both Render services and in the GitHub secret (`FRED_API_KEY`,
`EDINET_API_KEY`, `COMPANIES_HOUSE_API_KEY`). A source changed its shape
(`unreadable`): fix the connector in a PR and record the fixture again
(`record-economy.yml`, `record-filings.yml`). Then **Run workflow** on
`scheduled`.

**Roll back.** A task that failed after a deploy: revert it. The tasks
write only stored reference data; nothing to restore.

### RB-08 Supabase keep-alive failed

**Means.** The nightly list of the backup bucket (`python -m ops.scheduled
keepalive`) failed. Supabase's free project **pauses after a week without
activity**, taking the backups and uploads with it.

**Confirm.** Supabase dashboard → project status; `curl -s
https://api.variater.com/api/health/files` (the uploads project) too.

**Do.** Paused: restore the project in the dashboard (free, a few minutes), then
**Run workflow** on `scheduled`. A rotated key: update `SUPABASE_SERVICE_ROLE_KEY`
(GitHub) and `SUPABASE_UPLOADS_KEY` (Render) — the dashboard shows the
*publishable* key first and the *secret* key below it; the secret one is right
(CLAUDE.md "Gotchas").

**Roll back.** Nothing.

### RB-09 A GitHub workflow failed without an incident

**Means.** Only the workflows above raise Better Stack incidents. The others
fail quietly in GitHub (and email you): `security` (`secrets`,
`python-dependencies`, `npm-dependencies`, Mondays and on pushes), `codeql`,
`monitoring` (the Better Stack sync), `pr` (PR title) and `tests` (the 12
required checks).

**Confirm.** Actions → the workflow → the failed run → the red job and step;
re-run it once (a network blip passes).

**Do.**
- `security` → *secrets* (gitleaks): a real secret in history is **SEV1**:
  rotate it first ([MAINTAINING.md "Rotating secrets"](MAINTAINING.md)), then
  allow it in `.gitleaks.toml` only if it is a slug, never a real key.
  *python-dependencies* / *npm-dependencies*: an advisory. Update the package
  (Dependabot usually has the PR open); with no fix, note why in the PR and
  SEV4 it.
- `codeql`: read the finding; fix or dismiss with a reason.
- `monitoring`: `python ops/betterstack.py check` found the monitor plan breaks
  a free limit, or the token is refused (`BETTERSTACK_API_TOKEN`). Until it is
  fixed the monitors stay as they were. Sync again by hand with **Run workflow**
  (`dry_run` first).
- `tests`: a red required check blocks merges (nothing reaches production).
  Fix the cause; never loosen a test or regenerate the golden file
  (CLAUDE.md "Working rules").

**Roll back.** A merged change that broke these: revert it.

### RB-10 Sentry error

**Means.** The API or the web app reported an exception (Sentry organization
`aadit-xc`, project `fse-api`). Events carry the route, the error and a request
ID, **never** bodies, queries or deal contents.

**Confirm.** The issue's tags: `environment` (production, staging), `release`
(the commit), `request_id`, `api_path`. In Render's Logs search the same
`request_id` for the JSON line (status, `duration_ms`, `model_ms`). A burst
of issues for one route right after a deploy: deploy fault. A single one: SEV4
unless it repeats.

**Do.** Reproduce with the route and the commit; fix in a PR. Mark the issue
resolved in the next release so a regression reopens it. A 4xx user mistake
should not be an event; if it is, that is a bug in the scrubbing/filter.

**Roll back.** A fresh error rate after a deploy → roll back that deploy.

**Test it** (staging only; production answers 404, and it is limited to one a
minute):

```bash
curl -H "X-Request-ID: drill-runbook-1" https://fse-api-staging.onrender.com/api/debug/error
```

### RB-11 Neon storage or compute near the free limit

**Means.** Neon free has **0.5 GB of storage** and **100 compute hours a
month**. `/api/health/database` reports storage against the 0.5 GB with a
`warning` at 80% and each feature's own budget (`company_data` 64 MB,
`economic_data` 8 MB, `benchmark_data` 8 MB, `coverage_data` 64 MB); the daily
check turns any of them into "Production checks failed" with a line saying
which. Compute hours are shown only in Neon's dashboard (**Billing → usage**):
a database that is polled often never sleeps.

**Confirm.** `python3 ops/check_database.py https://api.variater.com` prints
the line; the JSON from `/api/health/database` gives `used_fraction` and each
budget. Biggest users are usually saved-deal versions, audit entries, uploads'
rows and the stored reference data.

**Do.** (1) Is something polling the database? Only `/api/health/database`
and `/api/health/jobs` touch it; nothing may call them in a loop. (2) Storage:
a budget past 80% is lowered at its constant (the message names it:
`db.companies.MAX_COMPANIES`, `db.economy.KEEP_FX_DAYS`,
`economy.model.KEEP_OBSERVATIONS`, `db.coverage.MAX_PERIODS`); `deal_versions`
and `audit_events` are already compacted nightly by `job-maintenance`. (3) Free
space or upgrade ([DEPLOY.md "Paid switches"](../DEPLOY.md#paid-switches): Neon
paid plan, PLAN.md 12.2). At 100% writes fail: SEV2.

**Roll back.** Deleting data is not a rollback; do not delete user rows to
save space. Neon's restore window undoes a recent mistake.

### RB-12 Render hours or memory

**Means.** The three free services share **750 instance hours a month**; the
API sleeps after 15 idle minutes and takes about a minute to wake. The
monitor plan (`ops/betterstack.py check`) keeps uptime checks from burning the
hours (the API is checked every 30 minutes on purpose). The instance has
**512 MB**: one simulation at the 100,000-path cap fits, two would not
(`simulation_slot`).

**Confirm.** Render → the service → **Metrics** (memory) and the workspace
**Billing** (hours). A restart loop with "Killed" in Logs is memory. The staging
drill's answer (`live_drill`) fails the run when the API's peak reaches the
limit or the live sliders add more than 32 MB.

**Do.** Hours near the limit: look for something keeping the API awake (a new
poller, a check closer than 30 minutes; staging has no scheduled check on
purpose) and remove it. Memory: find the route in the logs (`model_ms`
large, `route` the simulation); lower the cap in `api/limits.py` and in
`web/src/lib/limits.ts` only after measuring (CLAUDE.md "Usage limits"); do not raise it.
Out of hours, the service is suspended until the month turns: the only fix
is a paid instance (PLAN.md 12.1).

**Roll back.** A deploy that raised memory or restarts: Render → **Events** →
**Rollback**.

### RB-13 Upstash unreachable or over budget

**Means.** The usage counters are shared through Upstash (free: **500,000
commands a month**). `/api/health/limits` answers `store: "upstash"` when it
works. A hard daily budget (8,000 production, 2,000 staging, 1,000 elsewhere)
makes a process use the database for the rest of the UTC day. **Limits keep
working either way**; the shared day count is just kept in Neon or memory.

**Confirm.** `curl -s https://api.variater.com/api/health/limits`: `store` is
`database` or `memory`, `upstash.reachable` false, or `over_budget` true.
`python -m api.usage` prints the monthly estimate (about 31% of the free
plan at the target traffic).

**Do.** Unreachable: Upstash status and the two variables
`UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` on both Render services
(and GitHub secrets for CI). Over budget on a quiet day: something is syncing
too often (`FLUSH_INTERVAL_S` is 300 s); look for a loop. The live check
fails until `store` is `upstash` again, which is by design.

**Roll back.** A deploy that changed `api/usage.py`: roll back.

### RB-14 AI allowance, busy, cost or the reader gate

**Means.** The results reader, document reader, memo and note writers use
Gemini. Free tier (`AI_FREE_TIER` not `0`): the day's allowance is small (20
requests a model a day), the service may read 30 filings a day and write 10
notes (`READS_ALL`, `WRITES_ALL`), a person 10 and 5. Users see:
`allowance_spent` (every model's day is used up, try tomorrow),
`provider_busy` (every model said "high demand"; jobs already wait and ask
again), `key_refused`, `reader_not_measured` (503: the reader's card is below
its bar so production refuses reads, by design).

**Confirm.** `curl -s https://api.variater.com/api/health/reader` (key accepted?
models listed? spends nothing). Render's log for `ai_model_refused` lines (model,
HTTP status, Google's status and quota name, never the message or the
document). The nightly `job-maintenance` log line has `ai_requests_yesterday`,
`ai_pages_yesterday`, `ai_cost_usd_yesterday` (about 0.09 dollars a filing).
Per person: `GET /api/account/ai-usage` (the caller's own rows).

**Do.** Spent or busy: wait, and tell the person; it passes. Key refused: a
new key at Google AI Studio (project `variater-reader`) into `GEMINI_API_KEY`
on both Render services and in GitHub. Cost near the cap on the paid tier
(`AI_FREE_TIER=0`): Google's **budget alert is the real backstop**; lower
`TIERS["billed"]` / `PLANS` in `api/plans.py` and redeploy ([DEPLOY.md "Paid
switches"](../DEPLOY.md#paid-switches)). Do not switch provider mid-incident.

**Roll back.** `AI_FREE_TIER` back to its previous value; a new model name
that Google retired: `GEMINI_MODELS_*` in `research/reader/providers.py`.

### RB-15 Resend limits and mention emails

**Means.** Mention emails go through Resend (free: **100 a day, 3,000 a
month**); the sender stays under 90 and 2,900 (`teams/notices.py`
`DAILY_EMAILS`, `MONTHLY_EMAILS`), one email a person an hour, drops them after
7 days, and never writes the comment or the deal's name. Without
`RESEND_API_KEY`, mentions wait and then expire: nobody is told.

**Confirm.** `GET /api/teams` (signed in) answers `"mail"` and `"directory"`
(`resend`, `clerk` when set up). The nightly `job-maintenance` summary
has `mentions_emails_sent`, `mentions_waiting`, `mentions_failed`,
`mentions_expired`, `mentions_no_address`. Resend dashboard → Emails (bounces,
the domain `send.variater.com` verified).

**Do.** A bounce or unverified domain: re-add the DNS records at Cloudflare
(**DNS only**, grey cloud) and verify. A refused key: a new key with *Sending
access* for the domain into `RESEND_API_KEY` on both services. Near the cap:
nothing to do; the sender holds back. Waiting mentions go out with the next
comment's run or the nightly one.

**Roll back.** Remove `RESEND_API_KEY` to stop all mail; teams keep working.

### RB-16 Uploads and Supabase

**Means.** Uploaded files (private bucket `uploads` in production) and the
backups sit in Supabase free: **1 GB storage, 5 GB egress a month**, paused
after a week idle. `GET /api/health/files` reports `store`, `key` and the
bucket; the staging drill stores, scans, links and deletes a small PDF and
fails the run on `failed:<step>`.

**Confirm.** `curl -s https://api.variater.com/api/health/files`: `status`
must be `ok` with `key` `sb_secret` or `jwt:service_role`; `sb_publishable`
is the wrong key (it cannot create a bucket). Users see scans stay
"checking" if the store cannot be reached: the owner's next look re-queues them
after 60 s, and **an outage never refuses a file**.

**Do.** Wrong key or rotated: set `SUPABASE_UPLOADS_KEY` on Render. Paused:
RB-08. Over budget: `FSE_UPLOADS_BUDGET_MB` (default 800) is the guard;
the nightly sweep removes expired files. Egress high: check
`USER_FILE_VIEWS` (100 PDF views a person a day).

**Roll back.** A scan rule change that refuses good files: revert it
(`uploads/scan.py`, `uploads/rules.py`).

### RB-17 Sign-in failing

**Means.** Clerk (production instance for `variater.com`, `pk_live_`). The API
checks the session token locally against Clerk's keys, so a Clerk dashboard
outage does not break already-signed-in sessions for long, but new sign-ins
fail.

**Confirm.** Clerk's status page; open `https://variater.com/sign-in`. API
side: an anonymous `POST /api/deal/run` must answer **401**, and a
development token (`Bearer dev:...`) must also answer 401 when Clerk is
configured; a 200 on either is **SEV1** (the development sign-in is on in a
deployed copy). Header scan: the CSP must allow Clerk's hosts
(`ops/check_headers.py`).

**Do.** The wrong key pair after a change (`NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
and `CLERK_SECRET_KEY` on Vercel; `CLERK_PUBLISHABLE_KEY` on `fse-api`): put
the production pair back **together**. Google sign-in failing: the Google
Cloud project `variater`'s OAuth client and `/privacy` (DEPLOY.md "Own
domain"). A rate-limit storm of refused sign-ins: `429` per address
(60 a minute) is working as designed.

**Roll back.** Read DEPLOY.md "Rollback" first: **the Clerk keys and the code
travel together**; prefer rolling forward for anything that touched them.

### RB-18 Schedules stopped

**Means.** GitHub **pauses scheduled workflows after 60 days without a commit**
to the repository. The nightly backup, scheduled tasks, daily checks and
monitor sync all stop and **raise nothing**.

**Confirm.** Actions → each scheduled workflow (`backup`, `scheduled`, `live`,
`monitoring`, `security`, `codeql`): the newest run is older than a day
(weekly ones: older than a week) or it says "disabled". `curl -s
https://api.variater.com/api/health/jobs`: each task's `started_at` is old.

**Do.** Make a commit (a docs change merged through a PR counts), enable each
disabled workflow in Actions, then **Run workflow** on `scheduled`, `backup`
and `live`. Put a calendar reminder at 45 days (PLAN.md 11.3 adds a
check).

**Roll back.** Nothing.

---

## 4. Status updates (Better Stack status page)

The public page is **https://variater.betteruptime.com**, with the resources
"Website" and "Model API" (kept in `ops/betterstack.py`; staging is not on it).
Subscribers who signed up there are emailed when a report or update is posted
with *Notify status page subscribers* ticked.

**When.** SEV1: post within one hour. SEV2: post when users can see it
(sign-in, model, uploads, notes). SEV3 and SEV4: no post. A monitor that goes
down is posted for you only if *Automatically create status page updates for new
incidents* is ticked on the page (Status Pages → the page → settings); check
it, and if it is on, only add the updates below.

**How (Better Stack, in the browser; the project holds no code for it).**

1. Open the incident in Better Stack → **Incidents** → *Create status report*
   (prefilled from the incident), or on the status page: **Status updates →
   New status report**.
2. Choose the affected resource and its state: *Downtime*, *Degraded* (or
   *Resolved* at the end). Tick *Notify status page subscribers* for SEV1.
3. Write the message. Next updates: the same report → **Add status update**
   (Better Stack has an API for status reports; the UI is the supported route here).
4. Update at least every hour for SEV1 while it lasts, even to say "no change".
5. When fixed: a final update with state *Resolved*, then resolve the incident
   itself in Better Stack (`ops/betterstack.py` only raises incidents;
   there is no command to close one).

**Wording.** Plain, no cause you are unsure of, no blame, no deal or user
details; times in UTC.

| Stage | Template |
|---|---|
| Investigating | `We are investigating {what users see}. Started about {HH:MM} UTC. Next update by {HH:MM} UTC.` |
| Identified | `We found the cause: {one plain sentence}. We are {rolling back / fixing}. Next update by {HH:MM} UTC.` |
| Monitoring | `A fix is live at {HH:MM} UTC. We are watching it. Saved deals were {not affected / restored}.` |
| Resolved | `Resolved at {HH:MM} UTC. {What was affected, for how long}. A written review will follow {where, if any}.` |

The API alone being slow to wake is not an incident: do not post for it.

---

## 5. Rolling back, in one place

The default is a revert PR; the dashboards are faster. The full steps, and
what to do about Clerk keys and API fields that the older code refuses, are
in [DEPLOY.md "Rollback"](../DEPLOY.md#rollback).

| What broke | Fastest | Then |
|---|---|---|
| A web deploy | Vercel → `fse-ml` → Deployments → ⋯ → **Instant Rollback** | Revert PR (route A); undo the rollback or promote after |
| An API deploy | Render → `fse-api` → Events → **Rollback**; turn Auto-Deploy back on after | Revert PR |
| Either, with a record | `git log --first-parent main`, `git revert -m 1 <merge-sha>`, PR titled "Roll back …", merge | Watch `/api/health` show the commit change; run `live` |
| A migration | Roll the code back; `python -m db.migrate downgrade -1` **only** if the rollback is permanent | Never drop data to roll back |
| A recent bad data change | Neon's restore window | Else restore the newest backup ([DEPLOY.md "Restoring"](../DEPLOY.md#restoring)) |
| A model change that moved numbers | Revert it; `ENGINE_VERSION` and `MODEL_CHANGELOG.md` say what each version did | Saved deals show "Results changed since saved" |
| A bad env variable | Put the previous value back on the service; Vercel needs a redeploy | |

After any rollback: `/api/health` shows the expected commit and
`environment`, `python3 ops/check_database.py <api>` is clean, and the daily
`live` workflow passes when run by hand.

---

## 6. The incident note and the review

Keep one note per SEV1 or SEV2 (and any incident that taught something): a
short file `docs/incidents/YYYY-MM-DD-<slug>.md` in a PR, or the PR
description when it was a deploy. During the incident write the **timeline** as
you go (UTC); after it, fill the rest within a week. Blameless: what the system
allowed, not who.

```markdown
# Incident: <short title>   (SEV<n>)

- Date / duration: <start UTC> → <end UTC> (<minutes>)
- Detected by: <alert name / user / by chance> after <minutes>
- Who handled it: <name>
- Status updates posted: <links / times, or none>

## Impact
Who and what was affected, for how long. Any data lost or exposed (yes/no, and
what). No deal contents, names or emails.

## Timeline (UTC)
- HH:MM  <alert / first sign>
- HH:MM  <acknowledged>
- HH:MM  <what was checked, with the runbook step>
- HH:MM  <mitigation: roll back / fix / wait>
- HH:MM  <resolved, confirmed by: triage commands / live workflow>

## Cause
What actually happened, down to the change or the limit. One paragraph.
Why was it possible (no test, no check, a free-plan limit)?

## What went well / what did not
- Went well:
- Did not (alert late, runbook wrong or missing, wrong first guess):

## Actions
| Action | Owner | Due | Done |
|---|---|---|---|
| Fix the cause (PR link) | | | |
| Make the alert earlier or add one | | | |
| Fix this file (RB-nn) where it was wrong | | | |
| Add a test that fails on this | | | |

## Did the runbook work?
RB-nn: yes / partly / no, and what to change.
```

Anything the review changes about how alerts are handled goes into this file
in the same PR; a trap that cost time goes in CLAUDE.md "Gotchas".

---

## 7. Simulated incidents (runbook drills)

PLAN.md 11.2's "done when". Only ever on **staging**, never on production.

### 2026-11-03: a failing staging API (SEV3), drilled with RB-04 and RB-10

Run from a laptop against `https://fse-api-staging.onrender.com`, following the
runbooks as written.

| UTC | Step (runbook) | Result |
|---|---|---|
| 20:50:54 | Triage, section 1: `GET /api/health` on a sleeping API | 200 after **43 s** (the free wake-up; a monitor would have waited it out, as RB-02 says). `environment: staging`, `commit: 5130de3` |
| 20:50:54–20:51:00 | The other five health endpoints | `database` ok in 4.9 s (migrations `0027` current, role restricted, storage 2.7% of 0.5 GB, every budget under 80%); `limits` store `upstash`, reachable, not over budget; `jobs` queue `database`, no failed jobs, every task's newest run `succeeded`; `files` ok with `sb_secret`; `reader` key accepted, four models listed |
| 20:55:55 | The incident: `GET /api/debug/error` with `X-Request-ID: drill-runbook-11-2` (RB-10's test) | **500**, body `{"detail":"Internal server error","request_id":"drill-runbook-11-2"}`, the same ID in `x-request-id`: the fault is reproducible and traceable |
| 20:56 | RB-04 confirm: `python3 ops/check_database.py https://fse-api-staging.onrender.com --require-restricted-role` | exit 0: "database ok; migrations current; storage 13.9 MB (2.7% of free); … role restricted": the database is not the cause |
| 20:57 | Severity (section 2) | SEV3: staging only, no user affected; no status update |
| 20:57 | Rollback step (RB-04 and section 5) | Nothing live to roll back; checked that the commit staging runs resolves in the repository and that route A's commands apply (`git log --first-parent`) |

Findings, fixed in this file: staging's `/api/health` commit is the `staging`
branch's own merge commit, not a commit of `main`, so section 1 now says where
to look it up; the first call on a sleeping staging took 43 s, so the triage
commands carry `-m 150`.

### Still open: the parts that need the user

These need an account or a secret this session does not have (the Better Stack
token and the Sentry login live only in GitHub secrets and the user's
browser). Do them once, on staging, then fill in the row below:

1. **Better Stack alert end to end.** In Actions, **staging** → **Run workflow**
   cannot raise one by itself, so from a terminal with a Better Stack Uptime
   API token in `BETTERSTACK_API_TOKEN` (Better Stack → Uptime → API tokens):

   ```bash
   python3 ops/betterstack.py incident --name "Runbook drill (staging)" --summary "Drill for docs/RUNBOOKS.md, ignore. Staging only."
   ```

   Check: the email and push arrive; the incident shows in Better Stack →
   Incidents. Then follow section 4: **Create status report** on the incident
   (Investigating, then Resolved) and see it on `https://variater.betteruptime.com`
   only if you want to see the page change (tick no *Notify subscribers*); then
   resolve the incident. If the page shows "Model API" or "Website" as down
   afterwards, edit the report back to *Resolved*.
2. **Sentry.** In Sentry (`aadit-xc` / `fse-api`), find the issue
   "Deliberate test error (PLAN.md 1.2)" with `request_id`
   `drill-runbook-11-2`, `environment: staging`; mark it resolved.

| Date | Simulated | Runbook | Result |
|---|---|---|---|
| 2026-11-03 | Staging API error, triage and rollback check | RB-04, RB-10 | Handled as above; see the table |
| (open) | Better Stack incident, status report, resolve | Section 4 | Waits on the user (steps above) |
