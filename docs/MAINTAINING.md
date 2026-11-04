# Maintaining Variater without Claude

This handbook is for **you**, the owner. It assumes you can use a terminal and
GitHub, and nothing else. Every command here was checked against the code on
2026-11-04; if a command stops working, the code is right and this page is
stale: fix the page.

How the pieces fit, and why they were built this way, is in
[ARCHITECTURE.md](ARCHITECTURE.md). Setup of each outside service, step by
step, is in [DEPLOY.md](../DEPLOY.md); this handbook points into it rather
than repeating it. The development cycle is [WORKFLOW.md](WORKFLOW.md).

**Contents**

1. [Set up a new Mac](#1-set-up-a-new-mac)
2. [Run everything locally](#2-run-everything-locally)
3. [Ship a small change by hand](#3-ship-a-small-change-by-hand)
4. [Update dependencies](#4-update-dependencies)
5. [What each alert and failing workflow means](#5-what-each-alert-and-failing-workflow-means)
6. [Rotate each secret](#6-rotate-each-secret)
7. [Restore a backup](#7-restore-a-backup)
8. [The accounts table](#8-the-accounts-table)
9. [The monthly 15-minute checklist](#9-the-monthly-15-minute-checklist)

---

## 1. Set up a new Mac

You need about 20 minutes and a GitHub login. Everything installs with
Homebrew.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

```bash
brew install git gh python@3.12 node@24
```

Python must be **3.12** (the Docker image and CI use it) and Node **24**.
`node@24` is "keg-only" on some Homebrew versions: if `node --version` does
not say 24, run `brew link --overwrite node@24`.

Sign in to GitHub and clone:

```bash
gh auth login
gh repo clone aadit-raptor/FSE_ML ~/FSE_ML
cd ~/FSE_ML
```

Python side (the virtual environment lives in `.venv`, which git ignores):

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt pgserver
```

`pgserver` is Postgres in a pip package, so you need no Docker. Web side:

```bash
npm --prefix web ci
npx --prefix web playwright install chromium
```

(`playwright install` is only needed for the browser tests, section 2.)

macOS has no `python` on the PATH: always write `.venv/bin/python`.

**Optional tools** (only if you need them): `brew install poppler` (reads
tables out of the PDF rate studies in PLAN.md 4.5) and
`brew install --cask libreoffice` (lets the Excel-formula tests use a real
spreadsheet engine; they run by themselves when `soffice` is on the PATH).
The backup tests need a `pg_dump` at least as new as Neon's Postgres (18):
`brew install postgresql@18` and, if it is not on the PATH, set
`FSE_PG_BIN` to its `bin` folder.

**Check it worked:**

```bash
.venv/bin/python -m pytest tests/test_core_parity.py -q
```

It should pass without a database (database tests skip).

**Things that are not in git and that you carry yourself:** your password
manager's copy of `FSE_BACKUP_KEY` (section 6), your logins to the accounts in
section 8, and, only if you work on the results reader, the `research-samples/`
folder (git ignores it; it holds filings that must never be committed).

## 2. Run everything locally

Three things can run: a database, the API and the web app. The API works
without a database (saved deals, accounts and teams then answer that the
database is unavailable); start the database when you want those.

**The database (optional)**

```bash
.venv/bin/python -m db.local
```

It starts Postgres in `.localdb/` and prints `DATABASE_URL=...` and
`TEST_DATABASE_URL=...`. On a Mac these addresses use a Unix socket and hold
`?` characters, so **put them in single quotes** when you export them:

```bash
export DATABASE_URL='postgresql://postgres:@/fse?host=/path/printed/above'
```

Stop it with `.venv/bin/python -m db.local stop`.

**The API** (leave it running in its own terminal tab):

```bash
FSE_AUTH_DEV=1 .venv/bin/python -m uvicorn api.main:app --reload --port 8000
```

`FSE_AUTH_DEV=1` is the development sign-in: it accepts `dev:<any name>`
tokens, which is how the web app signs you in locally. The API refuses it in
production or when Clerk is configured. Docs are at
<http://localhost:8000/api/docs>; health at `/api/health`.

**The web app** (another tab):

```bash
npm --prefix web run dev
```

Open <http://localhost:3000> (use `localhost`, not `127.0.0.1`: Next's dev
server blocks its scripts for the second). It proxies `/api` to
`http://127.0.0.1:8000`. Without a Clerk key it uses the development sign-in.

**Uploads, the AI reader and mail locally** are off unless you set
`FSE_UPLOADS_DIR=<a folder>` (stands in for the file bucket),
`AI_PROVIDER=replay` (recorded answers, no key) and `FSE_MAIL_DIR=<a folder>`
(emails written as files). All three are refused in production.

**Tests**

```bash
# all Python tests. Export TEST_DATABASE_URL only, not DATABASE_URL
.venv/bin/python -m pytest

npm --prefix web run lint
npm --prefix web run typecheck
npm --prefix web run build
```

Run pytest with **`TEST_DATABASE_URL` set and `DATABASE_URL` unset**; with
both, seven "no database" tests fail on purpose. Browser tests (they start
their own API and web server, so stop yours first, and `build` first):

```bash
export DATABASE_URL='...' FSE_AUTH_DEV=1
npm --prefix web run build
npm --prefix web run test:e2e
```

If a browser test fails with `ChunkLoadError`, an old `next start` is still
holding port 3000: `lsof -iTCP:3000 -sTCP:LISTEN`, then `kill <pid>`. When
the machine is busy some browser specs time out; rerun just the failed one
before suspecting the code. `npm --prefix web run test:live` runs the
read-only checks against the real site (it needs `E2E_LIVE=1`; the daily
`live` workflow does this for you).

**After changing an API schema**, refresh the two generated files (a test and
CI fail if you forget):

```bash
.venv/bin/python -m api.export_openapi web/openapi.json
npm --prefix web run api:types
```

**Database changes** use Alembic through `db.migrate`:

```bash
.venv/bin/python -m db.migrate upgrade      # to the latest
.venv/bin/python -m db.migrate current
.venv/bin/python -m db.migrate downgrade -1
.venv/bin/python -m db.migrate revision -m "what it adds"
```

Read CLAUDE.md "Adding a table" before writing a migration: it must work with
the code that is still running while the new deploy starts.

## 3. Ship a small change by hand

The full cycle is [WORKFLOW.md](WORKFLOW.md); this is the short version for a
one-line change. `main` is protected: nothing is pushed to it directly.

1. **Branch from an up-to-date `main`.**

   ```bash
   git switch main && git pull
   git switch -c fix/short-name
   ```

   Names are `<type>/<short-name>` with a type from WORKFLOW.md step 7 (`feat`,
   `fix`, `docs`, `test`, `chore`, `refactor`, `perf`, `ci`).
2. **Make the change and run what it touches**: `pytest` for the folder you
   changed, `npm --prefix web run lint` and `typecheck` for web code. For a
   change to what the user sees, run the app (section 2) and look.
3. **A word on text.** Every word on screen lives in `web/messages/en.json`,
   not in a component. If you changed `docs/methodology.md`, also run
   `.venv/bin/python -m ops.help_articles` and commit what it writes.
4. **Commit** with a type: `git commit -am "docs: fix a typo in the handbook"`.
   Write why, not just what.
5. **Push and open the pull request.** The title must start with the same
   type (`pr.yml` checks it; it becomes the merge commit's message).

   ```bash
   git push -u origin HEAD
   gh pr create --fill
   ```

6. **Let the checks run.** `main` needs 12 green checks: `core`, `ml`, `web`,
   `e2e`, `docker`, `secrets`, `python-dependencies`, `npm-dependencies`, the
   three `analyze (...)` CodeQL checks and `title`. Watch them with
   `gh pr checks --watch`. The `core` and `ml` checks also enforce the
   coverage floors in `.coverage-floor` (they only go up).
7. **Merge when green**, with a merge commit (never squash):

   ```bash
   gh pr merge --auto --merge
   ```

   `--auto` makes GitHub merge it the moment the last check passes.
8. **Production deploys by itself.** Render rebuilds the API only when
   something the API uses changed (`buildFilter` in `render.yaml`); Vercel
   rebuilds the web app. Check `https://api.variater.com/api/health`: its
   `commit` is the one you merged.
9. **Bring staging level with `main`** (this is how staging is always a copy of
   what is live):

   ```bash
   gh pr create --base staging --head main --title "chore: bring staging level with main" --body "Keeps staging level with main."
   gh pr merge --merge
   ```

   Then watch the `staging` workflow in the Actions tab. It waits until
   both staging copies run that commit, then runs the browser checks, the
   database check, the header scan and the job drill. Green means staging
   is healthy. The `staging` workflow starts on a push to `staging`; you can
   also press **Run workflow** on it in the Actions tab.

**To try a change on staging before `main`**: push your branch to `staging`
fast-forward only (`git push origin <branch>:staging`), wait for the `staging`
workflow, then open the pull request into `main` as above. Staging's web
side is behind Vercel's login.

**If something goes wrong after a merge**, roll back with a revert pull
request (DEPLOY.md "Rollback", route A): `git revert -m 1 <merge-sha>`,
branch, PR, merge. It leaves a record and always works. The dashboard
rollbacks (Render Events, Vercel Instant Rollback) are faster but need you
to turn Render's auto-deploy back on afterwards. Prefer rolling forward for
anything that touches sign-in keys.

**Do not**: push to `main`; regenerate `tests/golden/golden.json`; lower
`.coverage-floor`; skip a failing check. A change to the model's numbers
(`core/`, `lbo_engine/`, `simulation/`) is not a small change: it needs an
engine version bump and a changelog entry (CLAUDE.md "Model version").

## 4. Update dependencies

**Dependabot** opens grouped pull requests every week (`.github/dependabot.yml`):
one for Python (`pip`), one for the web app (`npm`), one for GitHub Actions
and one for the Docker base image, titled `chore(deps): Bump ...`. Each runs
the full checks like any other change.

- Green and the update looks ordinary (patch or minor versions): merge it with
  `gh pr merge <n> --merge`.
- Red: read the failing check's log (`gh pr checks <n>`, then the run's
  link). Most failures are one library's new rule; fix it on the Dependabot
  branch or close the pull request and wait for the next week's.
- The `ml` job needs the Upstash test secrets, and Dependabot's runs cannot read
  Actions secrets. They must also be stored under **Settings → Secrets and
  variables → Dependabot** (`UPSTASH_REDIS_REST_URL`,
  `UPSTASH_REDIS_REST_TOKEN`), or that job fails on every Dependabot PR.
- **Major versions** (Next.js, React, a Python library's 2.0) are not merged
  on faith. Open the library's migration notes, update on a branch, run the
  whole local set (section 2) and the browser tests.
- **Next.js majors** differ from what most guides say; read
  `web/node_modules/next/dist/docs/` for the installed version before changing
  an API (for example, `params` is a Promise). After a Next upgrade, check the
  built proxy still matches pages: `.next/server/functions-config-manifest.json`
  must list the proxy's matcher, otherwise the content security policy
  silently disappears.
- **The Docker base image** is held to patch updates of Python 3.12 on
  purpose (`ignore` in `dependabot.yml`). Moving to a new Python minor version
  is a planned task: it changes what the numerics libraries and the saved
  model files run on. Do it on a branch, run everything, update
  `python-version` in the workflows and the `FROM` line together.
- **Node**: CI uses 24. A new LTS means changing `node-version` in the
  workflows (a change Claude's token cannot push, and you can) and installing
  it locally.
- **Python libraries pinned on purpose**: `oletools==0.60.2`, `pypdf<7`,
  `pypdfium2<5`, `pdfjs-dist==4.10.38`. Do not widen them casually; the
  comments in `requirements.txt` and CLAUDE.md "Gotchas" say why.
- **Security alerts** arrive as Dependabot security PRs (and the weekly
  `security` workflow). Treat them as urgent: merge as soon as the checks pass.
- After any dependency merge to `main`, bring staging level (section 3,
  step 9).

## 5. What each alert and failing workflow means

Alerts reach you two ways: **Better Stack** emails (an *incident*), and GitHub
emails about failed workflows. Before anything else, open the failing
workflow run's page (Actions tab) and read the first red step.

**First rule for a free service: it may just be asleep.** The Render API
sleeps after 15 idle minutes and takes about a minute to wake. A one-off
failure that passes on a second try is a wake-up, not an outage. The uptime
monitors already wait for that before raising an incident.

| What you see | What it means | What to do |
|---|---|---|
| Better Stack: **Website** monitor down (`variater.com/healthz`) | The web app is not answering. | Check <https://www.vercel-status.com> and the Vercel project's Deployments. If a deploy just went out, roll back (DEPLOY.md "Rollback", B) |
| Better Stack: **Model API** monitor down (`api.variater.com/api/health`) | The API is not answering after 4 minutes of retries. | Render dashboard → `fse-api` → Logs and Events. A failed deploy: roll back there. Out of free hours (750 a month across services): see section 9 |
| Incident **"Production checks failed"** (`live` workflow) | The daily check failed in one of: the browser checks, the database check, the usage-limits check, the header scan, the base-rates check. | Open the run; the red step names it. See the rows below |
| ...database check | Database unreachable, migrations not applied, role not restricted, or storage ≥ 80% of the plan. | `curl https://api.variater.com/api/health/database`. Storage: DEPLOY.md "Database"; look at which budget grew (`company_data`, `economic_data`, `benchmark_data` in that answer). A "privileged" role means `DATABASE_URL` was changed to the owner's string: put `fse_api`'s back |
| ...usage-limits check | Upstash is not answering or is over its daily command budget. | `curl https://api.variater.com/api/health/limits`. Limits keep working from the database meanwhile. Rotate the token if refused (section 6) |
| ...header scan | A security header or the content security policy changed. | Run `python3 ops/check_headers.py --web https://variater.com --api https://api.variater.com` and read the line that fails. Usually a new third-party script not added to `web/src/lib/security/headers.ts` |
| ...base-rates check | The default-rate studies are due a yearly check (`ops/check_base_rates.py`). | Not an outage. Look for a newer edition of S&P's and Global Credit Data's studies, transcribe it, or move `checked_on` (PLAN.md 4.5) |
| Incident **"Staging API broken"** or **"Staging checks failed"** (`staging` workflow) | The commit just pushed to `staging` broke something on the staging copy: it never reaches production unless you merge it. | Read the failing step. Fix and push again, or revert the change. Production is unaffected |
| Incident **"Scheduled task failed (production or staging)"** (`scheduled` workflow, nightly 04:10 UTC) | A nightly job failed: job maintenance, company refresh, economy refresh, benchmark refresh or validation report. | Open the run; the failed step names the task. `curl https://api.variater.com/api/health/jobs` shows each task's last run. Common causes: the FRED key refused (rotate it, section 6), fewer than 20 economies current (a source was down: usually fixes itself the next night), an EDINET key refused |
| Incident **"Supabase keep-alive failed"** | The free Supabase project paused or its key changed. | Open the project in Supabase and press Restore if paused; check `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` |
| Incident **"Nightly backup failed"** | Tonight's backup did not complete or could not be read back. | Open the `backup` run. Do not ignore: you have no safety net until it passes. Check the secrets in section 6 (the database URL, the Supabase key) and Supabase's storage (700 MB budget) |
| Incident **"Restore drill failed"** (first of the month) | The backups may not be restorable. | An incident in itself. Run the drill by hand (section 7) and read the message |
| `tests` workflow red on a PR | A check failed. | `gh pr checks <n>`, then the log. A job "cancelled" near its `timeout-minutes` (25 for `core`) is a time-out, not a failing test: push again, and if it repeats raise the limit in `tests.yml` |
| `security` workflow red | gitleaks found something that looks like a secret, or `pip-audit`/`npm audit` found a vulnerable package. | A leaked secret: rotate it first (section 6), then fix the history question. A vulnerable package: update it (section 4) |
| `codeql` red | CodeQL found a possible vulnerability in the code. | Read the alert under the repository's Security tab; fix or dismiss with a reason |
| `monitoring` workflow red | The uptime monitors in `ops/betterstack.py` could not be synced or checked. | Check `BETTERSTACK_API_TOKEN` (section 6) |
| `record-economy`, `record-filings`, `ml` red | Fixture recordings or the model gate (a model card got worse than `main`'s). | Only run by hand or when those files change; the `ml` gate failing is on purpose: read what got worse |
| Sentry email | An unhandled error in the API or the browser. | Open it: the route and `request_id` are in the tags. Search `request_id` in Render's logs. No deal contents are ever sent |
| Sentry/Neon/Upstash/Supabase/Resend/Google "you are near your limit" email | A free plan is filling up. | Section 8 says what each limit is and what to do; section 9 is where you check regularly |
| Banner "Results changed since saved" on a deal | The engine version changed since the deal was saved. | Not a fault; the user dismisses it. See MODEL_CHANGELOG.md |

**Useful public pages** (no sign-in): `https://api.variater.com/api/health`
(`environment` and `commit`), `/api/health/database`, `/api/health/limits`,
`/api/health/jobs`, `/api/health/files`, `/api/health/reader`. The same
paths work on `https://fse-api-staging.onrender.com`. The status page is
<https://variater.betteruptime.com>.

**Logs.** Render → the service → Logs: one JSON line per request with a
`request_id`. Sentry: search `request_id:<id>`.

## 6. Rotate each secret

Rotate a secret when it may have leaked (it appeared in a log, a screenshot,
a public commit), when someone who knew it leaves, or yearly. Never paste a
secret into a pull request, a chat or a command line you will keep in history:
put it in the service's dashboard or in GitHub's secret form.

Where a secret lives matters; "Render" means **both** services (`fse-api` and
`fse-api-staging`) unless noted; staging usually has its own value. GitHub
secrets are under repository → **Settings → Secrets and variables →
Actions**. Dashboard menus move: DEPLOY.md has the click paths that were
true when each was set up.

**After any change to a Render variable**, Render redeploys. After any change to
a Vercel variable, **trigger a new deploy**: web values are baked in at build
time.

| Secret | Where it lives | How to rotate | Check |
|---|---|---|---|
| **Clerk keys** (`pk_`/`sk_`) | Vercel (`NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`); Render (`CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`). **Two instances**: production (`pk_live_`) in Vercel Production scope and `fse-api`; development (`pk_test_`) in Preview scope and `fse-api-staging` | Clerk dashboard → the instance → API keys → regenerate the secret key; paste it into Vercel and `fse-api` (production) or Vercel Preview and staging (development); redeploy Vercel. Never mix the instances. Sign-in is briefly broken between the web build and the API redeploy | Sign in on the site |
| **Neon `fse_api` password** | `DATABASE_URL` on Render | Neon SQL Editor, on the right branch: `ALTER ROLE fse_api PASSWORD '...';`, then update `DATABASE_URL` on Render (DEPLOY.md "Least-privilege database role") | `/api/health/database` shows `"role": {"status": "restricted"}` |
| **Neon owner password** (`neondb_owner`) | `DATABASE_MIGRATION_URL` (Render); GitHub secrets `BACKUP_DATABASE_URL` (production, unpooled), `BACKUP_STAGING_DATABASE_URL` | Neon console → Roles → reset the password; update **all three** places and `fse_api`'s string if it shares the host | `/api/health/database`; run the `backup` workflow by hand |
| **Upstash token** | Render and GitHub secrets: `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN`; also the same two under *Dependabot* secrets | Upstash console → the database → reset the token; update all of those | `/api/health/limits` says `"store": "upstash"` and reachable |
| **Supabase backups key** | GitHub secrets `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | Supabase → Project Settings → API Keys → rotate the secret key; update the GitHub secret | Run the `backup` workflow by hand |
| **Supabase uploads key** | Render: `SUPABASE_UPLOADS_URL`, `SUPABASE_UPLOADS_KEY` (a different project from the backups'). Must start `sb_secret_` | Same, in the uploads project; update both Render services | `/api/health/files` answers `"key": "sb_secret"` |
| **`FSE_BACKUP_KEY`** | GitHub secret, **and a copy in your password manager** | Make a new one: `.venv/bin/python -c "import base64, secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"`. **Warning:** backups already stored were encrypted with the old key and cannot be read with the new one. Keep the old key too until the old backups have rotated out (up to a year), or accept that older backups are gone. This happened on 2026-10-12 | `backup` workflow by hand (it downloads and decrypts what it stored) |
| **Gemini API key** (`GEMINI_API_KEY`) | Render; a GitHub secret (recording only) | Google AI Studio, project `variater-reader`: create a key, delete the old one | `/api/health/reader` says the key works |
| **Resend key** (`RESEND_API_KEY`) | Render | Resend → API keys → create (sending access, domain `send.variater.com`), delete the old | `GET /api/teams` while signed in says `"mail": "resend"` |
| **Companies House, EDINET, FRED keys** | Render; GitHub secrets (`COMPANIES_HOUSE_API_KEY`, `EDINET_API_KEY`, `FRED_API_KEY`) | Each provider's site: make a new key, replace everywhere | The nightly `scheduled` run; `/api/health/jobs` |
| **Better Stack token** (`BETTERSTACK_API_TOKEN`) | GitHub secret | Better Stack → API tokens → create, replace, delete the old | Run the `monitoring` workflow by hand |
| **Vercel bypass secret** (`VERCEL_AUTOMATION_BYPASS_SECRET`) | GitHub secret, and Vercel's "Protection Bypass for Automation" for the project | Regenerate it on Vercel and paste into the GitHub secret | The next `staging` run |
| **Sentry DSN** (`SENTRY_DSN`) | Render; Vercel (Production and Preview) | It is an address, not a password, but it can be abused to send junk. Sentry → project → Client Keys → make a new key; update all; redeploy Vercel | A new error appears in Sentry |
| **Google OAuth secret** | Clerk production → SSO connections → Google | Google Cloud (project `variater`) → Clients → the client → Add secret; paste into Clerk | Sign in with Google |
| **GitHub token for `gh`** | Your Mac's keychain | `gh auth login` again (a fine-grained token for this repository only) | `gh auth status` |

If a secret was **committed to git**, rotating it is the fix; deleting the
commit is not (the repository is public and the `security` workflow scans all
history on purpose).

## 7. Restore a backup

First decide what happened.

| What happened | Use |
|---|---|
| A bad migration, a wrong delete, something you noticed within a day | **Neon's restore window** (the fast way, loses nothing else): Neon console → **Branches** → `production` → **Restore**, pick a time |
| Older than that, or the Neon project is gone | **The encrypted nightly backup** in Supabase Storage |

Everything below needs, in your shell (a git-ignored `.env` file loaded with
`set -a; source .env; set +a`, never typed into a command): `FSE_BACKUP_KEY`,
`SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`. And a `pg_restore` at least as
new as Neon's Postgres (18): `brew install postgresql@18`.

**1. See what exists, and prove the newest one decrypts:**

```bash
.venv/bin/python -m ops.backup list --environment production
.venv/bin/python -m ops.backup verify --environment production
```

**2. Restore into a new, empty database** (create a Neon project or branch
first, and use its **owner** connection string, unpooled):

```bash
.venv/bin/python -m ops.backup restore --name production/<name from list> --into "$NEW_DATABASE_URL"
```

Over a database that still holds wrong data, add `--clean`.

**3. Put the app's rights back.** A restore brings the tables and rows, not
the permissions. Run the `GRANT`/`REVOKE` block in DEPLOY.md "Restoring" as
the owner (copy it from there: it must match migration `0005` and `0010`).
A brand-new Neon project also needs the roles `fse_app` and `fse_api` made
first (DEPLOY.md "Least-privilege database role").

**4. Point the API at it.** Set `DATABASE_URL` (and `DATABASE_MIGRATION_URL`)
on `fse-api`, using the `fse_api` role as before. Render redeploys.

**5. Check:**

```bash
python3 ops/check_database.py https://api.variater.com --require-restricted-role
```

It must report migrations current **and** `role: restricted`. Then sign in
and open a saved deal: its IRR must be what it was.

**Practice it.** The `backup` workflow does a restore drill on the first of
each month into a throwaway database on Neon's staging branch. To run it now:
GitHub → Actions → **backup** → **Run workflow**, tick *Run the restore
drill*. By hand: `.venv/bin/python -m ops.backup drill --environment production --target "$STAGING_OWNER_URL"`.
It prints counts and "match", never a deal.

**Remember:** if `FSE_BACKUP_KEY` is lost, every backup is unreadable.

## 8. The accounts table

Every outside service, what it is for, and what to do when it fills up. Free
limits are those the project was built around; providers change them, so read
the provider's pricing page before a decision (the date this was checked is at
the top of this page). "Upgrade" rows link to the PLAN.md phase 12 task and to
DEPLOY.md "Paid switches", which says the exact setting each switch needs.

| Service | What it does here | Variables / secrets | Free limit that matters | When and how to upgrade |
|---|---|---|---|---|
| **GitHub** (repo, Actions, Dependabot) | Code, pull requests, every check, every nightly job, secrets | Actions secrets (section 6); `gh` token on your Mac | Actions minutes are free on a public repository; **scheduled workflows pause after 60 days with no commit** | Not needed while public. Making the repository private changes this: Actions minutes become metered |
| **Vercel** (web, Hobby) | Hosts the Next.js app at `variater.com`; previews are staging | `FSE_API_URL`, `FSE_STAGING_API_URL`, `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`, `SENTRY_DSN` | Hobby is **for non-commercial use** | Pro **before charging anyone**: [12.3](../PLAN.md) |
| **Render** (API, free web services) | Runs the API (`fse-api`) and staging (`fse-api-staging`) from the Docker image | `DATABASE_URL`, `DATABASE_MIGRATION_URL`, `CLERK_*`, `UPSTASH_*`, `FRED_API_KEY`, `COMPANIES_HOUSE_API_KEY`, `EDINET_API_KEY`, `SUPABASE_UPLOADS_*`, `AI_PROVIDER`, `GEMINI_API_KEY`, `AI_FREE_TIER`, `RESEND_API_KEY`, `FSE_MAIL_FROM`, `FSE_APP_URL`, `SENTRY_DSN`, `FSE_CORS_ORIGINS`, `FSE_ADMINS`, `FSE_PLAN` | 750 instance hours a month across services; sleeps after 15 idle minutes (a minute to wake); 512 MB of memory (why simulations are capped) | Always-on instance when the wake-up annoys users or memory is the limit: [12.1](../PLAN.md); "Paid switches" in DEPLOY.md |
| **Neon** (Postgres) | All saved data; branches `production` and `staging` | `DATABASE_URL`, `DATABASE_MIGRATION_URL`, and the GitHub secrets `BACKUP_DATABASE_URL`, `BACKUP_STAGING_DATABASE_URL` | 0.5 GB storage; 100 compute hours a month; a 24-hour restore window | When storage nears 80% (`/api/health/database` warns) or you want a longer restore window: [12.2](../PLAN.md) |
| **Upstash** (Redis) | Shared daily usage counters | `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` (Render, Actions, Dependabot) | 500,000 commands a month (the app uses ~30% and stops at a daily budget) | Pay-as-you-go if `/api/health/limits` reports over budget regularly; limits keep working from the database meanwhile |
| **Clerk** (sign-in) | Email and Google sign-in; names for teams | `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`, `CLERK_SECRET_KEY`, `CLERK_PUBLISHABLE_KEY`, `CLERK_ISSUER` | The free plan's monthly active users; the development instance is limited to 100 users | A paid plan when the monthly user limit nears; later company logins: [12.6](../PLAN.md) |
| **Cloudflare** (DNS for `variater.com`) | The domain and its records (all **DNS only**, grey cloud) | none | Domain registration renews yearly | Keep auto-renew on and a valid card (section 9) |
| **Sentry** | Error reports, scrubbed of deal contents | `SENTRY_DSN` | A monthly event quota on the free plan | If errors are dropped: fix the noisy error first, then upgrade |
| **Better Stack** | Uptime monitors, status page, incident emails | `BETTERSTACK_API_TOKEN` (GitHub), set up by `ops/betterstack.py` | Free monitors and status page; check frequency | The API monitor is deliberately 30 minutes so the free API can sleep (`tests/test_betterstack.py` guards it) |
| **Supabase** (two projects) | One holds encrypted **backups**; one holds uploaded **files** | `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (backups, GitHub); `SUPABASE_UPLOADS_URL`, `SUPABASE_UPLOADS_KEY` (uploads, Render) | 1 GB storage each; 5 GB downloads a month; a project **pauses after a week with no activity** (the nightly jobs prevent this) | Paid plan for the uploads project (`FSE_UPLOADS_BUDGET_MB`), and larger backups need resumable uploads: [12.2](../PLAN.md) |
| **Google AI (Gemini)** | Reads results filings and documents; writes linking sentences for notes and memos | `AI_PROVIDER`, `GEMINI_API_KEY`, `AI_FREE_TIER`, `GEMINI_MODELS_A/_B/_WRITE`, `READER_HIGH_RESOLUTION` | Free tier: 20 requests a day per model; Google may use free-tier content, so every read needs the person to confirm the file is public | Billing with a budget alert **before the first outside showing**, then `AI_FREE_TIER=0`; about 0.09 US dollars per filing. See DEPLOY.md "Paid switches"; another provider for confidential documents: [12.4](../PLAN.md) |
| **Resend** (email) | Batched mention emails for teams | `RESEND_API_KEY`, `FSE_MAIL_FROM`, `FSE_APP_URL` | 100 emails a day, 3,000 a month (the app stops at 90 and 2,900) | Paid plan if teams grow; until then mentions wait up to a week, then are dropped |
| **Google Cloud** (OAuth) | The "Sign in with Google" client for production | in Clerk (production) | Free | Nothing to upgrade. Its consent screen links to `/privacy` |
| **Microsoft Partner Center** | Listing the Excel add-in in the store | none (manifests are generated: `python -m ops.addin_manifest`) | Free | Not set up yet; DEPLOY.md "Listing it in Microsoft's store" |
| **Companies House, EDINET, FRED** (data keys) | UK and Japanese filings; US rates | `COMPANIES_HOUSE_API_KEY`, `EDINET_API_KEY`, `FRED_API_KEY` | Their published rate limits (the code paces itself inside them) | Free; licensed market data for corporate credit spreads is [12.9](../PLAN.md) |
| **SEC, ECB, IMF, World Bank, BIS, OECD, GLEIF, Damodaran** (no key) | Company and economic data | none | Be restrained; the code paces calls | Free |

**Payments and higher service limits** (a Stripe account, live payments) come
with [12.5](../PLAN.md) and [12.8](../PLAN.md); nothing in the app charges
anyone today.

## 9. The monthly 15-minute checklist

Do it on the first of the month, the day the restore drill runs. Tick each line.

1. **Scheduled workflows are still enabled** (2 min). GitHub → Actions → left
   column: `scheduled`, `backup`, `live`, `monitoring`, `security`,
   `codeql`. GitHub **disables schedules after 60 days with no commit**; a
   workflow shown with a "disabled" banner is re-enabled with its **Enable
   workflow** button. Normal merging prevents it. If a month passes with no
   commit, make a small `docs:` commit.
2. **Last night's jobs passed** (1 min).
   `curl -s https://api.variater.com/api/health/jobs` shows each task's newest
   run and status.
3. **Backups are fresh and the drill passed** (2 min).
   `.venv/bin/python -m ops.backup list --environment production` shows last
   night's backup; the `backup` workflow's first-of-the-month run (the drill)
   is green. Confirm your password manager still holds `FSE_BACKUP_KEY`.
4. **Free-limit usage** (5 min). Look at each:
   - `curl -s https://api.variater.com/api/health/database`: storage under
     80% of 0.5 GB, and the `company_data` / `economic_data` / `benchmark_data`
     budgets.
   - Neon console: compute hours used (of 100).
   - Render dashboard: instance hours used (of 750).
   - `curl -s https://api.variater.com/api/health/limits`: Upstash reachable,
     not over budget.
   - Supabase (both projects): storage and monthly downloads; both projects
     active (not paused).
   - Sentry: events this month, and any new unresolved issue.
   - Resend: emails sent this month.
   - Google AI Studio: usage and, once billing is on, the budget alert; the
     nightly summary has yesterday's `ai_cost_usd_yesterday`.
   - Clerk: monthly active users.
5. **Dependabot is merged** (2 min). `gh pr list --label dependencies` (or
   the Pull requests tab): merge the green ones, deal with the red ones
   (section 4). Check **Security → Dependabot alerts** is empty.
6. **Domain and certificates** (1 min). Open <https://variater.com> and
   <https://api.variater.com/api/health> and see no browser warning. Vercel
   and Render renew certificates themselves while the Cloudflare records stay
   **DNS only**; Cloudflare → Domain registration shows the renewal date and a
   working card.
7. **The status page and uptime** (1 min). <https://variater.betteruptime.com>
   shows both monitors up for the month; no open incident.
8. **Bring staging level** if it is behind `main` (section 3, step 9).

Once a quarter: rotate nothing unless something leaked, but re-read the
accounts table and delete accounts and keys for services you no longer use.
Once a year: the yearly data checks (base-rate editions, Damodaran's January
tables, the `firm_growth` rerun in CLAUDE.md "Growth calibrator") and secret
rotation (section 6).

---

*Last full check of the commands and variable names against the code:
2026-11-04. Update this page in the same pull request as any change to how
the app is built, run or deployed (PLAN.md 11.4).*
