# Threat model

PLAN.md 1.7, written 2026-09-17. Review it whenever a task adds a new kind of
data, a new outside service, a new way in (an upload, a webhook, an AI
provider) or moves to a paid plan (phase 12). Each "Open" item names the plan
task expected to close it.

## What the system is

```
                    ┌──────────── Vercel (Hobby) ────────────┐
 browser ──HTTPS──> │ Next.js pages   proxy.ts: sign-in,     │
   │                │                 CSP nonce, headers     │
   │                │ /api/* rewrite ─────────────────────────┼──HTTPS──> Render (free): FastAPI
   │                └─────────────────────────────────────────┘             │  auth.py   limits.py
   │                                                                        │  security.py
   ├──HTTPS──> Clerk (sign-in, session tokens)                              ├──TLS──> Neon Postgres (free)
   └──HTTPS──> Sentry (browser errors, scrubbed)                            ├──HTTPS─> Upstash Redis (free)
                                                                            ├──HTTPS─> Clerk JWKS (public keys)
 GitHub Actions: CI, live and staging checks, monitor sync                  ├──HTTPS─> SEC EDGAR, FRED (public data)
   (secrets: Upstash, Better Stack, Vercel bypass)                          └──HTTPS─> Sentry (API errors, scrubbed)
```

## What we protect

| Asset | Where it lives | Why it matters |
|---|---|---|
| Users' deals and versions (inputs, settings) | Neon `deals`, `deal_versions` | Confidential deal terms; the main thing a user trusts us with |
| Account profile (Clerk user id, country, currency, locale, time zone) | Neon `users` | Links a person to their deals; names and emails stay in Clerk |
| Sessions | Clerk (cookie in the browser, short-lived token sent as a bearer token) | Whoever holds one acts as the user |
| Secrets (Clerk secret key, database URLs, Upstash token, Better Stack token, Vercel bypass) | Vercel, Render, GitHub Actions secrets | Each opens a service or its data |
| The public repository | GitHub | Anyone can read it; anything committed is public forever |
| Service availability and free-plan quotas | Render hours, Neon compute hours and 0.5 GB, Upstash commands, Sentry events | Exhausting a quota is an outage on the free plans |
| Model integrity (code, trained model files, defaults) | Repository, Docker image | Wrong numbers look like right numbers |

## Who might attack, and how

| Actor | Can | Wants |
|---|---|---|
| Anonymous internet user | Reach both sites, read the public repo | Data, a free compute resource, defacement, to knock the free services over |
| Signed-in user (anyone can sign up) | Everything a user can, with a valid token | Other users' deals, more than their share of compute |
| Malicious page or browser extension | Run script in a user's browser if the app lets it (XSS) or via the extension | The session, deal contents |
| Compromised dependency or action | Run code in CI, the build or the running app | Secrets, data, a foothold |
| Someone with a leaked secret | Whatever the secret opens | Data or service abuse |
| Contributor mistake | Commit a key, add an open route, log deal contents | (accidental) |

## Threats and what stops them

Status: **Done** (in place and tested), **Partial**, **Open**.

### Spoofing (pretending to be someone)

| Threat | Protection | Status |
|---|---|---|
| Calling the API as someone else | Every route except `PUBLIC_PATHS` needs a Clerk session token, verified locally (RS256 signature against the instance's JWKS, issuer, expiry, not-before); dependency on `include_router`, so new routes are protected by default (`api/auth.py`, `tests/test_auth.py`) | Done |
| The development sign-in (`dev:<name>`) reaching production | Refused in production and whenever Clerk is configured; `staging.yml` checks staging refuses it | Done |
| Forged cross-site requests | The API reads a bearer token, not cookies, and CORS allows only the app's own origin without credentials (`api/security.py`) | Done |
| Account takeover through weak sign-in | Clerk handles passwords, email verification, Google sign-in and bot protection | Partial: Clerk **development** instance; a production instance with its own domain comes with a custom domain (phase 12.3) |

### Tampering (changing things we rely on)

| Threat | Protection | Status |
|---|---|---|
| Editing another user's deal | Every read and write matches the caller's subject in the same statement; another account's deal answers 404 (`db/deals.py`, `tests/test_deals.py`) | Done |
| SQL injection | SQLAlchemy Core with bound parameters throughout; no string-built SQL from user input | Done |
| A bug or injection in a request changing the schema or history | The API's database role (`fse_app`) can only select, insert, update and delete rows; migrations run as the owner (`DATABASE_MIGRATION_URL`); `tests/test_security.py` proves `CREATE`, `DROP`, `ALTER`, `TRUNCATE` and writing `alembic_version` are refused | Done: production and staging connect as `fse_api` since 2026-09-17; the daily and staging database checks fail if either doesn't |
| Traffic tampered with in transit | HTTPS everywhere (HSTS on web and API); a deployed API refuses a database URL that allows plain text and adds `sslmode=require` when missing (`db/engine.py`) | Done |
| Malicious code in a dependency | Lockfile for npm; Dependabot updates; `pip-audit` and `npm audit` on every PR and weekly; CodeQL on every PR | Partial: Python requirements are ranges, not a lockfile, so a build can pick up a new release (a hash-pinned lock is worth adding with 11.3) |
| A tampered GitHub Action | Only first-party (`actions/*`, `github/codeql-action`) actions are used; gitleaks is downloaded from its release and checksum-verified; workflow tokens are read-only by default | Partial: actions are pinned to major versions, not commit SHAs (Dependabot keeps them current) |
| Tampered trained model files (`*.pkl` load with pickle, which can run code) | Loaded only from files in the repository and image, never from users; changes go through a reviewed PR | Partial: no integrity hash recorded; 5.1 (ML evaluation) should record hashes and prefer safe formats |
| A backup altered in storage and restored unnoticed | Each 1 MiB chunk's AES-GCM tag covers the header, the chunk's number and whether the file ends there, so an altered, reordered, duplicated or truncated backup fails to decrypt instead of restoring wrong data; the manifest's SHA-256 is checked before anything is restored (`tests/test_backups.py`) | Done (1.8) |
| Backups that quietly stop, or can't actually be restored | The nightly run reads back and decrypts what it just uploaded; a monthly drill restores into a throwaway database and checks every deal gives the same model result as the manifest recorded from the dump's own snapshot; either failing raises a Better Stack incident | Done (1.8) |

### Repudiation (denying what happened)

| Threat | Protection | Status |
|---|---|---|
| No record of who did what | JSON request logs with request ID, route template, status and timing (no bodies or query strings); deal versions record changes per deal | Partial: no audit log of account actions (sign-in history is in Clerk) — PLAN.md 7.2 (teams) should add one |

### Information disclosure

| Threat | Protection | Status |
|---|---|---|
| Deal contents or personal data in logs or error reports | No request bodies, query strings, local variables, user details, IP addresses or console output in logs or Sentry; tested (`tests/test_observability.py`, `e2e/monitoring.spec.ts`) | Done |
| XSS stealing a session or deal | React escaping; a strict content security policy with a per-request nonce and `'strict-dynamic'` (no inline or third-party script without the nonce, no plugins, no framing, no `<base>` changes); `e2e/security.spec.ts` shows an injected inline handler is refused | Done. Styles allow `'unsafe-inline'` (Clerk injects styles), so CSS injection is still possible but can't run script |
| Clickjacking | `frame-ancestors 'none'` and `X-Frame-Options: DENY` on web and API | Done |
| API answers cached by a browser or proxy | `Cache-Control: no-store` on every API answer | Done |
| Secrets committed to the public repository | gitleaks over the full history on every PR (with a planted-key self-test); `.gitignore` for `.env`, local databases and browser state; review of what's public (below) | Done |
| Framework fingerprinting | No `X-Powered-By` (Next) and no `server: uvicorn` header | Done |
| Timing or enumeration of deals | Other users' deals and missing deals answer the same 404; ids are random UUIDs | Done |
| Data in Upstash identifying users | Keys hash the user; no personal data sent | Done |
| A stolen database URL | Neon requires TLS and a password; the API role can't change the schema. Neon's free plan has no IP allow list | Open: IP allow lists are a paid Neon feature (phase 12.2) |
| A stolen **owner** database URL (the backup secrets) | `BACKUP_DATABASE_URL` and `BACKUP_STAGING_DATABASE_URL` are the schema owner, because a dump must read every table and a restore must create them. They are GitHub Actions secrets only, never in Render or Vercel, and no workflow prints them | Accepted: the highest-value secrets here, alongside the backup key |
| Confidential documents sent to a free AI tier | Rule in PLAN.md: free AI tier only on public documents | Open until 6.2 and 12.4 |
| A backup read by whoever can reach the storage | Every backup is AES-256-GCM before it leaves the runner; the key is a GitHub Actions secret, held nowhere else; the Supabase bucket is private; backups are never Actions artifacts (public repositories hand those to anyone) | Done (1.8) |
| The backup key lost or leaked | A leak needs a new key and re-encrypted backups; a loss makes every stored backup unreadable, so a copy is kept outside GitHub (DEPLOY.md). Neon's own restore window covers recent mistakes either way | Accepted: one key, rotated by hand |
| Production deals copied onto the staging branch by the restore drill | The drill restores into a **new** database it creates and drops, on the same Neon project (one account, one trust boundary); it reports counts and a one-way fingerprint, never deal contents, and needs no access to the production database | Done (1.8) |

### Denial of service and quota exhaustion

| Threat | Protection | Status |
|---|---|---|
| Flooding the API | Per-address limit before sign-in (1,200 requests and 60 refused sign-ins a minute), per-user limits after (240 requests and 30 runs a minute, 1,000 runs a day), 1 MB bodies, 100,000-path simulations, one simulation at a time, 100 s timeout (`api/limits.py`) | Done |
| Burning Neon compute hours or Render hours | Health checks never touch the database; no connection at start-up; storage warning at 80% | Done |
| Burning Upstash commands | Counts batched every 5 minutes under a daily command budget, database fallback | Done |
| Burning Sentry events | Deliberate test error limited to one a minute and never in production | Done |
| Large-scale DDoS | Vercel's and Render's edge protection only | Open: a WAF or paid tier (phase 12) |

### Elevation of privilege

| Threat | Protection | Status |
|---|---|---|
| A new route shipped without sign-in or limits | Both are dependencies on `include_router`, not per route | Done |
| The container running as root | The image runs as user `app` (checked by CI's `docker` job) | Done |
| Workflow token abuse from a pull request | `permissions: contents: read` on every workflow; secrets are not given to forks' pull requests by GitHub | Done |

## Open items, by plan task

| Item | Task |
|---|---|
| Automated security testing against staging (DAST, dependency review gate) | 10.2 |
| Record trained-model hashes; prefer non-pickle formats | 5.1 |
| Hash-pinned Python lockfile | 11.3 |
| Audit log of account and team actions | 7.2 |
| Clerk production instance on a custom domain | 12.3 |
| Neon IP allow list, longer restore window | 12.2 |
| Paid AI provider with no training on inputs, for confidential documents | 12.4 |
| Company sign-in (SSO) | 12.6 |

## What's public in this repository (reviewed 2026-09-17)

The repository is public. Reviewed for data, keys and internal notes:

- **Keys:** none. gitleaks over all 75 commits of history and the working
  tree found nothing outside git-ignored folders (the local virtualenv and
  build output). Secret *names* (`DATABASE_URL`, `UPSTASH_REDIS_REST_TOKEN`…)
  appear in docs and workflows by design; values never do.
- **Data:** no user or customer data. `tests/golden/golden.json`,
  `web/e2e/fixtures/ml-responses.json` and `ml/anomaly_deals.json` hold model
  outputs and figures for four well-known public buyouts (the inception-era
  examples PLAN.md 2.1 labels as illustrative). `Simulation_Model_Template.xlsx`
  is an empty template (author metadata: openpyxl).
- **Internal notes:** `CLAUDE.md`, `PLAN.md` and `DEPLOY.md` describe the
  infrastructure (service names, public URLs, the Neon region, which free
  plans are used). That helps an attacker only as much as the public URLs
  already do, and it keeps the project reproducible; no credentials, private
  URLs that grant access, or personal details are in them. The design mockup
  link in `CLAUDE.md` is a private claude.ai artifact that only its owner can
  open.
- **Personal details:** the commit author email of earlier commits is a
  personal address. Rewriting published history would break every clone and
  pull request, so it stays; new commits should use the GitHub
  `…@users.noreply.github.com` address (DEPLOY.md "Security" says how).
- **Contact string:** `ml/edgar_extractor.py` sends the User-Agent the SEC
  requires with a placeholder contact address. Not a secret; SEC's fair-access
  rules want a real contact, so it should become configuration when EDGAR use
  grows (4.1, company filings).
