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
| Calling the scheduler's endpoints (`/api/scheduled/*`) | A GitHub Actions OIDC token, verified locally against GitHub's keys: this repository's id and name, an allowed workflow file on a protected branch, and an audience naming the environment (a staging token is refused by production). No shared secret exists to leak; forks' pull requests get no token (`api/github_oidc.py`, `tests/test_jobs.py`) | Done |
| Reading or cancelling another account's background job | Every job read and write matches the owner in the same statement; another account's job answers 404 (`jobs/`, `tests/test_jobs.py`) | Done |
| Using the Excel add-in's sign-in as more than it is (PLAN.md R3) | The pane's pass is made only for a caller signed in on the site itself (a pass can't make another), reaches only `ADDIN_PATHS`, by method (research, the three upload steps of the PDF it reads, its own session; not the file list, download links or deleting; anything else is 403), lasts 12 hours, at most five a person, ends at sign-out at once (the process's 60-second cache is cleared with it), and is stored only as its SHA-256 (`db/addin.py`, `api/auth.py`, `tests/test_addin.py`). It is kept in the pane's browser storage, which the pane's strict policy guards like the rest of the site | Done |
| Framing the site to trick a click (clickjacking) | `frame-ancestors 'none'` and `X-Frame-Options: DENY` on every page except the add-in's pane (`/excel`), which names Office's own hosts and no other (its sign-in dialog is never framed); the header scan holds both to that (`ops/check_headers.py`, `web/e2e/excel.spec.ts`) | Done |
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
| No record of who did what | JSON request logs with request ID, route template, status and timing (no bodies or query strings); deal versions record changes per deal | Done: an append-only audit history of deal, account and team actions (PLAN.md 3.3, 7.2: sharing, comments, invitations, roles), shown per deal with who did it; sign-in history is in Clerk |

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

### Uploaded files (PLAN.md 6.1)

| Threat | Protection | Status |
|---|---|---|
| A file that does harm when opened (macros, scripts, embedded objects, remote templates) | Every upload is scanned before it can be downloaded (`uploads/scan.py`), read the way its own program reads it: PDFs object by object with pypdf, object streams included, beside a search of the raw bytes; Office XML with a real XML parser (character references, UTF-16, document type declarations refused); older Office files with oletools. Anything that can act, anything encrypted and anything the scanner can't read is refused and its bytes deleted; a scanner failure refuses too | Done, as a **structural** scan, after two rounds of review whose every bypass is a test. Two readers of one file can still be made to disagree by a file built for it (an object stream only a PDF reader in repair mode would find is refused when the bytes declare it, and missed when they don't) |
| Known malware with no active content, or an exploit of a reader's own bug | Not detected: ClamAV needs 1.7 GB (measured) and the free server has 512 MB. Files are private to their uploader, so the readers at risk are that person and this server's parser (6.2), which must treat every file as hostile | Open: signature scan on a worker with memory (12.1) |
| A file built to hang or bloat the scanner (the first version's patterns took minutes on a 30 KB file, found in review) | The scan runs in a child process with a CPU limit, an address-space limit (Linux) and a 150 s deadline, without the server's secrets in its environment; a child that dies or overruns is a file refused. XML is parsed in one pass; nothing is inflated past fixed caps; no entity expansion | Done; tested with the reviews' files. Open: a file that costs the scanner its full minute holds the slot simulations share for that minute, and a person may upload 20 an hour; dedicated workers (12.1) take scans off that slot |
| A file that is not what its name says (a program named `.pdf`) | Kind decided by contents, not name or declared type; macro-enabled formats not accepted at all | Done |
| Reading or deleting someone else's file | Owner matched in every statement (`db/files.py`): another account's file answers 404 like a missing one. The bucket is private; a download link is signed, for one object, for 60 seconds, and only made for the owner of a clean file | Done; tested |
| Links to other workbooks (`externalLinkPath`), which Excel asks about before following, and web links to click | Kept: nearly every finance model has them. A network path in one gives away the reader's Windows sign-in if they agree to update links | Accepted |
| Storing more than was counted, or swapping a file after its scan | No upload link is ever handed out: the bytes arrive at the API, are counted as they stream to disk and cut off past the declared size, and only the server writes to the bucket, once per name. The copy scanned is the copy stored; bytes whose row vanished while they were being stored are deleted at once | Done; tested |
| The store's secret key leaking | Only on the API services, in headers only, never in an address, a log or an error; redirects are never followed. DEPLOY.md asks for a Supabase project of its own for uploads, so that this key cannot reach the backups | Done in code; the separate project is a set-up choice |
| Filling the free 1 GB, or queueing scans for everyone to wait behind | Room reserved before a byte is taken, and a row counts until its bytes are deleted; 20 MB a file (also set on the bucket), 100 MB and 40 files a person, 800 MB in all; three files a person in flight, 20 uploads an hour, two arriving at once (with a deadline, and a limit on room reserved and never used); files deleted after 90 days, room for an upload that never came after 30 minutes | Done |
| A large request exhausting memory | A file's bytes are the one request body not read into memory: streamed to disk, only for a signed-in caller whose own file is waiting for them. Every other request still stops at 1 MB | Done; tested |
| A file's name or contents in logs | Rows hold the name the person gave and nothing from the file; names, links and scanner messages are never logged (tested); object names are `u<user id>/<file id>` | Done |
| The local directory store's unauthenticated download links reachable in production | The store is refused in production and whenever Supabase is configured; its route then answers 404 | Done; tested |

### Denial of service and quota exhaustion

| Threat | Protection | Status |
|---|---|---|
| Flooding the API | Per-address limit before sign-in (1,200 requests and 60 refused sign-ins a minute), per-user limits after (240 requests and 30 runs a minute, 1,000 runs a day), 1 MB bodies, 100,000-path simulations, one simulation at a time, 100 s timeout (`api/limits.py`) | Done |
| Burning Neon compute hours or Render hours | Health checks never touch the database; no connection at start-up; storage warning at 80% | Done |
| Burning Upstash commands | Counts batched every 5 minutes under a daily command budget, database fallback | Done |
| Filling the job queue or the database with jobs | Submitting counts as a model run; 4 unfinished jobs per account, 40 in the queue; inputs deleted when a job ends, results after 6 hours, rows after 7 days; results over 2 MB refused (`jobs/queue.py`) | Done |
| Background jobs keeping Neon awake | The runner starts only when a job is submitted or checked and stops after a minute idle | Done |
| Burning Sentry events | Deliberate test error limited to one a minute and never in production | Done |
| Large-scale DDoS | Vercel's and Render's edge protection only | Open: a WAF or paid tier (phase 12) |

### Elevation of privilege

| Threat | Protection | Status |
|---|---|---|
| A new route shipped without sign-in or limits | Both are dependencies on `include_router`, not per route | Done |
| The container running as root | The image runs as user `app` (checked by CI's `docker` job) | Done |
| Workflow token abuse from a pull request | `permissions: contents: read` on every workflow; secrets are not given to forks' pull requests by GitHub. `scheduled.yml` and `staging.yml` also have `id-token: write`, which only mints the OIDC token above; the API accepts it only from those workflows on `main`/`staging` | Done |

## Shared deals (PLAN.md 7.2)

| Threat | Control |
|---|---|
| A member reaches a deal not shared with their team, or does more than their role allows | Access is worked out in the same SQL statement as each read and write (`db/deals.py` `_reachable`, `_access`); a reader without the right gets 403, anyone else 404, as for a missing deal. `tests/test_teams.py` runs every deal endpoint for owner, team owner, editor, viewer, removed member and stranger |
| An invitation link leaks or is reused | Kept only as a SHA-256 hash; ends after a week or its number of uses; owners withdraw it; the token travels after `#`, so no server log sees the page request |
| A mention email leaks deal contents | It says who mentioned you and when, with a link; never the comment, the deal's name or a figure (tested) |
| Names or addresses stored or logged | Neither: read from Clerk per request (names cached in memory ten minutes, addresses never), and the secret key goes in a header only |

## Open items, by plan task

| Item | Task |
|---|---|
| Automated security testing against staging (DAST, dependency review gate) | 10.2 |
| Record trained-model hashes; prefer non-pickle formats | 5.1 |
| Hash-pinned Python lockfile | 11.3 |
| Clerk production instance on a custom domain | 12.3 |
| Neon IP allow list, longer restore window | 12.2 |
| Paid AI provider with no training on inputs, for confidential documents | 12.4 |
| Company sign-in (SSO) | 12.6 |
| Signature (antivirus) scan of uploaded files: ClamAV needs about 1.7 GB, so it waits for a dedicated worker | 12.1 |
| No cap on deals per account or saved versions per deal: automatic checkpoints are bounded, explicit saves are not, and a deal with twelve facilities is about 20 KB, so one account could fill Neon's free 0.5 GB in hours (found reviewing 2.4a) | 1.6 follow-up |
| `sens_*` settings are unbounded and the deal endpoints have no timeout: `sensitivity_holding_periods` builds `range(lo, hi + 1)` unchecked, and `/api/deal/run` is not in `RUN_PATHS`, so a raised `sens_hp_max` makes one request hold a thread for seconds (found reviewing 2.4a; predates it) | 1.6 follow-up |
| Exception messages still reach the JSON logs and Sentry: `JsonFormatter` includes `formatException` and `scrub_event` clears frame variables but not `exception.values[].value`, though both say type-and-traceback only. Nothing leaks today, because every refusal is returned rather than raised, but the hardening is worth doing | 10.2 |

## What's public in this repository (reviewed 2026-09-17)

The repository is public. Reviewed for data, keys and internal notes:

- **Keys:** none. gitleaks over all 75 commits of history and the working
  tree found nothing outside git-ignored folders (the local virtualenv and
  build output). Secret *names* (`DATABASE_URL`, `UPSTASH_REDIS_REST_TOKEN`…)
  appear in docs and workflows by design; values never do.
- **Data:** no user or customer data. `tests/golden/golden.json`,
  `web/e2e/fixtures/ml-responses.json` holds model
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
