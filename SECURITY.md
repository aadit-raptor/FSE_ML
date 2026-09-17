# Security policy

## Reporting a vulnerability

Please report security problems **privately**, not in a public issue:

1. Go to this repository's **Security** tab.
2. Click **Report a vulnerability** and describe what you found, how to
   reproduce it and what it affects.

You'll get an acknowledgement within 7 days. Please give us a reasonable time
to fix the problem before telling anyone else, and don't access other
people's data, run denial-of-service tests or send automated scans at the
live site (https://fse-ml.vercel.app, https://fse-api.onrender.com) beyond
what is needed to show the problem. Those run on free plans and are easy to
knock over.

## Supported versions

Only the current `main` branch, which is what production runs, gets fixes.

## What's in place

The threat model, with what each protection is for and what is still open,
is in [docs/security/threat-model.md](docs/security/threat-model.md). In short:

- Every API call except the health checks and the schema needs a signed-in
  user; tokens are verified locally against Clerk's keys (`api/auth.py`).
  One account can never open another's deals (it gets 404).
- Usage limits per user and per address, capped simulation sizes, request
  size limits and timeouts (`api/limits.py`).
- Security headers and a nonce-based content security policy on the web app
  (`web/src/lib/security/headers.ts`, `web/src/proxy.ts`), and locked-down
  headers and CORS on the API (`api/security.py`).
- TLS to the database, and an API database role that can read and write rows
  but not change the schema (`db/engine.py`, migration `0005_app_role`).
- No deal contents, names, emails or tokens in logs or error reports
  (`api/observability.py`, `web/src/lib/monitoring.ts`).
- CI: secret scanning of the whole git history (gitleaks), dependency audits
  (pip-audit, npm audit), CodeQL code scanning, and Dependabot updates
  (`.github/workflows/security.yml`, `codeql.yml`, `.github/dependabot.yml`).
- Daily checks of the live headers (`ops/check_headers.py`).

## For contributors

- **Never commit secrets.** Keys live in Render, Vercel and GitHub Actions
  secrets, and in a git-ignored `.env` locally. CI fails on anything that
  looks like a key, in any commit of the pull request's history.
- This repository is **public**: no customer data, real deal files, personal
  details or internal-only notes in commits, issues or pull requests.
- A new API route is protected and rate-limited by default; making one public
  means adding it to `api.auth.PUBLIC_PATHS` on purpose, with a reason.
