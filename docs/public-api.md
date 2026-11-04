# Variater's public API, version 1

Run deals and simulations, read your saved deals and the sourced market
defaults from your own scripts, and be told when a long run ends
(PLAN.md 7.7). Every endpoint, with its request and answer, is listed at
[api.variater.com/api/docs](https://api.variater.com/api/docs) under
**public api v1**. A complete script using only Python's standard library is
[`examples/variater_api.py`](examples/variater_api.py); the tests run it
against the API.

## Keys

Make a key on the site: **Account → Developer API → Make key**. It is shown
once (only its fingerprint is kept), so copy it then. Send it on every
request:

```
Authorization: Bearer vk_...
```

- A key acts as you, on `/api/v1/*` only. It can't reach the site's own
  endpoints, change your account, make another key or manage webhooks: those
  answer **403** `key_scope`.
- You can hold five keys. Make one per script, so revoking one stops only it.
- **Revoking is immediate.** Every request looks the key up again, so the
  next request with a revoked key answers **401**.
- Keep keys out of code and out of repositories: an environment variable or
  your platform's secret store.

## Endpoints

| Method | Path | What |
|---|---|---|
| GET | `/api/v1/me` | The version, how you're signed in, your limits, how many webhooks you have |
| POST | `/api/v1/deal/run` | Run a deal: body `{"inputs": {...}, "settings": {...}}`, as the site's Deal screen sends |
| POST | `/api/v1/montecarlo/run` | Simulate a deal (one simulation at a time on the server) |
| GET | `/api/v1/deals` | Your saved deals and those shared with your teams |
| GET | `/api/v1/deals/{id}` | One, with its inputs and Settings |
| POST | `/api/v1/deals/{id}/run` | Run a saved deal's working copy with its own Settings |
| POST | `/api/v1/jobs` | Queue a long run: `{"kind": "montecarlo.run", "input": {...}}` (also `montecarlo.scenarios`, `backtesting.run`, `forecasting.run`) |
| GET | `/api/v1/jobs/{id}` | Its progress; once `succeeded`, `result` is what the matching endpoint answers |
| POST | `/api/v1/jobs/{id}/cancel` | Cancel it |
| GET | `/api/v1/defaults/industries` | The industries the market defaults cover |
| GET | `/api/v1/defaults/starting?country=DE&industry=machinery&currency=EUR` | A deal's sourced starting figures, each with its source, sample and date |
| GET | `/api/v1/defaults/risk?country=DE&industry=machinery&currency=EUR` | The simulation's sourced ranges, correlations and scenarios |

**Units** are the site's: percentages as numbers (`60.0` is 60%), money in
the deal's `currency` and `unit` (US dollar millions unless the deal says
otherwise), engine outputs in engine units (an IRR of `0.157` is 15.7%).
Every result carries `model`, the engine version and Settings fingerprint
that made it (docs/methodology.md, "Model version").

**Version 1 is stable.** An answer may gain fields; nothing is renamed or
removed within `/api/v1`. A change that would break a script is a new
version beside it. Every answer says `Variater-API-Version: 1`.

## Limits

Sized for the free hosting the API runs on:

| What | Limit |
|---|---|
| Requests with one key | 60 a minute |
| Model runs with one key (simulations, queued jobs) | 200 a day |
| Your account, every key and the site together | 240 requests and 30 runs a minute, 1,000 runs a day |
| A simulation | 100,000 paths, one at a time on the server, 100 seconds |
| A request body | 1 MB |

A refusal is **429** with `Retry-After` and a sentence saying when to try
again. The free server sleeps after 15 minutes idle; the first request after
that takes about a minute.

## Webhooks

Add an address on **Account → Developer API → Add webhook**. It must be
`https` on the public internet (no private, local or link-local address, no
user name in it). You can have three. Its **signing secret** (`whsec_...`) is
shown once.

When a job you queued through `/api/v1/jobs` ends (succeeded, failed or
cancelled), each webhook is sent a `POST`:

```json
{"data": {"error_status": null, "job_id": "…", "kind": "montecarlo.run", "status": "succeeded"},
 "timestamp": "2026-11-04T10:00:00.000Z", "type": "job.finished"}
```

It never holds the job's result or its error text: fetch those with your key
(`GET /api/v1/jobs/{id}`). **Send a test** sends a `ping` message.

Messages are signed as [Standard Webhooks](https://www.standardwebhooks.com)
sign them, so its libraries check ours. The headers are `webhook-id`,
`webhook-timestamp` (Unix seconds) and `webhook-signature`: `v1,` and the
base64 HMAC-SHA256 of `<webhook-id>.<webhook-timestamp>.<body>`, keyed with
the secret's bytes (the base64 after `whsec_`). Check it, and that the time
is within five minutes, before trusting a message: `verify` in the example
script does both.

Answer with any 2xx within 10 seconds. Anything else, or no answer, is tried
again after 10 seconds, a minute and five minutes, then given up; the screen
shows how each webhook's newest message went. A message is kept for a week.
The same message keeps its `webhook-id` on every try, so a receiver can drop
repeats.
