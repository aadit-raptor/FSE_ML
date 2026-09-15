# Deploying

The web app (`web/`, Next.js) runs on **Vercel**; the API (`api/`, FastAPI in
the root `Dockerfile`) runs on **Render**. The browser only talks to Vercel:
Next proxies `/api/*` to the Render URL, so no CORS setup is needed.

```
browser ──> Vercel (Next.js) ──/api/*──> Render (FastAPI + model)
```

Do the API first; the web app needs its URL at build time.

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

## Checking a deploy

The CI `docker` job builds this same image, starts it with a host-assigned
`PORT`, and checks the default deal IRR (21.16%), a seeded Monte Carlo run
and an Excel export. The `e2e` job runs the browser tests against the API and
a production build of the web app.
