import type { NextConfig } from "next";

// The FastAPI app (api/main.py) serves everything under /api. Proxying it
// through Next keeps the browser on one origin, so no CORS setup is needed
// in development or behind a single deployment.
//
// Environments (DEPLOY.md): Vercel production builds from `main` and uses
// FSE_API_URL (the production API). Vercel preview builds (the `staging`
// branch and pull requests) are the staging web app and always use the
// staging API, FSE_STAGING_API_URL, so a preview never talks to the
// production API. Locally FSE_API_URL defaults to the uvicorn dev server.
const STAGING_API = "https://fse-api-staging.onrender.com";
const isPreview = process.env.VERCEL_ENV === "preview";
const apiOrigin = (
  isPreview ? (process.env.FSE_STAGING_API_URL ?? STAGING_API) : (process.env.FSE_API_URL ?? "http://127.0.0.1:8000")
).replace(/\/+$/, "");

// On Vercel production the API is a separate service: fail the build rather
// than ship a proxy that points at localhost. Rewrites are resolved at build
// time, so the variable must be set before deploying.
if (process.env.VERCEL_ENV === "production" && !process.env.FSE_API_URL) {
  throw new Error("Set FSE_API_URL (e.g. https://fse-api.onrender.com) in the Vercel project settings.");
}

// Error tracking (lib/monitoring.ts). Vercel has SENTRY_DSN in Production and
// Preview; a DSN is public by design, so it is inlined into the browser
// bundle at build time. No DSN (local, CI) means no Sentry.
const fseEnv = process.env.VERCEL_ENV === "production" ? "production" : isPreview ? "staging" : "local";

const nextConfig: NextConfig = {
  env: {
    NEXT_PUBLIC_SENTRY_DSN: process.env.SENTRY_DSN ?? "",
    NEXT_PUBLIC_FSE_ENV: fseEnv,
    NEXT_PUBLIC_FSE_COMMIT: process.env.VERCEL_GIT_COMMIT_SHA ?? "",
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiOrigin}/api/:path*` }];
  },
};

export default nextConfig;
