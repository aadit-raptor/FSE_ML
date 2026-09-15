import type { NextConfig } from "next";

// The FastAPI app (api/main.py) serves everything under /api. Proxying it
// through Next keeps the browser on one origin, so no CORS setup is needed
// in development or behind a single deployment.
const apiOrigin = (process.env.FSE_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

// On Vercel the API is a separate service: fail the build rather than ship a
// proxy that points at localhost. Rewrites are resolved at build time, so the
// variable must be set before deploying.
if (process.env.VERCEL && !process.env.FSE_API_URL) {
  throw new Error("Set FSE_API_URL (e.g. https://fse-api.onrender.com) in the Vercel project settings.");
}

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiOrigin}/api/:path*` }];
  },
};

export default nextConfig;
