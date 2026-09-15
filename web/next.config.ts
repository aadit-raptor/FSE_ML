import type { NextConfig } from "next";

// The FastAPI app (api/main.py) serves everything under /api. Proxying it
// through Next keeps the browser on one origin, so no CORS setup is needed
// in development or behind a single deployment.
const apiOrigin = process.env.FSE_API_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${apiOrigin}/api/:path*` }];
  },
};

export default nextConfig;
