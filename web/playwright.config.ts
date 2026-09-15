import { defineConfig, devices } from "@playwright/test";

/**
 * Browser tests that prove each control changes its output.
 *
 * Local and CI (default): starts the FastAPI app and the built Next.js app,
 * then drives the real UI.
 *
 *   npm run build && npm run test:e2e
 *
 * Live (read-only checks against a deployed site, no servers started):
 *
 *   npm run test:live                       # https://fse-ml.vercel.app
 *   E2E_BASE_URL=https://… npm run test:live
 *   E2E_EXPECT_ENV=staging                  # the environment /api/health must report
 *   VERCEL_AUTOMATION_BYPASS_SECRET=…       # to open Vercel preview (staging) URLs
 *
 * Uses `next start`, not `next dev`: the dev server compiles on demand (slow)
 * and blocks its client scripts for hosts other than localhost.
 * Locally there is no bundled browser download: set PW_CHANNEL=msedge (or
 * chrome) to use an installed browser.
 */
const isCI = !!process.env.CI;
const live = process.env.E2E_LIVE === "1";
const liveUrl = process.env.E2E_BASE_URL ?? "https://fse-ml.vercel.app";
// Vercel previews sit behind Vercel login; this header (Protection Bypass for
// Automation) lets the staging checks in. Production needs no header.
const bypass = process.env.VERCEL_AUTOMATION_BYPASS_SECRET;
// Resolved from the repo root (the API server's cwd)
const python = process.env.PYTHON ?? (process.platform === "win32" ? ".venv\\Scripts\\python.exe" : "python");

export default defineConfig({
  testDir: "./e2e",
  // Live runs only the read-only checks; normal runs never touch production
  testMatch: live ? "live.spec.ts" : undefined,
  testIgnore: live ? undefined : "live.spec.ts",
  // A sleeping free-tier API can take about a minute to wake
  timeout: live ? 180_000 : 60_000,
  expect: { timeout: live ? 120_000 : 15_000 },
  fullyParallel: false,
  workers: 1,
  retries: isCI || live ? 1 : 0,
  reporter: isCI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: live ? liveUrl : "http://localhost:3000",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
    ...(live && bypass
      ? { extraHTTPHeaders: { "x-vercel-protection-bypass": bypass, "x-vercel-set-bypass-cookie": "true" } }
      : {}),
    ...(process.env.PW_CHANNEL ? { channel: process.env.PW_CHANNEL } : {}),
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
  webServer: live
    ? undefined
    : [
        {
          command: `${python} -m uvicorn api.main:app --port 8000`,
          cwd: "..",
          url: "http://127.0.0.1:8000/api/health",
          reuseExistingServer: !isCI,
          timeout: 120_000,
        },
        {
          command: "npm run start -- --port 3000",
          url: "http://localhost:3000",
          reuseExistingServer: !isCI,
          timeout: 180_000,
        },
      ],
});
