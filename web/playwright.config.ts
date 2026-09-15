import { defineConfig, devices } from "@playwright/test";

/**
 * Browser tests that prove each control changes its output (step 5).
 * Starts the FastAPI app and the built Next.js app, then drives the real UI.
 *
 *   npm run build && npm run test:e2e
 *
 * Uses `next start`, not `next dev`: the dev server compiles on demand (slow)
 * and blocks its client scripts for hosts other than localhost.
 * Locally there is no bundled browser download: set PW_CHANNEL=msedge (or
 * chrome) to use an installed browser.
 */
const isCI = !!process.env.CI;
// Resolved from the repo root (the API server's cwd)
const python = process.env.PYTHON ?? (process.platform === "win32" ? ".venv\\Scripts\\python.exe" : "python");

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  // The API seeds numpy's global RNG (finding 9): keep simulations sequential
  workers: 1,
  retries: isCI ? 1 : 0,
  reporter: isCI ? [["list"], ["html", { open: "never" }]] : "list",
  use: {
    baseURL: "http://localhost:3000",
    viewport: { width: 1440, height: 900 },
    trace: "retain-on-failure",
    ...(process.env.PW_CHANNEL ? { channel: process.env.PW_CHANNEL } : {}),
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } }],
  webServer: [
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
