import { expect, test } from "@playwright/test";

import { kpi } from "./helpers";

/**
 * Read-only checks against the deployed site (npm run test:live).
 * Nothing here saves or changes data. The first request may wait for the
 * free-tier API to wake up.
 *
 * They run signed out: since PLAN.md 1.4 every screen needs an account, so
 * they check what can be seen without one -- the web app's health route, the
 * API through the website's proxy, the sign-in redirect and the API refusing
 * anonymous calls. Checking the live model output again needs a test
 * account's credentials in GitHub secrets (DEPLOY.md, "Live checks").
 */
const bypass = process.env.VERCEL_AUTOMATION_BYPASS_SECRET;
const bypassHeaders: Record<string, string> = bypass ? { "x-vercel-protection-bypass": bypass } : {};

test.describe("live site", () => {
  // Protected previews (staging). One request with the secret asks Vercel for
  // a bypass cookie for this host; the browser then sends the cookie, never
  // the secret. Adding the secret to the page's own requests leaked it:
  // Playwright keeps changed headers across redirects, and signed-out pages
  // redirect to Clerk. API requests here use maxRedirects: 0 for the same reason.
  test.beforeEach(async ({ page, baseURL }) => {
    if (!bypass) return;
    const origin = new URL(baseURL!).origin;
    page.on("request", (request) => {
      if (new URL(request.url()).origin !== origin) {
        expect(request.headers()["x-vercel-protection-bypass"], `bypass secret sent to ${new URL(request.url()).origin}`).toBeUndefined();
      }
    });
    const resp = await page.request.get("/healthz", {
      headers: { ...bypassHeaders, "x-vercel-set-bypass-cookie": "true" },
      maxRedirects: 0,
    });
    expect(resp.status(), "the bypass secret was not accepted").toBe(200);
  });

  // What the uptime monitor checks (ops/betterstack.py)
  test("the website answers its health check", async ({ page }) => {
    const resp = await page.request.get("/healthz", { headers: bypassHeaders, maxRedirects: 0 });
    expect(resp.status()).toBe(200);
    expect(await resp.text()).toContain('"service":"FSE/ML web"');
  });

  // E2E_EXPECT_ENV (production or staging) proves the site is wired to the
  // right API: a staging site must never reach production, nor the reverse
  test("the API is reachable through the website, in the expected environment", async ({ page }) => {
    const resp = await page.request.get("/api/health", { headers: bypassHeaders, maxRedirects: 0 });
    expect(resp.status()).toBe(200);
    const health = await resp.json();
    expect(health.status).toBe("ok");
    const expected = process.env.E2E_EXPECT_ENV;
    if (expected) expect(health.environment).toBe(expected);
  });

  test("the deal screens ask for a sign-in", async ({ page }) => {
    await page.goto("/deal/returns");
    await expect(page).toHaveURL(/\/sign-in/);
    await expect(kpi(page, "IRR")).toHaveCount(0);
  });

  test("the live API refuses a call without a token", async ({ page }) => {
    const resp = await page.request.post("/api/deal/run", {
      data: {},
      failOnStatusCode: false,
      maxRedirects: 0,
      headers: bypassHeaders,
    });
    expect(resp.status()).toBe(401);
    expect(await resp.text()).not.toContain("irr");
  });
});
