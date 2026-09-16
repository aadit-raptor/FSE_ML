import { expect, test } from "@playwright/test";

import { kpi } from "./helpers";

/**
 * Read-only checks against the deployed site (npm run test:live).
 * Nothing here saves or changes data. The first request may wait for the
 * free-tier API to wake up.
 */
const bypass = process.env.VERCEL_AUTOMATION_BYPASS_SECRET;

test.describe("live site", () => {
  // Protected previews: send the bypass secret to the site's own origin only.
  // A header on every request would go to third parties too (Sentry ingest),
  // leaking the secret and failing their CORS preflight.
  test.beforeEach(async ({ page, baseURL }) => {
    if (!bypass) return;
    const origin = new URL(baseURL!).origin;
    page.on("request", (request) => {
      if (new URL(request.url()).origin !== origin) {
        expect(request.headers()["x-vercel-protection-bypass"], `bypass secret sent to ${new URL(request.url()).origin}`).toBeUndefined();
      }
    });
    await page.route(
      (url) => url.origin === origin,
      (route) => route.continue({ headers: { ...route.request().headers(), "x-vercel-protection-bypass": bypass, "x-vercel-set-bypass-cookie": "true" } }),
    );
  });

  test("API is reachable through the website", async ({ page }) => {
    await page.goto("/deal/inputs");
    await expect(page.locator("footer").getByRole("status")).toHaveText(/API ok · v\d/);
  });

  // E2E_EXPECT_ENV (production or staging) proves the site is wired to the
  // right API: a staging page must never reach production, nor the reverse
  test("website talks to the expected API environment", async ({ page }) => {
    const expected = process.env.E2E_EXPECT_ENV;
    test.skip(!expected, "set E2E_EXPECT_ENV to check the environment");
    const status = page.locator("footer").getByRole("status");
    await page.goto("/deal/inputs");
    await expect(status).toHaveText(/API ok · v\d/);
    const health = await (await page.request.get("/api/health", bypass ? { headers: { "x-vercel-protection-bypass": bypass } } : {})).json();
    expect(health.environment).toBe(expected);
    if (expected === "production") await expect(status).toHaveText(/^API ok · v[\d.]+$/);
    else await expect(status).toContainText(`· ${expected}`);
  });

  // Since PLAN.md 1.4 the deal screens need an account, so these checks run
  // signed out: the site must ask for a sign-in and the API must refuse
  // anonymous calls. Checking the live model output again needs a test
  // account's credentials in GitHub secrets (DEPLOY.md, "Live checks").
  test("the deal screens ask for a sign-in", async ({ page }) => {
    await page.goto("/deal/returns");
    await expect(page).toHaveURL(/\/sign-in/);
    await expect(kpi(page, "IRR")).toHaveCount(0);
  });

  test("the live API refuses a call without a token", async ({ page }) => {
    const resp = await page.request.post("/api/deal/run", {
      data: {},
      failOnStatusCode: false,
      headers: bypass ? { "x-vercel-protection-bypass": bypass } : {},
    });
    expect(resp.status()).toBe(401);
    expect(await resp.text()).not.toContain("irr");
  });
});
