import { expect, test } from "@playwright/test";

import { kpi } from "./helpers";

/**
 * Read-only checks against the deployed site (npm run test:live).
 * Nothing here saves or changes data. The first request may wait for the
 * free-tier API to wake up.
 */
test.describe("live site", () => {
  test("API is reachable through the website", async ({ page }) => {
    await page.goto("/deal/inputs");
    await expect(page.locator("footer").getByRole("status")).toHaveText(/API ok · v\d/);
  });

  test("default deal returns the model's numbers", async ({ page }) => {
    await page.goto("/deal/returns");
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await expect(kpi(page, "MOIC")).toHaveText("2.61x");
  });

  test("Monte Carlo with seed 42 is reproducible", async ({ page }) => {
    await page.goto("/monte-carlo/distribution");
    await expect(kpi(page, "Mean IRR")).toHaveText("18.0%");
    await expect(kpi(page, "P(IRR > 20%)")).toHaveText("42.9%");
  });
});
