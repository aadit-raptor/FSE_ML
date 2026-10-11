import { expect, test } from "@playwright/test";

import { simulationSettled, stepLink } from "./helpers";

/**
 * PLAN.md 2.1: no screen presents inception-era numbers as market facts.
 *
 * Live's label is the last of them to go (PLAN.md 5.8): it no longer repeats a
 * fixed training deal, because the estimate is the simulation of the deal on
 * screen; it says how many paths instead (e2e/live-sliders.spec.ts).
 */

const ILLUSTRATIVE = "Illustrative defaults — not market data";

test.describe("Honest labels", () => {
  test("every Settings step says its defaults are illustrative", async ({ page }) => {
    for (const step of ["deal", "fees", "monte-carlo", "correlations", "presets"]) {
      await page.goto(`/settings/${step}`);
      await expect(page.locator("#content").getByRole("note").filter({ hasText: ILLUSTRATIVE })).toBeVisible();
    }
  });

  test("Monte Carlo labels its ranges, correlations and presets on every step", async ({ page }) => {
    await page.goto("/monte-carlo/distribution");
    await simulationSettled(page);
    const note = page.getByRole("complementary", { name: "Inputs" }).getByRole("note");
    await expect(note).toContainText(ILLUSTRATIVE);
    await expect(note).toContainText("driver correlations and the scenario presets");
    for (const step of ["Scenarios", "Drivers"]) {
      await stepLink(page, step).click();
      await expect(note).toContainText(ILLUSTRATIVE);
    }
  });

  test("Backtest says its deals are examples, counted from the deal list", async ({ page }) => {
    await page.goto("/backtest/predicted");
    await page.getByRole("radio", { name: /^Burger King/ }).click();
    const note = page.locator("#content").getByRole("note").filter({ hasText: "Examples, not evidence" });
    await expect(note).toContainText("4 example deals from the 2006–2013 US market; not a validation of the model");
    await stepLink(page, "Error attribution").click();
    await expect(note).toBeVisible();
  });

  test("Live says its estimate is a small simulation of this deal, from the API's own count", async ({ page }) => {
    await page.goto("/monte-carlo/live");
    const label = page.locator('[data-provenance="live"]');
    await expect(label).toContainText("reruns this deal on the same 8,000 simulated paths");
    await expect(label).toContainText("nothing learned from other deals");
    await expect(page.getByText("Trained on one fixed deal")).toHaveCount(0);
  });
});
