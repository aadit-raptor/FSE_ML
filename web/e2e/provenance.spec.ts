import { expect, type Page, test } from "@playwright/test";

import ml from "./fixtures/ml-responses.json";
import { simulationSettled, stepLink } from "./helpers";

/**
 * PLAN.md 2.1: no screen presents inception-era numbers as market facts.
 *
 * Live needs the optional ML layer, which CI's e2e job doesn't install. Its
 * test replays responses recorded from a real ML server
 * (fixtures/ml-responses.json, written by the API's own endpoints), so the
 * labels are still rendered from API fields by the real UI.
 */

const ILLUSTRATIVE = "Illustrative defaults — not market data";

async function replayML(page: Page) {
  await page.route("**/api/capabilities", (r) => r.fulfill({ json: ml.capabilities }));
  await page.route("**/api/ml/surrogate", (r) => r.fulfill({ json: ml.surrogate }));
}

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

  test("Live repeats the fixed deal the surrogate was trained on", async ({ page }) => {
    await replayML(page);
    await page.goto("/monte-carlo/live");
    await simulationSettled(page);
    await expect(page.getByText("Trained on one fixed deal")).toBeVisible();
    const terms = page.locator('[data-provenance="live"]');
    await expect(terms).toContainText("entry multiple 10.0x · holding period 5 yr · opex / revenue 18.0% · tax rate 25.0%");
    await expect(terms).toContainText("senior / total debt 70%");
  });
});
