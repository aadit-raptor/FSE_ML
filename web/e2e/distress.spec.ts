import { expect, type Page, test } from "@playwright/test";

import { modeTab, stepLink } from "./helpers";

/**
 * PLAN.md 5.3: the distress predictor gives each year a rating band from the
 * deal's coverage and leverage, and the chance of default where its model
 * card says it beats the year-one default risk. The card says so nowhere
 * today, so the chances read "n/a" and the tile says not enough data; the
 * last test replays an answer the card allows to show the figures.
 * Expected bands are the API's for the default deal (tests/test_distress.py
 * works them by hand): B, B, BB, BB, BB.
 */

function tile(page: Page) {
  return page.getByRole("region", { name: "Distress by year" });
}

function bands(page: Page, which: "deal" | "simulated" = "deal") {
  return tile(page).locator(`table[data-distress="${which}"] td[data-band-year]`);
}

test.describe("Distress by year", () => {
  test("the default deal's bands, and leverage and business risk move them", async ({ page }) => {
    await page.goto("/deal/debt");
    await expect(bands(page)).toHaveText(["B", "B", "BB", "BB", "BB"]);
    await expect(tile(page)).toContainText("Not enough data.");
    await expect(tile(page)).toContainText("global (Table 24)");
    await expect(tile(page).getByRole("row", { name: /Chance of default in the year/ })).toContainText("n/a");

    // More debt: every year is B
    await page.getByLabel("Debt / EV", { exact: true }).fill("80");
    await page.getByLabel("Debt / EV", { exact: true }).blur();
    await expect(bands(page)).toHaveText(["B", "B", "B", "B", "B"]);
    await page.getByLabel("Debt / EV", { exact: true }).fill("60");
    await page.getByLabel("Debt / EV", { exact: true }).blur();
    await expect(bands(page)).toHaveText(["B", "B", "BB", "BB", "BB"]);

    // An excellent business carries the same leverage at better bands
    await page.getByRole("combobox", { name: "Business risk (S&P)", exact: true }).selectOption("1");
    await expect(bands(page)).toHaveText(["B", "BB", "BB", "BBB", "BBB"]);
    await expect(tile(page)).toContainText("business risk 1 Excellent");
    await page.getByRole("combobox", { name: "Business risk (S&P)", exact: true }).selectOption("4");
    await expect(bands(page)).toHaveText(["B", "B", "BB", "BB", "BB"]);
  });

  test("a country picks its region's table and the card's verdict for it", async ({ page }) => {
    const answer = page.waitForResponse(async (r) => {
      if (!r.url().endsWith("/api/deal/run") || r.status() !== 200) return false;
      return (await r.json()).distress.region === "us";
    });
    await page.goto("/deal/debt");
    // The deal's country lives on Inputs' starting point; set it through the API's own field
    await page.route("**/api/deal/run", async (route) => {
      const body = route.request().postDataJSON();
      await route.continue({ postData: JSON.stringify({ ...body, inputs: { ...body.inputs, country: "US" } }) });
    });
    await page.getByLabel("Debt / EV", { exact: true }).fill("61");
    await page.getByLabel("Debt / EV", { exact: true }).blur();
    const us = await (await answer).json();
    expect(us.distress.card.verdict).toBe("does_not_beat_baseline");
    await expect(tile(page)).toContainText("the U.S. (Table 25)");
    await expect(tile(page)).toContainText("tested on 6 buyouts in U.S. and tax havens");
    await expect(tile(page)).toContainText("no better than the year-one default risk (0.25 against 0.38)");
  });

  test("where the card allows, the chances of default are shown", async ({ page }) => {
    await page.route("**/api/deal/run", async (route) => {
      const response = await route.fetch();
      const json = await response.json();
      json.distress.shown = true;
      json.distress.card = { verdict: "beats_baseline", cases: 6, model: 0.9, baseline: 0.5 };
      json.distress.years = json.distress.years.map((y: Record<string, unknown>, i: number) => ({
        ...y, probability: 0.01 * (i + 1), cumulative: 0.02 * (i + 1),
      }));
      await route.fulfill({ response, json });
    });
    await page.goto("/deal/debt");
    const row = tile(page).getByRole("row", { name: /Chance of default by year end/ });
    await expect(row).toContainText("2.00%");
    await expect(row).toContainText("10.00%");
    await expect(tile(page)).not.toContainText("Not enough data.");
    await expect(tile(page)).toContainText("it ranked the distressed ones higher than the year-one default risk did (0.90 against 0.50)");
  });

  test("Monte Carlo counts the simulated paths in each band", async ({ page }) => {
    await page.goto("/deal/debt");
    await modeTab(page, "Monte Carlo").click();
    await stepLink(page, "Distribution").click();
    const sim = tile(page).locator('table[data-distress="simulated"]');
    await expect(sim).toBeVisible({ timeout: 60_000 });
    await expect(sim.locator('tr[data-band="B"]')).toBeVisible();
    await expect(sim.getByRole("row", { name: /Mean chance of default by year end/ })).toContainText("n/a");
    await expect(tile(page)).toContainText("Not enough data.");
  });
});

test("business risk marks Monte Carlo stale, since its distress tile reads it", async ({ page }) => {
  await page.goto("/monte-carlo/distribution");
  await expect(page.locator('table[data-distress="simulated"]')).toBeVisible({ timeout: 60_000 });
  await modeTab(page, "Deal").click();
  await stepLink(page, "Debt & cash flow").click();
  await page.getByRole("combobox", { name: "Business risk (S&P)", exact: true }).selectOption("5");
  await expect(modeTab(page, "Monte Carlo")).toContainText("stale");
  await page.getByRole("combobox", { name: "Business risk (S&P)", exact: true }).selectOption("4");
});
