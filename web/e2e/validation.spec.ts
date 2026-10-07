import { expect, type Page, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

import { asUser, dealSettled, kpi, modeTab, setField, simulationSettled, stepLink } from "./helpers";

/**
 * Model validation (PLAN.md 4.6), through the real screens.
 *
 * Done when: the deal summary shows the full metric set (IRR, MOIC,
 * probability of loss, downside, coverage, default risk) and each follows the
 * deal; the report shows every check and split (replayed from
 * fixtures/validation.json, written by `python -m tests.e2e_validation`); a
 * deal's owner can let it count, and the choice is stored.
 */
const content = (page: Page) => page.locator("#content");
const REPORT = JSON.parse(readFileSync(join(__dirname, "fixtures", "validation.json"), "utf-8"));
const run = Date.now().toString(36);

test.describe("Model validation", () => {
  test("the deal summary shows the full metric set, and each figure follows the deal", async ({ page }) => {
    await page.goto("/deal/summary");
    await dealSettled(page);
    // The default deal: EBIT 88.85 over interest 46.20 = 1.92x, a B+, whose S&P five-year rate is 12.88%
    await expect(kpi(page, "IRR")).toBeVisible();
    await expect(kpi(page, "MOIC")).toBeVisible();
    await expect(kpi(page, "Interest cover")).toHaveText("1.92x");
    await expect(kpi(page, "Default risk")).toHaveText("12.88%");
    await expect(page.getByRole("region", { name: "Default risk" })).toContainText("B+ · S&P cumulative over 5 years");
    // The summary simulates the deal itself: the same figures the Monte Carlo screen then shows
    await expect(kpi(page, "Probability of loss")).toHaveText(/^\d+\.\d%$/, { timeout: 45_000 });
    const downside = await kpi(page, "Downside IRR").textContent();
    expect(downside).toMatch(/^-?\d+\.\d%$/);
    // (in-app navigation: the simulation lives in the page, not on the server)
    await modeTab(page, "Monte Carlo").click();
    await simulationSettled(page);
    await expect(kpi(page, "P5")).toHaveText(downside!);

    // More debt: coverage and default risk move, and the simulation runs again for the new deal
    await page.goto("/deal/debt");
    await setField(page, "Debt / EV", "75");
    await dealSettled(page);
    await stepLink(page, "Summary").click();
    await dealSettled(page);
    await expect(kpi(page, "Interest cover")).not.toHaveText("1.92x");
    await expect(kpi(page, "Default risk")).not.toHaveText("12.88%");
    await expect(page.getByRole("region", { name: "Probability of loss" })).toContainText(/Simulating|paths/);
    await expect(kpi(page, "Probability of loss")).toHaveText(/^\d+\.\d%$/, { timeout: 45_000 });
    await page.goto("/deal/debt");
    await setField(page, "Debt / EV", "60");
    await dealSettled(page);
  });

  test("the report shows every check and split, out of time first, with small groups hidden", async ({ page }) => {
    await page.route("**/api/validation/report", (r) => r.fulfill({ json: REPORT }));
    await page.goto("/backtest/validation");
    const irr = content(page).locator('[data-check="irr_range"][data-sample="out_of_time"]');
    // 13 users' deals: 8 of 13 inside the central 50%, 11 inside 80%, 12 inside 90%
    await expect(irr.locator('[data-overall="stats"]')).toHaveText(
      "13 cases: 61.5% inside the 50% range, 84.6% inside the 80% range, 92.3% inside the 90% range. Bias -1.0 points. Consistent with the claim.",
    );
    // Europe (six deals) shows its figures; emerging markets (two) and the U.S. with them are hidden
    const europe = irr.locator('[data-dimension="region"] [data-bucket="region:europe"]');
    await expect(europe.locator("td")).toHaveText(["6", "50%", "83%", "83%", "-2.8"]);
    for (const hidden of ["us", "emerging"]) {
      await expect(irr.locator(`[data-bucket="region:${hidden}"] td`).first()).toHaveText("Hidden");
    }
    for (const d of ["region", "sector", "size", "era"]) {
      for (const check of ["default", "irr_range", "loss"]) {
        await expect(content(page).locator(`[data-check="${check}"] [data-dimension="${d}"]`)).toBeVisible();
      }
    }
    // The ten reference transactions predate the tables' data: no newer data yet, so in-sample on request
    await expect(content(page).locator('[data-check="default"] [data-overall="not-enough"]')).toHaveText(
      "Not enough data: 0 cases, statistics need 5.",
    );
    await page.getByRole("radio", { name: "In-sample" }).click();
    await expect(content(page).locator('[data-check="default"][data-sample="in_sample"] [data-overall="stats"]')).toHaveText(
      "10 cases: predicted 16.7%, observed 10.0%, bias -6.7 points. Consistent with the claim.",
    );
    await expect(kpi(page, "Reference transactions")).toHaveText("10");
    await expect(kpi(page, "Users' deals")).toHaveText("13");
  });

  test("before the first report the screen says so", async ({ page }) => {
    await page.route("**/api/validation/report", (r) => r.fulfill({ json: { report: null } }));
    await page.goto("/backtest/validation");
    await expect(content(page)).toContainText("No report yet");
  });

  test("a deal's owner lets it count, anonymised, and the choice is stored", async ({ page }) => {
    await page.goto("/deal/inputs");
    const headers = await asUser(page);
    const created = await page.request.post("/api/deals", { data: { name: `Validation ${run}`, inputs: {} }, headers });
    expect(created.status()).toBe(201);
    const id = (await created.json()).id;
    try {
      await page.goto("/backtest/actuals");
      await page.getByRole("radio", { name: `Validation ${run}` }).click();
      const consent = page.getByRole("switch", { name: "Count this deal, anonymised" });
      await expect(consent).toHaveAttribute("aria-checked", "false");
      const [put] = await Promise.all([
        page.waitForRequest((r) => r.method() === "PUT" && r.url().endsWith(`/api/deals/${id}/validation`)),
        consent.click(),
      ]);
      expect(put.postDataJSON()).toEqual({ opt_in: true });
      await expect(consent).toHaveAttribute("aria-checked", "true");
      const stored = await page.request.get(`/api/deals/${id}/validation`, { headers });
      expect(await stored.json()).toEqual({ opt_in: true });
      // Reopened, the switch reads the stored choice
      await page.reload();
      await page.getByRole("radio", { name: `Validation ${run}` }).click();
      await expect(page.getByRole("switch", { name: "Count this deal, anonymised" })).toHaveAttribute("aria-checked", "true");
    } finally {
      await page.request.delete(`/api/deals/${id}`, { headers });
    }
  });
});
