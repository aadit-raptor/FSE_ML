import { expect, test } from "@playwright/test";

import { kpi, modeTab, setField, simulationSettled, stepLink } from "./helpers";

test.describe("Monte Carlo", () => {
  test("seed 42 matches the API", async ({ page }) => {
    await page.goto("/monte-carlo/distribution");
    await simulationSettled(page);
    await expect(kpi(page, "Mean IRR")).toHaveText("18.0%");
    await expect(kpi(page, "P(IRR > 20%)")).toHaveText("42.9%");
    await expect(kpi(page, "P5")).toHaveText("2.9%");
    await expect(kpi(page, "P95")).toHaveText("31.5%");
  });

  test("an input change marks results stale until rerun", async ({ page }) => {
    await page.goto("/monte-carlo/distribution");
    await simulationSettled(page);
    await setField(page, "Mean", "12", "Exit multiple");
    await expect(page.getByText("Monte Carlo is out of date")).toBeVisible();
    await expect(page.getByText("Exit multiple mean 10.0 → 12.0")).toBeVisible();
    await expect(modeTab(page, "Monte Carlo")).toContainText("stale");
    await expect(kpi(page, "Mean IRR")).toHaveText("18.0%");

    await page.getByRole("button", { name: "Run Monte Carlo" }).click();
    await expect(kpi(page, "Mean IRR")).toHaveText("23.5%");
    await expect(modeTab(page, "Monte Carlo")).not.toContainText("stale");
  });

  test("only deal inputs the simulation reads make it stale", async ({ page }) => {
    await page.goto("/monte-carlo/distribution");
    await simulationSettled(page);
    await modeTab(page, "Deal").click();
    await setField(page, "Exit multiple", "13");
    await expect(modeTab(page, "Monte Carlo")).not.toContainText("stale");
    await setField(page, "Opex", "20");
    await expect(modeTab(page, "Monte Carlo")).toContainText("stale");
    await modeTab(page, "Monte Carlo").click();
    await expect(page.getByText("Opex 18.0% → 20.0%")).toBeVisible();
  });

  test("scenarios, drivers and heatmap render simulated data", async ({ page }) => {
    await page.goto("/monte-carlo/scenarios");
    await simulationSettled(page);
    await expect(kpi(page, "Recession")).toHaveText(/-?\d+\.\d%/);
    await expect(kpi(page, "Bull")).toHaveText(/\d+\.\d%/);

    await stepLink(page, "Drivers").click();
    await expect(page.locator("main circle")).toHaveCount(2000);
    const firstX = await page.locator("main circle").first().getAttribute("cx");
    await page.getByRole("radio", { name: "Exit Multiple" }).click();
    await expect(page.locator("main circle").first()).not.toHaveAttribute("cx", firstX ?? "");

    await stepLink(page, "Heatmap").click();
    await expect(page.getByText("full deal-model run")).toBeVisible();
    await expect(page.locator("main td")).toHaveCount(56);
  });
});
