import { expect, test } from "@playwright/test";

import { asUser, kpi, modeTab, setField, simulationSettled, stepLink } from "./helpers";

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

  // Background jobs (PLAN.md 1.9): the run is queued on the server and polled
  test("a run goes on in the background while the rest of the app works", async ({ page }) => {
    await page.goto("/monte-carlo/distribution");
    await simulationSettled(page);
    await setField(page, "Paths", "100000");
    await setField(page, "Mean", "12", "Exit multiple");
    await page.getByRole("button", { name: "Run Monte Carlo" }).click();

    await expect(page.getByRole("progressbar", { name: "Simulation progress" })).toBeVisible();
    await expect(modeTab(page, "Monte Carlo")).toContainText("running");
    // Another mode is usable meanwhile: the deal model still answers
    await modeTab(page, "Deal").click();
    await setField(page, "Exit multiple", "12");
    await expect(kpi(page, "IRR")).toHaveText("23.7%");

    await modeTab(page, "Monte Carlo").click();
    await simulationSettled(page);
    await expect(modeTab(page, "Monte Carlo")).not.toContainText("running");
    // 100,000 paths at exit mean 12x, seed 42 (50,000 paths give 68.7% and 10.2%)
    await expect(kpi(page, "P(IRR > 20%)")).toHaveText("68.8%");
    await expect(kpi(page, "P5")).toHaveText("10.0%");
  });

  test("cancelling stops the run on the server and keeps the last result", async ({ page }) => {
    await page.goto("/monte-carlo/distribution");
    await simulationSettled(page);
    await setField(page, "Paths", "100000");
    const submitted = page.waitForResponse((r) => r.url().endsWith("/api/jobs") && r.request().method() === "POST");
    await page.getByRole("button", { name: "Run Monte Carlo" }).click();
    const { id } = (await (await submitted).json()) as { id: string };
    await page.getByRole("button", { name: "Cancel" }).click();

    await expect(page.getByText("Monte Carlo is out of date")).toBeVisible();
    await expect(kpi(page, "Mean IRR")).toHaveText("18.0%");
    const headers = await asUser(page);
    await expect.poll(async () => {
      const job = await (await page.request.get(`/api/jobs/${id}`, { headers })).json();
      return job.status;
    }, { timeout: 30_000 }).toBe("cancelled");
  });
});
