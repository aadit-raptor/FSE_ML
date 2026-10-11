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
    // From another workspace the Monte Carlo tab isn't there: the switcher says it instead
    const workspaces = page.getByRole("navigation", { name: "Workspaces" });
    await expect(workspaces.getByRole("link", { name: /^LBO/ })).not.toContainText("stale");
    await workspaces.getByRole("link", { name: /^Equity research/ }).click();
    await expect(workspaces.getByRole("link", { name: /^LBO/ })).toContainText("stale");
    await workspaces.getByRole("link", { name: /^LBO/ }).click();
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
    await page.getByRole("radio", { name: "Exit multiple" }).click();
    await expect(page.locator("main circle").first()).not.toHaveAttribute("cx", firstX ?? "");

    await stepLink(page, "Heatmap").click();
    await expect(page.getByText("full deal-model run")).toBeVisible();
    await expect(page.locator("main td")).toHaveCount(56);
  });

  // Driver explanations (PLAN.md 5.6): each tail's IRR from the IRR at the means, driver by driver
  test("driver explanations add up and follow the inputs", async ({ page }) => {
    await page.goto("/monte-carlo/distribution");
    await simulationSettled(page);
    await stepLink(page, "Drivers").click();
    const worst = page.getByRole("region", { name: "Worst 5% of paths" });
    const best = page.getByRole("region", { name: "Best 5% of paths" });
    const shown = async (tile: typeof worst) => {
      const figure = (text: string | null) => Number((text ?? "").replace(/[^0-9.+-]/g, ""));
      const base = figure(await tile.locator('[data-explained="base"]').textContent());
      const total = figure(await tile.locator('[data-explained="total"]').textContent());
      const bars = (await tile.locator("svg text.chart-value").allTextContents()).map(figure);
      return { base, total, bars };
    };

    // Seed 42, the default deal: the API's own figures (docs/methodology.md, section 9)
    await expect(worst.locator('[data-explained="base"]')).toHaveText("18.56%");
    await expect(worst.locator('[data-explained="total"]')).toHaveText("-2.24%");
    await expect(best.locator('[data-explained="total"]')).toHaveText("34.41%");
    // Largest first: the exit multiple, then growth
    await expect(worst.locator("svg text.chart-category").first()).toHaveText("Exit multiple");
    await expect(worst.locator("svg text.chart-value").first()).toHaveText("-10.49");
    for (const tile of [worst, best]) {
      const { base, total, bars } = await shown(tile);
      expect(bars).toHaveLength(5);
      // Six figures rounded to two decimals: they add up to within the rounding
      expect(Math.abs(base + bars.reduce((a, b) => a + b, 0) - total)).toBeLessThan(0.04);
    }

    // With next to no spread in growth (0.1 is the least the rail takes), growth explains next to
    // nothing of either tail, where it was 9.51 points of the worst one
    const growthBar = (tile: typeof worst) => tile.locator("svg g").filter({ hasText: "Growth" }).locator("text.chart-value");
    await expect(growthBar(worst)).toHaveText("-9.51");
    await setField(page, "Std dev", "0.1", "Revenue growth");
    await page.getByRole("button", { name: "Run Monte Carlo" }).click();
    await expect(worst.locator('[data-explained="total"]')).not.toHaveText("-2.24%");
    const after = await shown(worst);
    expect(after.base).toBe(18.56);
    expect(after.total).toBeGreaterThan(-2.24);
    expect(Math.abs(Number(await growthBar(worst).textContent()))).toBeLessThan(0.5);
    expect(Math.abs(after.base + after.bars.reduce((a, b) => a + b, 0) - after.total)).toBeLessThan(0.04);

    // Above 50,000 paths a tail is read at 2,500 of its paths, and the tile says so
    await expect(worst.getByText("evenly spaced by rank")).toHaveCount(0);
    await setField(page, "Paths", "100000");
    await page.getByRole("button", { name: "Run Monte Carlo" }).click();
    await expect(worst.getByText("Read at 2,500 of the tail's 5,000 paths, evenly spaced by rank.")).toBeVisible();
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
