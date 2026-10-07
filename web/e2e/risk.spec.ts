import { expect, type Page, test } from "@playwright/test";

import { dealSettled, field, kpi, modeTab, recordedBenchmarks, replayBenchmarks, replayRisk, simulationSettled, stepLink } from "./helpers";

/**
 * Sourced risk ranges, correlations and scenarios (PLAN.md 4.4): once a deal has a country and an
 * industry, the Monte Carlo rail offers the means, spreads, correlations and presets measured on
 * published history for them, each with its source and years, and the presets list the years they
 * are built from. The answers are the API's for the recorded data (web/e2e/fixtures/benchmarks.json).
 */
const RISK = recordedBenchmarks().risk;
const UK = RISK["GB|machinery|GBP"];
const INDIA = RISK["IN|machinery|INR"];

async function startFrom(page: Page, country: string, currency: string) {
  await page.getByRole("combobox", { name: "Deal country", exact: true }).selectOption({ label: country });
  await page.getByRole("combobox", { name: "Deal industry", exact: true }).selectOption({ label: "Machinery" });
  await page.getByRole("combobox", { name: "Deal currency", exact: true }).selectOption(currency);
  await page.getByRole("button", { name: "Use sourced figures" }).click();
  await dealSettled(page);
}

/** As an input shows a number: the shortest form with at least `min` decimals. */
const shown = (v: number, min: number) => {
  for (let d = min; d <= 4; d++) if (Math.abs(Number(v.toFixed(d)) - v) < 1e-9) return v.toFixed(d);
  return v.toFixed(4);
};

const periodsText = (risk: typeof UK, id: string) =>
  risk.scenarios[id].periods.map((p: { start: number; end: number }) => (p.start === p.end ? `${p.start}` : `${p.start}–${p.end}`)).join(", ");

test.describe("Sourced risk ranges", () => {
  test("a UK deal's simulation runs on sourced ranges, and each preset lists its years", async ({ page }) => {
    await replayBenchmarks(page);
    await replayRisk(page);
    await page.goto("/deal/inputs");
    await dealSettled(page);
    await startFrom(page, "United Kingdom", "GBP");

    await modeTab(page, "Monte Carlo").click();
    await simulationSettled(page);
    const rail = page.getByRole("group", { name: "Sources" });
    // 1. Until applied, the rail says the ranges are illustrative and offers the sourced ones
    await expect(rail).toContainText("Illustrative defaults");
    const before = await kpi(page, "Mean IRR").textContent();
    await rail.getByRole("button", { name: /^Use sourced figures \(\d+\)$/ }).click();

    // 2. The rail now shows the sourced figures, and says where they come from
    await expect(page.getByTestId("risk-sourced")).toContainText("Machinery, United Kingdom");
    await expect(field(page, "Std dev", "Exit multiple")).toHaveValue(shown(UK.settings.mc_exit_std, 2));
    await expect(field(page, "Mean", "Exit multiple")).toHaveValue(shown(UK.settings.mc_exit_mean, 1));
    await expect(page.getByText("Monte Carlo is out of date")).toBeVisible();

    // 3. ... and the simulation moves with them
    await page.getByRole("button", { name: "Run Monte Carlo" }).click();
    await simulationSettled(page);
    await expect(kpi(page, "Mean IRR")).not.toHaveText(before ?? "");

    // 4. Each preset lists the historical years it is built from, and every figure its source
    await stepLink(page, "Scenarios").click();
    for (const id of ["recession", "stagflation", "bull"]) {
      await expect(page.getByTestId(`periods-${id}`)).toHaveText(periodsText(UK, id));
    }
    await expect(page.getByTestId("periods-recession")).toContainText("2020");
    const tile = page.getByRole("region", { name: /^Sourced ranges and presets/ });
    await expect(tile.locator('tr[data-setting="mc_exit_std"]')).toContainText("Damodaran's archive, NYU Stern · Developed Europe");
    await expect(tile.locator('tr[data-setting="mc_growth_std"]')).toContainText("IMF World Economic Outlook · United Kingdom");
    await expect(tile).toContainText("rank correlations of yearly changes");
  });

  test("an Indian deal gets different sourced ranges from a UK one", async ({ page }) => {
    expect(INDIA.settings.mc_exit_std).not.toBe(UK.settings.mc_exit_std);
    await replayBenchmarks(page);
    await replayRisk(page);
    await page.goto("/deal/inputs");
    await dealSettled(page);
    await startFrom(page, "India", "INR");
    await modeTab(page, "Monte Carlo").click();
    await simulationSettled(page);
    await page.getByRole("group", { name: "Sources" }).getByRole("button", { name: /^Use sourced figures/ }).click();
    await expect(field(page, "Std dev", "Exit multiple")).toHaveValue(shown(INDIA.settings.mc_exit_std, 2));
    await expect(field(page, "Std dev", "Growth")).toHaveValue(shown(INDIA.settings.mc_growth_std, 1));
  });
});
