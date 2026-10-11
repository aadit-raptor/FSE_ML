import { expect, type Page, test } from "@playwright/test";

import recorded from "./fixtures/regime.json";
import { dealSettled, kpi, modeTab, replayBenchmarks, simulationSettled, stepLink } from "./helpers";

/**
 * PLAN.md 5.7: the economic regime of the deal's region on Monte Carlo -> Scenarios. CI never runs
 * the monthly refit, so the real endpoint has nothing stored; the regimes are replayed from what
 * the real endpoint gives for the recorded IMF figures (fixtures/regime.json, written by
 * `python -m tests.e2e_regime`, checked by tests/test_macro_regime.py). The first test uses the
 * real endpoint; "Use ... preset" is proved by the simulation's own answer.
 */

type Answer = (typeof recorded)["in"];
const REGIONS = ["us", "eurozone", "uk", "japan", "china", "india", "global"];

/** Replay the recorded answer for the country the request carries; `change` edits it first. */
async function replayRegime(page: Page, change: (a: Answer) => object = (a) => a) {
  await page.route("**/api/ml/macro-regime", (route) => {
    const { country } = route.request().postDataJSON();
    const answer = Object.values(recorded).find((a) => a.country === country);
    return answer
      ? route.fulfill({ json: change(structuredClone(answer) as Answer) })
      : route.fulfill({ status: 404, json: { detail: "Not recorded." } });
  });
}

/** A deal started from a country's sourced figures, which is what gives it a country. */
async function dealIn(page: Page, country: string, industry: string, currency: string) {
  await replayBenchmarks(page);
  await page.goto("/deal/inputs");
  await dealSettled(page);
  await startFrom(page, country, industry, currency);
}

async function startFrom(page: Page, country: string, industry: string, currency: string) {
  await page.getByRole("combobox", { name: "Deal country", exact: true }).selectOption({ label: country });
  await page.getByRole("combobox", { name: "Deal industry", exact: true }).selectOption({ label: industry });
  await page.getByRole("combobox", { name: "Deal currency", exact: true }).selectOption(currency);
  await page.getByRole("button", { name: "Use sourced figures" }).click();
  await dealSettled(page);
}

async function scenarios(page: Page) {
  await modeTab(page, "Monte Carlo").click();
  await simulationSettled(page);
  await stepLink(page, "Scenarios").click();
}

test.describe("Economic regime by region", () => {
  test("the real endpoint gets the deal's country and answers its region", async ({ page }) => {
    await dealIn(page, "India", "Machinery", "INR");
    const answer = page.waitForResponse((r) => r.url().endsWith("/api/ml/macro-regime") && r.status() === 200);
    await scenarios(page);
    const body = await (await answer).json();
    expect(body.country).toBe("IN");
    expect(body.region).toBe("india");
    const tile = page.getByRole("region", { name: "Economic regime — India" });
    // CI stores no fit (the monthly refit never runs there); a database with one shows India's
    if (body.status === "no_data") {
      await expect(tile).toContainText("No regimes are stored yet");
    } else {
      await expect(tile.locator('tr[data-regime-row="india"]')).toContainText(String(body.regions[5].year));
    }
  });

  test("an India deal shows India's regime, and every region shows a dated one", async ({ page }) => {
    const india = recorded.in;
    expect(india.regime.year).toBe(2026);
    await replayRegime(page);
    await dealIn(page, "India", "Machinery", "INR");
    await scenarios(page);
    const tile = page.getByRole("region", { name: "Economic regime — India" });
    await expect(tile.locator("[data-regime-region]")).toHaveAttribute("data-regime-region", "india");
    await expect(tile.getByTestId("regime-current")).toContainText("Base · 2026, IMF projection");
    await expect(tile.getByTestId("regime-current")).toContainText("read 11 Oct 2026");
    // India's own figures, not the world's (3.1%) or the euro area's (1.1%)
    await expect(tile.locator('[data-provenance="regime"]')).toContainText("India, 2026: real growth 6.5% against 7.6%");
    await expect(tile.locator('[data-provenance="regime"]')).toContainText("inflation 4.7% against 5.4%");
    await expect(tile.getByTestId("regime-recent")).toContainText("2020 Recession");
    for (const region of REGIONS) {
      const row = tile.locator(`tr[data-regime-row="${region}"]`);
      await expect(row).toContainText(/Recession|Stagflation|Base|Bull/);
      await expect(row).toContainText("2026, IMF projection");
    }
    await expect(tile.getByTestId("regime-card")).toContainText("named 61% of 23 years");
    await expect(tile.getByRole("link", { name: "International Monetary Fund, World Economic Outlook" })).toBeVisible();
    // Base is the preset in use already: nothing to switch to
    await expect(tile.getByRole("button", { name: /preset$/ })).toHaveCount(0);
  });

  test("a euro member is shown the euro area's regime and another country the world's, which says so", async ({ page }) => {
    await replayRegime(page);
    await dealIn(page, "Germany", "Machinery", "EUR");
    await scenarios(page);
    const euro = page.getByRole("region", { name: "Economic regime — Euro area" });
    await expect(euro.locator('[data-provenance="regime"]')).toContainText("Euro area, 2026: real growth 1.1%");
    await expect(euro.getByTestId("regime-stand-in")).toHaveCount(0);

    await modeTab(page, "Deal").click();
    await startFrom(page, "Canada", "Shipbuilding & Marine", "CAD");
    await modeTab(page, "Monte Carlo").click();
    await stepLink(page, "Scenarios").click();
    const world = page.getByRole("region", { name: "Economic regime — World" });
    await expect(world.getByTestId("regime-stand-in")).toContainText("No regime is worked out for Canada itself");
    await expect(world.locator('[data-provenance="regime"]')).toContainText("World, 2026: real growth 3.1%");
  });

  test("the regime's preset reruns the simulation as that scenario", async ({ page }) => {
    // The recorded year is a base year everywhere; a recession year offers its preset
    await replayRegime(page, (a) => ({ ...a, regime: { ...a.regime, regime: "recession" } }));
    await dealIn(page, "India", "Machinery", "INR");
    await modeTab(page, "Monte Carlo").click();
    await simulationSettled(page);
    const before = await kpi(page, "Mean IRR").textContent();
    await stepLink(page, "Scenarios").click();
    const tile = page.getByRole("region", { name: "Economic regime — India" });
    await tile.getByRole("button", { name: "Use Recession preset" }).click();
    await expect(page.getByText("Monte Carlo is out of date")).toBeVisible();
    await expect(tile.getByRole("button", { name: /preset$/ })).toHaveCount(0);
    await page.getByRole("button", { name: "Run Monte Carlo" }).click();
    await stepLink(page, "Distribution").click();
    await simulationSettled(page);
    await expect(kpi(page, "Mean IRR")).not.toHaveText(before ?? "");
  });

  test("a region whose card does not beat the baseline shows no regime", async ({ page }) => {
    await replayRegime(page, (a) => ({
      ...a,
      shown: false,
      hidden: "does_not_beat_baseline",
      regime: null,
      regions: a.regions.map((r) => (r.region === "india" ? { ...r, shown: false, hidden: "does_not_beat_baseline", regime: null } : r)),
    }));
    await dealIn(page, "India", "Machinery", "INR");
    await scenarios(page);
    const tile = page.getByRole("region", { name: "Economic regime — India" });
    await expect(tile).toContainText("Not shown. In India, naming the year from the IMF's projection did no better");
    await expect(tile.getByTestId("regime-current")).toHaveCount(0);
    await expect(tile.locator('tr[data-regime-row="india"]')).toContainText("Not shown");
    await expect(tile.locator('tr[data-regime-row="us"]')).toContainText("Base");
  });
});
