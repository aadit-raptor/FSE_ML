import { expect, type Page, test } from "@playwright/test";

import recorded from "./fixtures/growth.json";
import { dealSettled, field, kpi, modeTab, replayBenchmarks, setField, simulationSettled, stepLink } from "./helpers";

/**
 * PLAN.md 5.5: revenue growth calibrated from the deal's sector and region. CI stores no economic
 * data, so the real endpoint can't centre a range on a country's growth; the ranges are replayed
 * from what the real endpoint gives with the recorded economic series (fixtures/growth.json,
 * written by `python -m tests.e2e_growth`, checked by tests/test_growth_calibrator.py). The first
 * test uses the real endpoint; "Calibrate from sector and region" is proved by the simulation's
 * own answer.
 */

const DE = recorded.de;

/** Replay the recorded answer for the deal the request carries. */
async function replayGrowth(page: Page) {
  await page.route("**/api/ml/growth", (route) => {
    const inputs = route.request().postDataJSON().inputs;
    const answer = Object.values(recorded).find((a) => a.country === inputs.country && a.hold === inputs.hold);
    return answer ? route.fulfill({ json: answer }) : route.fulfill({ status: 404, json: { detail: "Not recorded." } });
  });
}

async function germanMachinery(page: Page) {
  await replayBenchmarks(page);
  await page.goto("/deal/inputs");
  await dealSettled(page);
  await page.getByRole("combobox", { name: "Deal country", exact: true }).selectOption({ label: "Germany" });
  await page.getByRole("combobox", { name: "Deal industry", exact: true }).selectOption({ label: "Machinery" });
  await page.getByRole("combobox", { name: "Deal currency", exact: true }).selectOption("EUR");
  await page.getByRole("button", { name: "Use sourced figures" }).click();
  await dealSettled(page);
}

/** As an input shows a number: the shortest form with at least `min` decimals. */
const shown = (v: number, min: number) => {
  for (let d = min; d <= 4; d++) if (Math.abs(Number(v.toFixed(d)) - v) < 1e-9) return v.toFixed(d);
  return v.toFixed(4);
};

test.describe("Growth by sector and region", () => {
  test("the real endpoint gets the deal's country and hold, and says when no economy is stored", async ({ page }) => {
    await germanMachinery(page);
    const answer = page.waitForResponse((r) => r.url().endsWith("/api/ml/growth") && r.status() === 200);
    await modeTab(page, "Monte Carlo").click();
    const body = await (await answer).json();
    expect(body.country).toBe("DE");
    expect(body.hold).toBe(5);
    expect(body.region).toBe("europe");
    // CI stores no economic data, so there the range can't be centred; a database with some offers it
    if (body.reason === "no_growth") {
      await expect(page.getByTestId("growth-calibration-hidden")).toContainText("No economic data is stored");
    } else {
      expect(body.shown).toBe(true);
      await expect(page.getByRole("button", { name: "Calibrate from sector and region" })).toBeVisible();
    }
  });

  test("calibrating sets the growth draw from the sector's companies and moves the simulation", async ({ page }) => {
    await replayGrowth(page);
    await germanMachinery(page);
    await modeTab(page, "Monte Carlo").click();
    await simulationSettled(page);
    const before = await kpi(page, "Mean IRR").textContent();
    const growth = page.getByRole("group", { name: "Revenue growth", exact: true });
    // A rail edit the calibration doesn't cover survives it
    await setField(page, "Paths", "20000");
    await growth.getByRole("button", { name: "Calibrate from sector and region" }).click();
    await expect(field(page, "Mean", "Revenue growth")).toHaveValue(shown(DE.settings!.mc_growth_mean, 1));
    await expect(field(page, "Std dev", "Revenue growth")).toHaveValue(shown(DE.settings!.mc_growth_std, 1));
    await expect(field(page, "Paths")).toHaveValue("20000");
    await expect(page.getByTestId("growth-calibrated")).toBeVisible();
    await expect(page.getByText("Monte Carlo is out of date")).toBeVisible();

    await page.getByRole("button", { name: "Run Monte Carlo" }).click();
    await simulationSettled(page);
    await expect(kpi(page, "Mean IRR")).not.toHaveText(before ?? "");

    // An edit of the growth draw makes the button come back
    await setField(page, "Std dev", "3", "Revenue growth");
    await expect(growth.getByRole("button", { name: "Calibrate from sector and region" })).toBeVisible();

    // The tile says what the range rests on, and what the card found
    await stepLink(page, "Scenarios").click();
    const tile = page.getByRole("region", { name: /^Revenue growth/ });
    await expect(tile).toContainText("Industrials, Europe");
    await expect(tile.getByTestId("growth-range")).toContainText("-7.2%");
    await expect(tile.getByTestId("growth-range")).toContainText("22.4%");
    await expect(tile.locator('[data-provenance="growth"]')).toContainText("40 companies in Industrials, Europe");
    await expect(tile).toContainText("Centred on Germany's nominal GDP growth, 3.52%");
    await expect(tile.getByTestId("growth-card")).toContainText("interval score 74.0 points against 102.9");
  });

  test("a hold the card didn't test gets no range", async ({ page }) => {
    await replayGrowth(page);
    await germanMachinery(page);
    await setField(page, "Hold", "9");
    await dealSettled(page);
    await modeTab(page, "Monte Carlo").click();
    await expect(page.getByTestId("growth-calibration-hidden")).toContainText("holds of 3 to 7 years only");
    await expect(page.getByRole("button", { name: "Calibrate from sector and region" })).toHaveCount(0);
  });
});
