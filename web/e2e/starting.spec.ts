import { expect, type Page, test } from "@playwright/test";

import { dealSettled, field, kpi, recordedBenchmarks, replayBenchmarks, setField } from "./helpers";

/**
 * Sourced starting figures (PLAN.md 4.3): a new deal asks its country, industry, currency and
 * size, then starts from published data -- Damodaran's industry averages for the region, the
 * IMF's growth and the country's tax rate -- each shown with its source, sample and date. The
 * answers are the API's for the recorded data (web/e2e/fixtures/benchmarks.json).
 */
const GERMAN_MACHINERY = recordedBenchmarks().starting["DE|machinery|EUR"].inputs;

const startingTile = (page: Page) => page.getByRole("region", { name: /^Starting figures/ });
const row = (page: Page, field: string) => startingTile(page).locator(`tr[data-field="${field}"]`);

async function startFrom(page: Page, country: string, industry: string, currency: string) {
  await page.getByRole("combobox", { name: "Deal country", exact: true }).selectOption({ label: country });
  await page.getByRole("combobox", { name: "Deal industry", exact: true }).selectOption({ label: industry });
  await page.getByRole("combobox", { name: "Deal currency", exact: true }).selectOption(currency);
  await page.getByRole("button", { name: "Use sourced figures" }).click();
  await dealSettled(page);
}

test.describe("Sourced starting figures", () => {
  test("a German industrials deal starts from sourced figures with sample sizes, and they reach the model", async ({ page }) => {
    await replayBenchmarks(page);
    await page.goto("/deal/inputs");
    await dealSettled(page);
    // 1. Before a country is chosen, the figures say they are illustrative
    await expect(page.getByRole("status").filter({ hasText: "Illustrative figures" })).toBeVisible();
    const irrBefore = await kpi(page, "IRR").textContent();

    // 2. Germany, machinery, euros: every input published data covers is filled
    await startFrom(page, "Germany", "Machinery", "EUR");
    await expect(page.getByRole("status").filter({ hasText: "Illustrative figures" })).toHaveCount(0);
    await expect(page.getByRole("status").filter({ hasText: "15 figures filled from published data" })).toBeVisible();
    await expect(field(page, "Entry multiple")).toHaveValue(String(GERMAN_MACHINERY.entry_mult));
    await expect(field(page, "Gross margin")).toHaveValue(String(GERMAN_MACHINERY.gross_margin));

    // 3. Each figure names its source, its sample and its date
    await expect(row(page, "gross_margin")).toContainText("Damodaran, NYU Stern · Developed Europe · 210 companies");
    await expect(row(page, "gross_margin").getByRole("link")).toHaveAttribute("href", /marginEurope\.xls$/);
    await expect(row(page, "growth")).toContainText("IMF World Economic Outlook");
    await expect(row(page, "tax")).toContainText("Tax Foundation, via Damodaran · Germany");
    await expect(startingTile(page)).toContainText("No free source publishes buyout leverage");

    // 4. ... and they reach the model: EV is 100 x the industry's 14.98x, the IRR moves
    await expect(kpi(page, "Enterprise value")).toHaveText("1,498.0");
    await expect(kpi(page, "IRR")).not.toHaveText(irrBefore ?? "");

    // 5. An override shows beside the sourced figure, and "Use" puts the sourced one back
    await setField(page, "Gross margin", "50");
    await dealSettled(page);
    await page.getByRole("button", { name: "Use the sourced Gross margin" }).click();
    await dealSettled(page);
    await expect(field(page, "Gross margin")).toHaveValue(String(GERMAN_MACHINERY.gross_margin));
    await expect(page.getByRole("button", { name: "Use the sourced Gross margin" })).toHaveCount(0);
  });

  test("a thin group falls back to the global figure and says why", async ({ page }) => {
    await replayBenchmarks(page);
    await page.goto("/deal/inputs");
    await dealSettled(page);
    await startFrom(page, "Canada", "Shipbuilding & Marine", "CAD");
    const gm = row(page, "gross_margin");
    await expect(gm).toContainText("Global · 359 companies");
    await expect(gm).toContainText("Australia, NZ and Canada has 8 companies, fewer than 20: a wider group is used.");
    await expect(field(page, "Gross margin")).toHaveValue("26.53");
  });
});
