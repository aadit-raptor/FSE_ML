import { expect, type Page, test } from "@playwright/test";

import { dealSettled, kpi, modeTab, setField, stepLink } from "./helpers";

/**
 * A deal's tax rules (PLAN.md 2.5), proved by what the model answers. Every
 * figure is the deal model's for the default deal in pounds (core/tax.py's
 * presets, tests/test_tax_rules.py).
 */
const combo = (page: Page, name: string) => page.getByRole("combobox", { name, exact: true });
const tile = (page: Page, title: string) => page.getByRole("region", { name: title, exact: true });
const taxRow = (page: Page, row: string) =>
  tile(page, "Tax").getByRole("row").filter({ has: page.getByRole("rowheader", { name: row, exact: true }) });

async function poundDeal(page: Page) {
  await page.goto("/deal/inputs");
  await dealSettled(page);
  await combo(page, "Deal currency").selectOption("GBP");
  await dealSettled(page);
  await expect(kpi(page, "IRR")).toHaveText("21.2%");
}

test.describe("Tax rules", () => {
  test("a country preset sets the rules, and each rule reaches the model", async ({ page }) => {
    await poundDeal(page);

    // The UK: 25%, interest capped at 30% of EBITDA over a GBP 2m allowance,
    // losses carried forward. The default deal's 46.2 of interest in year
    // one is over the cap, so tax rises and the IRR falls
    await combo(page, "Country preset").selectOption("GB");
    await dealSettled(page);
    await expect(kpi(page, "IRR")).toHaveText("20.9%");
    await expect(kpi(page, "MOIC")).toHaveText("2.59x");
    await expect(page.getByTestId("tax-preset-note")).toContainText("check with a tax adviser");
    await expect(page.getByTestId("tax-preset-note")).toContainText("TIOPA 2010");
    await expect(page.getByTestId("tax-preset-note")).not.toContainText("left at 0");

    await stepLink(page, "Summary").click();
    await expect(taxRow(page, "Interest carried forward")).toContainText("14.7");
    await expect(taxRow(page, "Taxes")).toContainText("14.3");

    // Without the limit this profitable deal pays the flat 25% again: exactly
    // the answer from before any rule was set
    await stepLink(page, "Deal inputs").click();
    await combo(page, "Interest limit").selectOption("none");
    await dealSettled(page);
    await expect(kpi(page, "IRR")).toHaveText("21.2%");

    // No preset: the rules are off, and the Summary has no tax table
    await combo(page, "Country preset").selectOption("");
    await dealSettled(page);
    await stepLink(page, "Summary").click();
    await expect(tile(page, "Tax")).toHaveCount(0);
  });

  test("a preset in another currency says its amounts were left out", async ({ page }) => {
    await poundDeal(page);
    await combo(page, "Country preset").selectOption("DE");
    await dealSettled(page);
    await expect(page.getByTestId("tax-preset-note")).toContainText("amounts are in EUR and this deal is in GBP");
    await expect(page.getByLabel("Always deductible", { exact: true })).toHaveValue("0.0");
    await expect(kpi(page, "IRR")).toHaveText("20.7%");
  });

  test("a fixed cap, a minimum tax and losses each reach the tax table", async ({ page }) => {
    await poundDeal(page);

    // A fixed cap of 20 a year: the rail names the field for what it is now
    await combo(page, "Interest limit").selectOption("fixed");
    await setField(page, "Cap a year", "20");
    await dealSettled(page);
    await expect(kpi(page, "IRR")).toHaveText("20.6%");
    await stepLink(page, "Summary").click();
    await expect(taxRow(page, "Interest deducted")).toContainText("20.0");
    await expect(taxRow(page, "Minimum tax top-up")).toHaveCount(0);

    // No cap, and a 30% minimum tax on book profit: above the 25% rate, so it
    // tops the tax up to 30% of the 42.65 of profit in year one
    await stepLink(page, "Deal inputs").click();
    await combo(page, "Interest limit").selectOption("none");
    await setField(page, "Minimum tax on profit", "30");
    await dealSettled(page);
    await stepLink(page, "Summary").click();
    await expect(taxRow(page, "Minimum tax top-up")).toContainText("2.1");
    await expect(taxRow(page, "Taxes")).toContainText("12.8");

    // 8.5x of debt at 11%: two years of losses, carried forward and used up
    // by year five, where the only tax is paid
    await stepLink(page, "Deal inputs").click();
    await setField(page, "Minimum tax on profit", "0");
    await setField(page, "Senior debt", "6");
    await setField(page, "Mezzanine debt", "2.5");
    await setField(page, "Senior rate", "11");
    await page.getByRole("switch", { name: "Carry losses forward" }).click();
    await dealSettled(page);
    await stepLink(page, "Summary").click();
    await expect(taxRow(page, "Losses carried forward")).toContainText("21.6");
    await expect(taxRow(page, "Losses used")).toContainText("11.3");
    await expect(taxRow(page, "Taxes")).toContainText("1.6");
  });

  test("the amounts follow the deal's unit, and the rules are saved with it", async ({ page }) => {
    const name = `Tax ${Date.now().toString(36)}`;
    await poundDeal(page);
    await combo(page, "Country preset").selectOption("GB");
    await dealSettled(page);
    await expect(page.getByLabel("Always deductible", { exact: true })).toHaveValue("2.0");

    // In thousands the same allowance is 2,000, and the deal answers the same
    await combo(page, "Deal money unit").selectOption("thousands");
    await dealSettled(page);
    await expect(page.getByLabel("Always deductible", { exact: true })).toHaveValue("2000.0");
    await expect(kpi(page, "IRR")).toHaveText("20.9%");

    await stepLink(page, "Saved deals").click();
    await page.getByLabel("Deal name", { exact: true }).fill(name);
    await page.getByRole("button", { name: "Save deal" }).click();
    await expect(page.locator('[data-deal-save="saved"]')).toBeVisible();
    await page.reload();
    await stepLink(page, "Deal inputs").click();
    await dealSettled(page);
    await expect(combo(page, "Country preset")).toHaveValue("GB");
    await expect(kpi(page, "IRR")).toHaveText("20.9%");

    await stepLink(page, "Saved deals").click();
    const row = page.locator(`tr[data-deal="${name}"]`);
    await row.getByRole("button", { name: "Delete" }).click();
    await row.getByRole("button", { name: "Confirm delete" }).click();
    await expect(row).toHaveCount(0);
  });

  test("the rules make Monte Carlo stale", async ({ page }) => {
    await page.goto("/monte-carlo/distribution");
    await expect(kpi(page, "Mean IRR")).toHaveText("18.0%", { timeout: 45_000 });
    await modeTab(page, "Deal").click();
    await combo(page, "Interest limit").selectOption("ebitda_share");
    await expect(modeTab(page, "Monte Carlo")).toContainText("stale");
    await modeTab(page, "Monte Carlo").click();
    await expect(page.getByText("Interest limit No limit → Share of EBITDA")).toBeVisible();
    await page.getByRole("button", { name: "Run Monte Carlo" }).click();
    // The cap on every path: 17.8%, and fewer paths above the hurdle
    await expect(kpi(page, "Mean IRR")).toHaveText("17.8%", { timeout: 45_000 });
    await expect(kpi(page, "P(IRR > 20%)")).toHaveText("42.4%");
  });
});
