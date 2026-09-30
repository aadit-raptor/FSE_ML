import { expect, type Page, test } from "@playwright/test";

import { dealSettled, kpi, modeTab, stepLink } from "./helpers";

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
    await expect(page.getByTestId("tax-preset-note")).toContainText("amounts are in EUR");
    await expect(page.getByLabel("Always deductible", { exact: true })).toHaveValue("0.0");
    await expect(kpi(page, "IRR")).toHaveText("20.7%");
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
