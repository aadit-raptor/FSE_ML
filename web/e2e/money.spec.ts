import { expect, type Page, test } from "@playwright/test";

import { dealSettled, field, kpi, modeTab, simulationSettled, stepLink } from "./helpers";

/**
 * PLAN.md 2.2: a EUR deal in thousands shows € and "k" everywhere; every
 * money figure is in its deal's (or company's) currency and unit.
 *
 * The deal lives in the page (unsaved), so these tests move between screens
 * with the app's own links rather than reloading.
 */

const content = (page: Page) => page.locator("#content");

async function dealInEuroThousands(page: Page) {
  await page.goto("/deal/inputs");
  await dealSettled(page);
  await page.getByLabel("Deal currency").selectOption("EUR");
  await page.getByLabel("Deal money unit").selectOption("thousands");
  await dealSettled(page);
}

/** Every money label on the screen is the deal's: "€k", never a dollar or another unit. */
async function showsEuroThousands(page: Page) {
  await expect(content(page)).toContainText("€k");
  await expect(content(page)).not.toContainText("$");
  await expect(content(page)).not.toContainText("£M");
  await expect(content(page)).not.toContainText("€M");
}

test.describe("Currency and money units", () => {
  test("a new deal starts in the account's currency", async ({ page }) => {
    // e2e/auth.setup.ts gives this account pounds sterling
    await page.goto("/deal/inputs");
    await dealSettled(page);
    await expect(page.getByLabel("Deal currency")).toHaveValue("GBP");
    await expect(page.getByLabel("Deal money unit")).toHaveValue("millions");
    await expect(content(page)).toContainText("£M");
  });

  test("a EUR deal in thousands shows € and k everywhere, with the same returns", async ({ page }) => {
    await dealInEuroThousands(page);

    // Same deal, counted in thousands: EBITDA 100 (millions) is 100,000
    await expect(field(page, "EBITDA")).toHaveValue("100000.0");
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await expect(kpi(page, "Enterprise value")).toHaveText("1,000,000.0");
    await showsEuroThousands(page);

    for (const step of ["Debt & cash flow", "Returns", "Summary"]) {
      await stepLink(page, step).click();
      await dealSettled(page);
      await showsEuroThousands(page);
    }
    await expect(kpi(page, "Equity in")).toHaveText("438,600.0");
    await expect(kpi(page, "Equity out")).toHaveText("1,145,210.0");

    // Settings are entered in the open deal's money too
    await modeTab(page, "Settings").click();
    await expect(field(page, "EBITDA")).toBeVisible();
    await showsEuroThousands(page);

    await modeTab(page, "Monte Carlo").click();
    await simulationSettled(page);
    await showsEuroThousands(page);
  });

  test("a download says what its money is counted in", async ({ page }) => {
    await dealInEuroThousands(page);
    await stepLink(page, "Summary").click();
    await dealSettled(page);
    const request = page.waitForRequest((r) => r.url().endsWith("/api/export/workbook"));
    await page.getByRole("button", { name: /All tables/ }).click();
    const body = (await request).postDataJSON();
    expect(body.money).toEqual({ currency: "EUR", unit: "thousands" });
    expect(JSON.stringify(body.sheets)).not.toContain("$");
  });

  test("changing the unit back gives the original deal", async ({ page }) => {
    await dealInEuroThousands(page);
    await page.getByLabel("Deal money unit").selectOption("billions");
    await expect(field(page, "EBITDA")).toHaveValue("0.1");
    await expect(content(page)).toContainText("€bn");
    await page.getByLabel("Deal money unit").selectOption("millions");
    await expect(field(page, "EBITDA")).toHaveValue("100.0");
    await dealSettled(page);
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await expect(content(page)).toContainText("€M");
  });

  test("the backtest shows its example deal's own currency, not the open deal's", async ({ page }) => {
    await dealInEuroThousands(page);
    await modeTab(page, "Backtest").click();
    await expect(kpi(page, "Actual IRR")).toBeVisible({ timeout: 45_000 });
    // The example deals are US dollar millions (the API says so with each one)
    await expect(content(page)).toContainText("$M");
    await expect(content(page)).not.toContainText("€k");
  });

  test("a forecast company carries its own reporting currency", async ({ page }) => {
    await page.goto("/forecast/historicals");
    const revenue = page.getByLabel("Revenue, LTM", { exact: true });
    await expect(revenue).toHaveValue("265.0");
    await page.getByLabel("Company currency").selectOption("INR");
    await page.getByLabel("Company money unit").selectOption("thousands");
    // Same company, counted in thousands
    await expect(revenue).toHaveValue("265000.0");
    await expect(content(page)).toContainText("Income statement, ₹k");
    await expect(content(page)).not.toContainText("$");

    await stepLink(page, "Statements").click();
    await expect(kpi(page, "Balance sheet")).toHaveText("Balances");
    await expect(content(page)).toContainText("₹k");
    await expect(content(page)).not.toContainText("$");
  });
});
