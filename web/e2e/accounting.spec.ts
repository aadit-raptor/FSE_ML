import { readFileSync } from "node:fs";
import path from "node:path";

import { expect, type Page, test } from "@playwright/test";

import { dealSettled, kpi, replayCompanies, setField, stepLink } from "./helpers";

/**
 * Accounting standards and leases (PLAN.md 2.6), proved by what the model
 * answers. Every figure is the deal model's for the default deal with IFRS
 * leases costing 10 a year and a liability of 60 (tests/test_accounting.py
 * checks the same cases by hand):
 *
 *   post-IFRS 16  EBITDA valued 100, EV 1,000, net debt at entry 660, IRR 22.1%
 *   pre-IFRS 16   EBITDA valued  90, EV   900, net debt at entry 540, IRR 21.2%
 *   US GAAP, post EBITDA valued 110, EV 1,100, net debt at entry 720, IRR 22.0%
 */
const combo = (page: Page, name: string) => page.getByRole("combobox", { name, exact: true });
const tile = (page: Page, title: string) => page.getByRole("region", { name: title, exact: true });
const row = (page: Page, tileTitle: string, label: string) =>
  tile(page, tileTitle).getByRole("row").filter({ has: page.getByRole("rowheader", { name: label, exact: true }) });

async function leasedDeal(page: Page) {
  await page.goto("/deal/inputs");
  await dealSettled(page);
  await expect(kpi(page, "IRR")).toHaveText("21.2%");
  // A standard alone moves nothing
  await combo(page, "Accounting standard").selectOption("ifrs");
  await dealSettled(page);
  await expect(kpi(page, "IRR")).toHaveText("21.2%");
  await setField(page, "Lease cost a year", "10");
  await setField(page, "Lease liability", "60");
  await dealSettled(page);
}

test.describe("Accounting standards", () => {
  test("the lease view moves EV, net debt and the returns as the hand-checked case says", async ({ page }) => {
    await leasedDeal(page);
    // IFRS is post-IFRS 16 by default: valued before lease costs, leases as debt
    await expect(kpi(page, "IRR")).toHaveText("22.1%");
    await expect(kpi(page, "Enterprise value")).toHaveText("1,000.0");
    await expect(tile(page, "Uses")).toContainText("Purchase price, less leases taken over");

    await stepLink(page, "Returns").click();
    await expect(row(page, "Leases", "Entry EV")).toContainText("1,000.0");
    await expect(row(page, "Leases", "Net debt at entry")).toContainText("660.0");
    await expect(row(page, "Leases", "EBITDA after lease costs")).toContainText("90.0");
    await expect(tile(page, "Leases")).toContainText("Counted as debt at entry and exit");

    // Pre-IFRS 16: valued after lease costs, leases not debt
    await stepLink(page, "Deal inputs").click();
    await combo(page, "Priced on").selectOption("pre_ifrs16");
    await dealSettled(page);
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await expect(kpi(page, "Enterprise value")).toHaveText("900.0");
    await expect(tile(page, "Uses")).not.toContainText("less leases taken over");
    await stepLink(page, "Returns").click();
    await expect(row(page, "Leases", "Net debt at entry")).toContainText("540.0");

    // US GAAP's EBITDA is already after lease costs: post-IFRS 16 adds them back
    await stepLink(page, "Deal inputs").click();
    await combo(page, "Accounting standard").selectOption("us_gaap");
    await combo(page, "Priced on").selectOption("post_ifrs16");
    await dealSettled(page);
    await expect(kpi(page, "IRR")).toHaveText("22.0%");
    await expect(kpi(page, "Enterprise value")).toHaveText("1,100.0");
  });

  test("the statements use the standard's words", async ({ page }) => {
    await leasedDeal(page);
    await stepLink(page, "Summary").click();
    await expect(row(page, "Income statement", "Finance costs")).toHaveCount(1);
    await expect(row(page, "Income statement", "Profit for the year")).toHaveCount(1);
    await expect(row(page, "Income statement", "EBITDA after lease costs")).toHaveCount(1);
    await stepLink(page, "Deal inputs").click();
    await combo(page, "Accounting standard").selectOption("us_gaap");
    await dealSettled(page);
    await stepLink(page, "Summary").click();
    await expect(row(page, "Income statement", "Interest expense")).toHaveCount(1);
    await expect(row(page, "Income statement", "Net income")).toHaveCount(1);
  });

  test("lease amounts follow the deal's unit", async ({ page }) => {
    await leasedDeal(page);
    await combo(page, "Deal money unit").selectOption("thousands");
    await dealSettled(page);
    await expect(kpi(page, "IRR")).toHaveText("22.1%");
    await expect(kpi(page, "Enterprise value")).toHaveText("1,000,000.0");
  });

  test("a company's standard names its forecast lines, and an IFRS filing becomes a deal", async ({ page }) => {
    // SAP SE's 2025 20-F as the API answers it (tests/fixtures/edgar, recorded)
    const sap = readFileSync(path.join(__dirname, "fixtures", "edgar-sap.json"), "utf-8");
    await page.route("**/api/edgar/**", (route) => route.fulfill({ status: 200, contentType: "application/json", body: sap }));

    await page.goto("/forecast/statements");
    await expect(row(page, "Income statement", "Interest expense")).toHaveCount(1);
    await combo(page, "Accounting standard").selectOption("ifrs");
    await expect(row(page, "Income statement", "Finance costs")).toHaveCount(1);
    await combo(page, "Accounting standard").selectOption("");

    // Found through the company search (PLAN.md 4.1b); a US filer's statements still come from EDGAR
    await replayCompanies(page);
    await page.getByLabel("Company search").fill("SAP");
    await page.getByRole("button", { name: "Search", exact: true }).click();
    await page.getByRole("button", { name: /^SAP SE/ }).click();
    await page.getByRole("button", { name: "Use in forecast" }).click();
    await expect(combo(page, "Accounting standard")).toHaveValue("ifrs");
    await expect(page.getByText("principal repaid on lease liabilities only")).toBeVisible();
    await page.getByRole("button", { name: "Use in deal" }).click();

    await expect(page).toHaveURL(/\/deal\/inputs/);
    await dealSettled(page);
    await expect(combo(page, "Accounting standard")).toHaveValue("ifrs");
    // Valued post-IFRS 16 on SAP's EBITDA before lease costs: 10 x 10,928
    await expect(kpi(page, "Enterprise value")).toHaveText("109,280.0");
    await stepLink(page, "Returns").click();
    await expect(row(page, "Leases", "Lease liability")).toContainText("1,684.0");
    await expect(row(page, "Leases", "EBITDA after lease costs")).toContainText("10,629.0");
  });
});
