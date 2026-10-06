import { readFileSync } from "node:fs";
import path from "node:path";

import { expect, type Page, test } from "@playwright/test";

import { dealSettled, kpi, replayCompanies, signInAs, stepLink } from "./helpers";

/**
 * Company search on the deal and the forecast (PLAN.md 4.1b), proved by what the models
 * answer. The registers' answers are recorded (web/e2e/fixtures/companies.json, the API's own
 * answers for real filings: tests/test_company_use.py keeps it current). Each test signs in as
 * its own user, because "Use in deal" changes the open deal.
 *
 *   Tesco, FY ending Feb 2026 (ESEF, IFRS, GBP m): revenue 73,712, EBITDA 2,985 + 1,895 = 4,880
 *   Toyota, FY ending Mar 2026 (EDINET, IFRS, JPY m): EBITDA 3,766,216 + 2,392,519 = 6,158,735
 */
test.use({ storageState: { cookies: [], origins: [] } });

const run = Date.now().toString(36);
const combo = (page: Page, name: string) => page.getByRole("combobox", { name, exact: true });
const rail = (page: Page) => page.getByRole("complementary");

async function find(page: Page, query: string) {
  await rail(page).getByLabel("Company search").fill(query);
  await rail(page).getByRole("button", { name: "Search", exact: true }).click();
}

test("a UK company found by ISIN fills the forecast and the deal from its filings", async ({ page }) => {
  await signInAs(page, `company-tesco-${run}`, "en-GB");
  await replayCompanies(page);
  await page.goto("/forecast/historicals");
  await find(page, "GB00BLGZ9862");
  await expect(rail(page).getByText("1 company found · read as an ISIN")).toBeVisible();
  await rail(page).getByRole("button", { name: /TESCO PLC/ }).click();

  // Its figures by year, each year linked to its filing
  const figures = page.getByRole("table", { name: "Filed figures, TESCO PLC" });
  await expect(figures.getByRole("row", { name: /^EBITDA/ })).toContainText("4,880.0");
  await expect(figures.getByRole("columnheader", { name: /FY2025\/26/ })).toBeVisible();
  const link = figures.getByRole("link", { name: "The filing behind FY2025/26" });
  await expect(link).toHaveAttribute("href", /^https:\/\/filings\.xbrl\.org\//);
  await expect(rail(page).getByText("Not in the filings: Capital expenditure")).toBeVisible();

  await rail(page).getByRole("button", { name: "Use in forecast" }).click();
  await expect(page.getByRole("textbox", { name: "Revenue, FY2025/26", exact: true })).toHaveValue(/73,?712/);
  await expect(page.getByRole("textbox", { name: "Revenue, FY2023/24", exact: true })).toHaveValue(/68,?187/);
  await expect(combo(page, "Accounting standard")).toHaveValue("ifrs");
  await expect(page.getByLabel("Company fiscal year end")).toHaveValue("2");
  await expect(rail(page).getByText(/Filled from the filings' summary figures/)).toBeVisible();
  // The forecast reads Tesco's own margin: 4,880 / 73,712
  await expect(rail(page).locator("dt", { hasText: /^EBITDA margin$/ }).locator("+ dd")).toHaveText("6.6%");
  await stepLink(page, "Statements").click();
  await expect(kpi(page, "Balance sheet")).toHaveText("Balances");

  await stepLink(page, "Historicals").click();
  await rail(page).getByRole("button", { name: "Use in deal" }).click();
  await expect(page).toHaveURL(/\/deal\/inputs/);
  await dealSettled(page);
  await expect(combo(page, "Deal currency")).toHaveValue("GBP");
  await expect(combo(page, "Accounting standard")).toHaveValue("ifrs");
  // Valued post-IFRS 16 on its EBITDA at the default 10x
  await expect(kpi(page, "Enterprise value")).toHaveText("48,800.0");
  await expect(rail(page).getByText("The filing tags no lease cost", { exact: false })).toBeVisible();
});

test("a Japanese company found by name becomes a deal in yen", async ({ page }) => {
  await signInAs(page, `company-toyota-${run}`, "en-GB");
  await replayCompanies(page);
  await page.goto("/deal/inputs");
  await dealSettled(page);
  await find(page, "Toyota");
  await expect(rail(page).getByText("9 companies found")).toBeVisible();
  await expect(rail(page).getByText("Not searched: Companies House (no match)")).toBeVisible();
  await rail(page).getByRole("button", { name: /^TOYOTA MOTOR CORPORATION/ }).click();
  await expect(page.getByRole("link", { name: "The filing behind FY2025/26" })).toHaveAttribute("href", /edinet-fsa\.go\.jp/);

  await rail(page).getByRole("button", { name: "Use in deal" }).click();
  await dealSettled(page);
  await expect(combo(page, "Deal currency")).toHaveValue("JPY");
  await expect(kpi(page, "Enterprise value")).toHaveText("61,587,350.0");
});

test("a company nobody has names the document upload, and one without figures says why", async ({ page }) => {
  await signInAs(page, `company-none-${run}`, "en-GB");
  await replayCompanies(page);
  await page.goto("/forecast/historicals");
  await find(page, "Zzyzx");
  await expect(rail(page).getByText("No company found")).toBeVisible();
  await expect(rail(page).getByText(/Uploading a company's own accounts as a document is planned/)).toBeVisible();

  // Cambridge United's latest accounts give no revenue or EBITDA
  await find(page, "00482197");
  await rail(page).getByRole("button", { name: /CAMBRIDGE UNITED/ }).click();
  await expect(rail(page).getByRole("button", { name: "Use in deal" })).toBeDisabled();
  await expect(rail(page).getByText("The latest year has no EBITDA")).toBeVisible();
  await rail(page).getByRole("button", { name: "Use in forecast" }).click();
  await expect(rail(page).getByRole("alert")).toContainText("Not every year of this company's filings gives revenue");
  // Nothing changed: still the sample
  await expect(page.getByRole("textbox", { name: /^Revenue, / }).last()).toHaveValue("265.0");
});

test("a source this server can't load is shown but can't be chosen", async ({ page }) => {
  await signInAs(page, `company-nokey-${run}`, "en-GB");
  await replayCompanies(page);
  // The same answer as a server without an EDINET key gives
  const recorded = JSON.parse(readFileSync(path.join(__dirname, "fixtures", "companies.json"), "utf-8"));
  const answer = recorded.search.Toyota;
  for (const r of answer.results) if (r.source === "edinet") r.loadable = false;
  await page.route("**/api/companies/search?*", (route) => route.fulfill({ json: answer }));
  await page.goto("/forecast/historicals");
  await find(page, "Toyota");
  const toyota = rail(page).getByRole("button", { name: /^TOYOTA MOTOR CORPORATION/ });
  await expect(toyota).toBeDisabled();
  await expect(toyota).toContainText("EDINET isn't set up on this server");
});
