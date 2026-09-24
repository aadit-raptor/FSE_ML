import { expect, type Page, test } from "@playwright/test";

import { asUser, dealSettled, field, kpi, setField, stepLink } from "./helpers";

/**
 * Numbers, dates and fiscal years follow the account (PLAN.md 2.3a), in
 * en-US, de-DE and en-IN. Each locale signs in as its own development user,
 * so the shared e2e account (en-GB) is never touched, and every expected
 * figure is a real model answer: the default deal, or the API's own run of
 * the same inputs.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const run = Date.now().toString(36);

type Grouping = "locale" | "thousands" | "lakh";

async function signInAs(page: Page, user: string, locale: string, grouping: Grouping = "locale") {
  await page.goto("/sign-in");
  await page.getByLabel("Development user").fill(user);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/(deal|account)/);
  await page.goto("/account");
  await page.getByLabel("Country").selectOption(locale.slice(-2));
  await page.getByLabel("Currency").selectOption("USD");
  await page.getByLabel("Number and date format").fill(locale);
  await page.getByLabel("Digit grouping").selectOption(grouping);
  await page.getByLabel("Time zone").selectOption("Europe/London");
  const saved = page.waitForResponse((r) => r.url().endsWith("/api/account") && r.request().method() === "POST" && r.ok());
  await page.getByRole("button", { name: /^Save/ }).click();
  await saved;
  // Settings live on the account: start from the defaults
  await page.request.put("/api/account/settings", { data: { settings: {} }, headers: await asUser(page) });
}

/** The API's IRR for the default deal with this exit multiple. */
async function irrWithExit(page: Page, exitMult: number): Promise<number> {
  const res = await page.request.post("/api/deal/run", { data: { inputs: { exit_mult: exitMult } }, headers: await asUser(page) });
  expect(res.ok()).toBe(true);
  return (await res.json()).returns.irr;
}

const CASES = [
  { locale: "en-US", irr: "21.2%", moic: "2.61x", equityOut: "1,145.2", typed: "11.5", grouped: /^\d,\d{3},\d{3}\.\d$/ },
  { locale: "de-DE", irr: "21,2\u00a0%", moic: "2,61x", equityOut: "1.145,2", typed: "11,5", grouped: /^\d\.\d{3}\.\d{3},\d$/ },
  { locale: "en-IN", irr: "21.2%", moic: "2.61x", equityOut: "1,145.2", typed: "11.5", grouped: /^\d{1,2},\d{2},\d{3}\.\d$/ },
];

for (const c of CASES) {
  test.describe(`Locale ${c.locale}`, () => {
    test("figures, typed numbers, grouping and dates follow the account", async ({ page }) => {
      await signInAs(page, `locale-${c.locale}-${run}`, c.locale);

      // The default deal's answer, written the locale's way
      await page.goto("/deal/returns");
      await dealSettled(page);
      await expect(kpi(page, "IRR")).toHaveText(c.irr);
      await expect(kpi(page, "MOIC")).toHaveText(c.moic);
      await expect(kpi(page, "Equity out")).toHaveText(c.equityOut);

      // A number typed the locale's way is the number the model runs
      await setField(page, "Exit multiple", c.typed);
      await dealSettled(page);
      const irr = await irrWithExit(page, 11.5);
      const shown = new Intl.NumberFormat(c.locale, { style: "percent", minimumFractionDigits: 1, maximumFractionDigits: 1 }).format(irr);
      await expect(kpi(page, "IRR")).toHaveText(shown);
      await expect(field(page, "Exit multiple")).toHaveValue(c.typed);
      await setField(page, "Exit multiple", "11");

      // Millions of the unit: the same deal in thousands groups its digits the locale's way
      await page.goto("/deal/inputs");
      await page.getByLabel("Deal money unit").selectOption("thousands");
      await expect(field(page, "EBITDA")).toHaveValue(c.locale === "de-DE" ? "100000,0" : "100000.0");
      await stepLink(page, "Returns").click();
      await dealSettled(page);
      await expect(kpi(page, "Equity out")).toHaveText(c.grouped);
      await page.goto("/deal/inputs");
      await page.getByLabel("Deal money unit").selectOption("millions");

      // A saved deal's time, in the locale's date format
      const name = `Locale ${c.locale} ${run}`;
      await stepLink(page, "Saved deals").click();
      await page.getByLabel("Deal name", { exact: true }).fill(name);
      await page.getByRole("button", { name: "Save deal" }).click();
      const row = page.locator(`tr[data-deal="${name}"]`);
      await expect(row).toBeVisible();
      const today = new Intl.DateTimeFormat(c.locale, { dateStyle: "medium", timeZone: "Europe/London" }).format(new Date());
      await expect(row).toContainText(today);
      await row.getByRole("button", { name: "Delete" }).click();
      await row.getByRole("button", { name: "Confirm delete" }).click();
      await expect(row).toHaveCount(0);
    });
  });
}

test.describe("Digit grouping", () => {
  test("lakh and crore in en-US, thousands in en-IN, when the account says so", async ({ page }) => {
    await signInAs(page, `grouping-us-${run}`, "en-US", "lakh");
    await page.goto("/deal/inputs");
    await page.getByLabel("Deal money unit").selectOption("thousands");
    await stepLink(page, "Returns").click();
    await dealSettled(page);
    await expect(kpi(page, "Equity out")).toHaveText(/^\d{1,2},\d{2},\d{3}\.\d$/);
    await page.goto("/deal/inputs");
    await page.getByLabel("Deal money unit").selectOption("millions");

    await signInAs(page, `grouping-in-${run}`, "en-IN", "thousands");
    await page.goto("/deal/inputs");
    await page.getByLabel("Deal money unit").selectOption("thousands");
    await stepLink(page, "Returns").click();
    await dealSettled(page);
    await expect(kpi(page, "Equity out")).toHaveText(/^\d,\d{3},\d{3}\.\d$/);
    await page.goto("/deal/inputs");
    await page.getByLabel("Deal money unit").selectOption("millions");
  });
});

test.describe("Fiscal years", () => {
  test("a March year-end deal labels its years FY2026/27 onwards, and the numbers don't move", async ({ page }) => {
    await signInAs(page, `fiscal-deal-${run}`, "en-IN");
    await page.goto("/deal/inputs");
    await dealSettled(page);
    await page.getByLabel("Deal fiscal year end").selectOption("3");
    await page.getByLabel("Deal first fiscal year").fill("2027");
    await expect(page.getByText("Fiscal years end in March")).toBeVisible();
    await stepLink(page, "Returns").click();
    await dealSettled(page);
    const table = page.getByRole("table", { name: "Operating summary by year" });
    await expect(table.getByRole("columnheader", { name: "FY2026/27" })).toBeVisible();
    await expect(table.getByRole("columnheader", { name: "FY2030/31" })).toBeVisible();
    // Labels only: the default deal's answer
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    // Back to plain years for the next run of this account
    await page.goto("/deal/inputs");
    await page.getByLabel("Deal first fiscal year").fill("");
    await page.getByLabel("Deal fiscal year end").selectOption("12");
  });

  test("a March year-end company from EDGAR labels its history and forecast", async ({ page }) => {
    await signInAs(page, `fiscal-company-${run}`, "en-IN");
    // A recorded answer: CI has no network to the SEC. Fields left out keep the sample's values.
    await page.route("**/api/edgar/*", (route) =>
      route.fulfill({
        json: {
          ticker: "MARCH", company_name: "MARCH YEAR-END CO", years: [2023, 2024, 2025], history: {}, warnings: [],
          fiscal_year_end_month: 3, money: { currency: "USD", unit: "millions" },
        },
      }),
    );
    await page.goto("/forecast/historicals");
    await page.getByPlaceholder("Ticker, e.g. DELL").fill("MARCH");
    await page.getByRole("button", { name: "Fetch" }).click();
    await expect(page.getByText("Fiscal years FY2022/23, FY2023/24, FY2024/25")).toBeVisible();
    await expect(page.getByLabel("Company fiscal year end")).toHaveValue("3");
    await expect(page.getByLabel("Company latest fiscal year")).toHaveValue("2025");

    await stepLink(page, "Statements").click();
    // Same model answer as the sample's F+5, now named for its fiscal year
    await expect(kpi(page, "Revenue FY2029/30")).toHaveText("354.6");
    await expect(page.getByRole("table", { name: "Forecast income statement" }).getByRole("columnheader", { name: "FY2024/25" })).toBeVisible();

    // A December year-end the user types in relabels the same columns
    await page.getByLabel("Company fiscal year end").selectOption("12");
    await expect(kpi(page, "Revenue FY2030")).toHaveText("354.6");
  });
});
