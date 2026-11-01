import { expect, type Page, test } from "@playwright/test";

import { asUser, modeTab, signInAs, stepLink } from "./helpers";

/**
 * Portfolio (PLAN.md 7.4): a saved deal marked as owned, quarterly results typed in, the alert they raise below the
 * plan, the reforecast job, and the dashboard's figures against the deal model's own answer. Conversion between
 * currencies at stated ECB rates is proven in tests/test_portfolio_api.py, which can store the rates; here the
 * account and the deal share one currency.
 */
const run = Date.now().toString(36);

test.use({ storageState: { cookies: [], origins: [] } });

async function typeQuarter(page: Page, quarter: string, ebitda: string) {
  await page.getByRole("button", { name: "Add quarter" }).click();
  const cell = page.getByRole("textbox", { name: `EBITDA, ${quarter}`, exact: true });
  await cell.fill(ebitda);
  await cell.blur();
}

test("an owned deal's quarters below plan raise an alert, reforecast and total on the dashboard", async ({ page }) => {
  test.setTimeout(120_000);
  await signInAs(page, `portfolio-${run}`, "en-GB");
  const headers = await asUser(page);
  const name = `Holding ${run}`;

  await page.goto("/portfolio/dashboard");
  await expect(page.locator('[data-guide="portfolio"]')).toBeVisible();

  // a saved deal, the default one: year 1's EBITDA is 105.0 in the plan, 26.25 a quarter
  await page.goto("/deal/saved");
  await page.getByLabel("Deal name", { exact: true }).fill(name);
  await page.getByRole("button", { name: "Save deal" }).click();
  await expect(page.locator('[data-deal-save="saved"]')).toBeVisible();

  await modeTab(page, "Portfolio").click();
  await stepLink(page, "Quarterly results").click();
  await page.getByRole("switch", { name: "In the portfolio" }).click();
  await expect(page.getByRole("switch", { name: "In the portfolio" })).toHaveAttribute("aria-checked", "true");

  // two quarters at 85% of the plan's: 44.6 against 52.5, 15% below, beyond the 5% tolerance
  await typeQuarter(page, "Y1 Q1", "22.3");
  await typeQuarter(page, "Y1 Q2", "22.3");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  const alert = page.locator('[data-alert="ebitda_below_plan"]');
  await expect(alert).toHaveText("EBITDA this year so far is 44.6 against a plan of 52.5 (-15.0%).");

  // a looser tolerance clears it
  const tolerance = page.getByLabel("Alert tolerance (%)");
  await tolerance.fill("20");
  await tolerance.press("Enter");
  await expect(page.locator('[data-alerts="0"]')).toBeVisible();
  await tolerance.fill("5");
  await tolerance.press("Enter");
  await expect(alert).toBeVisible();

  // four quarters at 90%: the reforecast's growth is 1.05 x 0.9 - 1 = -5.5%
  for (const q of ["Y1 Q1", "Y1 Q2"]) {
    const cell = page.getByRole("textbox", { name: `EBITDA, ${q}`, exact: true });
    await cell.fill("23.625");
    await cell.blur();
  }
  await typeQuarter(page, "Y1 Q3", "23.625");
  await typeQuarter(page, "Y1 Q4", "23.625");
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await expect(page.locator('[data-quarter="1-4"]')).toBeVisible();
  await expect(page.getByRole("button", { name: "Save", exact: true })).toBeDisabled();
  await page.getByRole("button", { name: "Reforecast from actuals" }).click();
  await expect(page.getByText("Growth 5.0% → -5.5%")).toBeVisible({ timeout: 60_000 });

  // the dashboard: the holding's figures are the deal model's, in the account's currency
  const deals = await (await page.request.get("/api/deals", { headers })).json();
  const saved = deals.deals.find((d: { name: string }) => d.name === name);
  const detail = await (await page.request.get(`/api/deals/${saved.id}`, { headers })).json();
  const plan = await (await page.request.post("/api/deal/run", {
    headers, data: { inputs: detail.inputs, settings: detail.settings },
  })).json();
  const money = (v: number) => v.toLocaleString("en-GB", { minimumFractionDigits: 1, maximumFractionDigits: 1 }); // locale-ok

  await stepLink(page, "Dashboard").click();
  const row = page.locator(`[data-holding="${name}"]`);
  await expect(row.locator('[data-col="entry_equity"]')).toHaveText(money(plan.returns.entry_equity));
  await expect(row.locator('[data-col="ltm_ebitda"]')).toHaveText(money(94.5));
  await expect(page.locator('[data-total-col="entry_equity"]')).toHaveText(money(plan.returns.entry_equity));
  await expect(row.locator('[data-alert="ebitda_below_plan"]')).toBeVisible();
  await expect(page.locator('[data-kpi="Alerts"]')).toHaveText("1");

  const history = await (await page.request.get(`/api/deals/${saved.id}/history`, { headers })).json();
  const actions = history.entries.map((e: { action: string }) => e.action);
  expect(actions).toEqual(expect.arrayContaining(["portfolio_added", "quarters_saved", "reforecast_run"]));
});
