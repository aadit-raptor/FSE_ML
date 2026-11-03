import { expect, type Page, test } from "@playwright/test";

import { kpi, signInAs } from "./helpers";

/**
 * The lender view (PLAN.md 7.5): a covenant added to a facility, its level stepped down in year 2 so the default
 * deal breaches there and nowhere else, the convention changing what a test reads, the breach odds read from the
 * simulation's own answer, the expected loss from the deal's, and the export. tests/test_covenants.py works the
 * definitions out by hand; here every figure is checked against the API's answer the screen was given.
 */
const run = Date.now().toString(36);

test.use({ storageState: { cookies: [], origins: [] } });

function covenantTest(page: Page, index: number) {
  return page.locator(`tr[data-covenant-test="${index}"]`);
}

async function setLevel(page: Page, year: string, value: string, nth = 0) {
  const cell = page.getByRole("textbox", { name: `Level for ${year}`, exact: true }).nth(nth);
  await cell.fill(value);
  await cell.blur();
}

const lenderOf = (page: Page) =>
  page.waitForResponse(async (r) => r.url().endsWith("/api/deal/run") && r.ok()).then(async (r) => (await r.json()).lender);

test("a covenant stepped down in year 2 breaches in year 2, with odds from the simulation and expected loss", async ({ page }) => {
  test.setTimeout(120_000);
  await signInAs(page, `lender-${run}`, "en-GB");
  await page.goto("/deal/lender");
  await expect(page.getByText("Add covenants in the rail to test them against this deal each year.")).toBeVisible();

  // The first covenant: leverage on the senior loan, starting at the deal's year one (5.45x) with 30% headroom
  let answer = lenderOf(page);
  await page.getByRole("button", { name: "Add covenant" }).click();
  let lender = await answer;
  expect(lender.covenants[0]).toMatchObject({ facility: "Senior Term Loan", metric: "leverage", first_breach: null });
  await expect(page.getByRole("textbox", { name: "Level for Y1", exact: true })).toHaveValue("7.00");

  // Year 2's leverage is 4.89x: a maximum of 4.5x in year 2 alone is breached in year 2 only
  answer = lenderOf(page);
  await setLevel(page, "Y2", "4.5");
  lender = await answer;
  expect(lender.covenants[0].years.map((y: { breach: boolean }) => y.breach)).toEqual([false, true, false, false, false]);
  const cells = covenantTest(page, 0).locator("td");
  await expect(cells.nth(1)).toHaveAttribute("data-breach", "");
  await expect(covenantTest(page, 0).locator("td[data-breach]")).toHaveCount(1);
  await expect(cells.nth(1)).toContainText("4.89x");
  await expect(cells.nth(1)).toContainText("breach of 4.50x");
  await expect(kpi(page, "First breach")).toContainText("Y2");

  // Breach odds: the screen ran the simulation once for its first covenant; the changed level made that run out
  // of date, so it is run again here, and the table shows its answer
  const breachOdds = page.getByRole("region", { name: "Breach odds from the simulation" });
  await breachOdds.getByRole("button", { name: "Run the simulation" }).click();
  const odds = page.locator('tr[data-breach-odds="0"] td[data-year="2"]');
  await expect(odds).toBeVisible({ timeout: 60_000 });
  const year2 = Number(await odds.getAttribute("data-odds"));
  expect(year2).toBeGreaterThan(0.05);
  expect(year2).toBeLessThan(0.99);
  const year1 = Number(await page.locator('tr[data-breach-odds="0"] td[data-year="1"]').getAttribute("data-odds"));
  expect(year1).toBeLessThan(year2);
  await expect(kpi(page, "Chance of any breach")).not.toContainText("Not simulated");

  // A second covenant: fixed-charge cover on the mezzanine. The LMA's cashflow cover and India's DSCR read different
  // cash flows, so changing the convention changes the figure
  answer = lenderOf(page);
  await page.getByRole("button", { name: "Add covenant" }).click();
  await answer;
  answer = lenderOf(page);
  await page.getByRole("combobox", { name: "Facility", exact: true }).nth(1).selectOption("Mezzanine");
  await answer;
  answer = lenderOf(page);
  await page.getByRole("combobox", { name: "Test", exact: true }).nth(1).selectOption("fixed_charge_cover");
  const lma = (await answer).covenants[1].years[0].value;
  await expect(covenantTest(page, 1).locator("td").first()).toContainText(`${lma.toFixed(2)}x`);
  answer = lenderOf(page);
  await page.getByRole("combobox", { name: "Market convention", exact: true }).selectOption("india");
  lender = await answer;
  expect(lender.convention).toBe("india");
  const india = lender.covenants[1].years[0].value;
  expect(india).not.toBeCloseTo(lma, 2);
  await expect(covenantTest(page, 1).locator("td").first()).toContainText(`${india.toFixed(2)}x`);
  await expect(kpi(page, "Definitions")).toContainText("Indian bank lending");

  // Expected loss: each facility's figure is the deal answer's
  const senior = lender.facilities.find((f: { name: string }) => f.name === "Senior Term Loan");
  await expect(page.locator('tr[data-loss-facility="Senior Term Loan"] td[data-expected-loss]'))
    .toHaveText(senior.expected_loss.toFixed(1));

  // The export carries the covenant tests, the breach odds and the expected loss
  const exported = page.waitForRequest((r) => r.url().endsWith("/api/export/workbook"));
  await page.getByRole("region", { name: "Covenant tests by year" }).getByRole("button", { name: /Excel/ }).click();
  const body = (await exported).postDataJSON();
  const names = body.sheets.map((s: { name: string }) => s.name);
  expect(names[0]).toBe("Covenant tests");
  expect(names.at(-1)).toBe("Expected loss");
  const breached = body.sheets[0].rows.filter((r: unknown[]) => r[5] === "Yes").map((r: unknown[]) => r.slice(0, 2));
  expect(breached).toContainEqual(["Senior term loan: Leverage", "Y2"]);
});
