import { expect, type Page, test } from "@playwright/test";

import { asUser, kpi, signInAs, SIGNED_IN_STATE, stepLink } from "./helpers";

/**
 * Plan vs actual for any deal (PLAN.md 2.7), through the real screens, API
 * and database.
 *
 * Done when: a user's own saved deal is backtested end to end; with the
 * library off the screen still works. The expected figures are worked out by
 * hand below, not read back from the API.
 */
const run = Date.now().toString(36);
const content = (page: Page) => page.locator("#content");
const LIBRARY_OFF = { enabled: false, examples: [] };

/** The default deal with a 12x exit: Deal → Returns shows IRR 23.7%, equity out 1,272.8 (saved-deals.spec.ts). */
const PLAN = { exit_mult: 12 };

/**
 * Five years of actuals and an exit. Exit equity 1,800 - 400 = 1,400 on an
 * equity cheque of 600: MOIC 2.33x, IRR 2.333^(1/5) - 1 = 18.5%. Against the
 * plan's exit equity of 1,272.8 the attribution adds up to +127.2.
 */
const CSV = [
  "line,Y1,Y2,Y3,Y4,Y5",
  "revenue,260,270,280,290,300",
  "ebitda,55,58,61,64,150",
  "net_income,20,22,24,26,28",
  "fcf,30,32,34,36,38",
  "total_debt,580,560,540,520,500",
  "exit_ev,1800",
  "net_debt_at_exit,400",
  "sponsor_equity_entry,600",
].join("\n");

async function createDeal(page: Page, name: string): Promise<string> {
  const res = await page.request.post("/api/deals", { data: { name, inputs: PLAN }, headers: await asUser(page) });
  expect(res.status()).toBe(201);
  return (await res.json()).id;
}

async function deleteDeal(page: Page, id: string) {
  await page.request.delete(`/api/deals/${id}`, { headers: await asUser(page) });
}

const libraryOff = (page: Page) => page.route("**/api/backtesting/examples", (r) => r.fulfill({ json: LIBRARY_OFF }));

const planRadio = (page: Page, name: string | RegExp) => page.getByRole("radio", { name });

test.describe("Backtest: plan vs actual", () => {
  test("a saved deal is backtested end to end, with the example library off", async ({ page, browser }) => {
    const name = `Backtest ${run}`;
    await page.goto("/deal/inputs");
    const id = await createDeal(page, name);
    await libraryOff(page);
    try {
      await page.goto("/backtest/actuals");
      // No library: no examples to pick, and the screen still works
      await expect(page.getByRole("radiogroup", { name: "Example library" })).toHaveCount(0);
      await planRadio(page, name).click();
      await expect(planRadio(page, name)).toHaveAttribute("aria-checked", "true");
      await expect(page.getByRole("textbox", { name: "Revenue, Y1" })).toHaveValue("");

      await page.getByLabel("Upload CSV").setInputFiles({ name: "actuals.csv", mimeType: "text/csv", buffer: Buffer.from(CSV) });
      await expect(content(page).getByRole("status").filter({ hasText: "Loaded 5 years." })).toBeVisible();
      await expect(page.getByRole("textbox", { name: "EBITDA, Y5" })).toHaveValue("150.0");
      await expect(page.getByLabel("Enterprise value at exit")).toHaveValue("1800.0");
      await expect(page.locator('[data-save-state="saved"]')).toBeVisible();

      await stepLink(page, "Plan vs actual").click();
      // The plan is the saved deal's own model result
      await expect(kpi(page, "Plan IRR")).toHaveText("23.7%");
      await expect(kpi(page, "Actual IRR")).toHaveText("18.5%");
      await expect(kpi(page, "Exit equity")).toHaveText("1,400.0");

      await stepLink(page, "Error attribution").click();
      await expect(page.getByRole("img", { name: "Error attribution" })).toContainText("Net debt at exit");
      // (the e2e account is en-GB, so dollars read "US$M")
      await expect(content(page)).toContainText("The parts add up to actual minus plan exit equity (+127.2 US$M)");

      // An edit is saved with the deal and changes the comparison
      await stepLink(page, "Plan and actuals").click();
      const revenue = page.getByRole("textbox", { name: "Revenue, Y1" });
      await revenue.fill("300");
      await revenue.blur();
      await expect(page.locator('[data-save-state="saved"]')).toBeVisible();
      await stepLink(page, "Year by year").click();
      const actualRevenue = content(page).locator("tr").filter({ hasText: "Revenue · actual" });
      await expect(actualRevenue.locator("td").first()).toHaveText("300.0");

      // Elsewhere: a browser with no open deal reads the actuals back from the database
      const elsewhere = await browser.newContext({ storageState: SIGNED_IN_STATE });
      const other = await elsewhere.newPage();
      try {
        await libraryOff(other);
        await other.goto("/backtest/predicted");
        await planRadio(other, name).click();
        await expect(kpi(other, "Actual IRR")).toHaveText("18.5%");
        await expect(kpi(other, "Plan IRR")).toHaveText("23.7%");
      } finally {
        await elsewhere.close();
      }
    } finally {
      await deleteDeal(page, id);
    }
  });

  test("a deal still held compares its years so far, and attribution waits for the exit", async ({ page }) => {
    const name = `Held ${run}`;
    await page.goto("/deal/inputs");
    const id = await createDeal(page, name);
    try {
      await page.goto("/backtest/actuals");
      await planRadio(page, name).click();
      await page.getByLabel("Years with results").selectOption("2");
      const ebitda = page.getByRole("textbox", { name: "EBITDA, Y2" });
      await ebitda.fill("70");
      await ebitda.blur();
      await expect(page.getByRole("textbox", { name: "EBITDA, Y3" })).toHaveCount(0);
      await stepLink(page, "Plan vs actual").click();
      await expect(kpi(page, "Plan IRR")).toHaveText("23.7%");
      await expect(kpi(page, "Latest EBITDA")).toHaveText("70.0");
      await expect(content(page)).toContainText("still held");
      await stepLink(page, "Error attribution").click();
      await expect(content(page)).toContainText("The split needs the exit");
    } finally {
      await deleteDeal(page, id);
    }
  });

  test("an account with no saved deals and no library is told what to do", async ({ page }) => {
    await signInAs(page, `backtest-empty-${run}`, "en-GB");
    await libraryOff(page);
    await page.goto("/backtest/predicted");
    await expect(content(page).getByRole("heading", { name: "Nothing to backtest yet" })).toBeVisible();
    // In the rail and as the empty screen's action
    await expect(content(page).getByRole("link", { name: "Go to saved deals" })).toHaveCount(2);
  });

  test("the example library runs each example as a plan", async ({ page }) => {
    await page.goto("/backtest/predicted");
    await planRadio(page, /^Burger King/).click();
    // The example's own recorded IRR, against its plan run through the deal model
    await expect(kpi(page, "Actual IRR")).toHaveText("19.0%");
    await expect(kpi(page, "Actual percentile")).toHaveText("74.6");
    await expect(content(page).getByRole("note").filter({ hasText: "Examples, not evidence" })).toBeVisible();

    await planRadio(page, /^Dell/).click();
    await expect(kpi(page, "Plan IRR")).toHaveText("11.8%");
    await stepLink(page, "Year by year").click();
    // Dell's plan year 1 EBITDA from the deal model (the old backtest's figure too)
    const planEbitda = content(page).locator("tr").filter({ hasText: "EBITDA · plan" });
    await expect(planEbitda.locator("td").first()).toHaveText("3,626.0");
    await stepLink(page, "Error attribution").click();
    await expect(content(page)).toContainText("add up to actual minus plan exit equity");
  });
});
