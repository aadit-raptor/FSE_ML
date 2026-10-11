import { expect, type Page, test } from "@playwright/test";

import { asUser, dealSettled, field, kpi, replayBenchmarks, signInAs } from "./helpers";

/**
 * Personalized defaults (PLAN.md 5.9), through the real screens, API and database: an account's
 * usual figures -- the median of its own saved deals in the deal's region and sector -- sit beside
 * the sourced starting figures, reach the model when used, and stop when switched off. That no
 * other account's deals are read is tested in tests/test_usual.py; here each test has an account
 * of its own, so the figures are worked out by hand from the three deals it saves.
 */
const run = Date.now().toString(36);

const startingTile = (page: Page) => page.getByRole("region", { name: /^Starting figures/ });
const usualCell = (page: Page, name: string) => startingTile(page).locator(`td[data-usual="${name}"]`);
const usualSwitch = (page: Page) => page.getByRole("switch", { name: "Suggest my usual figures" });

/** A saved deal in Europe's industrials: France, trucking, as a deal sourced there would carry. */
async function saveDeal(page: Page, name: string, grossMargin: number) {
  const inputs = { country: "FR", industry: "trucking", gross_margin: grossMargin, entry_mult: 8 };
  const res = await page.request.post("/api/deals", { data: { name, inputs }, headers: await asUser(page) });
  expect(res.status()).toBe(201);
}

async function germanMachinery(page: Page) {
  await page.goto("/deal/inputs");
  await dealSettled(page);
  await page.getByRole("combobox", { name: "Deal country", exact: true }).selectOption({ label: "Germany" });
  await page.getByRole("combobox", { name: "Deal industry", exact: true }).selectOption({ label: "Machinery" });
  await page.getByRole("combobox", { name: "Deal currency", exact: true }).selectOption("EUR");
  await page.getByRole("button", { name: "Use sourced figures" }).click();
  await dealSettled(page);
}

// Each test signs in as its own user; the shared signed-in state is not used
test.use({ storageState: { cookies: [], origins: [] } });

test.describe("Your usual figures", () => {
  test("three of the account's own deals in the region and sector suggest their median, and using it moves the model", async ({ page }) => {
    await signInAs(page, `usual-${run}`, "en-GB");
    await replayBenchmarks(page);

    // 1. One deal is not enough: the tile says how many are needed, and suggests nothing
    await saveDeal(page, "Usual one", 30);
    await germanMachinery(page);
    await expect(startingTile(page).locator('[data-usual-note="not_enough"]')).toHaveText(
      "Your usual figures appear once 3 of your other saved deals are in Industrials, Europe; so far 1 is.",
    );
    await expect(startingTile(page).getByRole("columnheader", { name: "Your usual" })).toHaveCount(0);

    // 2. Three deals with gross margins 30, 34 and 50: the median is 34, the range 30 to 50
    await saveDeal(page, "Usual two", 50);
    await saveDeal(page, "Usual three", 34);
    await germanMachinery(page);
    await expect(startingTile(page).getByRole("columnheader", { name: "Your usual" })).toBeVisible();
    await expect(usualCell(page, "gross_margin")).toContainText("34.00%");
    await expect(usualCell(page, "gross_margin")).toContainText("30.00% to 50.00%, 3 deals");
    await expect(usualCell(page, "entry_mult")).toContainText("8.00x");
    await expect(startingTile(page).locator('[data-usual-note="ok"]')).toContainText(
      "the median across your own 3 other saved deals in Industrials, Europe",
    );

    // 3. Using it sets the input and the model answers for it: EV is 100 x 8.0x
    const irrBefore = await kpi(page, "IRR").textContent();
    await page.getByRole("button", { name: "Use your usual Gross margin" }).click();
    await dealSettled(page);
    await expect(field(page, "Gross margin")).toHaveValue("34.0");
    await expect(kpi(page, "IRR")).not.toHaveText(irrBefore ?? "");
    await expect(page.getByRole("button", { name: "Use your usual Gross margin" })).toHaveCount(0);
    await page.getByRole("button", { name: "Use your usual Entry multiple" }).click();
    await dealSettled(page);
    await expect(kpi(page, "Enterprise value")).toHaveText("800.0");

    // 4. Off stops them, and stays off after a reload; the sourced figures are untouched
    await expect(usualSwitch(page)).toHaveAttribute("aria-checked", "true");
    await usualSwitch(page).click();
    await expect(usualSwitch(page)).toHaveAttribute("aria-checked", "false");
    await expect(startingTile(page).getByRole("columnheader", { name: "Your usual" })).toHaveCount(0);
    await expect(startingTile(page).locator("[data-usual-note]")).toHaveCount(0);
    await expect(startingTile(page).locator('td[data-sourced="gross_margin"]')).toBeVisible();
    await germanMachinery(page);
    await expect(usualSwitch(page)).toHaveAttribute("aria-checked", "false");
    await expect(startingTile(page).getByRole("columnheader", { name: "Your usual" })).toHaveCount(0);
    const off = await page.request.get("/api/account/suggestions?country=DE&industry=machinery", { headers: await asUser(page) });
    expect(await off.json()).toMatchObject({ enabled: false, status: "off", figures: [] });

    // 5. ... and on brings them back
    await usualSwitch(page).click();
    await expect(usualCell(page, "gross_margin")).toContainText("34.00%");
  });

  test("another account sees none of them", async ({ page }) => {
    await signInAs(page, `usual-other-${run}`, "en-GB");
    await replayBenchmarks(page);
    await germanMachinery(page);
    await expect(usualSwitch(page)).toHaveAttribute("aria-checked", "true");
    await expect(startingTile(page).getByRole("columnheader", { name: "Your usual" })).toHaveCount(0);
    await expect(startingTile(page).locator("[data-usual-note]")).toHaveCount(0);
  });
});
