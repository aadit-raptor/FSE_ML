import { expect, type Page, test } from "@playwright/test";

import { dealSettled, field, recordedEconomy, replayEconomy, setField } from "./helpers";

/**
 * Current reference rates (PLAN.md 4.2): a floating facility starts at today's level of its
 * benchmark, says where that comes from, and the level reaches the model. The rates are the
 * API's answer for the recorded series (web/e2e/fixtures/economy.json); each interest figure
 * below is 100 of a non-amortising facility the senior loan's sweep never reaches, so year
 * one's interest is 100 x (reference + 5% margin), shown as cash paid (negative).
 */
const { rates } = recordedEconomy().reference_rates;
const SOFR = rates.SOFR.value as number; // 3.89, New York Fed via FRED
const SONIA = rates.SONIA.value as number; // 3.7317, Bank of England via FRED

const tile = (page: Page, title: string) => page.getByRole("region", { name: title, exact: true });
const interestYear1 = (page: Page) =>
  tile(page, "Unitranche").getByRole("row").filter({ has: page.getByRole("rowheader", { name: "Interest", exact: true }) }).getByRole("cell").first();
const reference = (page: Page) => page.getByRole("combobox", { name: "Reference rate", exact: true });

test.describe("Current reference rates", () => {
  test("a SONIA facility picks up today's SONIA, sourced, and the rate reaches the model", async ({ page }) => {
    await replayEconomy(page);
    await page.goto("/deal/debt");
    await dealSettled(page);
    await page.getByRole("button", { name: "Use explicit tranches" }).click();
    await dealSettled(page);

    // 1. A new floating facility in a sterling deal (the test account's currency) starts on
    //    SONIA at today's level, in every year, and says where that comes from
    await page.getByLabel("Kind of facility to add").selectOption({ label: "Unitranche" });
    await page.getByRole("button", { name: "Add", exact: true }).click();
    await expect(reference(page)).toHaveValue("SONIA");
    await expect(field(page, "Reference Y1")).toHaveValue(String(SONIA));
    await expect(field(page, "Reference Y5")).toHaveValue(String(SONIA));
    const source = page.getByTestId("current-reference");
    await expect(source).toContainText("Today's level 3.73%");
    await expect(source.getByRole("link", { name: "FRED" })).toHaveAttribute("href", "https://fred.stlouisfed.org/series/IUDSOIA");
    await expect(page.getByRole("button", { name: "Use today's level" })).toHaveCount(0);
    await setField(page, "Size", "100");
    await setField(page, "Margin", "5");
    await dealSettled(page);
    await expect(interestYear1(page)).toHaveText("-8.7"); // 100 x (3.7317% + 5%)

    // 2. Another benchmark brings its own level: SOFR's
    await reference(page).selectOption("SOFR");
    await expect(field(page, "Reference Y1")).toHaveValue(String(SOFR));
    await dealSettled(page);
    await expect(interestYear1(page)).toHaveText("-8.9"); // 100 x (3.89% + 5%)
    await reference(page).selectOption("SONIA");
    await dealSettled(page);
    await expect(interestYear1(page)).toHaveText("-8.7");

    // 3. A typed rate is the user's; today's level is one click away and gives the same answer
    await setField(page, "Reference Y1", "0");
    await dealSettled(page);
    await expect(interestYear1(page)).toHaveText("-5.0");
    // Looking at another benchmark keeps the rates typed year by year
    await reference(page).selectOption("SOFR");
    await expect(field(page, "Reference Y1")).toHaveValue("0.00");
    await reference(page).selectOption("SONIA");
    await dealSettled(page);
    await expect(interestYear1(page)).toHaveText("-5.0");
    await page.getByRole("button", { name: "Use today's level" }).click();
    await dealSettled(page);
    await expect(field(page, "Reference Y1")).toHaveValue(String(SONIA));
    await expect(interestYear1(page)).toHaveText("-8.7");
  });

  test("a benchmark with no free source says what stands in for it", async ({ page }) => {
    await replayEconomy(page);
    await page.goto("/deal/debt");
    await dealSettled(page);
    await page.getByRole("button", { name: "Use explicit tranches" }).click();
    await page.getByLabel("Kind of facility to add").selectOption({ label: "Unitranche" });
    await page.getByRole("button", { name: "Add", exact: true }).click();
    await reference(page).selectOption("TONA");
    await expect(field(page, "Reference Y1")).toHaveValue(String(rates.TONA.value));
    await expect(page.getByTestId("current-reference")).toContainText("this is the central bank's policy rate");
    await expect(page.getByTestId("current-reference").getByRole("link", { name: "BIS" })).toBeVisible();
  });
});
