import { expect, type Page, test } from "@playwright/test";

import { dealSettled, setField } from "./helpers";

/**
 * PLAN.md 2.8: risk warnings are computed from the deal and published data,
 * and each shows its sources. Every figure checked here moves with an input,
 * so a warning written into the page would fail.
 */

const warning = (page: Page, id: string) => page.locator(`[data-warning="${id}"]`);

test.describe("Risk warnings", () => {
  test("leverage above the supervisors' 6.0x appears, cites both, and goes when debt falls", async ({ page }) => {
    await page.goto("/deal/inputs");
    await dealSettled(page);
    await setField(page, "Senior debt", "5");
    await setField(page, "Mezzanine debt", "2");
    await dealSettled(page);
    const lev = warning(page, "leverage_above_guidance");
    await expect(lev).toContainText("Debt at close is 7.0x EBITDA, above the 6.0x");
    await expect(lev.locator('[data-source="ecb_leveraged_2017"]')).toContainText("European Central Bank");
    await expect(lev.locator('[data-source="us_leveraged_2013"]')).toContainText("Interagency Guidance on Leveraged Lending");

    await setField(page, "Senior debt", "6");
    await dealSettled(page);
    await expect(lev).toContainText("Debt at close is 8.0x EBITDA");

    await setField(page, "Senior debt", "2");
    await setField(page, "Mezzanine debt", "0.5");
    await dealSettled(page);
    await expect(lev).toHaveCount(0);
  });

  test("the implied rating reads S&P's default rate for the deal's own hold, with the study's sample", async ({ page }) => {
    await page.goto("/deal/inputs");
    await dealSettled(page);
    await setField(page, "Senior debt", "5");
    await setField(page, "Mezzanine debt", "2");
    await dealSettled(page);
    const rating = warning(page, "implied_rating");
    await expect(rating).toContainText("a speculative grade");
    await expect(rating).toContainText("within 5 years");
    const sp = rating.locator('[data-source="sp_default_study_2024"]');
    await expect(sp).toContainText("23,831 issuers, 1981 to 2024");
    await expect(rating.locator('[data-source="damodaran_ratings_2026"]')).toContainText("Damodaran");
    const before = await rating.locator("p").first().innerText();

    // The horizon is the deal's hold: a longer hold reads a later column
    await setField(page, "Hold", "7");
    await dealSettled(page);
    await expect(rating).toContainText("within 7 years");
    expect(await rating.locator("p").first().innerText()).not.toBe(before);
  });
});
