import { expect, type Page, test } from "@playwright/test";

import recorded from "./fixtures/deal-risk.json";
import { kpi } from "./helpers";

/**
 * PLAN.md 5.2: the deal risk score compares the deal with companies and deals
 * like it in its region. CI stores no industry averages, so the figures are
 * replayed from what the real endpoint gives for the recorded averages
 * (fixtures/deal-risk.json, written by `python -m tests.e2e_deal_risk`);
 * tests/test_deal_risk.py checks those answers against the averages. The
 * first test uses the real endpoint: the deal's own leverage reaches it.
 */

type Case = keyof typeof recorded;

async function replay(page: Page, name: Case) {
  await page.route("**/api/ml/deal-risk", (r) => r.fulfill({ json: recorded[name] }));
}

function tile(page: Page) {
  return page.getByRole("region", { name: "Deal risk" });
}

function row(page: Page, metric: string) {
  return tile(page).locator(`tr[data-metric="${metric}"]`);
}

test.describe("Deal risk", () => {
  test("the deal's leverage reaches the score, which needs a country to compare with", async ({ page }) => {
    const answer = page.waitForResponse((r) => r.url().endsWith("/api/ml/deal-risk") && r.status() === 200);
    await page.goto("/deal/inputs");
    const first = await (await answer).json();
    await expect(tile(page)).toContainText("Not enough data.");
    await expect(tile(page)).toContainText("Choose the deal's country under Starting point");

    const next = page.waitForResponse(async (r) => {
      if (!r.url().endsWith("/api/ml/deal-risk") || r.status() !== 200) return false;
      return (await r.json()).inputs.leverage !== first.inputs.leverage;
    });
    await page.getByLabel("Senior debt", { exact: true }).fill("9");
    await page.getByLabel("Senior debt", { exact: true }).blur();
    const after = await (await next).json();
    expect(after.inputs.leverage).toBeGreaterThan(first.inputs.leverage);
  });

  test("a US deal shows its place among its industry, the tested score and deals like it", async ({ page }) => {
    await replay(page, "us");
    await page.goto("/deal/inputs");
    const r = recorded.us;
    await expect(tile(page).locator('[data-provenance="risk"]')).toContainText(
      `Based on ${r.sample!.firms} companies in United States, Retail (Special Lines).`,
    );
    await expect(row(page, "leverage")).toContainText("4.2x");
    await expect(row(page, "leverage")).toContainText("2.9x");
    await expect(row(page, "leverage")).toContainText("in line");
    await expect(row(page, "ebitda_margin")).toContainText("above");
    await expect(kpi(page, "Risk score")).toHaveText("1.0");
    await expect(tile(page)).toContainText("Tested on 5 buyouts in U.S. and tax havens");
    const deals = tile(page).getByRole("list", { name: "Reference deals like it" });
    await expect(deals.getByRole("listitem")).toHaveCount(4);
    await expect(deals).toContainText("Toys");
    await expect(tile(page)).toContainText("Reference deals like it: 4 in U.S. and tax havens, Consumer discretionary");
  });

  test("with the library off the comparison is the same and no deal is listed", async ({ page }) => {
    await replay(page, "us_library_off");
    await page.goto("/deal/inputs");
    await expect(kpi(page, "Risk score")).toHaveText("1.0");
    await expect(row(page, "leverage")).toContainText("2.9x");
    await expect(tile(page)).toContainText("Based on 94 companies");
    await expect(tile(page).locator("[data-risk-deals]")).toHaveCount(0);
    await expect(tile(page)).not.toContainText("eference deals");
  });

  test("outside the US the comparison shows but the score says not enough data", async ({ page }) => {
    await replay(page, "de");
    await page.goto("/deal/inputs");
    await expect(tile(page).locator('[data-provenance="risk"]')).toContainText("Based on 210 companies in Developed Europe, Machinery.");
    await expect(row(page, "leverage")).toContainText("1.9x");
    await expect(row(page, "leverage")).toContainText("above");
    await expect(kpi(page, "Risk score")).toHaveText("Not enough data");
    await expect(tile(page)).toContainText("Tested on 1 buyout in Europe, too few to show a score");
  });

  test("a thin region says not enough data and names it", async ({ page }) => {
    await replay(page, "thin");
    await page.goto("/deal/inputs");
    await expect(tile(page).locator('[data-risk-status="not_enough_data"]')).toContainText(
      "Fewer than 20 Shipbuilding & Marine companies in Australia, NZ and Canada",
    );
    await expect(tile(page).locator("table")).toHaveCount(0);
  });
});
