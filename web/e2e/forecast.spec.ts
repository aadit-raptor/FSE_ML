import { expect, test } from "@playwright/test";

import { kpi, stepLink } from "./helpers";

test.describe("Forecast", () => {
  test("sample forecast matches the API and growth assumptions drive revenue", async ({ page }) => {
    await page.goto("/forecast/statements");
    await expect(kpi(page, "Revenue F+5")).toHaveText("354.6");
    await expect(kpi(page, "Balance sheet")).toHaveText("Balances");

    await stepLink(page, "Assumptions").click();
    const cells = page.getByRole("textbox", { name: /^Revenue growth, F\+\d$/ });
    await expect(cells).toHaveCount(5);
    for (let i = 0; i < 5; i++) {
      await cells.nth(i).fill("10");
      await cells.nth(i).blur();
    }
    await stepLink(page, "Statements").click();
    // 265.0 x 1.1^5
    await expect(kpi(page, "Revenue F+5")).toHaveText("426.8");
    await expect(kpi(page, "Balance sheet")).toHaveText("Balances");
  });

  test("simulation shows both fans and target probabilities", async ({ page }) => {
    await page.goto("/forecast/simulation");
    await expect(kpi(page, "Paths")).toHaveText("20,000");
    await expect(page.locator("main svg[role=img]")).toHaveCount(2);
    // Above plan is bull, below plan bear
    const rows = page.locator("main table tbody tr");
    await expect(rows.first()).toContainText("bear");
    await expect(rows.last()).toContainText("bull");
  });
});
