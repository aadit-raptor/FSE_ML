import { expect, test } from "@playwright/test";

import { kpi, modeTab, setField, simulationSettled, stepLink } from "./helpers";

test.describe("Settings", () => {
  test("fees change deal results and persist across reloads", async ({ page }) => {
    await page.goto("/settings/fees");
    await expect(kpi(page, "Current deal IRR")).toHaveText("21.2%");
    await setField(page, "Transaction fees, % of EV", "5");
    await expect(kpi(page, "Fees at entry")).toHaveText("65.6");
    await expect(kpi(page, "Current deal IRR")).toHaveText("19.7%");

    await page.reload();
    await expect(kpi(page, "Current deal IRR")).toHaveText("19.7%");
    await page.getByRole("button", { name: "Reset all" }).click();
    await expect(kpi(page, "Current deal IRR")).toHaveText("21.2%");
  });

  test("impossible correlations are flagged and reset clears them", async ({ page }) => {
    await page.goto("/settings/correlations");
    const cell = (a: string, b: string) => page.getByRole("textbox", { name: `Correlation ${a} and ${b}` });
    await cell("Growth", "Exit multiple").fill("0.95");
    await cell("Growth", "Interest").fill("0.9");
    await cell("Exit multiple", "Interest").fill("-0.9");
    await cell("Exit multiple", "Interest").blur();
    // Scoped to the content: Next.js adds its own role=alert route announcer
    const alert = page.locator("#content").getByRole("alert");
    await expect(alert).toContainText("Not a valid correlation matrix");
    await page.getByRole("button", { name: "Reset all" }).click();
    await expect(alert).toHaveCount(0);
  });

  test("deal defaults can be applied to the current deal", async ({ page }) => {
    await page.goto("/settings/deal");
    await setField(page, "Exit multiple", "12");
    await page.getByRole("button", { name: "Apply to current deal" }).click();
    await modeTab(page, "Deal").click();
    await stepLink(page, "Returns").click();
    await expect(kpi(page, "IRR")).toHaveText("23.7%");
    await modeTab(page, "Settings").click();
    await page.getByRole("button", { name: "Reset all" }).click();
  });

  test("senior amortisation and sensitivity ranges change deal output", async ({ page }) => {
    await page.goto("/settings/deal");
    await setField(page, "Senior amortisation", "12");
    await setField(page, "Hold from", "4");
    await setField(page, "Hold to", "6");
    await modeTab(page, "Deal").click();
    await stepLink(page, "Returns").click();
    await expect(kpi(page, "IRR")).toHaveText("23.0%");
    await expect(page.locator("main table").filter({ hasText: "11.0x" }).locator("thead th")).toHaveText(["Exit", "4y", "5y", "6y"]);
    await modeTab(page, "Settings").click();
    await page.getByRole("button", { name: "Reset all" }).click();
  });

  test("scenario presets feed Monte Carlo and mark it stale", async ({ page }) => {
    await page.goto("/monte-carlo/scenarios");
    await simulationSettled(page);
    const bullBefore = await kpi(page, "Bull").textContent();

    await modeTab(page, "Settings").click();
    await stepLink(page, "Scenario presets").click();
    const mult = page.getByRole("textbox", { name: "Bull bull_growth_mult" });
    await mult.fill("3");
    await mult.blur();
    await expect(modeTab(page, "Monte Carlo")).toContainText("stale");

    await modeTab(page, "Monte Carlo").click();
    await stepLink(page, "Scenarios").click();
    await expect(page.getByText("Settings changed")).toBeVisible();
    await page.getByRole("button", { name: "Run Monte Carlo" }).click();
    await expect(kpi(page, "Bull")).not.toHaveText(bullBefore ?? "");
    await simulationSettled(page);

    await modeTab(page, "Settings").click();
    await page.getByRole("button", { name: "Reset all" }).click();
  });
});
