import { expect, test } from "@playwright/test";

import { dealSettled, field, kpi, setField } from "./helpers";

test.describe("Deal", () => {
  test("default deal matches the API", async ({ page }) => {
    await page.goto("/deal/returns");
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await expect(kpi(page, "MOIC")).toHaveText("2.61x");
    await expect(kpi(page, "Equity in")).toHaveText("438.6");
    await expect(kpi(page, "Equity out")).toHaveText("1,145.2");
  });

  test("exit multiple reruns the model and moves the base cell", async ({ page }) => {
    await page.goto("/deal/returns");
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await setField(page, "Exit multiple", "12");
    await expect(kpi(page, "IRR")).toHaveText("23.7%");
    await expect(kpi(page, "MOIC")).toHaveText("2.90x");
    await expect(kpi(page, "Equity out")).toHaveText("1,272.8");
    await expect(page.locator('td[aria-current="true"]')).toHaveText("23.7%");
  });

  test("debt multiples reproduce the Streamlit default deal", async ({ page }) => {
    await page.goto("/deal/inputs");
    await dealSettled(page);
    await setField(page, "Senior debt", "3.4");
    await setField(page, "Mezzanine debt", "0.8");
    await expect(kpi(page, "Senior share")).toHaveText("81.0%");
    await expect(kpi(page, "IRR")).toHaveText("15.7%");
    await expect(kpi(page, "MOIC")).toHaveText("2.07x");
    await expect(kpi(page, "Sponsor equity")).toHaveText("613.9");
  });

  test("with auto-update off, edits wait for Run and Discard restores them", async ({ page }) => {
    await page.goto("/deal/returns");
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await page.getByRole("switch", { name: "Auto-update" }).click();

    await setField(page, "Hold", "6");
    await expect(page.getByText("1 input changed")).toBeVisible();
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await page.getByRole("button", { name: "Run model" }).click();
    await expect(kpi(page, "MOIC")).not.toHaveText("2.61x");
    await expect(page.getByText("1 input changed")).toHaveCount(0);

    await setField(page, "Hold", "7");
    await page.getByRole("button", { name: "Discard" }).click();
    await expect(field(page, "Hold")).toHaveValue("6");
  });

  test("an invalid entry is rejected and doesn't run a half-typed value", async ({ page }) => {
    await page.goto("/deal/returns");
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    const hold = field(page, "Hold");
    await hold.fill("2");
    await hold.fill("20");
    await expect(page.getByText("At most 15")).toBeVisible();
    // Give a wrongly committed "2" time to rerun (300 ms debounce) before checking
    await page.waitForTimeout(1500);
    await dealSettled(page);
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
  });

  test("minimum cash is funded by sponsor equity and the bridge closes", async ({ page }) => {
    await page.goto("/deal/debt");
    await dealSettled(page);
    await setField(page, "Minimum cash", "20");
    await page.locator('nav[aria-label$="steps"]').getByRole("link", { name: "Returns", exact: true }).click();
    await expect(kpi(page, "Equity in")).toHaveText("458.6");
    await expect(kpi(page, "IRR")).toHaveText("20.5%");
    await page.locator('nav[aria-label$="steps"]').getByRole("link", { name: "Summary", exact: true }).click();
    await expect(kpi(page, "Bridge residual")).toHaveText("0.0");
  });

  test("the interest loop converges", async ({ page }) => {
    await page.goto("/deal/debt");
    await expect(kpi(page, "Interest loop")).toHaveText("Converged");
    await expect(page.getByRole("region", { name: "Interest loop" })).toContainText("gap 0.0");
  });

  test("every sensitivity cell is a real run for that hold", async ({ page }) => {
    await page.goto("/deal/returns");
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    // Row 11.0x (the deal's exit multiple), column 6y: a full 6-year run gives 19.3%
    const row = page.locator("main table").filter({ hasText: "11.0x" }).locator("tr", { hasText: "11.0x" });
    await expect(row.locator("td").nth(3)).toHaveText("19.3%");
    await setField(page, "Hold", "6");
    await expect(kpi(page, "IRR")).toHaveText("19.3%");
  });

  test("working capital from days changes cash flow and which fields apply", async ({ page }) => {
    await page.goto("/deal/debt");
    await dealSettled(page);
    const before = await kpi(page, "Cumulative FCF").textContent();
    await expect(field(page, "Receivable days")).toBeDisabled();
    await page.getByRole("switch", { name: "Working capital from days" }).click();
    await expect(field(page, "Receivable days")).toBeEnabled();
    await expect(field(page, "NWC change")).toBeDisabled();
    await expect(kpi(page, "Cumulative FCF")).not.toHaveText(before ?? "");
  });
});
