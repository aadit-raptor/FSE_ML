import { expect, test } from "@playwright/test";

import { dealSettled, field, kpi, setField } from "./helpers";

test.describe("Deal", () => {
  test("default deal matches the API", async ({ page }) => {
    await page.goto("/deal/returns");
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await expect(kpi(page, "MOIC")).toHaveText("2.61x");
    await expect(kpi(page, "Equity in")).toHaveText("438.6");
    await expect(kpi(page, "Equity out")).toHaveText("1,145.5");
  });

  test("exit multiple reruns the model and moves the base cell", async ({ page }) => {
    await page.goto("/deal/returns");
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await setField(page, "Exit multiple", "12");
    await expect(kpi(page, "IRR")).toHaveText("23.8%");
    await expect(kpi(page, "MOIC")).toHaveText("2.90x");
    await expect(kpi(page, "Equity out")).toHaveText("1,273.2");
    await expect(page.locator('td[aria-current="true"]')).toHaveText("23.8%");
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

  test("minimum cash shows the finding 1 notice with the bridge residual", async ({ page }) => {
    await page.goto("/deal/debt");
    await dealSettled(page);
    await expect(page.getByRole("note")).toHaveCount(0);
    await setField(page, "Minimum cash", "20");
    await expect(page.getByRole("note")).toContainText("bridge residual 20.0");
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
