import { expect, type Page, test } from "@playwright/test";

import { dealSettled, kpi, modeTab, setField, simulationSettled, stepLink } from "./helpers";

/**
 * The Debt step's facility editor (PLAN.md 2.4b), proved by what the model
 * answers rather than by what the rail shows. Every figure below comes from
 * the deal model for the default deal (core/debt.py `equivalent_tranches`,
 * then the edits each test makes).
 */
const run = Date.now().toString(36);

const tile = (page: Page, title: string) => page.getByRole("region", { name: title, exact: true });
const facility = (page: Page, name: RegExp) => page.getByRole("group", { name: "Capital structure" }).getByRole("button", { name });
/** A row of a schedule tile, header and figures, as text */
const scheduleRow = (page: Page, tileTitle: string, row: string) =>
  tile(page, tileTitle).getByRole("row").filter({ has: page.getByRole("rowheader", { name: row, exact: true }) });

async function useExplicitTranches(page: Page) {
  await page.goto("/deal/debt");
  await dealSettled(page);
  await expect(kpi(page, "Net debt at exit")).toHaveText("258.7");
  await page.getByRole("button", { name: "Use explicit tranches" }).click();
  await expect(facility(page, /^Senior Term Loan · Amortising term loan/)).toBeVisible();
  await dealSettled(page);
}

async function addPikNotes(page: Page) {
  await page.getByLabel("Kind of facility to add").selectOption({ label: "PIK notes" });
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await setField(page, "Size", "100");
  await setField(page, "Fixed rate", "12");
  await dealSettled(page);
}

test.describe("Debt facilities", () => {
  test("writing the debt out as facilities changes nothing, and each edit reaches the model", async ({ page }) => {
    // 1. The first click: two facilities, the same answer
    await useExplicitTranches(page);
    await expect(kpi(page, "Net debt at exit")).toHaveText("258.7");
    await expect(kpi(page, "Debt at close")).toHaveText("600.0");
    await expect(tile(page, "Senior term loan")).toBeVisible();
    await expect(tile(page, "Mezzanine")).toBeVisible();
    await expect(tile(page, "Capital structure")).toContainText("Second lien, floating");
    await stepLink(page, "Returns").click();
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await expect(kpi(page, "MOIC")).toHaveText("2.61x");

    // 2. PIK notes: a third schedule, accruing, and a different deal
    await stepLink(page, "Debt & cash flow").click();
    await addPikNotes(page);
    await expect(tile(page, "PIK notes")).toBeVisible();
    await expect(scheduleRow(page, "PIK notes", "PIK accrued")).toContainText("12.0");
    await expect(kpi(page, "Debt at close")).toHaveText("700.0");
    await expect(kpi(page, "Net debt at exit")).toHaveText("419.6");
    // The plain loans' tables stay as they were: no rows for what they don't have
    await expect(scheduleRow(page, "Mezzanine", "PIK accrued")).toHaveCount(0);
    await stepLink(page, "Returns").click();
    await expect(kpi(page, "IRR")).toHaveText("23.6%");
    await expect(kpi(page, "MOIC")).toHaveText("2.88x");

    // 3. Removed: exactly the answer from before it was added
    await stepLink(page, "Debt & cash flow").click();
    await facility(page, /^PIK notes · PIK notes/).click();
    await page.getByRole("button", { name: "Remove" }).click();
    await dealSettled(page);
    await expect(tile(page, "PIK notes")).toHaveCount(0);
    await expect(kpi(page, "Net debt at exit")).toHaveText("258.7");
    await expect(kpi(page, "Debt at close")).toHaveText("600.0");
    await stepLink(page, "Returns").click();
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await expect(kpi(page, "MOIC")).toHaveText("2.61x");
  });

  test("the order of the list is the order of the sweep, and it is saved", async ({ page }) => {
    const name = `Facilities ${run}`;
    await useExplicitTranches(page);
    // The mezzanine sits behind the senior loan and never sees the sweep
    await expect(scheduleRow(page, "Mezzanine", "Cash sweep")).not.toContainText(/[1-9]/);
    await expect(kpi(page, "Debt at exit")).toHaveText("258.7");

    // 4. Moved to the front, it takes the cash first
    await facility(page, /^Mezzanine · Second lien/).click();
    await page.getByRole("button", { name: "Move up" }).click();
    await dealSettled(page);
    await expect(scheduleRow(page, "Mezzanine", "Cash sweep")).toContainText("16.9");
    await expect(kpi(page, "Debt at exit")).toHaveText("315.0");

    // 5. Saved and reopened, the list and its order survive
    await stepLink(page, "Saved deals").click();
    await page.getByLabel("Deal name", { exact: true }).fill(name);
    await page.getByRole("button", { name: "Save deal" }).click();
    await expect(page.locator('[data-deal-save="saved"]')).toBeVisible();
    await page.reload();
    await stepLink(page, "Debt & cash flow").click();
    await dealSettled(page);
    const rows = page.getByRole("group", { name: "Capital structure" }).getByRole("button", { name: / · / });
    await expect(rows).toHaveCount(2);
    await expect(rows.first()).toHaveAccessibleName(/^Mezzanine · Second lien/);
    await expect(kpi(page, "Debt at exit")).toHaveText("315.0");

    await stepLink(page, "Saved deals").click();
    const row = page.locator(`tr[data-deal="${name}"]`);
    await row.getByRole("button", { name: "Delete" }).click();
    await row.getByRole("button", { name: "Confirm delete" }).click();
    await expect(row).toHaveCount(0);
  });

  test("the other steps point to the facilities instead of offering fields the model ignores", async ({ page }) => {
    await useExplicitTranches(page);
    for (const step of ["Deal inputs", "Returns"]) {
      await stepLink(page, step).click();
      await expect(page.getByText("This deal lists its debt as 2 facilities, edited on the Debt step.")).toBeVisible();
      await expect(page.getByLabel("Debt / EV", { exact: true })).toHaveCount(0);
    }
    await stepLink(page, "Deal inputs").click();
    await expect(tile(page, "Sources")).toContainText("Mezzanine");
    await expect(kpi(page, "Facilities")).toHaveText("2");
  });

  test("Monte Carlo simulates the facilities: the same answer written out, a new one with PIK notes", async ({ page }) => {
    await page.goto("/monte-carlo/distribution");
    await simulationSettled(page);
    await expect(kpi(page, "Mean IRR")).toHaveText("18.0%");

    await modeTab(page, "Deal").click();
    await stepLink(page, "Debt & cash flow").click();
    await dealSettled(page);
    await page.getByRole("button", { name: "Use explicit tranches" }).click();
    await modeTab(page, "Monte Carlo").click();
    await expect(page.getByTestId("rate-tranches-note")).toContainText("the 2 floating facilities");
    await page.getByRole("button", { name: "Run Monte Carlo" }).click();
    await simulationSettled(page);
    await expect(kpi(page, "Mean IRR")).toHaveText("18.0%");
    await expect(kpi(page, "P5")).toHaveText("2.9%");

    await modeTab(page, "Deal").click();
    await stepLink(page, "Debt & cash flow").click();
    await addPikNotes(page);
    await modeTab(page, "Monte Carlo").click();
    await page.getByRole("button", { name: "Run Monte Carlo" }).click();
    await simulationSettled(page);
    await expect(kpi(page, "Mean IRR")).not.toHaveText("18.0%");
  });
});
