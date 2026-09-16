import { expect, type Page, test } from "@playwright/test";

import { dealSettled, kpi, resetAllSettings, setField, settingsSaved, SIGNED_IN_STATE, stepLink } from "./helpers";

/**
 * Saved deals, versions and settings on the account (PLAN.md 1.5), driven
 * through the real screens against the real API and database. "Elsewhere" is
 * a second browser context: same account, nothing shared with the first but
 * the server (no local storage, no open deal).
 *
 * Names carry a run id, because a local database keeps deals between runs;
 * each test deletes what it made.
 */
const run = Date.now().toString(36);

const dealRow = (page: Page, name: string) => page.locator(`tr[data-deal="${name}"]`);

async function saveAs(page: Page, name: string) {
  await stepLink(page, "Saved deals").click();
  await page.getByLabel("Deal name", { exact: true }).fill(name);
  await page.getByRole("button", { name: "Save deal" }).click();
  await expect(page.locator('[data-deal-save="saved"]')).toBeVisible();
  await expect(dealRow(page, name)).toBeVisible();
}

async function deleteDeal(page: Page, name: string) {
  const row = dealRow(page, name);
  await row.getByRole("button", { name: "Delete" }).click();
  await row.getByRole("button", { name: "Confirm delete" }).click();
  await expect(row).toHaveCount(0);
}

/** Wait until the open deal's latest edit is in the database. */
async function autosaved(page: Page) {
  await expect(page.locator('[data-deal-save="saved"]')).toBeVisible();
}

test.describe("Saved deals", () => {
  test("a saved deal reopens identically in another browser", async ({ page, browser }) => {
    const name = `Reopen ${run}`;
    await page.goto("/deal/returns");
    await expect(kpi(page, "IRR")).toHaveText("21.2%");
    await setField(page, "Exit multiple", "12");
    await expect(kpi(page, "IRR")).toHaveText("23.7%");
    await saveAs(page, name);

    const elsewhere = await browser.newContext({ storageState: SIGNED_IN_STATE });
    const other = await elsewhere.newPage();
    try {
      await other.goto("/deal/returns");
      await dealSettled(other);
      // A fresh browser starts on an unsaved default deal...
      await expect(kpi(other, "IRR")).toHaveText("21.2%");
      await stepLink(other, "Saved deals").click();
      await dealRow(other, name).getByRole("button", { name: "Open" }).click();
      await expect(dealRow(other, name)).toHaveAttribute("aria-current", "true");
      // ...and the saved one opens with the same result
      await stepLink(other, "Returns").click();
      await expect(kpi(other, "IRR")).toHaveText("23.7%");
      await expect(kpi(other, "MOIC")).toHaveText("2.90x");
      await expect(kpi(other, "Equity out")).toHaveText("1,272.8");

      await stepLink(other, "Saved deals").click();
      await deleteDeal(other, name);
    } finally {
      await elsewhere.close();
    }
  });

  test("restoring a version brings back its exact IRR", async ({ page }) => {
    const name = `Restore ${run}`;
    await page.goto("/deal/returns");
    await setField(page, "Exit multiple", "12");
    await expect(kpi(page, "IRR")).toHaveText("23.7%");
    await saveAs(page, name);
    await page.getByLabel("Version label", { exact: true }).fill("Exit 12x");
    await page.getByRole("button", { name: "Keep version" }).click();
    const history = page.getByRole("table", { name: "Version history" });
    await expect(history.locator("tr").filter({ hasText: "Exit 12x" })).toHaveCount(1);

    // Edit away from it; autosave keeps the edit
    await stepLink(page, "Returns").click();
    await setField(page, "Exit multiple", "9");
    await expect(kpi(page, "IRR")).not.toHaveText("23.7%");
    const edited = await kpi(page, "IRR").textContent();
    await autosaved(page);

    // A reload reopens the deal as last edited
    await page.reload();
    await expect(kpi(page, "IRR")).toHaveText(edited ?? "");

    await stepLink(page, "Saved deals").click();
    await history.locator("tr").filter({ hasText: "Exit 12x" }).getByRole("button", { name: "Restore" }).click();
    await expect(kpi(page, "IRR")).toHaveText("23.7%");
    await expect(history.locator("tr").filter({ hasText: "Restored version" })).toHaveCount(1);
    // The edit that was replaced is still in the history
    await expect(history.locator("tr").filter({ hasText: "Before restoring" })).toHaveCount(1);
    await stepLink(page, "Returns").click();
    await expect(kpi(page, "IRR")).toHaveText("23.7%");
    await expect(kpi(page, "MOIC")).toHaveText("2.90x");

    await stepLink(page, "Saved deals").click();
    await deleteDeal(page, name);
    await expect(page.locator('[data-deal-save="unsaved"]')).toBeVisible();
  });

  test("rename, duplicate, archive and delete from the deal list", async ({ page }) => {
    const name = `List ${run}`;
    const renamed = `Renamed ${run}`;
    await page.goto("/deal/inputs");
    await dealSettled(page);
    await saveAs(page, name);

    await page.getByLabel("Deal name", { exact: true }).fill(renamed);
    await page.getByRole("button", { name: "Rename" }).click();
    await expect(dealRow(page, renamed)).toBeVisible();
    await expect(dealRow(page, name)).toHaveCount(0);

    await dealRow(page, renamed).getByRole("button", { name: "Duplicate" }).click();
    const copy = `${renamed} (copy)`;
    await expect(dealRow(page, copy)).toBeVisible();
    await expect(dealRow(page, copy)).not.toHaveAttribute("aria-current", "true");

    await dealRow(page, copy).getByRole("button", { name: "Archive" }).click();
    await expect(dealRow(page, copy)).toHaveCount(0);
    await page.getByRole("switch", { name: "Show archived" }).click();
    await expect(dealRow(page, copy)).toContainText("archived");
    await dealRow(page, copy).getByRole("button", { name: "Unarchive" }).click();
    await expect(dealRow(page, copy)).not.toContainText("archived");

    await deleteDeal(page, copy);
    await deleteDeal(page, renamed);
  });

  test("settings follow the account to another browser", async ({ page, browser }) => {
    await page.goto("/settings/fees");
    await expect(kpi(page, "Current deal IRR")).toHaveText("21.2%");
    const saved = settingsSaved(page);
    await setField(page, "Transaction fees, % of EV", "5");
    await saved;
    await expect(kpi(page, "Current deal IRR")).toHaveText("19.7%");

    const elsewhere = await browser.newContext({ storageState: SIGNED_IN_STATE });
    const other = await elsewhere.newPage();
    try {
      await other.goto("/settings/fees");
      await expect(kpi(other, "Current deal IRR")).toHaveText("19.7%");
      await resetAllSettings(other);
      await expect(kpi(other, "Current deal IRR")).toHaveText("21.2%");
    } finally {
      await elsewhere.close();
    }
  });
});
