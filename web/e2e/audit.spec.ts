import { expect, type Page, test } from "@playwright/test";

import { dealSettled, setField, stepLink } from "./helpers";

/**
 * Audit history (PLAN.md 3.3) on Deal -> Saved deals: each action on the
 * screens is one line in the deal's activity, a download from the deal is one
 * of them, and the whole account's view keeps a deal after it is deleted.
 * Names carry a run id, because a local database keeps deals between runs.
 */
const run = Date.now().toString(36);

const dealRow = (page: Page, name: string) => page.locator(`tr[data-deal="${name}"]`);
const activity = (page: Page) => page.getByRole("table", { name: "Activity", exact: true });
const actions = (page: Page) => activity(page).locator("tbody tr").evaluateAll((rows) => rows.map((r) => r.getAttribute("data-action")));

test("each action on a deal is one line of its activity", async ({ page }) => {
  const name = `Audit ${run}`;
  await page.goto("/deal/inputs");
  await dealSettled(page);
  await stepLink(page, "Saved deals").click();
  await page.getByLabel("Deal name", { exact: true }).fill(name);
  await page.getByRole("button", { name: "Save deal" }).click();
  await expect(page.locator('[data-deal-save="saved"]')).toBeVisible();
  await expect.poll(() => actions(page)).toEqual(["created"]);

  // One edit, saved by autosave
  await stepLink(page, "Returns").click();
  await setField(page, "Exit multiple", "12");
  await expect(page.locator('[data-deal-save="saved"]')).toBeVisible();

  // A download from the deal names it, and goes in its history
  await stepLink(page, "Summary").click();
  await dealSettled(page);
  const [request] = await Promise.all([
    page.waitForRequest((r) => new URL(r.url()).pathname === "/api/export/workbook"),
    page.waitForEvent("download"),
    page.getByRole("button", { name: "↓ All tables" }).click(),
  ]);
  expect((await request.response())?.status()).toBe(200);
  expect(new URL(request.url()).searchParams.get("deal_id")).toBeTruthy();

  await stepLink(page, "Saved deals").click();
  await expect.poll(() => actions(page)).toEqual(["exported", "edited", "created"]);
  const edited = activity(page).locator('tr[data-action="edited"]');
  await expect(edited).toContainText("Edited");
  await expect(edited).toContainText("Exit multiple");
  await expect(activity(page).locator('tr[data-action="exported"]')).toContainText("Excel workbook");

  // The whole account names the deal, and keeps it once it is deleted
  await page.getByRole("switch", { name: "Whole account" }).click();
  await expect(activity(page).locator("tbody tr").filter({ hasText: name })).toHaveCount(3);
  await dealRow(page, name).getByRole("button", { name: "Delete" }).click();
  await dealRow(page, name).getByRole("button", { name: "Confirm delete" }).click();
  await expect(dealRow(page, name)).toHaveCount(0);
  const newest = activity(page).locator("tbody tr").first();
  await expect(newest).toHaveAttribute("data-action", "deleted");
  await expect(newest).toContainText("A deleted deal");
  await expect(activity(page).locator("tbody tr").filter({ hasText: name })).toHaveCount(0);
});
