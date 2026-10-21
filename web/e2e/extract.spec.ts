import { expect, type Page, test } from "@playwright/test";

import { asUser, field, kpi, onePagePdf, signInAs, stepLink } from "./helpers";

/**
 * A document read for a deal (PLAN.md 6.2), through the real screen, API, job runner and file store, with the AI
 * replaced by recorded answers (playwright.config.ts: any document is answered as MBB's 2025 annual report, in
 * German, read twice when recorded). No AI is called.
 *
 * "Done when": nothing enters a deal without acceptance; the confirmation blocks a document the user does not
 * confirm is public on the free provider.
 */
const run = Date.now().toString(36);

const readButton = (page: Page) => page.getByRole("button", { name: "Read the document" });
const confirm = (page: Page) => page.getByRole("checkbox", { name: "This document is public (a published annual report or filing)" });
const accept = (page: Page, id: string) => page.locator(`tr[data-item="${id}"] input[type="checkbox"]`);

async function uploadDocument(page: Page, name: string) {
  await page.goto("/deal/from-document");
  await expect(page.getByRole("button", { name: "Upload a PDF" })).toBeEnabled();
  await page.locator('input[type="file"]').setInputFiles({ name, mimeType: "application/pdf", buffer: onePagePdf() });
  await expect(page.getByRole("combobox", { name: "PDF" })).toContainText(name);
}

test.use({ storageState: { cookies: [], origins: [] } });

test.describe("Deal from a document", () => {
  test("a document the user does not confirm is public is not read", async ({ page }) => {
    await signInAs(page, `extract-public-${run}`, "en-GB");
    await uploadDocument(page, "annual-report.pdf");
    await expect(page.locator("[data-public-warning]")).toContainText("free AI tier");
    await expect(confirm(page)).not.toBeChecked();
    await expect(readButton(page)).toBeDisabled();
    await confirm(page).check();
    await expect(readButton(page)).toBeEnabled();
    await confirm(page).uncheck();
    await expect(readButton(page)).toBeDisabled();
    // the API refuses a read sent without the confirmation, whatever the screen did
    const files = await (await page.request.get("/api/files", { headers: await asUser(page) })).json();
    const refused = await page.request.post("/api/documents/reads", {
      headers: await asUser(page),
      data: { file_id: files.files[0].id, public: false },
    });
    expect(refused.status()).toBe(422);
    expect((await refused.json()).detail).toBe("confirm_public");
    const jobs = await (await page.request.get("/api/jobs", { headers: await asUser(page) })).json();
    expect(jobs.jobs).toEqual([]);
  });

  test("values are read with their pages and quotes and only the accepted ones enter the deal", async ({ page }) => {
    await signInAs(page, `extract-read-${run}`, "en-GB");
    await page.goto("/deal/inputs");
    await expect(field(page, "EBITDA")).toHaveValue(/^100(\.0+)?$/);
    await stepLink(page, "From a document").click();
    await expect(page.getByRole("button", { name: "Upload a PDF" })).toBeEnabled();
    await page.locator('input[type="file"]').setInputFiles({ name: "mbb-2025.pdf", mimeType: "application/pdf", buffer: onePagePdf() });
    await expect(page.getByRole("combobox", { name: "PDF" })).toContainText("mbb-2025.pdf");
    await confirm(page).check();
    await readButton(page).click();
    await expect(page.locator("[data-company]")).toContainText("MBB", { timeout: 30_000 });
    await expect(kpi(page, "Accepted")).toHaveText("0");
    // every value with its page and the document's own (German) words
    const ebitda = page.locator('tr[data-item="figure:ebitda:2025-12-31"]');
    await expect(ebitda.locator("[data-value]")).toHaveText("217,835");
    await expect(ebitda.locator("[data-quote]")).toContainText("217.835");
    await expect(page.locator('tr[data-item="label:currency"] [data-value]')).toHaveText("EUR");
    await expect(page.locator('tr[data-item="label:unit"] [data-value]')).toHaveText("thousands");
    // nothing accepted: nothing can be used, and the deal is as it was
    await expect(page.getByRole("button", { name: "Accept values first" })).toBeDisabled();
    // EBITDA alone is not enough: its money needs the currency and the unit accepted too
    await accept(page, "figure:ebitda:2025-12-31").check();
    await page.getByRole("button", { name: "Use 1 accepted value in the deal" }).click();
    await expect(page.locator("[data-applied]")).toContainText("Nothing the deal takes was accepted.");
    await expect(page.locator('[data-left-out="ebitda"]')).toContainText("2 values it needs were not accepted");
    await accept(page, "label:currency").check();
    await accept(page, "label:unit").check();
    await page.getByRole("button", { name: "Use 3 accepted values in the deal" }).click();
    await expect(page.locator('[data-input="ebitda"]')).toContainText("217,835.00");
    await expect(page.locator('[data-input="currency"]')).toContainText("EUR");
    // the revenue was not accepted, so growth did not enter the deal
    await expect(page.locator('[data-input="growth"]')).toHaveCount(0);
    await stepLink(page, "Deal inputs").click();
    await expect(field(page, "EBITDA")).toHaveValue(/^217,?835(\.0+)?$/);
  });
});
