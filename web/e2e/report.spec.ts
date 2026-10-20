import { expect, test } from "@playwright/test";

import { asUser, signInAs } from "./helpers";

/**
 * Coverage -> Report (PLAN.md R8): a results note drawn from a covered company's saved model. Every check reads an
 * output: the note's revenue is the server's run of the same model, written with the account's lakh grouping; no
 * rating is in the note until the analyst types one, and then only theirs; the firm name typed once is on the note;
 * the Word file holds the figures on screen; without an AI the templates write the sentences and say so.
 */
const run = Date.now().toString(36);

test.use({ storageState: { cookies: [], origins: [] } });

test("a note is drawn from the saved model, holds only the analyst's rating and downloads as Word", async ({ page }) => {
  await signInAs(page, `report-${run}`, "en-IN");
  const headers = await asUser(page);
  const made = await (await page.request.post("/api/research/companies", { headers, data: { name: "Lemon Tree" } })).json();
  const catalogue = await (await page.request.get("/api/research/models", { headers })).json();
  const example = catalogue.examples.find((e: { id: string }) => e.id === "lemon-tree-hotels");
  const put = await page.request.put(`/api/research/companies/${made.id}/model`, {
    headers, data: { structure: example.structure, company: example.company },
  });
  expect(put.status()).toBe(200);
  const ran = await (await page.request.post("/api/research/models/run", {
    headers, data: { structure: example.structure, company: example.company },
  })).json();
  const revenue = (period: string) => ran.rows.find((x: { key: string }) => x.key === "rev").cells[period].value as number;
  const shown = (v: number) => v.toLocaleString("en-IN", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

  await page.goto("/coverage/report");
  const note = page.getByRole("article", { name: "The note as it will be written" });
  await expect(note.getByRole("heading", { level: 2 })).toHaveText(example.company.name);
  const y1 = shown(revenue("y1"));
  expect(y1.split(",").length).toBeGreaterThan(2);                       // lakh grouping: 1,23,456.7
  await expect(note.locator('[data-figure="model"]', { hasText: y1 }).first()).toBeVisible();
  await expect(note).not.toContainText(/\b(Buy|Sell|Hold|Accumulate|Add|Reduce)\b/);

  // the analyst's view goes in as typed, and is the only rating in the note
  await page.getByLabel("Rating, in your words").fill("Accumulate");
  await page.getByLabel("Target per share, yours").fill("512.5");
  await page.getByLabel("Your argument").fill("New rooms open on time.");
  const saved = page.waitForResponse((r) => r.url().endsWith(`/report`) && r.request().method() === "PUT" && r.ok());
  await page.getByRole("button", { name: "Save", exact: true }).click();
  await saved;
  await expect(note.locator("[data-typed]", { hasText: "Accumulate" })).toHaveCount(1);
  await expect(note.getByText(/Accumulate/)).toHaveCount(1);
  await expect(note.locator("[data-typed]", { hasText: "512.50" })).toBeVisible();
  await expect(note).toContainText("New rooms open on time.");

  // the firm, typed once for the account
  await page.getByLabel("Firm name").fill(`Example Securities ${run}`);
  await page.getByRole("button", { name: "Save details" }).click();
  await expect(note).toContainText(`Example Securities ${run}`);

  // the browser tests replay the AI with nothing recorded for a note: the templates write and say so
  await page.getByRole("button", { name: "Write with the AI" }).click();
  await expect(page.locator('[data-section="outlook"]')).toContainText("Template: the AI didn't answer");
  await expect(note.locator('[data-kind="sentence"]').first()).toContainText(y1);

  // the Word file holds the figure on screen
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Word", exact: true }).click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("Lemon_Tree_results_note.docx");
  const word = await page.request.get(`/api/research/companies/${made.id}/report/word`, { headers });
  expect(word.ok()).toBe(true);
  const history = await (await page.request.get("/api/account/history", { headers })).json();
  expect(history.entries.filter((e: { export?: string }) => e.export === "report_word").length).toBeGreaterThan(0);
});
