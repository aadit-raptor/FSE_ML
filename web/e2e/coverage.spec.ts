import { execFileSync } from "node:child_process";
import { existsSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { expect, type Page, test } from "@playwright/test";

import { asUser, kpi, onePagePdf, signInAs } from "./helpers";

/**
 * Coverage (PLAN.md R2): a results PDF read (recorded answers: the replay provider answers any file as Avenue
 * Supermarts' September 2026 half-year filing, playwright.config.ts), reviewed with its page beside the
 * figures, the amber figure corrected, the quarter accepted into a company, and the checked workbook
 * downloaded. No AI is called.
 *
 * "Done when": a results PDF becomes an accepted quarter and a workbook that recalculates to the same totals;
 * a corrected figure is the one stored and downloaded.
 */
const run = Date.now().toString(36);
const root = join(__dirname, "..", "..");
const venvPython = process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python";
const python = process.env.PYTHON ?? (existsSync(join(root, venvPython)) ? join(root, venvPython) : "python");


async function readFiling(page: Page, company: string) {
  await page.goto("/coverage/results");
  await page.getByLabel("New company").fill(company);
  await page.getByRole("button", { name: "Add company" }).click();
  await expect(page.getByRole("combobox", { name: "Covered company" })).toContainText(company);
  await expect(page.getByRole("button", { name: "Upload a PDF" })).toBeEnabled();
  await page.locator('input[type="file"]').setInputFiles({ name: "dmart-h1.pdf", mimeType: "application/pdf", buffer: onePagePdf() });
  await expect(page.getByRole("combobox", { name: "Uploaded PDF" })).toContainText("dmart-h1.pdf");
  await page.getByLabel("Quarter end (optional)").fill("2026-09-30");
  await page.getByRole("checkbox", { name: "This is a listed company's published results" }).check();
  await page.getByRole("button", { name: "Read results" }).click();
  await expect(kpi(page, "Figures")).toHaveText("332", { timeout: 30_000 });
}

test.use({ storageState: { cookies: [], origins: [] } });

test.describe("Coverage", () => {
  test("a read is reviewed beside its page, corrected, accepted and downloaded as a workbook that recalculates", async ({ page }, info) => {
    await signInAs(page, `coverage-${run}`, "en-GB");
    await readFiling(page, "DMart");

    // nothing is kept yet, and the quarter can't be accepted while the amber figure is unanswered
    const acceptButton = page.getByRole("button", { name: "Accept quarter into DMart" });
    await expect(acceptButton).toBeDisabled();
    await expect(page.locator("[data-unanswered]")).toHaveAttribute("data-unanswered", "1");
    const before = await (await page.request.get("/api/research/companies", { headers: await asUser(page) })).json();
    expect(before.companies[0].periods).toEqual([]);

    // the page the figures came from, drawn in the browser from the stored PDF
    await expect(page.locator("[data-page-number]")).toHaveText("Page 1 of 1");
    await expect(page.locator("[data-page-view] canvas")).toHaveAttribute("data-drawn", "1");

    // the amber figure is selected first: correct it
    const amber = page.locator("td[data-amber]");
    await expect(amber).toHaveCount(1);
    const key = await amber.getAttribute("data-cell");
    await expect(page.locator("[data-selected]")).toHaveAttribute("data-selected", key ?? "");
    await page.getByLabel("Correct figure").fill("4321.5");
    await page.getByRole("button", { name: "Use correction" }).click();
    await expect(amber).toHaveAttribute("data-answer", "corrected");
    await expect(amber).toHaveText("4,321.50");
    await expect(page.locator("[data-unanswered]")).toHaveAttribute("data-unanswered", "0");

    await acceptButton.click();
    await expect(page.locator("[data-accepted]")).toHaveAttribute("data-accepted", "1");

    // the corrected figure is the one stored ...
    const listed = await (await page.request.get("/api/research/companies", { headers: await asUser(page) })).json();
    const summary = listed.companies[0].periods[0];
    expect([summary.period_end, summary.months, summary.scope, summary.unit]).toEqual(["2026-09-30", 3, "consolidated", "crores"]);
    const stored = await (await page.request.get(`/api/research/periods/${summary.id}`, { headers: await asUser(page) })).json();
    const [t, r, c] = (key ?? "").split(":").map(Number);
    expect(stored.figures.tables[t].rows[r].values[c]).toBe(4321.5);
    expect(stored.counts).toEqual({ figures: 332, corrected: 1, accepted: 0 });

    // ... and downloaded: the workbook's formulas give every stored figure, the subtotals recalculated
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Workbook, Consolidated" }).click(),
    ]);
    expect(download.suggestedFilename()).toBe("DMart_2026-09-30_consolidated.xlsx");
    const workbook = info.outputPath("checked.xlsx");
    await download.saveAs(workbook);
    const periodFile = info.outputPath("period.json");
    writeFileSync(periodFile, JSON.stringify(stored));
    const checked = JSON.parse(execFileSync(python, ["-m", "tests.coverage_workbook_check", workbook, periodFile], { cwd: root }).toString());
    expect(checked.figures).toBe(332 + (await zeroDashes(stored)));
    expect(checked.formulas).toBeGreaterThan(40);

    // Coverage -> Companies keeps it, and shows it again with its page
    await page.goto("/coverage/companies");
    const row = page.locator('[data-company-tile="DMart"] tr[data-period="2026-09-30|consolidated"]');
    await expect(row).toContainText(/3 months to 30 Sept? 2026/);
    await row.getByRole("button", { name: "Show" }).click();
    await expect(page.locator(`[data-stored-period="${summary.id}"] td[data-mark="corrected"]`)).toHaveText("4,321.50");
    await expect(page.locator("[data-page-view] canvas")).toHaveAttribute("data-drawn", "1");
  });

  test("another account sees none of it", async ({ page }) => {
    await signInAs(page, `coverage-owner-${run}`, "en-GB");
    await readFiling(page, "Owned Co");
    await page.getByRole("button", { name: "Accept as read" }).click();
    await page.getByRole("button", { name: "Accept quarter into Owned Co" }).click();
    await expect(page.locator("[data-accepted]")).toHaveAttribute("data-accepted", "1");
    const owned = await (await page.request.get("/api/research/companies", { headers: await asUser(page) })).json();
    const period = owned.companies[0].periods[0];

    await signInAs(page, `coverage-other-${run}`, "en-GB");
    await page.goto("/coverage/companies");
    await expect(page.getByText("No companies yet")).toBeVisible();
    for (const path of [`/api/research/periods/${period.id}`, `/api/research/periods/${period.id}/workbook`, `/api/files/${period.file_id}/content`]) {
      expect((await page.request.get(path, { headers: await asUser(page) })).status(), path).toBe(404);
    }
  });
});

/** Figures the workbook holds that the reader does not count as figures: printed zeros (a dash read as 0). */
async function zeroDashes(stored: { figures: { tables: { rows: { values: (number | null)[] }[] }[] } }): Promise<number> {
  return stored.figures.tables.reduce((n, table) => n + table.rows.reduce((m, row) => m + row.values.filter((v) => v === 0).length, 0), 0);
}
