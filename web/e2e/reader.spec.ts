import { expect, type Page, test } from "@playwright/test";

import { asUser, kpi, signInAs } from "./helpers";

/**
 * The results reader (PLAN.md R1), through the real screen, API, job runner and file store, with
 * the AI replaced by recorded answers (playwright.config.ts: AI_PROVIDER=replay answers any file as
 * Avenue Supermarts' September 2026 half-year filing, read twice in R0). No AI is called.
 *
 * "Done when": the confirmation blocks a file the user does not confirm is public.
 */
const run = Date.now().toString(36);

/** A well-formed one-page PDF: the upload's scan passes it; the replayed answers stand for its reads. */
function pdf(): Buffer {
  const bodies = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>"];
  let out = "%PDF-1.4\n";
  const offsets: number[] = [];
  bodies.forEach((body, i) => {
    offsets.push(out.length);
    out += `${i + 1} 0 obj\n${body}\nendobj\n`;
  });
  const table = out.length;
  out += `xref\n0 ${bodies.length + 1}\n0000000000 65535 f \n`;
  out += offsets.map((o) => `${String(o).padStart(10, "0")} 00000 n \n`).join("");
  out += `trailer\n<< /Size ${bodies.length + 1} /Root 1 0 R >>\nstartxref\n${table}\n%%EOF\n`;
  return Buffer.from(out, "latin1");
}

const readButton = (page: Page) => page.getByRole("button", { name: "Read results" });
const confirm = (page: Page) => page.getByRole("checkbox", { name: "This is a listed company's published results" });

async function uploadResults(page: Page, name: string) {
  await page.goto("/coverage/results");
  // a read is for a covered company (PLAN.md R2)
  await page.getByLabel("New company").fill("Results Co");
  await page.getByRole("button", { name: "Add company" }).click();
  await expect(page.getByRole("combobox", { name: "Covered company" })).toContainText("Results Co");
  await expect(page.getByRole("button", { name: "Upload a PDF" })).toBeEnabled();
  await page.locator('input[type="file"]').setInputFiles({ name, mimeType: "application/pdf", buffer: pdf() });
  await expect(page.getByRole("combobox", { name: "Uploaded PDF" })).toContainText(name);
}

test.use({ storageState: { cookies: [], origins: [] } });

test.describe("Results reader", () => {
  test("a file the user does not confirm is public is not read", async ({ page }) => {
    await signInAs(page, `reader-public-${run}`, "en-GB");
    await uploadResults(page, "results-q2.pdf");
    await expect(page.locator("[data-public-warning]")).toContainText("free AI tier");
    // The screen keeps the button off until the box is ticked ...
    await expect(confirm(page)).not.toBeChecked();
    await expect(readButton(page)).toBeDisabled();
    await confirm(page).check();
    await expect(readButton(page)).toBeEnabled();
    await confirm(page).uncheck();
    await expect(readButton(page)).toBeDisabled();
    // ... and the API refuses a read sent without the confirmation, whatever the screen did
    const files = await (await page.request.get("/api/files", { headers: await asUser(page) })).json();
    const refused = await page.request.post("/api/research/reads", {
      headers: await asUser(page),
      data: { file_id: files.files[0].id, public: false },
    });
    expect(refused.status()).toBe(422);
    expect((await refused.json()).detail).toBe("confirm_public");
    const jobs = await (await page.request.get("/api/jobs", { headers: await asUser(page) })).json();
    expect(jobs.jobs).toEqual([]);
  });

  test("a confirmed public filing is read twice and shown with its amber figure", async ({ page }) => {
    await signInAs(page, `reader-read-${run}`, "en-GB");
    await uploadResults(page, "dmart-h1.pdf");
    const pagesLeft = page.locator("[data-pages-left]");
    await expect(pagesLeft).toHaveText(/^[\d,]+ of [\d,]+ pages left today$/);
    const left = async () => Number(((await pagesLeft.textContent()) ?? "").split(" of ")[0].replace(/,/g, ""));
    const before = await left();
    await page.getByLabel("Quarter end (optional)").fill("2026-09-30");
    await confirm(page).check();
    await readButton(page).click();
    await expect(kpi(page, "Figures")).toHaveText("332", { timeout: 30_000 });
    await expect(kpi(page, "Amber")).toHaveText("1");
    await expect(page.locator("[data-company]")).toHaveText("Avenue Supermarts Limited");
    // the three statements, revenue as printed, the one figure the reads disagree on amber with its reason
    await expect(page.locator("table[data-statement]")).toHaveCount(3);
    const revenue = page.locator('table[data-statement="results"] tr[data-line="revenue_from_operations"] td').first();
    await expect(revenue).toHaveText("19,644.01");
    const amber = page.locator("td[data-amber]");
    await expect(amber).toHaveCount(1);
    await expect(amber).toHaveAttribute("data-amber", "reads_differ");
    // selected first, its reason beside it
    await expect(page.locator("[data-selected]")).toContainText("The two reads give this figure differently");
    // the filing's one page is counted against the day's pages
    await expect.poll(left).toBe(before - 1);
  });
});
