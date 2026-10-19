import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { type BrowserContext, expect, type Page, test } from "@playwright/test";

import { acceptedQuarter, asUser, onePagePdf, signInAs } from "./helpers";

/**
 * The Excel add-in (PLAN.md R3): its task pane is /excel, signed in through Office's dialog
 * (/excel/connect), and imports a covered company's accepted quarter as cells. Excel itself is the stand-in
 * in fake-office.js, served where Microsoft's Office.js would be: it opens the dialog in a window of its own,
 * as Excel on the web does, and records every cell the pane writes. The read is the recorded one
 * (playwright.config.ts: the replay provider answers any PDF as Avenue Supermarts' September 2026 filing).
 *
 * "Done when": signing in and importing a recorded quarter writes the same cells as R2's download
 * (tests/addin_cells_check.py compares them); the header scan passes on every other route.
 */
const run = Date.now().toString(36);
const root = join(__dirname, "..", "..");
const venvPython = process.platform === "win32" ? ".venv\\Scripts\\python.exe" : ".venv/bin/python";
const python = process.env.PYTHON ?? (existsSync(join(root, venvPython)) ? join(root, venvPython) : "python");
const FAKE_OFFICE = readFileSync(join(__dirname, "fake-office.js"), "utf-8");

/** Serve the stand-in wherever the pane or the dialog asks Microsoft's CDN for Office.js. */
async function fakeOffice(context: BrowserContext) {
  await context.route("https://appsforoffice.microsoft.com/**", (route) =>
    route.fulfill({ contentType: "application/javascript", body: FAKE_OFFICE }));
}

/** Click Sign in in the pane and wait for Office's dialog to hand it a pass. */
async function signInThroughDialog(page: Page, inDialog?: (dialog: Page) => Promise<void>) {
  const opened = page.waitForEvent("popup");
  await page.getByRole("button", { name: "Sign in" }).click();
  const dialog = await opened;
  await expect(dialog).toHaveURL(/\/excel\/connect\?platform=OfficeOnline$/);
  if (inDialog) await inDialog(dialog);
  await expect(page.locator('[data-pane-state="signed-in"]')).toBeVisible();
  await expect.poll(() => dialog.isClosed()).toBe(true);
}

function compare(info: { outputPath: (name: string) => string }, workbook: Buffer, written: unknown) {
  const workbookFile = info.outputPath("download.xlsx");
  writeFileSync(workbookFile, workbook);
  const writtenFile = info.outputPath("written.json");
  writeFileSync(writtenFile, JSON.stringify(written));
  try {
    return JSON.parse(execFileSync(python, ["-m", "tests.addin_cells_check", workbookFile, writtenFile], { cwd: root }).toString());
  } catch (error) {
    return JSON.parse((error as { stdout: Buffer }).stdout.toString());
  }
}

test.use({ storageState: { cookies: [], origins: [] } });

test.describe("Excel add-in", () => {
  test("signing in through Office's dialog and importing a quarter writes the download's own cells", async ({ page, context }, info) => {
    await fakeOffice(context);
    await signInAs(page, `excel-${run}`, "en-GB");
    // a name that looks like a formula must arrive as words, as it does in the download
    const period = await acceptedQuarter(page, "=DMart");

    await page.goto("/excel");
    await expect(page.locator('[data-pane-state="signed-out"]')).toBeVisible();
    await expect(page.getByText(/Google may refuse to sign in inside Excel/)).toBeVisible();
    // the site's own session is signed in in this browser, so the dialog hands over a pass at once
    await signInThroughDialog(page);
    await expect(page.getByText(`Signed in as excel-${run}`)).toBeVisible();

    await expect(page.getByRole("combobox", { name: "Covered company" }).first()).toContainText("=DMart");
    await expect(page.getByRole("combobox", { name: "Accepted quarter" })).toContainText("2026-09-30, 3 months, Consolidated");
    await page.getByRole("button", { name: "Import into workbook" }).click();
    await expect(page.locator("[data-written]")).toBeVisible();
    const written = await page.evaluate(() => (window as unknown as { __excelWritten: () => { sheets: { name: string; active: boolean }[] } }).__excelWritten());
    expect(written.sheets[0].name).toBe("About");
    expect(written.sheets.find((s) => s.active)?.name).toBe(written.sheets[1].name);

    // the same cells as the checked workbook's download: values, formulas, formats, notes, widths, frozen panes
    const download = await page.request.get(`/api/research/periods/${period}/workbook`, { headers: await asUser(page) });
    const checked = compare(info, await download.body(), written);
    expect(checked.differences).toEqual([]);
    expect(checked.cells).toBeGreaterThan(300);

    // a second import goes into new sheets beside the first, never over them
    await page.getByRole("button", { name: "Import into workbook" }).click();
    await expect(page.locator("[data-written]")).toContainText("About 2");

    // every call carried the add-in's pass, which reaches research only
    const pass = await page.evaluate(() => JSON.parse(window.localStorage.getItem("fse.excel.pass") ?? "{}").token as string);
    expect(pass).toMatch(/^addin:/);
    const deals = await page.request.get("/api/deals", { headers: { Authorization: `Bearer ${pass}` } });
    expect(deals.status()).toBe(403);

    // signing out ends the pass on the server too
    await page.getByRole("button", { name: "Sign out" }).click();
    await expect(page.locator('[data-pane-state="signed-out"]')).toBeVisible();
    const after = await page.request.get("/api/research/companies", { headers: { Authorization: `Bearer ${pass}` } });
    expect(after.status()).toBe(401);
  });

  test("signed out of the site, the dialog signs in itself; a PDF is read, reviewed and imported from the pane", async ({ page, context }, info) => {
    await fakeOffice(context);
    await signInAs(page, `excel-read-${run}`, "en-GB");
    await context.clearCookies();

    await page.goto("/excel");
    await signInThroughDialog(page, async (dialog) => {
      await expect(dialog.getByText("Sign in as you do on variater.com")).toBeVisible();
      await dialog.getByLabel("Development user").fill(`excel-read-${run}`);
      await dialog.getByRole("button", { name: "Sign in" }).click();
    });

    await page.getByLabel("New company").fill("DMart");
    await page.getByRole("button", { name: "Add company" }).click();
    await expect(page.getByRole("combobox", { name: "Covered company" }).last()).toContainText("DMart");
    await page.locator('input[type="file"]').setInputFiles({ name: "dmart-h1.pdf", mimeType: "application/pdf", buffer: onePagePdf() });
    await page.getByLabel("Quarter end (optional)").fill("2026-09-30");
    await page.getByRole("checkbox", { name: "This is a listed company's published results" }).check();
    await page.getByRole("button", { name: "Read results" }).click();

    // the amber figure is answered in the pane before anything is kept
    await expect(page.locator("[data-unanswered]")).toHaveAttribute("data-unanswered", "1", { timeout: 30_000 });
    const acceptButton = page.getByRole("button", { name: "Accept quarter and import" });
    await expect(acceptButton).toBeDisabled();
    const amber = page.locator("[data-amber]");
    const key = await amber.getAttribute("data-amber");
    await amber.getByLabel("Correct figure").fill("4321.5");
    await amber.getByRole("button", { name: "Use correction" }).click();
    await expect(amber).toHaveAttribute("data-answer", "corrected");
    await expect(page.locator("[data-unanswered]")).toHaveAttribute("data-unanswered", "0");
    await acceptButton.click();
    await expect(page.locator("[data-written]")).toBeVisible({ timeout: 30_000 });

    // stored with the correction, and written as the download holds it
    const pass = await page.evaluate(() => JSON.parse(window.localStorage.getItem("fse.excel.pass") ?? "{}").token as string);
    const auth = { Authorization: `Bearer ${pass}` };
    const listed = await (await page.request.get("/api/research/companies", { headers: auth })).json();
    const summary = listed.companies[0].periods[0];
    const stored = await (await page.request.get(`/api/research/periods/${summary.id}`, { headers: auth })).json();
    const [t, r, c] = (key ?? "").split(":").map(Number);
    expect(stored.figures.tables[t].rows[r].values[c]).toBe(4321.5);
    const written = await page.evaluate(() => (window as unknown as { __excelWritten: () => unknown }).__excelWritten());
    const download = await page.request.get(`/api/research/periods/${summary.id}/workbook`, { headers: auth });
    expect(compare(info, await download.body(), written).differences).toEqual([]);
  });

  test("Load model writes a sector model as the download's own cells, its assumptions named", async ({ page, context }, info) => {
    await fakeOffice(context);
    await signInAs(page, `excel-model-${run}`, "en-GB");
    await page.goto("/excel");
    await signInThroughDialog(page);
    await page.getByRole("combobox", { name: "Model for" }).selectOption("example:lemon-tree-hotels");
    await page.getByRole("button", { name: "Load model" }).click();
    await expect(page.locator("[data-model-written]")).toBeVisible();
    const written = await page.evaluate(() => (window as unknown as { __excelWritten: () => { sheets: { name: string; active: boolean; names: Record<string, string> }[] } }).__excelWritten());
    expect(written.sheets.map((s) => s.name)).toEqual(["Model", "Notes"]);
    expect(written.sheets[0].active).toBe(true);
    expect(Object.keys(written.sheets[0].names).length).toBe(105);       // with the valuation's five (PLAN.md R6)

    const headers = await asUser(page);
    const catalogue = await (await page.request.get("/api/research/models", { headers })).json();
    const example = catalogue.examples.find((e: { id: string }) => e.id === "lemon-tree-hotels");
    const download = await page.request.post("/api/research/models/workbook", {
      headers, data: { structure: example.structure, company: example.company },
    });
    const checked = compare(info, await download.body(), written);
    expect(checked.differences).toEqual([]);
    expect(checked.cells).toBeGreaterThan(700);
  });

  test("Save to Variater reads typed assumptions back, again after a second change, and refuses a restructured sheet", async ({ page, context }) => {
    await fakeOffice(context);
    await signInAs(page, `excel-save-${run}`, "en-GB");
    await acceptedQuarter(page, "DMart");
    await page.goto("/excel");
    await signInThroughDialog(page);
    await page.getByRole("button", { name: "Load model" }).click();
    await expect(page.locator("[data-model-written]")).toBeVisible();

    type Written = { sheets: { name: string; active: boolean; names: Record<string, string>; cells: Record<string, { value?: unknown }> }[] };
    const written = await page.evaluate(() => (window as unknown as { __excelWritten: () => Written }).__excelWritten());
    const sheet = written.sheets.find((s) => s.active)!;
    const headers = await asUser(page);
    const companies = await (await page.request.get("/api/research/companies", { headers })).json();
    const companyId: string = companies.companies[0].id;
    expect(sheet.cells.B1.value).toBe(`Saved in Variater: ${companyId}`);   // the sheet says whose saved model it is
    const at = (name: string) => sheet.names[name].replace(/\$/g, "");
    const saved = async () => (await (await page.request.get(`/api/research/companies/${companyId}/model`, { headers })).json()).company;
    const type = (address: string, value: number) => page.evaluate(([n, a, v]) =>
      (window as unknown as { __excelType: (s: string, a: string, v: number) => void }).__excelType(n as string, a as string, v as number),
    [sheet.name, address, value]);

    await type(at("mat_sh_y1"), 0.31);
    await page.getByRole("button", { name: "Save to Variater" }).click();
    await expect(page.locator('[data-saved-outcome="status"]')).toHaveText(/Saved: 1 cell changed/);
    expect((await saved()).values.mat_sh.y1).toBe(0.31);

    // changed again in Excel and saved again: the changed assumptions come back
    await type(at("mat_sh_y1"), 0.35);
    await type(at("unit_rupees"), 1000000);
    await page.getByRole("button", { name: "Save to Variater" }).click();
    await expect(page.locator('[data-saved-outcome="status"]')).toHaveText(/Saved: 2 cells changed/);
    const after = await saved();
    expect(after.values.mat_sh.y1).toBe(0.35);
    expect(after.constants.unit_rupees).toBe(1000000);

    // a row inserted: refused, saying what changed, and nothing saved
    await type(at("mat_sh_y2"), 0.5);
    await page.evaluate((n) => (window as unknown as { __excelInsertRow: (s: string, r: number) => void }).__excelInsertRow(n, 20), sheet.name);
    await page.getByRole("button", { name: "Save to Variater" }).click();
    await expect(page.locator('[data-saved-outcome="alert"]')).toContainText("structure was changed");
    await expect(page.locator('[data-saved-outcome="alert"]')).toContainText("row 20");
    expect((await saved()).values.mat_sh.y2).toBe(after.values.mat_sh.y2);
  });

  test("outside Office the pane says where to open it", async ({ page }) => {
    await page.route("https://appsforoffice.microsoft.com/**", (route) => route.fulfill({ contentType: "application/javascript", body: "" }));
    await page.goto("/excel");
    await expect(page.locator('[data-pane-state="outside"]')).toBeVisible();
  });

  test("only the add-in's pages may be framed by Office, and only they load Office.js", async ({ page }) => {
    const pane = await page.request.get("/excel");
    const csp = pane.headers()["content-security-policy"];
    const directive = (policy: string, name: string) => policy.split(";").map((d) => d.trim()).find((d) => d.startsWith(`${name} `)) ?? "";
    expect(directive(csp, "frame-ancestors")).toBe("frame-ancestors https://*.officeapps.live.com https://*.office.com https://*.office365.com https://*.cloud.microsoft https://*.microsoft365.com https://*.sharepoint.com https://onedrive.live.com");
    expect(directive(csp, "script-src")).toContain("https://appsforoffice.microsoft.com");
    expect(directive(csp, "script-src")).toContain("'strict-dynamic'");
    expect(pane.headers()["x-frame-options"]).toBeUndefined();
    expect(pane.headers()["cross-origin-opener-policy"]).toBe("unsafe-none");
    // the dialog is a window of its own: it may load Office.js and keep its opener, and is never framed
    const dialog = await page.request.get("/excel/connect");
    expect(directive(dialog.headers()["content-security-policy"], "frame-ancestors")).toBe("frame-ancestors 'none'");
    expect(directive(dialog.headers()["content-security-policy"], "script-src")).toContain("https://appsforoffice.microsoft.com");
    expect(dialog.headers()["x-frame-options"]).toBe("DENY");
    expect(dialog.headers()["cross-origin-opener-policy"]).toBe("unsafe-none");

    for (const path of ["/sign-in", "/privacy", "/start", "/coverage/results", "/excelsior"]) {
      const other = await page.request.get(path, { maxRedirects: 0 });
      const policy = other.headers()["content-security-policy"] ?? "";
      if (policy) {
        expect(directive(policy, "frame-ancestors"), path).toBe("frame-ancestors 'none'");
        expect(directive(policy, "script-src"), path).not.toContain("appsforoffice");
      }
      expect(other.headers()["x-frame-options"], path).toBe("DENY");
      expect(other.headers()["cross-origin-opener-policy"], path).toBe("same-origin-allow-popups");
    }

    // the manifests a tester sideloads are served beside the pane
    const manifest = await page.request.get("/excel/manifest.xml");
    expect(manifest.status()).toBe(200);
    expect(await manifest.text()).toContain("<SourceLocation DefaultValue=\"https://variater.com/excel\"/>");
  });
});
