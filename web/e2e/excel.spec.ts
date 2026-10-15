import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { type BrowserContext, expect, type Page, test } from "@playwright/test";

import { asUser, signInAs } from "./helpers";

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

/** A well-formed one-page PDF: the upload's scan passes it. */
function pdf(): Buffer {
  const content = "0 0 1 rg 72 600 200 100 re f";
  const bodies = ["<< /Type /Catalog /Pages 2 0 R >>", "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>",
    `<< /Length ${content.length} >>\nstream\n${content}\nendstream`];
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

/** An accepted quarter for `company`, made through the API as the Coverage screens make one. */
async function acceptedQuarter(page: Page, company: string): Promise<string> {
  const headers = await asUser(page);
  const made = await (await page.request.post("/api/research/companies", { headers, data: { name: company } })).json();
  const bytes = pdf();
  const begun = await (await page.request.post("/api/files", { headers, data: { name: "dmart-h1.pdf", size: bytes.length } })).json();
  const sent = await page.request.put(`/api/files/${begun.id}/content`, {
    headers: { ...headers, "Content-Type": "application/octet-stream" }, data: bytes,
  });
  expect(sent.ok()).toBe(true);
  await expect.poll(async () => (await (await page.request.get(`/api/files/${begun.id}`, { headers })).json()).status,
    { timeout: 30_000 }).toBe("clean");
  const started = await (await page.request.post("/api/research/reads", {
    headers, data: { file_id: begun.id, public: true, quarter_end: "2026-09-30" },
  })).json();
  let done: { status: string; result: { tables: { rows: { cells: { amber: string | null; value: number | null }[] }[] }[] } } | null = null;
  await expect.poll(async () => {
    done = await (await page.request.get(`/api/research/reads/${started.id}`, { headers })).json();
    return done!.status;
  }, { timeout: 30_000 }).toBe("succeeded");
  const amber: number[][] = [];
  done!.result.tables.forEach((t, ti) => t.rows.forEach((r, ri) => r.cells.forEach((c, ci) => {
    if (c.amber && c.value !== null) amber.push([ti, ri, ci]);
  })));
  const accepted = await page.request.post(`/api/research/companies/${made.id}/periods`, {
    headers, data: { read_id: started.id, accepted: amber, corrected: [] },
  });
  expect(accepted.status()).toBe(201);
  return (await accepted.json()).periods[0].id;
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
    const period = await acceptedQuarter(page, "DMart");

    await page.goto("/excel");
    await expect(page.locator('[data-pane-state="signed-out"]')).toBeVisible();
    await expect(page.getByText(/Google may refuse to sign in inside Excel/)).toBeVisible();
    // the site's own session is signed in in this browser, so the dialog hands over a pass at once
    await signInThroughDialog(page);
    await expect(page.getByText(`Signed in as excel-${run}`)).toBeVisible();

    await expect(page.getByRole("combobox", { name: "Covered company" }).first()).toContainText("DMart");
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
    await page.locator('input[type="file"]').setInputFiles({ name: "dmart-h1.pdf", mimeType: "application/pdf", buffer: pdf() });
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

  test("outside Office the pane says where to open it", async ({ page }) => {
    await page.route("https://appsforoffice.microsoft.com/**", (route) => route.fulfill({ contentType: "application/javascript", body: "" }));
    await page.goto("/excel");
    await expect(page.locator('[data-pane-state="outside"]')).toBeVisible();
  });

  test("only the add-in's pages may be framed by Office, and only they load Office.js", async ({ page }) => {
    const pane = await page.request.get("/excel");
    const csp = pane.headers()["content-security-policy"];
    const directive = (policy: string, name: string) => policy.split(";").map((d) => d.trim()).find((d) => d.startsWith(`${name} `)) ?? "";
    expect(directive(csp, "frame-ancestors")).toBe("frame-ancestors https://*.officeapps.live.com https://*.office.com https://*.office365.com https://*.cloud.microsoft https://*.microsoft365.com https://*.sharepoint.com https://*.live.com");
    expect(directive(csp, "script-src")).toContain("https://appsforoffice.microsoft.com");
    expect(directive(csp, "script-src")).toContain("'strict-dynamic'");
    expect(pane.headers()["x-frame-options"]).toBeUndefined();
    expect(pane.headers()["cross-origin-opener-policy"]).toBe("unsafe-none");
    const dialog = await page.request.get("/excel/connect");
    expect(directive(dialog.headers()["content-security-policy"], "frame-ancestors")).toContain("https://*.officeapps.live.com");
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
