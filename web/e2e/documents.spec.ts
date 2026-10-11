import { readFileSync } from "node:fs";

import { expect, type Page, test } from "@playwright/test";

import { asUser, kpi, signInAs } from "./helpers";

/**
 * File uploads (PLAN.md 6.1), through the real screen, API, job runner (the scan runs in its own
 * process, as it does when deployed), database and a real (directory) file store: a document goes up, is checked, comes back byte for byte and can be
 * deleted; a file that can run something, an oversized one and a wrong kind are refused; another
 * account sees nothing; and the screen warns before the space runs out. Each test has an account
 * of its own, so its counts start from nothing.
 */
const run = Date.now().toString(36);

const PAGE_OBJECTS = [
  "<< /Type /Catalog /Pages 2 0 R >>",
  "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
  "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
];
/** A well-formed one-page PDF with `extra` as a further object, padded to about `size` bytes when given. */
function pdf(extra = "", size = 0): Buffer {
  const bodies = [...PAGE_OBJECTS, ...(extra ? [extra] : [])];
  if (size) {
    const filler = "x".repeat(Math.max(0, size - 600));
    bodies.push(`<< /Length ${filler.length} >>\nstream\n${filler}\nendstream`);
  }
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
const SCRIPTED = "<< /S /JavaScript /JS (app.alert\\(1\\)) >>";

const row = (page: Page, name: string) => page.locator(`tr[data-file="${name}"]`);
const picker = (page: Page) => page.locator('input[type="file"]');

async function open(page: Page) {
  await page.goto("/deal/documents");
  await expect(page.getByRole("button", { name: "Choose files" })).toBeEnabled();
}

async function choose(page: Page, name: string, buffer: Buffer) {
  await picker(page).setInputFiles({ name, mimeType: "application/octet-stream", buffer });
}

// Each test signs in as its own user; the shared signed-in state is not used
test.use({ storageState: { cookies: [], origins: [] } });

test.describe("Documents", () => {
  test("a document is uploaded, checked, downloaded byte for byte and deleted", async ({ page }) => {
    await signInAs(page, `docs-${run}`, "en-GB");
    await open(page);
    await expect(page.getByText("No documents yet. Choose files on the left to upload one.")).toBeVisible();
    await expect(kpi(page, "Files")).toHaveText("0");
    await expect(kpi(page, "Largest file")).toHaveText("20 MB");

    // Bigger than any other request may be (1 MB): a file's bytes are streamed, not read into memory
    const report = pdf("", 1_600_000);
    await choose(page, "Annual report 2025.pdf", report);
    const line = row(page, "Annual report 2025.pdf");
    await expect(line).toHaveAttribute("data-status", "clean");
    await expect(line).toContainText("Passed");
    await expect(line).toContainText("1.5 MB");
    await expect(kpi(page, "Files")).toHaveText("1");
    await expect(kpi(page, "Space used")).toHaveText("1.5 MB");
    await expect(page.locator("[data-usage]")).toHaveText("1.5 MB of 100 MB");

    const download = page.waitForEvent("download");
    await line.getByRole("button", { name: "Download" }).click();
    const saved = await download;
    expect(saved.suggestedFilename()).toBe("Annual report 2025.pdf");
    expect(readFileSync(await saved.path()).equals(report)).toBe(true);

    // It is still there after a reload, and gone for good once deleted
    await page.reload();
    await expect(line).toContainText("Passed");
    await line.getByRole("button", { name: "Delete" }).click();
    await expect(line).toContainText("Delete for good?");
    await line.getByRole("button", { name: "Delete" }).click();
    await expect(line).toHaveCount(0);
    await expect(kpi(page, "Files")).toHaveText("0");
    await page.reload();
    await expect(page.getByText("No documents yet.")).toBeVisible();
  });

  test("a file that can run something, an oversized file and a wrong kind are refused", async ({ page }) => {
    await signInAs(page, `docs-bad-${run}`, "en-GB");
    await open(page);

    // 1. A PDF with a script: sent, checked, refused, with the reason, and never downloadable
    await choose(page, "Scripted.pdf", pdf(SCRIPTED));
    const scripted = row(page, "Scripted.pdf");
    await expect(scripted).toHaveAttribute("data-status", "refused");
    await expect(scripted).toContainText("Refused: it contains something that can run or fetch from elsewhere");
    await expect(scripted.getByRole("button", { name: "Download" })).toHaveCount(0);
    await expect(kpi(page, "Files")).toHaveText("0"); // its bytes are deleted, so it takes no space

    // 2. A program named .pdf: its contents are not what its name says
    await choose(page, "Tool.pdf", Buffer.concat([Buffer.from("MZ\x90\x00", "latin1"), Buffer.alloc(4000)]));
    await expect(row(page, "Tool.pdf")).toContainText("Refused: its contents are not the kind of file its name says.");

    // 3. Over 20 MB: refused before anything is sent
    const asked: string[] = [];
    page.on("request", (r) => {
      if (r.method() !== "GET" && r.url().includes("/api/files")) asked.push(r.url());
    });
    await choose(page, "Huge.pdf", pdf("", 21 * 1024 * 1024));
    await expect(page.locator('[data-upload-problem="too_large"]')).toHaveText("Huge.pdf: the file is larger than 20 MB.");
    await expect(row(page, "Huge.pdf")).toHaveCount(0);

    // 4. A kind that is not accepted, and the server says the same when asked directly
    await choose(page, "Model.xlsm", Buffer.from("PK\x03\x04"));
    await expect(page.locator('[data-upload-problem="bad_type"]')).toContainText("Model.xlsm: only PDF, Excel");
    expect(asked).toEqual([]);
    const direct = await page.request.post("/api/files", { data: { name: "Huge.pdf", size: 21 * 1024 * 1024 }, headers: await asUser(page) });
    expect(direct.status()).toBe(413);
    expect((await direct.json()).detail.code).toBe("too_large");
  });

  test("another account sees none of it and cannot fetch it", async ({ page }) => {
    await signInAs(page, `docs-owner-${run}`, "en-GB");
    await open(page);
    await choose(page, "Private memo.pdf", pdf());
    await expect(row(page, "Private memo.pdf")).toHaveAttribute("data-status", "clean");
    const listed = await (await page.request.get("/api/files", { headers: await asUser(page) })).json();
    const id = listed.files[0].id as string;

    await page.context().clearCookies();
    await signInAs(page, `docs-other-${run}`, "en-GB");
    await open(page);
    await expect(page.getByText("No documents yet.")).toBeVisible();
    await expect(row(page, "Private memo.pdf")).toHaveCount(0);
    const headers = await asUser(page);
    expect((await page.request.get(`/api/files/${id}`, { headers })).status()).toBe(404);
    expect((await page.request.post(`/api/files/${id}/link`, { headers })).status()).toBe(404);
    expect((await page.request.delete(`/api/files/${id}`, { headers })).status()).toBe(404);
  });

  test("the screen warns near the limit and stops at it", async ({ page }) => {
    test.setTimeout(240_000);
    await signInAs(page, `docs-quota-${run}`, "en-GB");
    const headers = await asUser(page);
    // 31 small documents, sent as the screen sends them and each checked before the next
    const store = async (n: number) => {
      const begun = await page.request.post("/api/files", { data: { name: `Report ${n}.pdf`, size: pdf().length }, headers });
      expect(begun.status()).toBe(201);
      const { id } = await begun.json();
      const sent = await page.request.put(`/api/files/${id}/content`, { data: pdf(), headers: { ...headers, "Content-Type": "application/octet-stream" } });
      expect(sent.status()).toBe(200);
      await expect.poll(async () => (await (await page.request.get(`/api/files/${id}`, { headers })).json()).status, { timeout: 30_000 }).toBe("clean");
    };
    for (let n = 1; n <= 31; n++) await store(n);
    await open(page);
    await expect(page.locator("[data-quota-message]")).toHaveCount(0);

    await choose(page, "Thirty-second.pdf", pdf());
    await expect(row(page, "Thirty-second.pdf")).toHaveAttribute("data-status", "clean");
    const warning = page.getByRole("status").filter({ hasText: "Nearly full" });
    await expect(warning).toContainText("32 of 40 files. Delete documents you no longer need to make room.");
    await expect(kpi(page, "Files")).toHaveText("32");

    for (let n = 33; n <= 40; n++) await store(n);
    await page.reload();
    await expect(page.getByRole("status").filter({ hasText: "Space full" })).toContainText("40 of 40 files");
    await expect(page.getByRole("button", { name: "Choose files" })).toBeDisabled();
    const refused = await page.request.post("/api/files", { data: { name: "One more.pdf", size: 1000 }, headers });
    expect(refused.status()).toBe(409);
    expect((await refused.json()).detail.code).toBe("quota_files");
  });

  test("the bytes go to the API's own address, and the page's policy allows only that", async ({ page }) => {
    await signInAs(page, `docs-direct-${run}`, "en-GB");
    await open(page);
    const sent = page.waitForRequest((r) => r.method() === "PUT" && /\/api\/files\/[0-9a-f-]{36}\/content$/.test(r.url()));
    await choose(page, "Direct.pdf", pdf());
    const request = await sent;
    // Past this site's proxy: a different origin, reached with the session token and CORS
    expect(new URL(request.url()).origin).not.toBe(new URL(page.url()).origin);
    expect(request.headers()["authorization"]).toMatch(/^Bearer /);
    await expect(row(page, "Direct.pdf")).toHaveAttribute("data-status", "clean");
    const policy = (await page.request.get("/deal/documents")).headers()["content-security-policy"] ?? "";
    const connect = policy.split(";").find((d: string) => d.trim().startsWith("connect-src")) ?? "";
    // The API's own origin is there, exactly, and nothing in the list is a wildcard
    // (CI adds its stand-in for error tracking, which is one exact origin too)
    const sources = connect.trim().split(/\s+/).slice(1);
    expect(sources).toContain(new URL(request.url()).origin);
    expect(sources.filter((source: string) => source.includes("*") || /^https?:$/.test(source))).toEqual([]);
  });
});
