import { expect, type Page, test } from "@playwright/test";

import { asUser, dealSettled, kpi, stepLink } from "./helpers";

/**
 * Model version on every result (PLAN.md 3.1). The API's side -- stamps on
 * every result, stored with saved deals, compared on reopening -- is proven in
 * tests/test_model_version.py against a real database, including a deal whose
 * stored stamp is aged to an older engine. Here: the screen shows that check,
 * an export sends the stamp of the result on screen, and the status bar names
 * the engine version.
 */
const run = Date.now().toString(36);
const dealRow = (page: Page, name: string) => page.locator(`tr[data-deal="${name}"]`);
const notice = (page: Page) => page.locator('[data-model-check="changed"]');

type Stamp = { engine_version: string; settings_fingerprint: string | null; data_vintage: string };

async function health(page: Page) {
  return (await (await page.request.get("/api/health")).json()) as { engine_version: string };
}

async function createDeal(page: Page, name: string): Promise<string> {
  const res = await page.request.post("/api/deals", {
    data: { name, inputs: { exit_mult: 12 } },
    headers: await asUser(page),
  });
  expect(res.status()).toBe(201);
  return (await res.json()).id;
}

async function openFromList(page: Page, name: string) {
  await page.goto("/deal/saved");
  await dealRow(page, name).getByRole("button", { name: "Open" }).click();
  await expect(dealRow(page, name)).toHaveAttribute("aria-current", "true");
}

test.describe("Model version", () => {
  test("the status bar names the engine version", async ({ page }) => {
    const { engine_version } = await health(page);
    await page.goto("/deal/returns");
    await expect(page.locator(`[data-engine-version="${engine_version}"]`)).toHaveText(`· model ${engine_version}`);
  });

  test("a deal whose results have not changed opens without a notice", async ({ page }) => {
    const name = `Unchanged ${run}`;
    const id = await createDeal(page, name);
    try {
      const opened = page.waitForResponse((r) => r.url().endsWith(`/api/deals/${id}`) && r.request().method() === "GET");
      await openFromList(page, name);
      const body = await (await opened).json();
      expect(body.model_check.status).toBe("unchanged");
      await stepLink(page, "Returns").click();
      await expect(kpi(page, "IRR")).toHaveText("23.7%");
      await expect(notice(page)).toHaveCount(0);
    } finally {
      await page.request.delete(`/api/deals/${id}`, { headers: await asUser(page) });
    }
  });

  test("reopening a deal saved by an older model says its results changed", async ({ page }) => {
    const name = `Older model ${run}`;
    const id = await createDeal(page, name);
    // The API's own answer, with the deal's stored stamp aged the way an old
    // deal's would be: an older engine that gave a lower IRR and MOIC
    await page.route(`**/api/deals/${id}`, async (route) => {
      if (route.request().method() !== "GET") return route.fallback();
      const response = await route.fetch();
      const body = await response.json();
      const now = body.model_check.now;
      body.model_check = {
        status: "changed",
        saved: { ...now, engine_version: "0.9.0", irr: 0.2251, moic: 2.75 },
        now,
        causes: ["engine_version"],
      };
      await route.fulfill({ response, json: body });
    });
    try {
      await openFromList(page, name);
      await stepLink(page, "Returns").click();
      await dealSettled(page);
      const shown = notice(page);
      await expect(shown).toBeVisible();
      await expect(page.getByText("Results changed since saved")).toBeVisible();
      // Saved figures beside today's, which are the ones on screen
      await expect(shown).toContainText("IRR 22.5% → 23.7%");
      await expect(shown).toContainText("MOIC 2.75x → 2.90x");
      const { engine_version } = await health(page);
      await expect(shown).toContainText(`the model changed from version 0.9.0 to ${engine_version}`);
      await expect(kpi(page, "IRR")).toHaveText("23.7%");

      await page.getByRole("button", { name: "Dismiss" }).click();
      await expect(notice(page)).toHaveCount(0);
    } finally {
      await page.unroute(`**/api/deals/${id}`);
      await page.request.delete(`/api/deals/${id}`, { headers: await asUser(page) });
    }
  });

  test("an export sends the stamp of the result on screen", async ({ page }) => {
    await page.goto("/deal/summary");
    await dealSettled(page);
    const direct = await (await page.request.post("/api/deal/run", { data: {}, headers: await asUser(page) })).json();
    const [req] = await Promise.all([
      page.waitForRequest((r) => r.url().endsWith("/api/export/workbook")),
      page.waitForEvent("download"),
      page.getByRole("button", { name: "↓ All tables" }).click(),
    ]);
    expect((await req.response())?.status()).toBe(200);
    const sent = (req.postDataJSON() as { model?: Stamp }).model;
    expect(sent?.engine_version).toBe((await health(page)).engine_version);
    expect(sent?.settings_fingerprint).toBe(direct.model.settings_fingerprint);
    expect(sent?.data_vintage).toBe(direct.model.data_vintage);
  });
});
