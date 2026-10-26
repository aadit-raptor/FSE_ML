import { expect, test } from "@playwright/test";

import { asUser, kpi, setField, signInAs, stepLink } from "./helpers";

/**
 * Deal -> Memo (PLAN.md 6.3): an investment memo of a saved version. Every check reads an output: the memo's IRR
 * and MOIC are the deal model's for that version (the same the Returns step shows), its simulated figures are the
 * Monte Carlo endpoint's on the screen's seed, the linking sentences hold the run's figures, and the Word file and
 * the deal's history follow. The browser tests replay the AI with nothing recorded for a memo, so the templates
 * write and say so.
 */
const run = Date.now().toString(36);

test.use({ storageState: { cookies: [], origins: [] } });

test("a saved version's memo shows the model run's figures and downloads as Word", async ({ page }) => {
  await signInAs(page, `memo-${run}`, "en-GB");
  const headers = await asUser(page);
  const name = `Memo ${run}`;

  await page.goto("/deal/memo");
  await expect(page.getByText("Save this deal first")).toBeVisible();

  await page.goto("/deal/returns");
  await setField(page, "Exit multiple", "12");
  await expect(kpi(page, "IRR")).toHaveText("23.7%");
  await stepLink(page, "Saved deals").click();
  await page.getByLabel("Deal name", { exact: true }).fill(name);
  await page.getByRole("button", { name: "Save deal" }).click();
  await expect(page.locator('[data-deal-save="saved"]')).toBeVisible();
  await page.getByLabel("Version label", { exact: true }).fill("Board pack");
  await page.getByRole("button", { name: "Keep version" }).click();
  await expect(page.getByRole("table", { name: "Version history" }).locator("tr").filter({ hasText: "Board pack" }))
    .toHaveCount(1);

  await stepLink(page, "Memo").click();
  await expect(page.getByRole("combobox", { name: "Draft from" })).toContainText("Board pack");
  await page.getByRole("button", { name: "Draft memo" }).click();
  const memo = page.getByRole("article", { name: "The memo as it will be written" });
  await expect(memo.getByRole("heading", { level: 2 })).toHaveText(name, { timeout: 60_000 });

  // the figures are the version's model run: the deal model's IRR, and the simulation on the screen's seed
  const deals = await (await page.request.get("/api/deals", { headers })).json();
  const saved = deals.deals.find((d: { name: string }) => d.name === name);
  const detail = await (await page.request.get(`/api/deals/${saved.id}`, { headers })).json();
  const sim = await (await page.request.post("/api/montecarlo/run", {
    headers, data: { mc: { n: 50000, hurdle: 20, ebitda: detail.inputs.ebitda, entry_mult: detail.inputs.entry_mult,
      hold: detail.inputs.hold }, deal: detail.inputs, settings: detail.settings, seed: 42 },
  })).json();
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const figure = (text: string) => memo.locator('[data-figure="run"]', { hasText: text }).first();
  await expect(figure("23.7%")).toBeVisible();
  await expect(figure(pct(sim.summary.median_irr))).toBeVisible();
  await expect(figure(pct(sim.summary.p_loss))).toBeVisible();
  await expect(memo).toContainText(/Version \d+: Board pack\./);
  await expect(memo).not.toContainText(/\b(Buy|Sell|Hold|Accumulate)\b/);

  // the sentences: the templates, with the run's figures in them
  await page.getByRole("button", { name: "Write the sentences" }).click();
  await expect(page.locator('[data-section="summary"]')).toContainText("Template: the AI didn't answer");
  await expect(memo.locator('[data-kind="sentence"]').first()).toContainText("23.7%");

  // the Word file, and the deal's history
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Word", exact: true }).click();
  const file = await download;
  expect(file.suggestedFilename()).toMatch(new RegExp(`^Memo_${run}_v\\d+_memo\\.docx$`));
  const history = await (await page.request.get(`/api/deals/${saved.id}/history`, { headers })).json();
  const actions = history.entries.map((e: { action: string; export?: string }) => e.export ?? e.action);
  expect(actions).toContain("memo_word");
  expect(actions).toContain("memo_drafted");

  // deleting it leaves the version ready to draft again
  await page.locator("[data-memo]").getByRole("button", { name: "Delete" }).click();
  await expect(page.locator("[data-memo]")).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Draft memo" })).toBeVisible();
});
