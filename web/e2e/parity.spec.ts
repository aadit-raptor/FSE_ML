import { expect, type Page, test } from "@playwright/test";

import { dealSettled, kpi, simulationSettled, stepLink } from "./helpers";

type Sheet = { name: string; columns: string[]; rows: (string | number | null)[][] };

/** Click a download and return the saved filename and the workbook request the page sent. */
async function download(page: Page, button: ReturnType<Page["getByRole"]>, path = "/api/export/workbook") {
  const [req, dl] = await Promise.all([page.waitForRequest((r) => r.url().endsWith(path)), page.waitForEvent("download"), button.click()]);
  const res = await req.response();
  expect(res?.status()).toBe(200);
  expect(res?.headers()["content-type"]).toContain("spreadsheetml");
  return { filename: dl.suggestedFilename(), body: req.postDataJSON() as { sheets?: Sheet[] } & Record<string, unknown> };
}

const row = (s: Sheet | undefined, label: string) => s?.rows.find((r) => r[0] === label);

async function capabilities(page: Page) {
  const res = await page.request.get("/api/capabilities");
  return (await res.json()) as { anomaly_detector: boolean; surrogate: boolean };
}

test.describe("Streamlit parity", () => {
  test("summary exports every table with the model's numbers", async ({ page }) => {
    await page.goto("/deal/summary");
    await dealSettled(page);
    const api = await (await page.request.post("/api/deal/run", { data: {} })).json();
    const { filename, body } = await download(page, page.getByRole("button", { name: "↓ All tables" }));
    expect(filename).toBe("lbo_summary.xlsx");
    const names = body.sheets!.map((s) => s.name);
    expect(names).toEqual(expect.arrayContaining(["P&L", "Cash flow", "Debt schedule", "Senior Term Loan", "Mezzanine", "Equity bridge", "PP&E", "Adjusted EBITDA"]));
    const pl = body.sheets!.find((s) => s.name === "P&L");
    expect(row(pl, "Revenue")?.[1]).toBeCloseTo(api.operating_model.revenue[0], 6);
    expect(row(pl, "Net income")?.[5]).toBeCloseTo(api.operating_model.net_income[4], 6);
  });

  test("schedules: adjusted EBITDA follows SBC, working capital appears with days", async ({ page }) => {
    await page.goto("/deal/summary");
    await dealSettled(page);
    const adj = page.getByRole("region", { name: "Adjusted EBITDA" });
    const before = await adj.getByRole("row", { name: /^Adjusted EBITDA/ }).textContent();
    await adj.getByLabel("Stock-based comp").fill("5");
    await expect(adj.getByRole("row", { name: /^Adjusted EBITDA/ })).not.toHaveText(before ?? "");
    const wc = page.getByRole("region", { name: "Working capital" });
    await expect(wc).toContainText("working capital from days");

    await stepLink(page, "Debt & cash flow").click();
    await page.getByRole("switch", { name: "Working capital from days" }).click();
    await dealSettled(page);
    await stepLink(page, "Summary").click();
    // Y1 revenue 403.8 (100 EBITDA / 26% margin, +5%): receivables at 45 days = 49.8
    await expect(page.getByRole("region", { name: "Working capital" }).getByRole("row", { name: /^Receivables/ })).toContainText("49.8");
  });

  test("forecast schedules tie to the statements", async ({ page }) => {
    await page.goto("/forecast/statements");
    await expect(kpi(page, "Revenue F+5")).toHaveText("354.6");
    const netIncome = await page.getByRole("region", { name: "Income statement" }).getByRole("row", { name: /^Net income/ }).locator("td").last().textContent();
    const ppe = await page.getByRole("region", { name: "Balance sheet" }).getByRole("row", { name: /^PP&E, net/ }).locator("td").last().textContent();
    await stepLink(page, "Schedules").click();
    await expect(page.getByRole("region", { name: "PP&E roll-forward" }).getByRole("row", { name: /^Closing PP&E/ }).locator("td").last()).toHaveText(ppe ?? "");
    await expect(page.getByRole("img", { name: /EBITDA to net income bridge/ })).toContainText(netIncome ?? "");
    const { filename, body } = await download(page, page.getByRole("button", { name: "↓ All schedules" }));
    expect(filename).toBe("supporting_schedules.xlsx");
    expect(body.sheets!.map((s) => s.name)).toEqual(["PP&E", "Retained earnings", "Working capital", "Interest", "Revolver"]);
  });

  test("monte carlo exports the paths behind the results", async ({ page }) => {
    await page.goto("/monte-carlo/distribution");
    await simulationSettled(page);
    const { filename, body } = await download(page, page.getByRole("button", { name: "↓ 10k paths" }), "/api/export/montecarlo-sample");
    expect(filename).toBe("mc_simulation.xlsx");
    expect(body.seed).toBe(42);
  });

  test("deal risk score reacts to leverage (ML layer)", async ({ page }) => {
    test.skip(!(await capabilities(page)).anomaly_detector, "server has no anomaly detector");
    await page.goto("/deal/inputs");
    const score = kpi(page, "Risk score");
    await expect(score).toHaveText(/\d+\.\d \/ 10/);
    const before = await score.textContent();
    await page.getByLabel("Senior debt", { exact: true }).fill("9");
    await page.getByLabel("Senior debt", { exact: true }).blur();
    await expect(score).not.toHaveText(before ?? "", { timeout: 20_000 });
    await expect(page.getByRole("list", { name: "Risk flags" })).toContainText("Leverage");
  });

  test("live mode updates as a slider moves (ML layer)", async ({ page }) => {
    test.skip(!(await capabilities(page)).surrogate, "server has no surrogate model");
    await page.goto("/monte-carlo/live");
    const median = kpi(page, "Median IRR");
    await expect(median).toHaveText(/%$/, { timeout: 40_000 });
    const before = await median.textContent();
    await page.getByRole("slider", { name: "Exit multiple mean" }).fill("14");
    await expect(median).not.toHaveText(before ?? "", { timeout: 20_000 });
  });
});
