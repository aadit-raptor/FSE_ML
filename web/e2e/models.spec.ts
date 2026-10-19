import { expect, type Page, test } from "@playwright/test";

import { acceptedQuarter, asUser, signInAs } from "./helpers";

/**
 * Coverage -> Model (PLAN.md R4): a sector model opened for a company solved by hand, an assumption typed, the
 * projection moving to what the server gives for that assumption; a line swapped; the workbook downloaded; and a
 * new company whose past is filled from an accepted quarter. Every check reads an output, never a render alone.
 */
const run = Date.now().toString(36);

const cell = (page: Page, row: string, period: string) => page.locator(`[data-cell="${row}:${period}"]`);

/** The server's own answer for the example with `changes` typed in. */
async function serverValue(page: Page, exampleId: string, changes: Record<string, Record<string, number>>, row: string, period: string) {
  const headers = await asUser(page);
  const catalogue = await (await page.request.get("/api/research/models", { headers })).json();
  const example = catalogue.examples.find((e: { id: string }) => e.id === exampleId);
  const values = { ...example.company.values };
  for (const [r, by] of Object.entries(changes)) values[r] = { ...values[r], ...by };
  const answer = await (await page.request.post("/api/research/models/run", {
    headers, data: { structure: example.structure, company: { ...example.company, values } },
  })).json();
  return answer.rows.find((x: { key: string }) => x.key === row).cells[period].value as number;
}

test.use({ storageState: { cookies: [], origins: [] } });

test.describe("Sector models", () => {
  test("an assumption typed on screen moves the projection to the server's answer, and the model says it is not vetted", async ({ page }) => {
    await signInAs(page, `models-${run}`, "en-US");
    await page.goto("/coverage/model");
    await page.getByLabel("Start from").selectOption("lemon-tree-hotels");
    await page.getByRole("button", { name: "Start the model" }).click();
    await expect(page.locator('[data-model-grid="occupancy"]')).toBeVisible();
    await expect(page.getByText("Not yet vetted", { exact: true })).toBeVisible();

    const before = await serverValue(page, "lemon-tree-hotels", {}, "pat_own", "y1");
    await expect(cell(page, "pat_own", "y1")).toHaveText(before.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
    const occupancy = cell(page, "s1_occ", "y1").locator("input");
    await occupancy.fill("80");
    await occupancy.press("Enter");
    const after = await serverValue(page, "lemon-tree-hotels", { s1_occ: { y1: 0.8 } }, "pat_own", "y1");
    expect(after).toBeGreaterThan(before + 1000);
    await expect(cell(page, "pat_own", "y1")).toHaveText(after.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }));

    // swap line 2 to growth: its unit rows go, its growth is still to type, the model stays whole
    await page.locator('[data-line="s2"]').getByRole("combobox").selectOption("growth");
    await expect(page.locator('[data-row="s2_g"]')).toBeVisible();
    await expect(page.locator('[data-row="s2_rpu"]')).toHaveCount(0);
    await expect(page.getByText("Assumptions to type")).toBeVisible();
    await expect(page.getByText("Left out", { exact: true })).toBeVisible();
    await expect(cell(page, "s2_rev", "y1")).toHaveText("n/a");          // its growth is not typed yet
    for (const period of ["cy", "y1"]) {                                    // y1 grows on the year in progress
      const growth = cell(page, "s2_g", period).locator("input");
      await growth.fill("10");
      await growth.press("Enter");
    }
    await expect(cell(page, "s2_rev", "y1")).not.toHaveText("n/a");

    // the workbook comes down as live formulas
    const download = page.waitForEvent("download");
    await page.getByRole("button", { name: "Download workbook" }).click();
    expect((await download).suggestedFilename()).toBe("Lemon_Tree_Hotels_Limited_occupancy_model.xlsx");
  });

  test("a new company's past is filled from its accepted quarter", async ({ page }) => {
    await signInAs(page, `models-fill-${run}`, "en-GB");
    await acceptedQuarter(page, "DMart");
    await page.goto("/coverage/model");
    await page.getByLabel("Sector").selectOption("hotels");
    await page.getByLabel("Fill the past from").selectOption({ label: "DMart" });
    await page.getByRole("button", { name: "Start the model" }).click();
    await expect(page.locator('[data-model-grid="occupancy"]')).toBeVisible();
    const headers = await asUser(page);
    const companies = await (await page.request.get("/api/research/companies", { headers })).json();
    const fill = await (await page.request.get(`/api/research/companies/${companies.companies[0].id}/fill`, { headers })).json();
    expect(fill.values.rev.ytd).toBeGreaterThan(0);
    // an input shows no grouping and the fewest decimals that keep the figure
    await expect(cell(page, "rev", "ytd").locator("input")).toHaveValue(String(Number(fill.values.rev.ytd.toFixed(4))));
    await expect(page.getByText("From accepted quarters").first()).toBeVisible();

    // valuation (PLAN.md R6): the country's 10-year yield is offered, and taken, as the discount rate's start
    // (the real endpoint answers from stored economic data: none on CI's empty database, a figure on a refreshed one)
    await expect(page.getByText(/Starting point for the discount rate|No risk-free rate to offer: no 10-year yield/)).toBeVisible();
    await page.route("**/risk-free", (route) => route.fulfill({ json: {
      status: "ok", currency: "INR", country: "IN", value: 0.0706, period: "2026-09", source: "oecd",
      source_series: "DF_FINMARK/IND.M.IRLT", url: "https://data-explorer.oecd.org/", current: true } }));
    await page.reload();
    await page.getByLabel("Fill the past from").selectOption({ label: "DMart" });
    await page.getByRole("button", { name: /^(Open the saved model|Start the model)$/ }).click();   // saved or not yet
    await expect(page.getByText(/India's 10-year government bond yield, 7.06%/)).toBeVisible();
    await page.getByRole("button", { name: "Start the discount rate from it" }).click();
    await expect(cell(page, "v_rate", "ly").locator("input")).toHaveValue("7.06");
  });

  test("the valuation: both methods per share as the server works them out, moved by what the analyst types", async ({ page }) => {
    await signInAs(page, `models-value-${run}`, "en-US");
    await page.goto("/coverage/model");
    await page.getByLabel("Start from").selectOption("lemon-tree-hotels");
    await page.getByRole("button", { name: "Start the model" }).click();
    const dcf = page.locator('[data-kpi="Value per share, discounted cash flow"]');
    const multiple = page.locator('[data-kpi="Value per share, target multiple"]');
    const two = (n: number) => n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    // the example carries the case solved by hand (tests/fixtures/research/valuation.json): 31.78 and 154.48
    await expect(dcf).toHaveText("31.78");
    await expect(multiple).toHaveText("154.48");
    await expect(page.getByText("35.0x P/E on FY28 · 10.3% against the price")).toBeVisible();
    await expect(page.getByText("on Oct 16, 2026")).toBeVisible();

    const rate = cell(page, "v_rate", "ly").locator("input");
    await rate.fill("14");
    await rate.press("Enter");
    const dearer = await serverValue(page, "lemon-tree-hotels", { v_rate: { ly: 0.14 } }, "v_ps", "ly");
    expect(dearer).toBeLessThan(31.78);
    await expect(dcf).toHaveText(two(dearer));

    // the multiple on another year, then of EBITDA: the server's value for that year
    await page.getByLabel("Target multiple: the year it is applied to").selectOption("y1");
    await expect(multiple).toHaveText(two(await serverValue(page, "lemon-tree-hotels", {}, "v_mv", "y1")));
    await page.getByLabel("Target multiple: of which figure").selectOption("ev_ebitda");
    await expect(page.getByText(/35.0x EV\/EBITDA on FY27/)).toBeVisible();
    const ev = (await multiple.textContent()) ?? "";
    expect(Number(ev.replace(/,/g, ""))).toBeGreaterThan(154.48);

    // the price's date is a date cell, kept as Excel keeps one
    const day = cell(page, "v_date", "ly").locator("input");
    await day.fill("2026-10-20");
    await day.blur();
    await expect(page.getByText("on Oct 20, 2026")).toBeVisible();
    const price = cell(page, "v_price", "ly").locator("input");
    await price.fill("");
    await price.press("Enter");
    await expect(page.getByText("Type the price and its date in the model below")).toBeVisible();
  });

  test("a saved model rolls forward on newer results, its estimate kept beside them and unchanged by later edits", async ({ page }) => {
    await signInAs(page, `models-roll-${run}`, "en-US");
    await acceptedQuarter(page, "DMart");                        // six months to 30 September 2026 accepted
    const headers = await asUser(page);
    const companyId: string = (await (await page.request.get("/api/research/companies", { headers })).json()).companies[0].id;
    const catalogue = await (await page.request.get("/api/research/models", { headers })).json();
    const example = catalogue.examples.find((e: { id: string }) => e.id === "lemon-tree-hotels");
    // the model as saved after the first quarter: three months of FY27 reported
    const put = await page.request.put(`/api/research/companies/${companyId}/model`, {
      headers, data: { structure: example.structure, company: { ...example.company, name: "DMart" },
        as_of: { ly_end: "2026-03-31", ytd_end: "2026-06-30", ytd_months: 3 } },
    });
    expect(put.status()).toBe(200);

    await page.goto("/coverage/model");
    await page.getByLabel("Fill the past from").selectOption({ label: "DMart" });
    await expect(page.getByText("This company has a saved model")).toBeVisible();
    await page.getByRole("button", { name: "Open the saved model" }).click();
    await expect(page.locator('[data-model-grid="occupancy"]')).toBeVisible();
    await expect(page.getByText("Newer results accepted")).toBeVisible();
    await page.getByRole("button", { name: "Roll forward" }).click();
    await expect(page.getByText("Rolled forward", { exact: true })).toBeVisible();

    const versions = async () => (await (await page.request.get(`/api/research/companies/${companyId}/model/versions`, { headers })).json()).versions;
    const rolled = (await versions()).find((v: { kind: string }) => v.kind === "rolled");
    const rev = rolled.comparison.lines.find((l: { row: string }) => l.row === "rev");
    const fmt = (v: number) => v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    const line = page.locator('[data-compare="rev"]');
    await expect(line.locator("[data-estimate]")).toHaveText(fmt(rev.estimate));
    await expect(line.locator("[data-actual]")).toHaveText(fmt(rev.actual));
    const model = await (await page.request.get(`/api/research/companies/${companyId}/model`, { headers })).json();
    expect(model.as_of).toEqual({ ly_end: "2026-03-31", ytd_end: "2026-09-30", ytd_months: 6 });
    expect(model.company.values.rev.ytd).toBeCloseTo(example.company.values.rev.ytd + rev.actual, 3);

    // an assumption changed afterwards is saved, and the estimate beside the results does not move
    const occupancy = cell(page, "s1_occ", "roy").locator("input");
    await occupancy.fill("10");
    await occupancy.press("Enter");
    await expect.poll(async () => (await (await page.request.get(`/api/research/companies/${companyId}/model`, { headers })).json())
      .company.values.s1_occ.roy).toBe(0.1);
    await page.reload();
    await page.getByLabel("Fill the past from").selectOption({ label: "DMart" });
    await page.getByRole("button", { name: "Open the saved model" }).click();
    await expect(line.locator("[data-estimate]")).toHaveText(fmt(rev.estimate));
    expect((await versions()).find((v: { kind: string }) => v.kind === "rolled").comparison).toEqual(rolled.comparison);
  });
});
