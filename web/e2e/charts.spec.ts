import { execFileSync } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";

import { expect, type Locator, type Page, type TestInfo, test } from "@playwright/test";

import { dealSettled, kpi, simulationSettled, stepLink } from "./helpers";

/**
 * PLAN.md 7.9: every download with a chart on screen holds the same chart as a native Excel chart over its own
 * cells. Each test downloads a workbook from a screen and reads its charts back with tests/chart_check.py: the
 * ranges each series reads and the values there, worked out from the file's cells and formulas by the tests'
 * own Excel reader (a waterfall and a fan draw from helper formulas over the table).
 */

type Charted = {
  charts: Record<string, { sheet: string; kinds: string[]; refs: string[]; values: Record<string, (number | string | null)[]> }>;
  colour_scales: Record<string, string[]>;
};

const root = join(__dirname, "..", "..");
const venv = join(root, ".venv", "bin", "python");
const python = process.env.PYTHON ?? (existsSync(venv) ? venv : "python");

async function charted(page: Page, button: Locator, info: TestInfo, name: string): Promise<Charted> {
  const downloading = page.waitForEvent("download");
  await button.click();
  const file = info.outputPath(name);
  await (await downloading).saveAs(file);
  return JSON.parse(execFileSync(python, ["-m", "tests.chart_check", file], { cwd: root }).toString()) as Charted;
}

/** A figure as the screen shows it ("1,234.5", "21.2%") as a number. */
const shownNumber = (text: string | null) => Number((text ?? "").replace(/[,%x\s]/g, ""));

test.describe("Charts in the downloads", () => {
  test("the deal summary's workbook draws revenue and EBITDA, debt paydown and the equity bridge", async ({ page }, info) => {
    await page.goto("/deal/summary");
    await dealSettled(page);
    const equityOut = shownNumber(await kpi(page, "Equity out").textContent());
    const book = await charted(page, page.getByRole("button", { name: /All tables/ }), info, "summary.xlsx");

    const pl = book.charts["Revenue and EBITDA"];
    expect(pl.sheet).toBe("P&L");
    // Revenue is the table's first row and EBITDA its fourth: rows 2 and 5 of the sheet, the five years across
    expect(pl.refs).toEqual(expect.arrayContaining(["'P&L'!$B$2:$F$2", "'P&L'!$B$5:$F$5", "'P&L'!$B$1:$F$1"]));

    const debt = book.charts["Debt paydown"];
    expect(debt.sheet).toBe("Debt by facility");
    expect(debt.kinds).toEqual(["barChart"]);
    expect(debt.refs).toEqual(expect.arrayContaining(["'Debt by facility'!$B$2:$G$2", "'Debt by facility'!$B$3:$G$3"]));

    // The equity bridge's waterfall ends at the equity out on screen, worked out from the sheet's formulas
    const bridge = book.charts["Equity bridge"];
    expect(bridge.sheet).toBe("Equity bridge");
    const totals = bridge.values[bridge.refs.filter((r) => /\$K\$\d+:\$K\$\d+$/.test(r))[0]] as number[];
    expect(totals.at(-1)).toBeCloseTo(equityOut, 1);
    expect(totals[0]).toBeGreaterThan(0);

    // Each facility's closing balance is charted on its own sheet
    expect(book.charts["Senior term loan: closing balance"].refs).toContain("'Senior term loan'!$E$2:$E$6");
  });

  test("the IRR sensitivity download is its table with a colour scale", async ({ page }, info) => {
    await page.goto("/deal/returns");
    await dealSettled(page);
    // The Returns step's one download is the sensitivity tile's
    const book = await charted(page, page.locator("#content").getByRole("button", { name: /Excel/ }), info, "sensitivity.xlsx");
    expect(book.colour_scales["IRR sensitivity"]?.[0]).toMatch(/^B2:[A-Z]+\d+$/);
  });

  test("the simulation's downloads chart the distribution, the percentiles, the drivers and the scenarios", async ({ page }, info) => {
    await page.goto("/monte-carlo/distribution");
    await simulationSettled(page);
    const book = await charted(page, page.getByRole("button", { name: /10k paths/ }), info, "paths.xlsx");
    const irr = book.charts["IRR distribution"];
    expect(irr.refs).toContain("'IRR distribution'!$C$2:$C$81"); // the screen's 80 bins
    expect(book.charts["IRR by percentile"].refs).toContain("'IRR by percentile'!$B$2:$B$102");
    expect(book.charts["What moves the IRR"].sheet).toBe("Drivers");
    expect(book.charts["IRR against Exit Multiple (sampled paths)"].kinds).toEqual(["scatterChart"]);
    expect(book.colour_scales.Heatmap).toHaveLength(1);

    await stepLink(page, "Scenarios").click();
    await simulationSettled(page);
    const scenarios = await charted(page, page.getByRole("button", { name: /Excel/ }).first(), info, "scenarios.xlsx");
    const range = scenarios.charts["IRR by scenario: P5, mean and P95"];
    expect(range.kinds).toEqual(["lineChart"]);
    // P5, mean and P95 columns of the scenario table, one point per scenario
    expect(range.refs).toEqual(expect.arrayContaining(["'Scenarios'!$D$2:$D$5", "'Scenarios'!$B$2:$B$5", "'Scenarios'!$E$2:$E$5"]));
  });

  test("the forecast's simulation download draws both fans, and its schedules the net income bridge", async ({ page }, info) => {
    await page.goto("/forecast/simulation");
    await expect(kpi(page, "Paths")).toHaveText("20,000");
    const book = await charted(page, page.getByRole("button", { name: /Excel/ }).first(), info, "fans.xlsx");
    for (const [title, sheet] of [
      ["Revenue fan", "Revenue bands"],
      ["EBITDA fan", "EBITDA bands"],
    ]) {
      const fan = book.charts[title];
      expect(fan.sheet).toBe(sheet);
      expect(fan.kinds).toEqual(["areaChart", "lineChart"]);
      // Five stacked bands from helper rows under the table (the Python tests check they add back up to
      // P95), the median and the plan drawn over them from the table's own rows
      const bands = fan.refs.filter((r) => /!\$B\$(\d+):/.exec(r) && Number(/!\$B\$(\d+):/.exec(r)?.[1]) > 8);
      expect(bands).toHaveLength(5);
      for (const band of bands.slice(1)) for (const v of fan.values[band] as number[]) expect(v).toBeGreaterThanOrEqual(-1e-9);
      expect(fan.refs.some((r) => r.startsWith(`'${sheet}'!$B$4:`))).toBe(true); // P50
      expect(fan.refs.some((r) => r.startsWith(`'${sheet}'!$B$7:`))).toBe(true); // the plan
    }
    expect(book.charts["Chance of reaching each EBITDA target"].sheet).toBe("Targets");

    await stepLink(page, "Schedules").click();
    const schedules = await charted(page, page.getByRole("button", { name: /All schedules/ }), info, "schedules.xlsx");
    const [title] = Object.keys(schedules.charts).filter((t) => t.startsWith("EBITDA to net income bridge"));
    expect(schedules.charts[title].sheet).toBe("Net income bridge");
  });

  test("the plan vs actual download charts EBITDA, the margin, where the IRR landed and the attribution", async ({ page }, info) => {
    await page.goto("/backtest/predicted");
    await page.getByRole("radio", { name: /^Burger King/ }).click();
    await expect(kpi(page, "Actual IRR")).toHaveText("19.0%");
    await stepLink(page, "Year by year").click();
    const book = await charted(page, page.getByRole("button", { name: /Excel/ }).first(), info, "backtest.xlsx");
    const ebitda = book.charts["EBITDA, plan vs actual"];
    expect(ebitda.sheet).toBe("Year by year");
    // EBITDA · plan and · actual are the fourth and fifth lines: sheet rows 5 and 6
    expect(ebitda.refs.some((r) => /^'Year by year'!\$B\$5:/.test(r))).toBe(true);
    expect(ebitda.refs.some((r) => /^'Year by year'!\$B\$6:/.test(r))).toBe(true);
    expect(book.charts["EBITDA margin"].sheet).toBe("EBITDA margin");
    expect(book.charts["Where the actual IRR landed"].refs).toContain("'Plan IRR distribution'!$C$2:$C$61");
    expect(book.charts["Where the plan missed"].sheet).toBe("Attribution");
  });
});
