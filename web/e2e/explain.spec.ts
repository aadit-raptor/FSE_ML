import { readFileSync } from "node:fs";
import path from "node:path";

import { expect, type Locator, type Page, test } from "@playwright/test";

import { dealSettled, kpi, setField, signInAs, simulationSettled, stepLink } from "./helpers";

/**
 * Plain-English explanations and the glossary (PLAN.md 6.4), proved by what the screen shows.
 *
 * An explanation is a template filled with the tile's own figures, so every figure it quotes (each `[data-quote]`)
 * must be, character for character, a figure the screen shows outside the explanation: the same formatting, the
 * same digit grouping. The account here groups digits the Indian way (lakh and crore) and the deal is in thousands,
 * so a quote that formatted its number another way, or rounded it differently, would not be found on the screen.
 *
 * The glossary: every label on the explained screens' tiles (titles, row and column headings, legend entries, chart
 * categories) that holds a word sits in a `[data-term]`, which the app draws only for a term the glossary defines,
 * and hovering one shows that definition.
 */
const run = Date.now().toString(36);
const GLOSSARY: Record<string, string> = JSON.parse(readFileSync(path.join(__dirname, "..", "messages", "en.json"), "utf-8")).glossary;

const tile = (page: Page, title: string) => page.getByRole("region", { name: title, exact: true });

/** Every text the screen shows outside explanations and glossary pop-ups, whole and split into words. */
async function shownTokens(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const root = document.querySelector("#content") ?? document.body;
    const hidden = (el: Element) => el.closest("[data-explanation], [data-glossary]");
    const texts: string[] = [];
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    for (let n = walker.nextNode(); n; n = walker.nextNode()) {
      if (n.parentElement && !hidden(n.parentElement)) texts.push(n.textContent ?? "");
    }
    root.querySelectorAll("input").forEach((i) => {
      if (!hidden(i)) texts.push((i as HTMLInputElement).value);
    });
    return texts
      .flatMap((t) => [t.trim(), ...t.split(/\s+/)])
      .map((t) => t.replace(/^[(·:;]+|[),·:;]+$/g, ""))
      .filter(Boolean);
  });
}

/** Opens a tile's explanation and checks each figure it quotes is shown on the screen exactly; answers the quotes. */
async function explained(page: Page, title: string, atLeast: number): Promise<string[]> {
  const section = tile(page, title);
  await section.getByRole("button", { name: "Explain", exact: true }).click();
  const note = section.locator("[data-explanation]");
  await expect(note).toBeVisible();
  await expect(section.getByRole("button", { name: "Hide explanation", exact: true })).toHaveAttribute("aria-expanded", "true");
  const quotes = (await note.locator("[data-quote]").allTextContents()).map((q) => q.trim());
  expect(quotes.length, `${title}: quotes`).toBeGreaterThanOrEqual(atLeast);
  const shown = await shownTokens(page);
  for (const q of quotes) expect(shown, `${title} quotes ${q}, which the screen does not show`).toContain(q);
  return quotes;
}

/** Labels holding a word, on the named tiles, that no glossary term covers (each tile must hold at least its title). */
async function labelsWithoutTerm(page: Page, titles: string[]): Promise<string[]> {
  const missing: string[] = [];
  for (const title of titles) {
    const section = tile(page, title);
    await expect(section, title).toBeVisible();
    missing.push(
      ...(await section.evaluate((el, name) => {
        const out: string[] = [];
        const labels = el.querySelectorAll("h2, th, dt, [data-legend], .chart-category");
        if (!labels.length) out.push(`${name}: no label read at all`);
        labels.forEach((label) => {
          if (label.closest("[data-explanation], [data-glossary]")) return;
          const text = (label.textContent ?? "").trim();
          if (!/[A-Za-z]{3}/.test(text)) return;                 // a year, a multiple, a hold: figures, not terms
          if (label.closest("[data-term]") || label.querySelector("[data-term]")) return;
          out.push(`${name}: ${text}`);
        });
        return out;
      }, title)),
    );
  }
  return missing;
}

/** Hover a term: its definition, word for word, from the glossary. */
async function definitionOnHover(page: Page, term: Locator) {
  const id = await term.getAttribute("data-term");
  expect(id && GLOSSARY[id], `glossary.${id}`).toBeTruthy();
  await term.hover();
  await expect(page.getByRole("tooltip")).toHaveText(GLOSSARY[id!]);
  await page.keyboard.press("Escape");
  await expect(page.locator("[data-glossary]")).toHaveCount(0);
}

test.use({ storageState: { cookies: [], origins: [] } });

test.describe("Explanations and glossary", () => {
  test("the deal's returns and debt: each explanation quotes the tiles' figures exactly, lakh grouping included", async ({ page }) => {
    await signInAs(page, `explain-deal-${run}`, "en-US", "lakh");
    await page.goto("/deal/inputs");
    await page.getByLabel("Deal money unit").selectOption("thousands");
    await stepLink(page, "Returns").click();
    await dealSettled(page);
    await expect(kpi(page, "Equity out")).toHaveText(/^\d{1,2},\d{2},\d{3}\.\d$/);   // grouped in lakhs

    const returns = await explained(page, "Equity bridge", 8);
    // the default deal's 21.2% against the 20% hurdle: the sentence says it clears it
    await expect(tile(page, "Equity bridge").locator("[data-explanation]")).toContainText("it clears it");
    for (const title of ["IRR", "MOIC", "Equity in", "Equity out", "Exit EV", "Net debt at exit"]) {
      expect(returns, title).toContain((await kpi(page, title).textContent())?.trim());
    }
    const grid = await explained(page, "IRR sensitivity", 4);
    expect(grid).toContain((await tile(page, "IRR sensitivity").locator('td[aria-current="true"]').textContent())?.trim());

    expect(await labelsWithoutTerm(page, ["IRR", "MOIC", "Equity in", "Equity out", "Exit EV", "Net debt at exit",
      "Equity bridge", "IRR sensitivity", "Debt paydown", "Operating summary"])).toEqual([]);
    await definitionOnHover(page, tile(page, "IRR").locator("[data-term]").first());
    // the money unit is a term too: a thousand of the deal's currency
    await definitionOnHover(page, tile(page, "Operating summary").locator('[data-term="unit_thousands"]'));

    await stepLink(page, "Debt & cash flow").click();
    await dealSettled(page);
    const debt = await explained(page, "Debt balance", 6);
    for (const title of ["Debt at close", "Debt at exit", "Interest, year 1", "Cumulative FCF"]) {
      expect(debt, title).toContain((await kpi(page, title).textContent())?.trim());
    }
    const credit = await explained(page, "Distress by year", 4);
    expect(credit).toContain((await page.locator('[data-band-year="1"]').textContent())?.trim());
    expect(await labelsWithoutTerm(page, ["Debt at close", "Debt at exit", "Net debt at exit", "Interest, year 1",
      "Cumulative FCF", "Interest loop", "Debt balance", "Cash flow", "Distress by year"])).toEqual([]);
    // each facility's own schedule: its tile title and rows
    const facilities = await page.locator("[data-facility-tile]").evaluateAll((els) => els.map((e) => e.getAttribute("aria-label") ?? ""));
    expect(facilities.length).toBeGreaterThan(0);
    expect(await labelsWithoutTerm(page, facilities)).toEqual([]);

    // a change on the rail changes what the explanation says, figure for figure
    await stepLink(page, "Returns").click();
    await dealSettled(page);
    const exitMultiple = await page.getByLabel("Exit multiple", { exact: true }).inputValue();
    const irrBefore = (await kpi(page, "IRR").textContent())?.trim() ?? "";
    await setField(page, "Exit multiple", "13");
    await expect(kpi(page, "IRR")).not.toHaveText(irrBefore);
    await dealSettled(page);
    const after = await explained(page, "Equity bridge", 8);
    expect(after).toContain((await kpi(page, "IRR").textContent())?.trim());
    expect(after).not.toContain(irrBefore);
    await setField(page, "Exit multiple", exitMultiple);
    await page.goto("/deal/inputs");
    await page.getByLabel("Deal money unit").selectOption("millions");
  });

  test("leases and IFRS: the regional statement words are glossary terms", async ({ page }) => {
    await signInAs(page, `explain-ifrs-${run}`, "en-US");
    await page.goto("/deal/inputs");
    await dealSettled(page);
    await page.getByRole("combobox", { name: "Accounting standard", exact: true }).selectOption("ifrs");
    await setField(page, "Lease cost a year", "10");
    await setField(page, "Lease liability", "60");
    await dealSettled(page);
    await stepLink(page, "Returns").click();
    await dealSettled(page);
    const summary = tile(page, "Operating summary");
    await expect(summary.locator('[data-term="ebitdaAfterLeases"]')).toHaveText("EBITDA after lease costs");
    await expect(summary.locator('[data-term="financeCosts"]')).toHaveText("Finance costs");
    await definitionOnHover(page, summary.locator('[data-term="ebitdaAfterLeases"]'));
    expect(await labelsWithoutTerm(page, ["Operating summary", "Leases"])).toEqual([]);
    await stepLink(page, "Debt & cash flow").click();
    await dealSettled(page);
    await expect(tile(page, "Cash flow").locator('[data-term="profitForYear"]')).toHaveText("Profit for the year");
    await page.goto("/deal/inputs");
    await setField(page, "Lease cost a year", "0");
    await setField(page, "Lease liability", "0");
    await page.getByRole("combobox", { name: "Accounting standard", exact: true }).selectOption("");
  });

  test("Monte Carlo: the distribution and the drivers explained from the figures on screen", async ({ page }) => {
    await signInAs(page, `explain-mc-${run}`, "en-US", "lakh");
    await page.goto("/monte-carlo/distribution");
    await simulationSettled(page);
    const dist = await explained(page, "IRR distribution", 6);
    for (const title of ["Mean IRR", "P5", "P95"]) expect(dist, title).toContain((await kpi(page, title).textContent())?.trim());
    const kpis = ["Mean IRR", "P(IRR > 20%)", "P5", "P95", "Wipeout", "Run time"];
    expect(await labelsWithoutTerm(page, [...kpis, "IRR distribution", "IRR by percentile", "MOIC distribution",
      "What was simulated", "Distress by year"])).toEqual([]);

    await stepLink(page, "Drivers").click();
    await expect(tile(page, "What moves IRR")).toBeVisible();
    await explained(page, "What moves IRR", 2);
    const worst = page.locator("[data-explained-case]").first();
    const worstTitle = (await worst.getAttribute("aria-label")) ?? "";
    const tail = await explained(page, worstTitle, 3);
    expect(tail).toContain((await worst.locator('[data-explained="total"]').textContent())?.trim());
    expect(tail).toContain((await worst.locator('[data-explained="base"]').textContent())?.trim());
    const cases = await page.locator("[data-explained-case]").evaluateAll((els) => els.map((e) => e.getAttribute("aria-label") ?? ""));
    expect(await labelsWithoutTerm(page, ["What moves IRR", "IRR against growth", "Correlations in the simulated paths", ...cases])).toEqual([]);
    await definitionOnHover(page, tile(page, "What moves IRR").locator("[data-term]").first());
  });

  test("a covered company's valuation and model, in lakhs and crores", async ({ page }) => {
    await signInAs(page, `explain-model-${run}`, "en-US", "lakh");
    await page.goto("/coverage/model");
    await page.getByLabel("Start from").selectOption("lemon-tree-hotels");          // figures in Rs lakh
    await page.getByRole("button", { name: "Start the model" }).click();
    await expect(page.locator('[data-model-grid="occupancy"]')).toBeVisible();
    const value = await explained(page, "Valuation", 5);
    expect(value).toContain("31.78");
    expect(value).toContain("154.48");
    const grid = page.locator("section[data-model-tile]");
    const model = await explained(page, (await grid.getAttribute("aria-label")) ?? "", 4);
    expect(model).toContain((await page.locator('[data-cell="rev:cy"]').textContent())?.trim());
    expect(model.some((q) => q.split(",").length > 2)).toBe(true);                    // lakh grouping: 1,23,456.78
    expect(await labelsWithoutTerm(page, ["Value per share, discounted cash flow", "Value per share, target multiple",
      "Share price typed", "Valuation"])).toEqual([]);
    await definitionOnHover(page, grid.locator('[data-term="unit_lakhs"]'));

    await page.getByLabel("Start from").selectOption("avenue-supermarts");          // figures in Rs crore
    await page.getByRole("button", { name: /^(Start the model|Open the saved model)$/ }).click();
    await expect(page.locator('[data-model-grid="stores"]')).toBeVisible();
    await definitionOnHover(page, page.locator('section[data-model-tile] [data-term="unit_crores"]'));
  });
});
