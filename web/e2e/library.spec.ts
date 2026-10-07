import { expect, type Page, test } from "@playwright/test";

import { MODES } from "../src/lib/nav";

import { kpi, modeTab, stepLink } from "./helpers";

/**
 * The optional reference library (PLAN.md 4.5), through the real screens and API.
 *
 * Done when: with the library off, every screen and model still works; base rates show citations;
 * the coverage page reports counts. The figures below are the sources' own (S&P's 2024 study,
 * Global Credit Data's 2020 report), not read back from the API.
 *
 * Switching the library off for real would hide it from every other spec running alongside, so
 * "off" is the API's own off answer replayed here; tests/test_library.py switches it for real.
 */
const content = (page: Page) => page.locator("#content");
type State = { enabled: boolean; locked_off: boolean; can_switch: boolean; updated_at: string | null };
const STATE: State = { enabled: true, locked_off: false, can_switch: false, updated_at: null };
const OFF = { ...STATE, enabled: false };

async function libraryOff(page: Page) {
  await page.route("**/api/library", (r) => r.fulfill({ json: OFF }));
  await page.route("**/api/library/base-rates*", (r) =>
    r.fulfill({ json: { enabled: false, sources: [], tables: [], country: null, sp_region: null, default: null, recovery: null } }),
  );
  await page.route("**/api/library/coverage", (r) => r.fulfill({ json: { enabled: false, collections: [], base_rates: [] } }));
  await page.route("**/api/backtesting/examples", (r) => r.fulfill({ json: { enabled: false, examples: [] } }));
}

const row = (page: Page, table: string, key: string) => content(page).getByRole("table", { name: table }).locator(`[data-row="${key}"]`);

test.describe("Reference library", () => {
  test("base rates show each table's figures with its citation", async ({ page }) => {
    await page.goto("/library/base-rates");
    await expect(modeTab(page, "Library")).toHaveAttribute("aria-current", "page");

    // Cumulative defaults: S&P Table 24 worldwide, Table 25 for Europe
    await page.getByRole("radio", { name: "Global" }).click();
    const global = "Cumulative default rate by rating · Global";
    await expect(row(page, global, "B").locator("td").nth(4)).toHaveText("15.60%");
    await expect(row(page, global, "speculative_grade").locator("td").first()).toHaveText("3.54%");
    await expect(content(page).locator('[data-cite="cumulative_global"]')).toHaveText(
      "S&P Global Ratings · Table 24: Global corporate average cumulative default rates, 1981-2024",
    );
    await page.getByRole("radio", { name: "Europe" }).click();
    const europe = "Cumulative default rate by rating · Europe";
    await expect(row(page, europe, "B").locator("td").first()).toHaveText("1.75%");
    await expect(row(page, europe, "B").locator("td")).toHaveCount(10);
    await expect(content(page).locator('[data-cite="cumulative_by_region"]')).toContainText("Table 25");

    // Recovery: Global Credit Data, Table 2 and Table 4
    await expect(row(page, "Recovery by seniority and collateral", "unsecured_subordinated")).toContainText("38%");
    await expect(row(page, "Recovery by seniority and collateral", "unsecured_subordinated")).toContainText("62%");
    await expect(row(page, "Recovery by region", "north_america")).toContainText("4,781");
    await expect(content(page).locator('[data-cite="lgd_by_region"]')).toHaveText("Global Credit Data · Table 4: LGD by region");

    // Every source in the rail, with its sample, link and the day it was last checked
    const sp = page.locator('[data-source="sp_default_study_2024"]');
    await expect(sp).toContainText("23,831 rated companies, 1981-2024");
    await expect(sp.getByRole("link", { name: "S&P Global Ratings" })).toHaveAttribute("href", /maalot\.co\.il/);
    await expect(page.locator('[data-source="gcd_lgd_2020"]')).toContainText("11,527 defaulted borrowers, defaults 2000-2016");
    await expect(page.locator('[data-source="gcd_lgd_2020"]')).toContainText("last checked 7 Oct 2026");
    await expect(sp).toContainText("Published 27 Mar 2025");
    await expect(content(page).getByRole("img", { name: "Speculative-grade defaults each year by region" })).toBeVisible();
  });

  test("coverage reports counts by region, size, era and outcome", async ({ page }) => {
    await page.goto("/library/coverage");
    const examples = content(page).locator('[data-collection="examples"]');
    await expect(examples.locator('[data-bucket="region:us"] td')).toHaveText("4");
    await expect(examples.locator('[data-bucket="region:europe"] td')).toHaveText("0");
    await expect(examples.locator('[data-bucket="size:over_10bn"] td')).toHaveText("3");
    await expect(examples.locator('[data-bucket="era:2000_2007"] td')).toHaveText("2");
    await expect(examples.locator('[data-bucket="outcome:distress"] td')).toHaveText("1");
    await expect(content(page)).toContainText("4 example deals");
    await expect(content(page)).toContainText("Sourced reference deals: none yet");
    const lgd = content(page).getByRole("table", { name: "Base rates covered" }).locator('[data-table="lgd_by_region"]');
    await expect(lgd).toContainText("11,527");
    await expect(lgd).toContainText("2000-2016");
    // The shared e2e account is no administrator
    await expect(page.getByRole("switch")).toHaveCount(0);
    await expect(page.getByText("Only administrators can show or hide the reference library.")).toBeVisible();
  });

  test("examples are listed as unsourced and lead to Backtest", async ({ page }) => {
    await page.goto("/library/examples");
    const bk = content(page).getByRole("region", { name: "Burger King" });
    await expect(bk).toContainText("Unsourced");
    await expect(bk.locator("[data-example]")).toContainText("3,893.8");
    await expect(content(page).getByRole("note").filter({ hasText: "Examples, not evidence" })).toContainText("4 example deals");
    await page.getByRole("link", { name: "Open Backtest" }).click();
    await expect(page).toHaveURL(/\/backtest\/actuals$/);
  });

  test("with the library off, its tab is gone and every other screen and result works", async ({ page }) => {
    await page.goto("/deal/returns");
    const irr = await kpi(page, "IRR").textContent();
    const tabs = MODES.filter((m) => !m.optional).length;
    await expect(page.getByRole("navigation", { name: "Modes" }).getByRole("link")).toHaveCount(tabs + 1);

    await libraryOff(page);
    await page.goto("/deal/returns");
    await expect(kpi(page, "IRR")).toHaveText(irr ?? "");
    await expect(page.getByRole("navigation", { name: "Modes" }).getByRole("link")).toHaveCount(tabs);
    await expect(modeTab(page, "Library")).toHaveCount(0);

    // Every step of every other mode opens without an error screen
    for (const mode of MODES.filter((m) => !m.optional)) {
      for (const step of mode.steps) {
        await page.goto(`/${mode.slug}/${step.slug}`);
        await expect(page.locator(`nav[aria-label$="steps"] a[aria-current="page"]`)).toHaveCount(1);
        await expect(page.getByText("This screen hit an error")).toHaveCount(0);
      }
    }

    // A link to the library says it is off rather than failing
    await page.goto("/library/base-rates");
    await expect(content(page).getByRole("heading", { name: "The reference library is off" })).toBeVisible();
    await expect(content(page)).toContainText("Every other screen and every result works as usual.");
    // Search no longer finds it
    await page.keyboard.press("Control+k");
    await page.getByRole("combobox", { name: "Search screens" }).fill("base rates");
    await expect(page.getByRole("option")).toHaveCount(0);
  });

  test("an administrator's switch sends the choice and the screens follow it", async ({ page }) => {
    let state: State = { ...STATE, can_switch: true };
    const sent: unknown[] = [];
    await page.route("**/api/library", (r) => r.fulfill({ json: state }));
    await page.route("**/api/library/switch", async (r) => {
      sent.push(r.request().postDataJSON());
      state = { ...state, enabled: false, updated_at: "2026-10-07T12:00:00Z" };
      await r.fulfill({ json: state });
    });
    await page.goto("/library/coverage");
    const toggle = page.getByRole("switch", { name: "Show the library to everyone" });
    await expect(toggle).toHaveAttribute("aria-checked", "true");
    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-checked", "false");
    expect(sent).toEqual([{ enabled: false }]);
    await expect(page.getByTestId("library-switch-state")).toContainText("Hidden from every account.");
    // Off, an administrator still sees the tab, to turn it back on
    await expect(modeTab(page, "Library")).toBeVisible();
    await stepLink(page, "Base rates").click();
    await expect(content(page)).toContainText("It is hidden from every account.");
    await expect(content(page).getByRole("switch", { name: "Show the library to everyone" })).toBeVisible();
  });
});
