import { expect, type Page, test } from "@playwright/test";

import recorded from "./fixtures/multiples.json";
import { field, kpi, setField } from "./helpers";

/**
 * PLAN.md 5.4: entry and exit multiples by region. CI stores no industry
 * history, so the ranges are replayed from what the real endpoint gives for
 * the recorded multiples (fixtures/multiples.json, written by
 * `python -m tests.e2e_multiples`); tests/test_multiples.py checks those
 * answers. The first test uses the real endpoint, and "Use suggestion" is
 * proved by the deal model's own answer.
 */

type Case = keyof typeof recorded;

async function replay(page: Page, name: Case) {
  await page.route("**/api/ml/multiples", (r) => r.fulfill({ json: recorded[name] }));
}

function tile(page: Page) {
  return page.getByRole("region", { name: "Multiples", exact: true });
}

function row(page: Page, range: "entry" | "exit") {
  return tile(page).locator(`tr[data-range="${range}"]`);
}

test.describe("Multiples", () => {
  test("the deal's hold reaches the endpoint, which needs a country to read a region", async ({ page }) => {
    const answer = page.waitForResponse((r) => r.url().endsWith("/api/ml/multiples") && r.status() === 200);
    await page.goto("/deal/returns");
    const first = await (await answer).json();
    await expect(tile(page)).toContainText("Not enough data.");
    const next = page.waitForResponse(async (r) => {
      if (!r.url().endsWith("/api/ml/multiples") || r.status() !== 200) return false;
      return (await r.json()).hold !== first.hold;
    });
    await setField(page, "Hold", "7");
    expect((await (await next).json()).hold).toBe(7);
  });

  test("a German deal shows its region's ranges, and Use suggestion moves the IRR", async ({ page }) => {
    await replay(page, "de");
    await page.goto("/deal/returns");
    const r = recorded.de;
    await expect(tile(page).locator('[data-provenance="multiples"]')).toContainText(
      "Machinery: 15.0x in 2025, the latest of 210 companies in Developed Europe.",
    );
    await expect(row(page, "entry")).toContainText("2026");
    await expect(row(page, "entry")).toContainText("10.9x");
    await expect(row(page, "entry")).toContainText("15.4x");
    await expect(row(page, "entry")).toContainText("20.5x");
    await expect(row(page, "exit")).toContainText("2031");
    await expect(row(page, "exit")).toContainText("16.1x");
    await expect(tile(page).locator('[data-card="entry"]')).toContainText("Entry: tested on 672 industry-years in Europe");
    const regions = tile(page).getByRole("list", { name: "The industry in every region" });
    await expect(regions.getByRole("listitem")).toHaveCount(r.regions.length);
    await expect(tile(page).getByRole("list", { name: "Industries in the same sector" })).toContainText("Electrical Equipment");

    const irr = await kpi(page, "IRR").textContent();
    await tile(page).getByRole("button", { name: "Use suggestion" }).click();
    await expect(field(page, "Entry multiple")).toHaveValue("15.43");
    await expect(field(page, "Exit multiple")).toHaveValue("16.12");
    await expect(kpi(page, "IRR")).not.toHaveText(irr ?? "");
    await expect(tile(page).getByRole("button", { name: "Use suggestion" })).toBeDisabled();
  });

  test("with the library on, reference deals like it are listed; off, the ranges are the same", async ({ page }) => {
    await replay(page, "us");
    await page.goto("/deal/returns");
    const deals = tile(page).getByRole("list", { name: "Reference deals like it, with their entry multiples" });
    await expect(deals.getByRole("listitem")).toHaveCount(4);
    await expect(row(page, "entry")).toContainText("12.4x");

    await page.unroute("**/api/ml/multiples");
    await replay(page, "us_library_off");
    await page.reload();
    await expect(row(page, "entry")).toContainText("12.4x");
    await expect(tile(page).locator("[data-multiples-deals]")).toHaveCount(0);
  });

  test("a thin region says not enough data and never borrows the global group", async ({ page }) => {
    await replay(page, "thin");
    await page.goto("/deal/returns");
    await expect(tile(page).locator('[data-multiples-status="not_enough_data"]')).toContainText(
      "Fewer than 20 Shipbuilding & Marine companies in Australia, NZ and Canada",
    );
    await expect(tile(page).getByRole("button", { name: "Use suggestion" })).toHaveCount(0);
    await expect(tile(page).locator("table")).toHaveCount(0);
  });
});
