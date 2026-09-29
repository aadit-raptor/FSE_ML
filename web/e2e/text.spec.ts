import { readFileSync } from "node:fs";
import { join } from "node:path";

import { expect, test } from "@playwright/test";

import { dealSettled, kpi, signInAs } from "./helpers";

/**
 * Interface text comes from the translation files, and the layout runs the way
 * the account's locale does (PLAN.md 2.3b).
 *
 * The expected words are read from `web/messages/en.json` itself, not typed in
 * here, so a screen that stops reading the catalogue fails even if someone
 * writes the same English back into the component.
 *
 * Right to left is checked with an Arabic account. English is the only
 * catalogue (PLAN.md 7.8 adds languages), so the words stay English while the
 * layout mirrors -- which is exactly the case worth proving: the mirroring is
 * the part that breaks, and it is proved before any translation depends on it.
 */
test.use({ storageState: { cookies: [], origins: [] } });

const run = Date.now().toString(36);

/**
 * A figure with the direction marks removed. `Intl` wraps a per-cent sign and a
 * sign in U+200E for a right-to-left locale -- invisible, and exactly what
 * keeps "21.2%" from being reordered -- so an exact comparison has to allow
 * for them.
 */
const plain = (text: string | null) => (text ?? "").replace(/[\u200e\u200f]/g, "");

const en = JSON.parse(readFileSync(join(__dirname, "..", "messages", "en.json"), "utf8")) as {
  app: Record<string, string>;
  nav: Record<string, string>;
  deal: Record<string, string>;
  shell: Record<string, string>;
};

test.describe("Interface text", () => {
  test("screens, the step row and the tab title all read from the catalogue", async ({ page }) => {
    await signInAs(page, `text-en-${run}`, "en-GB");

    await page.goto("/deal/returns");
    await dealSettled(page);

    // The step row and the tab name the step with the catalogue's words
    const steps = page.locator('nav[aria-label$="steps"]');
    await expect(steps.getByRole("link", { name: en.nav.dealReturns, exact: true })).toBeVisible();
    await expect(page).toHaveTitle(`${en.nav.modeDeal} · ${en.nav.dealReturns} · ${en.app.brand}`);

    // A result tile, a rail group and a note, each by its catalogue entry
    await expect(page.getByRole("region", { name: en.deal.tileEquityBridge })).toBeVisible();
    await expect(page.getByRole("group", { name: en.deal.groupOperations })).toBeVisible();
    await expect(page.getByText(en.deal.sensitivityNote)).toBeVisible();

    // The mode tabs and the search box, above the routes
    await expect(page.getByRole("navigation", { name: en.shell.modes })).toBeVisible();
    await expect(page.getByRole("button", { name: en.shell.searchPlaceholderShort })).toBeVisible();

    // Still the model's own answer
    await expect(kpi(page, en.deal.kpiIrr)).toHaveText("21.2%");
  });

  test("a left-to-right account keeps the rail on the left", async ({ page }) => {
    await signInAs(page, `text-ltr-${run}`, "de-DE");
    await page.goto("/deal/returns");
    await dealSettled(page);

    await expect(page.locator("html")).toHaveAttribute("dir", "ltr");
    await expect(page.locator("html")).toHaveAttribute("lang", "de-DE");
    const rail = page.locator("#content aside");
    const box = (await rail.boundingBox())!;
    const width = page.viewportSize()!.width;
    expect(box.x).toBeLessThan(width / 2);
  });

  test("a right-to-left account mirrors the layout and keeps the figures readable", async ({ page }) => {
    await signInAs(page, `text-rtl-${run}`, "ar-EG");
    await page.goto("/deal/returns");
    await dealSettled(page);

    // The document itself runs right to left
    await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
    await expect(page.locator("html")).toHaveAttribute("lang", "ar-EG");

    // The input rail has moved to the right-hand side
    const rail = page.locator("#content aside");
    const box = (await rail.boundingBox())!;
    const width = page.viewportSize()!.width;
    expect(box.x).toBeGreaterThan(width / 2);
    expect(box.x + box.width).toBeGreaterThan(width - 40);

    // Figures stay left-to-right islands, so a leading minus or a trailing
    // per-cent sign isn't moved to the other end of the number (globals.css)
    const direction = (selector: string) => page.locator(selector).first().evaluate((el) => getComputedStyle(el).direction);
    expect(await direction("#content aside")).toBe("rtl");
    expect(await direction("[data-kpi]")).toBe("ltr");
    expect(await direction("#content table td")).toBe("ltr");

    // And they are the same figures, in Latin digits
    expect(plain(await kpi(page, en.deal.kpiIrr).textContent())).toBe("21.2%");
    expect(plain(await kpi(page, en.deal.kpiMoic).textContent())).toBe("2.61x");
    // The marks really are there: that is what makes the per-cent sign stay put
    expect(await kpi(page, en.deal.kpiIrr).textContent()).toMatch(/\u200e/);

    // A signed figure keeps its sign in front: the IRR tile's line against the
    // hurdle is "+1.2 pts vs 20% hurdle", and the "+" is a neutral character
    // that a right-to-left paragraph would otherwise push to the other end
    // Exactly "IRR": by default the name matches "IRR sensitivity" as well
    const hurdleLine = page.getByRole("region", { name: en.deal.kpiIrr, exact: true }).locator("p").last();
    expect(plain(await hurdleLine.textContent())).toMatch(/^\+\d/);
  });
});
