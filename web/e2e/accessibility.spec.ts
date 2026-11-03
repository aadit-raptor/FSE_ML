import AxeBuilder from "@axe-core/playwright";
import { expect, type Page, test } from "@playwright/test";

import { MODES } from "../src/lib/nav";

import { dealSettled, kpi, replayBenchmarks, setField, signInAs, simulationSettled } from "./helpers";

/**
 * Phones, tablets and accessibility (PLAN.md 7.6).
 *
 * Every screen of every mode is checked by axe-core against WCAG 2.1 A and AA at a desktop, a tablet
 * and a phone size, and fails on any serious or critical finding. At the smaller sizes no screen may
 * scroll sideways: a wide table scrolls inside its own box, never the page or the pane around it. The
 * key journeys are then walked at phone size with touch, and the layout is checked right to left.
 */

const SIZES = {
  desktop: { width: 1440, height: 900 },
  tablet: { width: 820, height: 1180 },
  phone: { width: 390, height: 844 },
} as const;

/** Findings that block: axe's own "serious" and "critical". Moderate and minor ones are advice. */
async function serious(page: Page): Promise<string[]> {
  const result = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "best-practice"]).analyze();
  return result.violations
    .filter((v) => v.impact === "serious" || v.impact === "critical")
    .map((v) => `${v.id} (${v.impact}): ${v.nodes.slice(0, 3).map((n) => n.target.join(" ")).join(" | ")}`);
}

/**
 * Anything wider than its box that is not meant to scroll sideways: the page itself, and every pane
 * that scrolls (they scroll down, so content wider than them would scroll sideways too). A box marked
 * `overflow-x-auto` is a table that scrolls on its own, which is how a wide table fits a phone.
 */
async function sideways(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    window.scrollTo(1000, 0);
    const found: string[] = window.scrollX > 0 ? [`the page scrolls sideways by ${window.scrollX}px`] : [];
    window.scrollTo(0, 0);
    const boxes = [document.querySelector("header"), document.querySelector("#content"), ...document.querySelectorAll("#content *")];
    for (const el of boxes as (HTMLElement | null)[]) {
      if (!el || el.clientWidth === 0 || el.classList.contains("overflow-x-auto")) continue;
      const style = getComputedStyle(el);
      if (!["auto", "scroll"].includes(style.overflowX)) continue;
      if (el.scrollWidth > el.clientWidth + 1) found.push(`${el.tagName.toLowerCase()}.${String(el.className).slice(0, 60)}: ${el.scrollWidth} > ${el.clientWidth}`);
    }
    return found;
  });
}

/**
 * The deal has its answer. On a phone the rail, which says so, may be the hidden view, so its status
 * is looked for whether shown or not.
 */
async function phoneDealSettled(page: Page) {
  await expect(page.getByRole("status", { includeHidden: true }).filter({ hasText: /^Up to date/ })).toBeAttached();
  await expect(page.getByRole("status", { includeHidden: true }).filter({ hasText: /^Updating/ })).toHaveCount(0);
}

/** The screens outside the modes. */
const OTHER = ["/start", "/account", "/help", "/help/whats-new", "/teams"];

/**
 * One screen at the three sizes. It is opened once and the window resized, which reflows it as a
 * phone or tablet would draw it: axe at desktop and phone size (the tablet's layout holds the same
 * elements as one or the other), sideways scrolling at tablet and phone size, and on a phone both of
 * its views, the results and the inputs.
 */
async function everySize(page: Page, path: string) {
  await page.setViewportSize(SIZES.desktop);
  await page.goto(path);
  await page.waitForLoadState("networkidle");
  expect(await serious(page), `${path} at desktop size`).toEqual([]);
  await page.setViewportSize(SIZES.tablet);
  expect(await sideways(page), `${path} at tablet size`).toEqual([]);
  await page.setViewportSize(SIZES.phone);
  expect(await serious(page), `${path} at phone size`).toEqual([]);
  expect(await sideways(page), `${path} at phone size`).toEqual([]);
  const inputs = page.locator("[data-screen-view] button").first();
  if (await inputs.isVisible()) {
    await inputs.click();
    expect(await serious(page), `${path} at phone size, inputs`).toEqual([]);
    expect(await sideways(page), `${path} at phone size, inputs`).toEqual([]);
  }
}

test.describe("Accessibility and small screens", () => {
  for (const mode of MODES) {
    test(`every ${mode.slug} screen`, async ({ page }) => {
      test.setTimeout(240_000);
      await replayBenchmarks(page);
      for (const step of mode.steps) await everySize(page, `/${mode.slug}/${step.slug}`);
    });
  }

  test("the launcher, account, help and teams", async ({ page }) => {
    test.setTimeout(120_000);
    for (const path of OTHER) await everySize(page, path);
  });
});

test.describe("Accessibility, the shell", () => {
  test("the first tab stop skips to the screen, which has a heading", async ({ page }) => {
    await page.goto("/deal/returns");
    await dealSettled(page);
    // From the top of the page, as on arriving: nothing has focus
    await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
    await page.keyboard.press("Tab");
    const skip = page.getByRole("link", { name: "Skip to content" });
    await expect(skip).toBeFocused();
    await expect(skip).toBeVisible();
    await page.keyboard.press("Enter");
    await expect(page.locator("#content")).toBeFocused();
    await expect(page.getByRole("heading", { level: 1, name: "Deal: Returns", exact: true })).toBeAttached();
  });

  test("a wide table is reached and scrolled from the keyboard", async ({ page }) => {
    await page.setViewportSize(SIZES.phone);
    await page.goto("/forecast/assumptions");
    await page.waitForLoadState("networkidle");
    const boxes = page.locator('#content .overflow-x-auto[tabindex="0"]');
    await expect(boxes.first()).toBeAttached();
    const wide = await boxes.evaluateAll((els) => els.findIndex((el) => el.scrollWidth > el.clientWidth + 1));
    expect(wide).toBeGreaterThanOrEqual(0);
    const table = boxes.nth(wide);
    await table.focus();
    await expect(table).toBeFocused();
    await page.keyboard.press("ArrowRight");
    await expect.poll(() => table.evaluate((el) => el.scrollLeft)).toBeGreaterThan(0);
  });
});

test.describe("Key journeys on a phone", () => {
  // A phone: touch, its width, and an account of its own so the edits stay out of the shared one
  test.use({ viewport: SIZES.phone, hasTouch: true, isMobile: true, storageState: { cookies: [], origins: [] } });

  test("edit a deal, see its returns, then run the simulation", async ({ page }) => {
    await signInAs(page, `phone-${Date.now().toString(36)}`, "en-GB");
    await page.goto("/deal/returns");
    await phoneDealSettled(page);
    const irr = kpi(page, "IRR");
    await expect(irr).toHaveText("21.2%");
    await expect(irr).toBeInViewport();

    // The inputs are a tap away, the results another
    await page.getByRole("button", { name: "Inputs", exact: true }).tap();
    await expect(page.locator("#content aside")).toBeVisible();
    await expect(irr).toBeHidden();
    await setField(page, "Exit multiple", "13");
    await page.getByRole("button", { name: "Results", exact: true }).tap();
    await phoneDealSettled(page);
    await expect(irr).not.toHaveText("21.2%");
    const edited = await irr.textContent();

    // The mode tabs and step row are reached by touch on their own row
    await page.getByRole("navigation", { name: "Modes" }).getByRole("link", { name: /^Monte Carlo/ }).tap();
    await expect(page).toHaveURL(/\/monte-carlo\/distribution$/);
    await simulationSettled(page);
    await expect(kpi(page, "Mean IRR")).toBeInViewport();

    // Back on the deal the edit is still there
    await page.getByRole("navigation", { name: "Modes" }).getByRole("link", { name: /^Deal/ }).tap();
    await page.locator('nav[aria-label$="steps"]').getByRole("link", { name: "Returns", exact: true }).tap();
    await phoneDealSettled(page);
    await expect(irr).toHaveText(edited!);
    expect(await sideways(page)).toEqual([]);
  });

  test("the top bar fits, and search, help and the account are reached by their icons", async ({ page }) => {
    await signInAs(page, `phone-bar-${Date.now().toString(36)}`, "en-GB");
    await page.goto("/deal/inputs");
    await phoneDealSettled(page);
    const header = page.locator("header").first();
    expect(await header.evaluate((el) => el.scrollWidth <= el.clientWidth + 1)).toBe(true);

    await page.getByRole("button", { name: "Search screens and actions" }).tap();
    await expect(page.getByRole("dialog")).toBeVisible();
    await page.keyboard.press("Escape");

    await page.getByRole("link", { name: "Help" }).tap();
    await expect(page).toHaveURL(/\/help$/);
    await page.getByRole("link", { name: "3. The deal model" }).first().tap();
    await expect(page.getByRole("heading", { level: 1 })).toHaveText("3. The deal model");
    expect(await sideways(page)).toEqual([]);

    await page.getByRole("link", { name: /^phone-bar/ }).tap();
    await expect(page).toHaveURL(/\/account$/);
    await expect(page.getByRole("heading", { name: "Account" })).toBeVisible();
  });

  test("a right-to-left account on a phone: mirrored, readable, and nothing serious for axe", async ({ page }) => {
    await signInAs(page, `phone-rtl-${Date.now().toString(36)}`, "ar-EG");
    await page.goto("/deal/returns");
    await phoneDealSettled(page);
    await expect(page.locator("html")).toHaveAttribute("dir", "rtl");
    expect(await serious(page)).toEqual([]);
    expect(await sideways(page)).toEqual([]);
    // The brand sits at the start of the line, which is the right
    const brand = (await page.getByRole("link", { name: /variater/i }).first().boundingBox())!;
    expect(brand.x).toBeGreaterThan(SIZES.phone.width / 2);
    await page.getByRole("button", { name: "Inputs", exact: true }).tap();
    expect(await serious(page)).toEqual([]);
    expect(await sideways(page)).toEqual([]);
  });
});
