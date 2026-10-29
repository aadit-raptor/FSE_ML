import { expect, type Page, test } from "@playwright/test";

import { MODES } from "../src/lib/nav";

import { dealSettled, replayBenchmarks, signInAs } from "./helpers";

/**
 * Every input has a tip (PLAN.md 7.1): on every screen of every mode, each form control a person can
 * reach -- inputs, selects, text areas and switches -- is described by a tip whose words come from
 * the catalogue (`tips` in web/messages/en.json), so a screen reader reads it on focus, and hovering
 * the control's label shows it. tests/test_onboarding.py holds the source to the same rule for
 * controls only some states show.
 */

/** The controls on the page without a tip: a description naming a `[data-tip]` element with words. */
async function untipped(page: Page): Promise<string[]> {
  return page.evaluate(() => {
    const controls = [...document.querySelectorAll<HTMLElement>('input:not([type="hidden"]), select, textarea, [role="switch"]')];
    return controls
      .filter((el) => {
        // Hidden ones a person can't reach (a file picker behind its button is reached through the button's label)
        const style = window.getComputedStyle(el);
        if (style.display === "none" || style.visibility === "hidden") return false;
        if (el.closest("[hidden], [aria-hidden='true']")) return false;
        const ids = (el.getAttribute("aria-describedby") ?? "").split(/\s+/).filter(Boolean);
        const tips = ids.map((id) => document.getElementById(id)).filter((t) => t?.hasAttribute("data-tip"));
        return !tips.some((t) => (t?.textContent ?? "").trim().split(/\s+/).length >= 5);
      })
      .map((el) => `${el.tagName.toLowerCase()} ${el.getAttribute("aria-label") ?? el.id ?? ""} ${el.getAttribute("name") ?? ""}`.trim());
  });
}

async function controlCount(page: Page): Promise<number> {
  return page.locator('input:not([type="hidden"]), select, textarea, [role="switch"]').count();
}

test.describe("Every input has a tip", () => {
  for (const mode of MODES) {
    test(`on every ${mode.slug} screen`, async ({ page }) => {
      test.setTimeout(120_000);
      await replayBenchmarks(page);
      for (const step of mode.steps) {
        await page.goto(`/${mode.slug}/${step.slug}`);
        await page.waitForLoadState("networkidle");
        expect(await untipped(page), `/${mode.slug}/${step.slug}`).toEqual([]);
      }
    });
  }

  test("on the launcher, the account, help and the guided deal", async ({ page }) => {
    for (const path of ["/start", "/account", "/help", "/deal/guided"]) {
      await page.goto(path);
      await page.waitForLoadState("networkidle");
      expect(await untipped(page), path).toEqual([]);
    }
  });

  test("a tip shows on hover over its label and is read as the input's description", async ({ page }) => {
    await page.goto("/deal/inputs");
    await dealSettled(page);
    const hold = page.getByLabel("Hold", { exact: true });
    await expect(hold).toHaveAccessibleDescription(/years the sponsor owns the company/i);
    await page.locator('label[for]').filter({ hasText: /^Hold$/ }).getByText("Hold", { exact: true }).hover();
    await expect(page.locator('[data-tip-popup="field_hold"]')).toContainText(/years the sponsor owns the company/i);
  });
});

test.describe("Every input has a tip, in the states only some deals reach", () => {
  // An account of its own: the edits below must not reach the shared account's open deal
  test.use({ storageState: { cookies: [], origins: [] } });

  test("tax rules, leases and listed facilities", async ({ page }) => {
    await signInAs(page, `tips-${Date.now().toString(36)}`, "en-GB");
    await page.goto("/deal/inputs");
    await dealSettled(page);
    await page.getByRole("combobox", { name: "Country preset", exact: true }).selectOption({ label: "Germany" });
    await page.getByRole("combobox", { name: "Accounting standard", exact: true }).selectOption("ifrs");
    await dealSettled(page);
    expect(await untipped(page)).toEqual([]);

    // The facility editor: each facility's own name, kind, rate and sweep controls
    await page.goto("/deal/debt");
    await dealSettled(page);
    const before = await controlCount(page);
    await page.getByRole("button", { name: "Use explicit tranches" }).click();
    await dealSettled(page);
    // Open the first facility's editor
    await page.getByRole("group", { name: "Capital structure" }).getByRole("button", { name: /^Senior Term Loan/ }).click();
    await expect(page.getByRole("combobox", { name: "Kind", exact: true })).toBeVisible();
    expect(await controlCount(page)).toBeGreaterThan(before);
    expect(await untipped(page)).toEqual([]);
  });
});
