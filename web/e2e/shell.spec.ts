import { expect, test } from "@playwright/test";

import { modeTab, stepLink } from "./helpers";

test("home opens deal inputs and the API is reachable through the proxy", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/deal\/inputs$/);
  await expect(page.getByRole("status").filter({ hasText: /^API ok · v/ })).toBeVisible();
});

test("mode tabs and step links change the screen", async ({ page }) => {
  await page.goto("/deal/inputs");
  await modeTab(page, "Backtest").click();
  await expect(page).toHaveURL(/\/backtest\/predicted$/);
  await expect(modeTab(page, "Backtest")).toHaveAttribute("aria-current", "page");
  await stepLink(page, "Year by year").click();
  await expect(page).toHaveURL(/\/backtest\/years$/);
});

test("keyboard: Alt+number switches mode, ] moves to the next step", async ({ page }) => {
  await page.goto("/deal/inputs");
  await page.locator("main").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("Alt+2");
  await expect(page).toHaveURL(/\/monte-carlo\/distribution$/);
  await page.keyboard.press("]");
  await expect(page).toHaveURL(/\/monte-carlo\/scenarios$/);
});

test("Ctrl K search filters screens and Enter opens the first match", async ({ page }) => {
  await page.goto("/deal/inputs");
  await page.keyboard.press("Control+k");
  const box = page.getByRole("combobox", { name: "Search screens" });
  await box.fill("correlation");
  await expect(page.getByRole("option")).toHaveCount(2);
  await box.press("Enter");
  await expect(page).toHaveURL(/\/monte-carlo\/drivers$/);
  await expect(page.getByRole("dialog")).toHaveCount(0);
});

test("unknown addresses show the not-found screen", async ({ page }) => {
  await page.goto("/deal/nope");
  await expect(page.getByRole("heading", { name: "Screen not found" })).toBeVisible();
});
