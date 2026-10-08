import { expect, type Page, test } from "@playwright/test";

import { modeTab, stepLink } from "./helpers";

const workspaces = (page: Page) => page.getByRole("navigation", { name: "Workspaces" });

test("home carries on at the last screen and the API is reachable through the proxy", async ({ page }) => {
  await page.goto("/backtest/years");
  await expect(page.getByRole("navigation", { name: "Modes" })).toBeVisible();
  await page.goto("/");
  await expect(page).toHaveURL(/\/backtest\/years$/);
  await expect(page.getByRole("status").filter({ hasText: /^API ok · v/ })).toBeVisible();
});

test("home offers the launcher when this browser remembers no screen", async ({ page }) => {
  await page.goto("/start");
  await page.evaluate(() => window.localStorage.clear());
  await page.goto("/");
  await expect(page).toHaveURL(/\/start$/);
});

test("the launcher opens each workspace, and the brand mark leads back to it", async ({ page }) => {
  await page.goto("/start");
  await expect(page.getByRole("heading", { name: "Choose a workspace" })).toBeVisible();
  // No mode tabs here: choosing the workspace comes first
  await expect(page.getByRole("navigation", { name: "Modes" }).getByRole("link")).toHaveCount(0);

  const lbo = page.locator('[data-workspace="lbo"]');
  await expect(lbo.getByRole("navigation", { name: "LBO screens" }).getByRole("link")).toHaveText(
    ["Deal", "Monte Carlo", "Backtest", "Settings", "Library"],
  );
  await lbo.getByRole("link", { name: "Backtest" }).click();
  await expect(page).toHaveURL(/\/backtest\/actuals$/);

  await page.getByRole("link", { name: "Variater" }).click();
  await expect(page).toHaveURL(/\/start$/);
  await page.getByRole("link", { name: "Open Equity research" }).click();
  await expect(page).toHaveURL(/\/forecast\/historicals$/);
});

test("each workspace shows only its own mode tabs, and the switcher moves between them", async ({ page }) => {
  await page.goto("/deal/inputs");
  const tabs = page.getByRole("navigation", { name: "Modes" }).getByRole("link");
  await expect(tabs).toHaveText([/^Deal/, /^Monte Carlo/, /^Backtest/, /^Settings/, /^Library/]);
  await expect(workspaces(page).getByRole("link", { name: "LBO" })).toHaveAttribute("aria-current", "true");

  await workspaces(page).getByRole("link", { name: "Equity research" }).click();
  await expect(page).toHaveURL(/\/forecast\/historicals$/);
  await expect(tabs).toHaveText([/^Forecast/]);
  await expect(workspaces(page).getByRole("link", { name: "Equity research" })).toHaveAttribute("aria-current", "true");

  // The account screen keeps the tabs of the workspace last worked in, even opened afresh
  await page.goto("/account");
  await expect(page.getByRole("heading", { name: "Account" })).toBeVisible();
  await expect(tabs).toHaveText([/^Forecast/]);
});

test("mode tabs and step links change the screen", async ({ page }) => {
  await page.goto("/deal/inputs");
  await modeTab(page, "Backtest").click();
  await expect(page).toHaveURL(/\/backtest\/actuals$/);
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
  // Settings is the LBO workspace's fourth mode
  await page.keyboard.press("Alt+4");
  await expect(page).toHaveURL(/\/settings\/deal$/);
});

test("keyboard: Alt+number counts within the current workspace", async ({ page }) => {
  await page.goto("/forecast/statements");
  await page.locator("main").click({ position: { x: 5, y: 5 } });
  // Equity research has one mode: Alt+2 leads nowhere, Alt+1 is Forecast
  await page.keyboard.press("Alt+2");
  await expect(page).toHaveURL(/\/forecast\/statements$/);
  await page.keyboard.press("Alt+1");
  await expect(page).toHaveURL(/\/forecast\/historicals$/);
});

test("Ctrl K searches every workspace, not just the current one", async ({ page }) => {
  await page.goto("/deal/inputs");
  // The shortcut listens once the signed-in shell is up
  await expect(page.getByRole("navigation", { name: "Modes" })).toBeVisible();
  await page.keyboard.press("Control+k");
  const box = page.getByRole("combobox", { name: "Search screens" });
  // The workspace's name finds its screens, and each result names its workspace
  await box.fill("equity research");
  await expect(page.getByRole("option")).toHaveCount(5);
  await expect(page.getByRole("option").first()).toContainText("Equity research");
  await page.getByRole("option", { name: /Forecast\s*Statements/ }).click();
  await expect(page).toHaveURL(/\/forecast\/statements$/);
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
