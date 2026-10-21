import { expect, test } from "@playwright/test";

import { dealSettled, kpi } from "./helpers";

/**
 * Sign-in, the account, and what a signed-out visitor can reach (PLAN.md 1.4).
 *
 * This spec starts signed out -- it drops the shared signed-in state the
 * other specs use -- and checks the doors as well as the rooms: no token, no
 * model output, whichever way you ask.
 */
test.use({ storageState: { cookies: [], origins: [] } });

test("a signed-out visitor is sent to sign-in and sees no deal", async ({ page }) => {
  await page.goto("/deal/returns");
  await expect(page).toHaveURL(/\/sign-in/);
  await expect(page.getByRole("heading", { name: "Sign in" })).toBeVisible();
  await expect(kpi(page, "IRR")).toHaveCount(0);
  // The mode tabs aren't there to click: there is nothing behind them yet
  await expect(page.getByRole("navigation", { name: "Modes" })).toHaveCount(0);
});

test("the API refuses a call without a token, whoever asks", async ({ request }) => {
  const resp = await request.post("/api/deal/run", { data: {}, failOnStatusCode: false });
  expect(resp.status()).toBe(401);
  expect(await resp.text()).not.toContain("irr");
  // The health check stays open: the status bar and the uptime monitor use it
  expect((await request.get("/api/health")).status()).toBe(200);
});

test("the web app's health route answers signed out, for the uptime monitor", async ({ request }) => {
  const resp = await request.get("/healthz", { maxRedirects: 0 });
  expect(resp.status()).toBe(200);
  // The keyword ops/betterstack.py's web monitor looks for
  expect(await resp.text()).toContain('"service":"Variater web"');
  // Pages stay closed: the same kind of request to a screen doesn't get the app
  const page = await request.get("/deal/returns", { maxRedirects: 0 });
  expect(page.status()).not.toBe(200);
});

test("signing in reaches the deal screens, and signing out closes them again", async ({ page }) => {
  await page.goto("/deal/returns");
  await expect(page).toHaveURL(/\/sign-in/);

  await page.getByLabel("Development user").fill("e2e");
  await page.getByRole("button", { name: "Sign in" }).click();

  // Straight to the deal it asked for, with real model output on screen
  await expect(page).toHaveURL(/\/deal\/returns/);
  await expect(kpi(page, "IRR")).toHaveText("21.2%");
  await expect(kpi(page, "MOIC")).toHaveText("2.61x");

  await page.goto("/account");
  await expect(page.locator('[data-account="subject"]')).toHaveText("dev:e2e");
  await page.getByRole("button", { name: "Sign out" }).click();

  await expect(page).toHaveURL(/\/sign-in/);
  await page.goto("/deal/returns");
  await expect(page).toHaveURL(/\/sign-in/);
});

test("every sign-in lands on the launcher; an open session carries on where it was", async ({ page }) => {
  // Opening the site signed out asks for no particular screen
  await page.goto("/");
  await expect(page).toHaveURL(/\/sign-in$/);
  await page.getByLabel("Development user").fill("e2e");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/start$/);
  await expect(page.getByRole("heading", { name: "Choose a workspace" })).toBeVisible();

  // Work somewhere, then come back to the site still signed in: straight back there
  await page.getByRole("link", { name: "Open Equity research" }).click();
  await expect(page).toHaveURL(/\/forecast\/historicals$/);
  await page.goto("/");
  await expect(page).toHaveURL(/\/forecast\/historicals$/);

  // Signing out and in again is a fresh start: the launcher, not the last screen
  await page.goto("/account");
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page).toHaveURL(/\/sign-in/);
  await page.getByLabel("Development user").fill("e2e");
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/start$/);
});

test("a new account answers four questions before the deal screens open", async ({ page }) => {
  // A name of its own, so this account has never set a profile
  const name = `new${Date.now().toString(36)}`;
  await page.goto("/sign-in");
  await page.getByLabel("Development user").fill(name);
  await page.getByRole("button", { name: "Sign in" }).click();

  await expect(page).toHaveURL(/\/account/);
  await expect(page.getByRole("heading", { name: "Finish setting up your account" })).toBeVisible();
  // Every other screen leads back here until the questions are answered
  await page.goto("/deal/returns");
  await expect(page).toHaveURL(/\/account/);

  await page.getByLabel("Country").selectOption("JP");
  await page.getByLabel("Currency").selectOption("JPY");
  await page.getByLabel("Number and date format").fill("ja-JP");
  await page.getByLabel("Time zone").selectOption("Asia/Tokyo");
  // The answers are in use on the screen before they are even saved
  await expect(page.locator('[data-account="preview"]')).toContainText("￥1,234,568");

  await page.getByRole("button", { name: "Save and start" }).click();
  // A finished account starts at the launcher, then the chosen workspace
  await expect(page).toHaveURL(/\/start$/);
  await page.getByRole("link", { name: "Open LBO" }).click();
  await expect(page).toHaveURL(/\/deal\/inputs/);
  await dealSettled(page);

  // Saved, not just held in the page: it survives a reload
  await page.goto("/account");
  await expect(page.getByLabel("Currency")).toHaveValue("JPY");
  await expect(page.getByLabel("Time zone")).toHaveValue("Asia/Tokyo");
  await expect(page.getByRole("heading", { name: "Account" })).toBeVisible();
});

test("one account never sees another's answers", async ({ page }) => {
  const first = `alpha${Date.now().toString(36)}`;
  const second = `beta${Date.now().toString(36)}`;

  await page.goto("/sign-in");
  await page.getByLabel("Development user").fill(first);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/account/);
  await page.getByLabel("Country").selectOption("BR");
  await page.getByLabel("Currency").selectOption("BRL");
  await page.getByLabel("Number and date format").fill("pt-BR");
  await page.getByLabel("Time zone").selectOption("America/Sao_Paulo");
  await page.getByRole("button", { name: "Save and start" }).click();
  await expect(page).toHaveURL(/\/start$/);

  await page.goto("/account");
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(page).toHaveURL(/\/sign-in/);
  await page.getByLabel("Development user").fill(second);
  await page.getByRole("button", { name: "Sign in" }).click();

  // The second account starts empty, not with Brazil's answers
  await expect(page).toHaveURL(/\/account/);
  await expect(page.locator('[data-account="subject"]')).toHaveText(`dev:${second}`);
  await expect(page.getByLabel("Currency")).toHaveValue("");
});

// Google's consent screen links to it, so it must open for anyone (PLAN.md 0.2)
test("the privacy policy opens signed out, and says what is stored", async ({ page }) => {
  const resp = await page.goto("/privacy");
  expect(resp?.status()).toBe(200);
  await expect(page).toHaveURL(/\/privacy$/);
  await expect(page.getByRole("heading", { name: "Privacy policy" })).toBeVisible();
  await expect(page.getByText("never your name or email address", { exact: false })).toBeVisible();
  // how to have data removed: Microsoft's store requires it, and it once rendered without its link
  await expect(page.getByRole("link", { name: "github.com/aadit-raptor/FSE_ML/issues" })).toHaveAttribute(
    "href",
    "https://github.com/aadit-raptor/FSE_ML/issues",
  );
  await expect(page.getByRole("navigation", { name: "Modes" })).toHaveCount(0);
});

// The Excel add-in's manifest and Microsoft's store listing link to it, so it must open for anyone
test("the support page opens signed out, with a web address for help and the account it needs", async ({ page }) => {
  const resp = await page.goto("/support");
  expect(resp?.status()).toBe(200);
  expect(resp?.headers()["content-security-policy"]).toContain("frame-ancestors 'none'");
  await expect(page).toHaveURL(/\/support$/);
  await expect(page.getByRole("heading", { name: "Support", exact: true })).toBeVisible();
  // the store refuses an email address: help is an https page
  await expect(page.getByRole("link", { name: "github.com/aadit-raptor/FSE_ML/issues" })).toHaveAttribute(
    "href",
    "https://github.com/aadit-raptor/FSE_ML/issues",
  );
  await expect(page.locator("article")).not.toContainText("@");
  await expect(page.getByRole("heading", { name: "You need a Variater account" })).toBeVisible();
  await expect(page.getByText("The Excel add-in brings that work into your workbook", { exact: false })).toBeVisible();
  await expect(page.getByRole("link", { name: "privacy policy" })).toHaveAttribute("href", "/privacy");
  await expect(page.getByRole("navigation", { name: "Modes" })).toHaveCount(0);
});
