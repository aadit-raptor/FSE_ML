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
  expect(await resp.text()).toContain('"service":"FSE/ML web"');
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
  await expect(page).toHaveURL(/\/deal\/inputs/);

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
