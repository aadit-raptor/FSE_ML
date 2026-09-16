import { expect, test as setup } from "@playwright/test";

import { asUser, SIGNED_IN_STATE } from "./helpers";

/**
 * Signs in once, so the other specs start where a real user does: signed in,
 * with a finished account (PLAN.md 1.4).
 *
 * These runs have no Clerk instance, so this is the development sign-in and
 * the API sees `dev:e2e` (api/auth.py accepts it only outside production and
 * only while Clerk is unconfigured). auth.spec.ts is the spec that drives the
 * sign-in and sign-out screens themselves.
 */
setup("sign in and finish the account", async ({ page, request }) => {
  const health = await (await request.get("/api/health")).json();
  expect(
    health.database?.configured,
    "the account needs a database: set DATABASE_URL (python -m db.local) before running e2e",
  ).toBe(true);

  await page.goto("/sign-in");
  await page.getByLabel("Development user").fill("e2e");
  await page.getByRole("button", { name: "Sign in" }).click();
  // Signed in: the app has left the sign-in page (straight to the deal, or to
  // the account screen when this account hasn't answered the questions yet)
  await expect(page).toHaveURL(/\/(deal|account)/);

  // Answer the four account questions (saving again is harmless, so this
  // works whether or not a previous run already did it)
  await page.goto("/account");
  await page.getByLabel("Country").selectOption("GB");
  await page.getByLabel("Currency").selectOption("GBP");
  await page.getByLabel("Number and date format").fill("en-GB");
  await page.getByLabel("Time zone").selectOption("Europe/London");
  const saved = page.waitForResponse(
    (r) => r.url().includes("/api/account") && r.request().method() === "POST" && r.ok(),
  );
  await page.getByRole("button", { name: /^Save/ }).click();
  await saved;

  // Settings live on the account (PLAN.md 1.5) and a local database keeps
  // them between runs: every run starts from the defaults
  const cleared = await page.request.put("/api/account/settings", { data: { settings: {} }, headers: await asUser(page) });
  expect(cleared.ok()).toBe(true);

  // Finished: the deal screens no longer send us back to the account
  await page.goto("/deal/inputs");
  await expect(page).toHaveURL(/\/deal\/inputs$/);
  await expect(page.getByRole("navigation", { name: "Modes" })).toBeVisible();

  await page.context().storageState({ path: SIGNED_IN_STATE });
});
