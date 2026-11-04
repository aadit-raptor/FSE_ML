import { expect, test } from "@playwright/test";

import { asUser, signInAs } from "./helpers";

/**
 * The developer API's screen (PLAN.md 7.7), through the real API and database: a key made on the screen is shown
 * once, runs a deal through /api/v1 as its maker, and is refused the moment it is revoked; a webhook's address
 * must be https on the public internet, and its secret is shown once. The keys, scopes, limits, signatures and a
 * script receiving a webhook are tests/test_public_api.py.
 */
const run = Date.now().toString(36);

test.use({ storageState: { cookies: [], origins: [] } });

test("a key made on the screen runs a deal through /api/v1 and stops when revoked", async ({ page }) => {
  await signInAs(page, `developer-${run}`, "en-GB");
  await page.goto("/account");
  await page.locator('[data-account="developer"]').click();
  await expect(page).toHaveURL(/\/account\/developer$/);
  await expect(page.locator('[data-guide="api-keys"]')).toBeVisible();

  await page.getByLabel("New key's name").fill("pricing script");
  await page.getByRole("button", { name: "Make key" }).click();
  const shown = page.locator('[data-developer="key"]');
  await expect(shown).toHaveText(/^vk_[A-Za-z0-9_-]{43}$/);
  const key = (await shown.textContent()) ?? "";
  const row = page.locator('[data-key="pricing script"]');
  await expect(row).toContainText(`…${key.slice(-4)}`);
  await expect(row).toContainText("not used yet");

  // The key runs a deal, and gets exactly what the site's own sign-in gets
  const deal = { inputs: { ebitda: 80, entry_mult: 8, exit_mult: 9, hold: 5 } };
  const withKey = await page.request.post("/api/v1/deal/run", { data: deal, headers: { Authorization: `Bearer ${key}` } });
  expect(withKey.status()).toBe(200);
  expect(withKey.headers()["variater-api-version"]).toBe("1");
  const onSite = await page.request.post("/api/deal/run", { data: deal, headers: await asUser(page) });
  expect((await withKey.json()).returns.irr).toBe((await onSite.json()).returns.irr);
  // ... and nothing outside version 1
  expect((await page.request.get("/api/deals", { headers: { Authorization: `Bearer ${key}` } })).status()).toBe(403);

  // Once shown, the key is gone from the screen; the list says it was used
  await page.getByRole("button", { name: "Done" }).click();
  await expect(shown).toHaveCount(0);
  await page.reload();
  await expect(page.locator('[data-key="pricing script"]')).toContainText("last used");
  await expect(page.locator('[data-developer="key"]')).toHaveCount(0);

  await page.locator('[data-key="pricing script"]').getByRole("button", { name: "Revoke" }).click();
  await expect(page.locator('[data-key="pricing script"]')).toHaveCount(0);
  const after = await page.request.get("/api/v1/me", { headers: { Authorization: `Bearer ${key}` } });
  expect(after.status()).toBe(401);
});

test("a webhook must be https on the public internet, and its secret is shown once", async ({ page }) => {
  await signInAs(page, `hooks-${run}`, "en-GB");
  await page.goto("/account/developer");
  await expect(page.locator('[data-guide="webhooks"]')).toBeVisible();

  await page.getByLabel("New webhook's address").fill("http://hooks.example.com/variater");
  await page.getByRole("button", { name: "Add webhook" }).click();
  await expect(page.locator("#content").getByRole("alert")).toContainText("must start with https");
  await page.getByLabel("New webhook's address").fill("https://10.0.0.8/variater");
  await page.getByRole("button", { name: "Add webhook" }).click();
  await expect(page.locator("#content").getByRole("alert")).toContainText("public internet");

  await page.getByLabel("New webhook's address").fill("https://hooks.example.com/variater");
  await page.getByRole("button", { name: "Add webhook" }).click();
  await expect(page.locator('[data-developer="secret"]')).toHaveText(/^whsec_[A-Za-z0-9+/=]{44}$/);
  const hook = page.locator('[data-webhook="https://hooks.example.com/variater"]');
  await expect(hook.locator("[data-webhook-state]")).toHaveAttribute("data-webhook-state", "none");
  await expect(page.locator("#content").getByRole("alert")).toHaveCount(0);

  await hook.getByRole("button", { name: "Remove" }).click();
  await expect(hook).toHaveCount(0);
  await expect(page.locator('[data-guide="webhooks"]')).toBeVisible();
});
