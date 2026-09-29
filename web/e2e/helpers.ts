import { expect, type Locator, type Page } from "@playwright/test";

/** A headline number by its tile title (Kpi renders data-kpi). */
export const kpi = (page: Page, title: string): Locator => page.locator(`[data-kpi="${title}"]`);

/** Rail input by its visible label, optionally inside a titled group. */
export function field(page: Page, label: string, group?: string): Locator {
  const scope = group ? page.getByRole("group", { name: group }) : page;
  return scope.getByLabel(label, { exact: true });
}

/** Replace an input's value the way a user would. */
export async function setField(page: Page, label: string, value: string, group?: string) {
  const input = field(page, label, group);
  await input.fill(value);
  await input.blur();
}

export const modeTab = (page: Page, name: string) => page.getByRole("navigation", { name: "Modes" }).getByRole("link", { name: new RegExp(`^${name}`) });

export const stepLink = (page: Page, name: string) => page.locator('nav[aria-label$="steps"]').getByRole("link", { name, exact: true });

/** A promise for the next successful save of the account's settings. */
export function settingsSaved(page: Page) {
  return page.waitForResponse((r) => r.url().endsWith("/api/account/settings") && r.request().method() === "PUT" && r.ok());
}

/**
 * Settings > Reset all, waiting until the account has the defaults again.
 * Settings are saved to the account (PLAN.md 1.5), so a test that closed its
 * page before the save went out would hand its settings to the next test.
 */
export async function resetAllSettings(page: Page) {
  const saved = settingsSaved(page);
  await page.getByRole("button", { name: "Reset all" }).click();
  await saved;
}

/** Wait for a deal result, and for any pending rerun to finish. */
export async function dealSettled(page: Page) {
  await expect(page.getByRole("status").filter({ hasText: /^Up to date/ })).toBeVisible();
}

/** Monte Carlo has a result and isn't running. */
export async function simulationSettled(page: Page) {
  await expect(kpi(page, "Mean IRR").or(kpi(page, "Base"))).toBeVisible({ timeout: 45_000 });
  await expect(page.getByText("Running simulation")).toHaveCount(0, { timeout: 45_000 });
}

export type Grouping = "locale" | "thousands" | "lakh";

/**
 * Sign in as a development user of this account's own, and give it a locale.
 * The locale chooses the number and date format (PLAN.md 2.3a) and the
 * interface language and its direction (PLAN.md 2.3b), so a spec that wants
 * one of those signs in with its own user and never touches the shared e2e
 * account.
 */
export async function signInAs(page: Page, user: string, locale: string, grouping: Grouping = "locale") {
  await page.goto("/sign-in");
  await page.getByLabel("Development user").fill(user);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page).toHaveURL(/\/(deal|account)/);
  await page.goto("/account");
  await page.getByLabel("Country").selectOption(locale.slice(-2));
  await page.getByLabel("Currency").selectOption("USD");
  await page.getByLabel("Number and date format").fill(locale);
  await page.getByLabel("Digit grouping").selectOption(grouping);
  await page.getByLabel("Time zone").selectOption("Europe/London");
  const saved = page.waitForResponse((r) => r.url().endsWith("/api/account") && r.request().method() === "POST" && r.ok());
  await page.getByRole("button", { name: /^Save/ }).click();
  await saved;
  // Settings live on the account: start from the defaults
  await page.request.put("/api/account/settings", { data: { settings: {} }, headers: await asUser(page) });
}

/** Where the signed-in browser state from e2e/auth.setup.ts is kept. */
export const SIGNED_IN_STATE = "e2e/.auth/signed-in.json";

/**
 * Headers for calling the API straight from a test, as the signed-in user.
 *
 * The app itself sends the token from JavaScript, so a `page.request` call
 * carries none: this reads whoever the browser is signed in as (the
 * development sign-in, lib/auth/dev.ts) and sends their token. It is added
 * per request, never as `extraHTTPHeaders`, which would send it to every host
 * the page talks to.
 */
export async function asUser(page: Page): Promise<Record<string, string>> {
  const cookie = (await page.context().cookies()).find((c) => c.name === "fse_dev_user");
  return cookie ? { Authorization: `Bearer dev:${cookie.value}` } : {};
}
