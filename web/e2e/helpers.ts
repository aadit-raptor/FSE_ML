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

/** Wait for a deal result, and for any pending rerun to finish. */
export async function dealSettled(page: Page) {
  await expect(page.getByRole("status").filter({ hasText: /^Up to date/ })).toBeVisible();
}

/** Monte Carlo has a result and isn't running. */
export async function simulationSettled(page: Page) {
  await expect(kpi(page, "Mean IRR").or(kpi(page, "Base"))).toBeVisible({ timeout: 45_000 });
  await expect(page.getByText("Running simulation")).toHaveCount(0, { timeout: 45_000 });
}
