import { expect, test, type Page } from "@playwright/test";

import { dealSettled, kpi } from "./helpers";

/**
 * Security headers and the content security policy (PLAN.md 1.7), in a real
 * browser: the app works under the policy with nothing refused, and script
 * the page didn't ship -- the shape an injected XSS payload takes -- is
 * refused.
 */

/** Collect the page's CSP violations from before its first script runs. */
async function recordViolations(page: Page) {
  await page.addInitScript(() => {
    const seen: string[] = [];
    (window as unknown as { __cspViolations: string[] }).__cspViolations = seen;
    document.addEventListener("securitypolicyviolation", (e) => seen.push(`${e.effectiveDirective} ${e.blockedURI}`));
  });
  return () => page.evaluate(() => (window as unknown as { __cspViolations: string[] }).__cspViolations);
}

test("pages carry a strict policy with a fresh nonce on every script", async ({ page }) => {
  const first = await page.goto("/deal/returns");
  const csp = first!.headers()["content-security-policy"];
  const nonce = /'nonce-([^']+)'/.exec(csp)?.[1];
  expect(nonce).toBeTruthy();
  const scriptSrc = csp.split(";").map((d) => d.trim()).find((d) => d.startsWith("script-src"))!;
  expect(scriptSrc).toContain("'strict-dynamic'");
  expect(scriptSrc).not.toMatch(/'unsafe-inline'|'unsafe-eval'|https:|\*/);
  expect(csp).toContain("object-src 'none'");
  expect(csp).toContain("frame-ancestors 'none'");

  // Next stamped the nonce on every script it rendered
  const scripts = await page.evaluate(() => [...document.scripts].map((s) => s.nonce));
  expect(scripts.length).toBeGreaterThan(0);
  expect(scripts.every((n) => n === nonce)).toBe(true);

  // A new request gets a new nonce
  const second = await page.request.get("/deal/returns");
  expect(/'nonce-([^']+)'/.exec(second.headers()["content-security-policy"])?.[1]).not.toBe(nonce);

  const headers = first!.headers();
  expect(headers["x-frame-options"]).toBe("DENY");
  expect(headers["x-content-type-options"]).toBe("nosniff");
  expect(headers["referrer-policy"]).toBe("strict-origin-when-cross-origin");
  expect(headers["strict-transport-security"]).toMatch(/max-age=63072000/);
  expect(headers["permissions-policy"]).toContain("camera=()");
  expect(headers["x-powered-by"]).toBeUndefined();
});

test("the app runs under the policy with nothing refused", async ({ page }) => {
  const violations = await recordViolations(page);
  await page.goto("/deal/returns");
  await dealSettled(page);
  await expect(kpi(page, "IRR")).toHaveText("21.2%");
  await page.goto("/monte-carlo/distribution");
  await expect(page.getByRole("navigation", { name: "Modes" })).toBeVisible();
  expect(await violations()).toEqual([]);
});

test("script the page didn't ship is refused", async ({ page }) => {
  const violations = await recordViolations(page);
  await page.goto("/deal/returns");
  await dealSettled(page);
  // An injected inline event handler, the classic XSS payload, may not run
  await page.evaluate(() => {
    document.body.insertAdjacentHTML("beforeend", `<img src="data:," onerror="window.__injected = 1">`);
  });
  await expect.poll(violations).toEqual(expect.arrayContaining([expect.stringMatching(/^script-src-attr/)]));
  expect(await page.evaluate(() => (window as unknown as { __injected?: number }).__injected)).toBeUndefined();
});

test("API answers through the app carry the API's own headers", async ({ page }) => {
  const resp = await page.request.get("/api/health");
  expect(resp.status()).toBe(200);
  const headers = resp.headers();
  expect(headers["content-security-policy"]).toContain("default-src 'none'");
  expect(headers["cache-control"]).toBe("no-store");
  expect(headers["x-content-type-options"]).toBe("nosniff");
  // next.config.ts leaves /api to the API: the page's policy isn't layered on top
  expect(headers["referrer-policy"]).toBe("no-referrer");
});
