import { expect, test } from "@playwright/test";

import { kpi } from "./helpers";

/**
 * Monitoring (PLAN.md 1.2): one request ID shared by the browser and the API,
 * errors reported with that ID and without deal contents, and UTC times shown
 * in the viewer's time zone.
 */

test("browser and API share each call's request ID", async ({ page }) => {
  const response = page.waitForResponse((r) => r.url().endsWith("/api/deal/run") && r.request().method() === "POST");
  await page.goto("/deal/returns");
  const res = await response;
  const sent = await res.request().headerValue("x-request-id");
  expect(sent).toMatch(/^[0-9a-f]{32}$/);
  expect(res.headers()["x-request-id"]).toBe(sent);
  expect(res.headers()["server-timing"]).toMatch(/^model;dur=[\d.]+$/);
  await expect(kpi(page, "IRR")).toHaveText("21.2%");
});

test.describe("viewer's time zone", () => {
  test.use({ timezoneId: "Asia/Kolkata", locale: "en-GB" });

  test("the API's UTC clock is shown in local time", async ({ page }) => {
    await page.route("**/api/health", (route) =>
      route.fulfill({ json: { status: "ok", version: "0.1.0", environment: "local", commit: null, time: "2026-01-02T03:04:05.000Z" } }),
    );
    await page.goto("/deal/inputs");
    const status = page.locator("footer").getByRole("status");
    await expect(status).toHaveText(/API ok/);
    // 03:04 UTC is 08:34 in India (UTC+5:30)
    await expect(status).toHaveAttribute("title", /^API checked 2 Jan 2026, 08:34/);
  });
});

// Needs a build with SENTRY_DSN equal to E2E_SENTRY_DSN, a local address that
// the test intercepts (CI's e2e job sets both), so nothing reaches Sentry
const sentryDsn = process.env.E2E_SENTRY_DSN;

test("an API failure reaches error tracking with its request ID and no deal contents", async ({ page }) => {
  test.skip(!sentryDsn, "build with SENTRY_DSN and set E2E_SENTRY_DSN to the same value");
  const ingest = new URL(sentryDsn!);
  const envelopes: string[] = [];
  await page.route(`${ingest.protocol}//${ingest.host}/**`, async (route) => {
    envelopes.push(route.request().postData() ?? "");
    await route.fulfill({ status: 200, headers: { "access-control-allow-origin": "*" }, json: {} });
  });

  let requestId = "";
  let dealBody = "";
  await page.route("**/api/deal/run", async (route) => {
    requestId = (await route.request().headerValue("x-request-id")) ?? "";
    dealBody = route.request().postData() ?? "";
    await route.fulfill({ status: 500, headers: { "x-request-id": requestId }, json: { detail: "Internal server error", request_id: requestId } });
  });

  await page.goto("/deal/returns");
  await expect.poll(() => envelopes.find((e) => e.includes("API 500 on /api/deal/run")), { timeout: 20_000 }).toBeTruthy();

  const envelope = envelopes.find((e) => e.includes("API 500 on /api/deal/run"))!;
  const event = envelope.split("\n").map((line) => JSON.parse(line)).find((item) => item.message || item.logentry);
  expect(requestId).toMatch(/^[0-9a-f]{32}$/);
  expect(event.tags).toMatchObject({ request_id: requestId, api_path: "/api/deal/run", api_status: "500" });
  expect(event.environment).toBe("local");
  // The deal inputs sent with the failing call never appear in the report
  expect(dealBody.length).toBeGreaterThan(20);
  expect(envelope).not.toContain(dealBody);
  for (const key of Object.keys(JSON.parse(dealBody).inputs ?? {})) expect(envelope).not.toContain(`"${key}"`);
});
