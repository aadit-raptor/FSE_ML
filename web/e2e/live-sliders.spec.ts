import { expect, type Page, type Response, test } from "@playwright/test";

import { asUser, dealSettled, kpi, modeTab, setField, stepLink } from "./helpers";

/**
 * Monte Carlo -> Live (PLAN.md 5.8): the estimate is the simulation itself on a
 * small, fixed set of paths, for any deal. Proved by output: what the screen
 * shows is what the API answers, a slider is an input (the same figure as the
 * run with that input changed), and a deal with listed facilities is estimated
 * as itself, with no "trained on one deal" warning anywhere.
 */
type Estimate = { paths: number; prediction: Record<string, number>; median_interval: number[] };
type Body = { mc: Record<string, number>; deal: Record<string, unknown>; settings: Record<string, unknown>; sliders: Record<string, number> };

const LIVE = "/api/montecarlo/live";
const pct = (v: number) => `${(v * 100).toFixed(1)}%`; // locale-ok: the test's own expectation, en-US

/** The next live answer the page receives, with the request that asked for it. */
async function nextEstimate(page: Page, act: () => Promise<unknown>): Promise<{ sent: Body; got: Estimate }> {
  const [response] = await Promise.all([
    page.waitForResponse((r: Response) => r.url().endsWith(LIVE) && r.status() === 200),
    act(),
  ]);
  return { sent: response.request().postDataJSON() as Body, got: (await response.json()) as Estimate };
}

/** The API's own answer, asked directly. */
async function estimateOf(page: Page, body: Partial<Body>): Promise<Estimate> {
  const res = await page.request.post(LIVE, { headers: await asUser(page), data: body });
  expect(res.status()).toBe(200);
  return (await res.json()) as Estimate;
}

const slider = (page: Page, name: string) => page.getByRole("slider", { name, exact: true });

test.describe("Live sliders", () => {
  test("the estimate is the simulation's own answer, and each slider is an input", async ({ page }) => {
    const first = await nextEstimate(page, () => page.goto("/monte-carlo/live"));
    await expect(kpi(page, "Median IRR")).toHaveText(pct(first.got.prediction.irr_p50));
    // The default deal on 8,000 seeded paths (jobs/drill.py EXPECTED_LIVE pins the same figure)
    expect(first.got.paths).toBe(8000);
    await expect(kpi(page, "Median IRR")).toHaveText("18.6%");
    await expect(kpi(page, "P5")).toHaveText("2.6%");
    await expect(kpi(page, "P95")).toHaveText("31.4%");
    await expect(page.getByRole("region", { name: "Median IRR" })).toContainText("95% interval 18.4% to 18.8%");
    await expect(kpi(page, "P(IRR > 20%)")).toHaveText("43.3%");
    // Where the sliders start is the deal itself: the same answer with no sliders at all
    const own = await estimateOf(page, { mc: first.sent.mc, deal: first.sent.deal, settings: first.sent.settings });
    expect(own.prediction).toEqual(first.got.prediction);

    // Exit multiple 14x: the run with the rail's exit multiple at 14x
    const moved = await nextEstimate(page, () => slider(page, "Exit multiple mean").fill("14"));
    expect(moved.sent.sliders.exit_mean).toBe(14);
    const asInput = await estimateOf(page, { mc: { ...first.sent.mc, exit_mean: 14 }, deal: first.sent.deal, settings: first.sent.settings });
    expect(moved.got.prediction).toEqual(asInput.prediction);
    await expect(kpi(page, "Median IRR")).toHaveText(pct(asInput.prediction.irr_p50));
    expect(asInput.prediction.irr_p50).toBeGreaterThan(first.got.prediction.irr_p50 + 0.05);

    // Every other slider moves the answer too
    let before = moved.got.prediction.irr_p50;
    for (const [name, value] of [
      ["Revenue growth mean", "9"],
      ["Interest rate mean", "10"],
      ["Gross margin mean", "60"],
      ["Debt / EV", "75"],
    ] as const) {
      const next = await nextEstimate(page, () => slider(page, name).fill(value));
      expect(Math.abs(next.got.prediction.irr_p50 - before), name).toBeGreaterThan(0.002);
      await expect(kpi(page, "Median IRR")).toHaveText(pct(next.got.prediction.irr_p50));
      before = next.got.prediction.irr_p50;
    }
    // The spread of the exit multiple widens the range, not the middle
    const wide = await nextEstimate(page, () => slider(page, "Exit multiple std dev").fill("4"));
    const range = (p: Record<string, number>) => p.irr_p95 - p.irr_p5;
    expect(range(wide.got.prediction)).toBeGreaterThan(range(first.got.prediction) + 0.05);

    // Back to the deal
    const reset = await nextEstimate(page, () => page.getByRole("button", { name: "Reset to rail" }).click());
    expect(reset.got.prediction).toEqual(first.got.prediction);
    await expect(kpi(page, "Median IRR")).toHaveText("18.6%");

    // The rail edited under moved sliders: they restart from the deal as it now is, so the
    // estimate is never for an assumption the rail no longer shows
    await nextEstimate(page, () => slider(page, "Revenue growth mean").fill("9"));
    const railed = await nextEstimate(page, () => setField(page, "Mean", "12", "Exit multiple"));
    expect(railed.sent.sliders).toMatchObject({ exit_mean: 12, growth_mean: 5 });
    await expect(slider(page, "Exit multiple mean")).toHaveValue("12");
    await expect(slider(page, "Revenue growth mean")).toHaveValue("5");
    expect(railed.got.prediction.irr_p50).toBeGreaterThan(first.got.prediction.irr_p50 + 0.02);
  });

  test("it says what it is, and warns about no training deal", async ({ page }) => {
    await nextEstimate(page, () => page.goto("/monte-carlo/live"));
    const note = page.locator("#content").getByRole("note").filter({ hasText: "The simulation itself" });
    await expect(note).toContainText("reruns this deal on the same 8,000 simulated paths");
    await expect(note).toContainText("Run Monte Carlo uses 50,000");
    for (const gone of ["Trained on one fixed deal", "Directional only", "Downside tail unreliable", "surrogate"]) {
      await expect(page.locator("#content").getByText(gone)).toHaveCount(0);
    }
  });

  test("a deal with listed facilities, tax rules and another hold is estimated as itself", async ({ page }) => {
    // The default deal written out as facilities is the same deal ...
    await page.goto("/deal/debt");
    await dealSettled(page);
    await page.getByRole("button", { name: "Use explicit tranches" }).click();
    await dealSettled(page);
    const listed = await nextEstimate(page, async () => {
      await modeTab(page, "Monte Carlo").click();
      await stepLink(page, "Live").click();
    });
    expect((listed.sent.deal.tranches as unknown[]).length).toBe(2);
    await expect(kpi(page, "Median IRR")).toHaveText("18.6%");
    await expect(slider(page, "Debt / EV")).toHaveValue("60");
    // ... and its debt slider scales the facilities: 72% is the percentages deal at 72%
    const scaled = await nextEstimate(page, () => slider(page, "Debt / EV").fill("72"));
    const byShare = await estimateOf(page, { mc: listed.sent.mc, deal: { ...listed.sent.deal, tranches: [], debt_pct: 72 }, settings: listed.sent.settings });
    expect(scaled.got.prediction.irr_p50).toBeCloseTo(byShare.prediction.irr_p50, 12);
    await expect(kpi(page, "Median IRR")).toHaveText(pct(byShare.prediction.irr_p50));
    expect(byShare.prediction.irr_p50).toBeGreaterThan(listed.got.prediction.irr_p50 + 0.01);

    // A different hold changes the deal the sliders estimate, with nothing to warn about
    await modeTab(page, "Deal").click();
    await stepLink(page, "Deal inputs").click();
    await dealSettled(page);
    const held = await nextEstimate(page, async () => {
      await setField(page, "Hold", "7");
      await dealSettled(page);
      await modeTab(page, "Monte Carlo").click();
      await stepLink(page, "Live").click();
    });
    expect(held.sent.mc.hold).toBe(7);
    expect(held.got.paths).toBe(8000);
    expect(Math.abs(held.got.prediction.irr_p50 - listed.got.prediction.irr_p50)).toBeGreaterThan(0.01);
    await expect(kpi(page, "Median IRR")).toHaveText(pct(held.got.prediction.irr_p50));
    await expect(page.locator("#content").getByText("Directional only")).toHaveCount(0);
  });
});
