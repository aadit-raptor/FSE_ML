import { expect, type Page, test } from "@playwright/test";

import { asUser, dealSettled, field, kpi, recordedBenchmarks, replayBenchmarks, signInAs } from "./helpers";

/**
 * Onboarding and in-app help (PLAN.md 7.1), through the real screens, API and database, each test as
 * a new account of its own: the guided first deal asks region, sector, size and currency, then
 * starts from the sourced figures and saves the deal; a sample deal opens the same way; a new
 * account's empty screens say what to do; the methodology reads as help articles; "What's new"
 * lists the releases. The sourced figures are the API's for the recorded data
 * (web/e2e/fixtures/benchmarks.json), so the deal's figures below are worked out from that file.
 */
const run = Date.now().toString(36);
const GERMAN_MACHINERY = recordedBenchmarks().starting["DE|machinery|EUR"].inputs;

const guide = (page: Page) => page.getByRole("region", { name: "Guided first deal" });
const next = (page: Page) => guide(page).getByRole("button", { name: "Next", exact: true });

async function savedDeals(page: Page): Promise<{ name: string; id: string }[]> {
  const res = await page.request.get("/api/deals", { headers: await asUser(page) });
  expect(res.ok()).toBeTruthy();
  return (await res.json()).deals;
}

// Each test signs in as its own new user; the shared signed-in state is not used
test.use({ storageState: { cookies: [], origins: [] } });

test.describe("Onboarding", () => {
  test("a new user completes the guided first deal: region, sector, size, currency, then sourced figures, saved", async ({ page }) => {
    await signInAs(page, `guided-${run}`, "en-GB");
    await replayBenchmarks(page);

    // 1. The launcher offers the guided deal to an account with no deals
    await page.goto("/start");
    await page.getByRole("link", { name: "Build a guided first deal" }).click();
    await expect(page).toHaveURL(/\/deal\/guided$/);
    await expect(guide(page).getByText("Step 1 of 5")).toBeVisible();

    // 2. Region: nothing to go on until a country is chosen
    const country = guide(page).getByRole("combobox", { name: "Country of the company", exact: true });
    await country.selectOption("");
    await expect(next(page)).toBeDisabled();
    await country.selectOption({ label: "Germany" });
    await next(page).click();

    // 3. Sector
    await expect(guide(page).getByText("Step 2 of 5")).toBeVisible();
    await guide(page).getByRole("combobox", { name: "Industry of the company", exact: true }).selectOption({ label: "Machinery" });
    await next(page).click();

    // 4. Size: EBITDA in millions; a size of nothing can't go on
    await expect(guide(page).getByText("Step 3 of 5")).toBeVisible();
    const size = guide(page).getByLabel("EBITDA", { exact: true });
    await size.fill("0");
    await expect(next(page)).toBeDisabled();
    await size.fill("80");
    await next(page).click();

    // 5. Currency: the country's own is offered, euros for Germany
    await expect(guide(page).getByText("Step 4 of 5")).toBeVisible();
    await expect(guide(page).getByRole("combobox", { name: "Guided deal currency", exact: true })).toHaveValue("EUR");
    // Back keeps the answers
    await guide(page).getByRole("button", { name: "Back" }).click();
    await expect(guide(page).getByLabel("EBITDA", { exact: true })).toHaveValue("80");
    await next(page).click();
    await next(page).click();

    // 6. The sourced figures, with their sources, before anything is created
    await expect(guide(page).getByText("Step 5 of 5")).toBeVisible();
    const preview = guide(page).locator('tr[data-field="gross_margin"]');
    await expect(preview).toContainText("Damodaran, NYU Stern");
    await expect(guide(page).locator('tr[data-field="entry_mult"]')).toContainText("14.98x");
    expect(await savedDeals(page)).toHaveLength(0);
    await guide(page).getByLabel("Deal name", { exact: true }).fill("My first buyout");
    await guide(page).getByRole("button", { name: "Create the deal" }).click();

    // 7. The deal is open on Inputs, saved, with every sourced figure and the answers given
    await expect(page).toHaveURL(/\/deal\/inputs$/);
    await dealSettled(page);
    await expect(page.locator("[data-deal-save]")).toContainText("My first buyout");
    await expect(field(page, "EBITDA")).toHaveValue("80.0");
    await expect(page.getByRole("combobox", { name: "Deal currency", exact: true })).toHaveValue("EUR");
    await expect(page.getByRole("combobox", { name: "Deal country", exact: true })).toHaveValue("DE");
    await expect(field(page, "Gross margin")).toHaveValue(String(GERMAN_MACHINERY.gross_margin));
    await expect(field(page, "Entry multiple")).toHaveValue(String(GERMAN_MACHINERY.entry_mult));
    // EV = 80 x the industry's 14.98x, in euro millions, and the model answers for it
    await expect(kpi(page, "Enterprise value")).toHaveText("1,198.4");
    await expect(kpi(page, "IRR")).toHaveText(/%$/);
    await expect(page.getByRole("status").filter({ hasText: "Illustrative figures" })).toHaveCount(0);

    const deals = await savedDeals(page);
    expect(deals.map((d) => d.name)).toEqual(["My first buyout"]);
    const detail = await (await page.request.get(`/api/deals/${deals[0].id}`, { headers: await asUser(page) })).json();
    expect(detail.inputs).toMatchObject({ country: "DE", industry: "machinery", currency: "EUR", unit: "millions", ebitda: 80 });
    expect(detail.inputs.gross_margin).toBe(GERMAN_MACHINERY.gross_margin);

    // 8. Once the account has a deal, the launcher no longer pushes the guide
    await page.goto("/start");
    await expect(page.getByRole("link", { name: "Build a guided first deal" })).toHaveCount(0);
  });

  test("a sample deal opens from its region's sourced figures, saved under its own name", async ({ page }) => {
    await signInAs(page, `sample-${run}`, "en-GB");
    await replayBenchmarks(page);
    await page.goto("/deal/guided");
    const samples = page.getByRole("list", { name: "Sample deals" });
    // Several regions, each in its own currency
    await expect(samples.getByRole("listitem")).toHaveCount(6);
    await expect(samples).toContainText("Germany");
    await expect(samples).toContainText("India");
    await expect(samples).toContainText("Japan");
    await samples.getByRole("button", { name: "Open the sample German machinery maker" }).click();

    await expect(page).toHaveURL(/\/deal\/inputs$/);
    await dealSettled(page);
    await expect(page.locator("[data-deal-save]")).toContainText("Sample: German machinery maker");
    await expect(field(page, "Gross margin")).toHaveValue(String(GERMAN_MACHINERY.gross_margin));
    await expect(page.getByRole("combobox", { name: "Deal currency", exact: true })).toHaveValue("EUR");
    expect((await savedDeals(page)).map((d) => d.name)).toEqual(["Sample: German machinery maker"]);
  });

  test("a new account's empty screens say what to do next", async ({ page }) => {
    await signInAs(page, `empty-${run}`, "en-GB");
    await page.goto("/deal/saved");
    const empty = page.locator('[data-guide="saved-deals"]');
    await expect(empty).toContainText("No saved deals yet");
    await expect(empty.getByRole("link", { name: "Build a guided first deal" })).toHaveAttribute("href", "/deal/guided");

    await page.goto("/deal/documents");
    await expect(page.locator('[data-guide="documents"]')).toBeVisible();
    await expect(page.locator('[data-guide="documents"]').getByRole("link", { name: "Read a document for a deal" })).toHaveAttribute("href", "/deal/from-document");
    await page.goto("/coverage/companies");
    await expect(page.locator('[data-guide="coverage"]')).toContainText("Add a company you cover");
  });
});

test.describe("Help", () => {
  test.use({ storageState: "e2e/.auth/signed-in.json" });

  test("the methodology reads as help articles, linked from the top bar, and search finds them", async ({ page }) => {
    await page.goto("/deal/inputs");
    await page.getByRole("link", { name: "Help", exact: true }).click();
    await expect(page).toHaveURL(/\/help$/);
    const list = page.getByRole("navigation", { name: "Help articles" });
    await list.getByRole("link", { name: "The deal model" }).click();
    await expect(page).toHaveURL(/\/help\/the-deal-model$/);
    await expect(page.getByRole("heading", { level: 1 })).toHaveText("3. The deal model");
    // The methodology's own formulas and worked example
    await expect(page.locator("pre").filter({ hasText: "entry EV" }).first()).toBeVisible();
    await expect(page.getByText("its IRR is 21.16% and its MOIC 2.61x")).toBeVisible();
    // A link to another section stays in the help
    await page.getByRole("link", { name: "finding 11" }).first().click();
    await expect(page).toHaveURL(/\/help\/known-limitations$/);

    // Ctrl K finds an article
    await page.keyboard.press("Control+k");
    await page.getByRole("combobox").fill("Tax rules");
    await expect(page.getByRole("option", { name: /7\. Tax rules/ })).toBeVisible();
  });

  test("what's new lists the releases newest first and marks what this browser has not seen", async ({ page }) => {
    await page.goto("/deal/inputs");
    await page.evaluate(() => window.localStorage.removeItem("fse.whatsNewSeen.v1"));
    await page.reload();
    const help = page.getByRole("link", { name: "Help", exact: true });
    await expect(help.locator("[data-unseen]")).toBeVisible();
    await page.goto("/help/whats-new");
    const entries = page.locator("[data-whats-new]");
    await expect(entries.first()).toContainText("Guided first deal");
    const dates = await entries.evaluateAll((els) => els.map((e) => e.getAttribute("data-whats-new")));
    expect(dates).toEqual([...dates].sort().reverse());
    // Seen now: the marker goes
    await page.goto("/deal/inputs");
    await expect(page.getByRole("link", { name: "Help", exact: true }).locator("[data-unseen]")).toHaveCount(0);
  });
});
