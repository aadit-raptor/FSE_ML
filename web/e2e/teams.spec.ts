import { readdirSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { type Browser, expect, type Page, test } from "@playwright/test";

import { asUser, signInAs } from "./helpers";

/**
 * Teams (PLAN.md 7.2), through the real screens, API and database, with two people in two browsers: the
 * owner makes a team and an invitation link, a viewer joins by it, the owner shares a deal, the viewer reads
 * it (nothing they change is saved) and comments mentioning the owner, who is emailed (the API writes mail to
 * a directory here: teams/mail.py) and sees who did what in the deal's activity. The permission rules
 * themselves, every endpoint for every role, are tests/test_teams.py.
 */
const run = Date.now().toString(36);
const OWNER = `teamowner-${run}`;
const VIEWER = `teamviewer-${run}`;
const MAIL = join(tmpdir(), "fse-e2e-mail");

test.use({ storageState: { cookies: [], origins: [] } });

async function person(browser: Browser, user: string): Promise<Page> {
  const page = await (await browser.newContext()).newPage();
  await signInAs(page, user, "en-GB");
  return page;
}

function mailTo(address: string): { subject: string; text: string } | null {
  let files: string[] = [];
  try {
    files = readdirSync(MAIL);
  } catch {
    return null;
  }
  for (const file of files) {
    const mail = JSON.parse(readFileSync(join(MAIL, file), "utf-8"));
    if (mail.to === address) return mail;
  }
  return null;
}

test("a viewer joins by a link, reads a shared deal without changing it, and a mention emails its owner", async ({ browser }) => {
  test.setTimeout(120_000);
  const owner = await person(browser, OWNER);
  const viewer = await person(browser, VIEWER);
  const made = await owner.request.post("/api/deals", {
    data: { name: `Heron ${run}`, inputs: { growth: 7.5 } }, headers: await asUser(owner),
  });
  expect(made.status()).toBe(201);
  const dealId = (await made.json()).id as string;

  // 1. The owner makes a team and an invitation link for viewers
  await owner.goto("/teams");
  await expect(owner.locator('[data-guide="teams"]')).toBeVisible();
  await owner.getByLabel("New team's name").fill(`Desk ${run}`);
  await owner.getByRole("button", { name: "Make the team" }).click();
  await expect(owner.locator(`[data-team-open="Desk ${run}"]`)).toBeVisible();
  await owner.getByLabel("They join as").selectOption("viewer");
  await owner.getByRole("button", { name: "Make a link" }).click();
  const link = (await owner.locator("[data-invite-link]").textContent()) ?? "";
  expect(link).toMatch(/\/teams\/join#inv_/);

  // 2. The viewer opens it and is in the team, as a viewer
  await viewer.goto(link.replace(/^https?:\/\/[^/]+/, ""));
  await expect(viewer.getByText(`You are in Desk ${run} as viewer.`)).toBeVisible();
  await expect(viewer).toHaveURL(/\/teams\/join$/);

  // 3. The owner shares the deal with the team; the viewer is listed as reading it
  await owner.goto(`/deal/team?deal=${dealId}`);
  await expect(owner.locator('[data-access="owner"]')).toBeVisible();
  await owner.getByRole("switch", { name: `Desk ${run}` }).click();
  await expect(owner.getByRole("switch", { name: `Desk ${run}` })).toHaveAttribute("aria-checked", "true");
  await expect(owner.locator(`[data-person="${VIEWER}"]`)).toContainText("reads");

  // 4. The viewer sees it among their deals, opens it read-only, and an edit is not saved
  await viewer.goto("/deal/saved");
  await expect(viewer.locator(`[data-deal="Heron ${run}"] [data-shared="viewer"]`)).toBeVisible();
  await viewer.goto(`/deal/team?deal=${dealId}`);
  await expect(viewer.locator('[data-access="viewer"]')).toBeVisible();
  await expect(viewer.getByRole("note").filter({ hasText: "nothing you change is saved" })).toBeVisible();
  await viewer.goto("/deal/inputs");
  await expect(viewer.locator("[data-view-only]")).toHaveText("View only");
  const refused = await viewer.request.put(`/api/deals/${dealId}/draft`, {
    data: { inputs: { growth: 99 }, settings: {} }, headers: await asUser(viewer),
  });
  expect(refused.status()).toBe(403);

  // 5. The viewer comments, mentioning the owner; the owner gets an email that says who, never what
  await viewer.goto("/deal/team");
  await viewer.getByRole("button", { name: `@${OWNER}` }).click();
  await viewer.getByLabel("Comment", { exact: true }).fill(`Exit at 11x is aggressive ${run}`);
  await viewer.getByRole("button", { name: "Post the comment" }).click();
  await expect(viewer.locator("[data-comment]")).toContainText(`Exit at 11x is aggressive ${run}`);
  await expect.poll(() => mailTo(`${OWNER}@example.test`), { timeout: 30_000 }).not.toBeNull();
  const mail = mailTo(`${OWNER}@example.test`)!;
  expect(mail.text).toContain(`${VIEWER} mentioned you in a comment`);
  expect(mail.text).toContain(`/deal/team?deal=${dealId}`);
  expect(mail.text).not.toContain("11x");
  expect(mail.text).not.toContain("Heron");

  // 6. The owner reads the comment and the deal's activity says who did what
  await owner.goto(`/deal/team?deal=${dealId}`);
  await expect(owner.locator("[data-comment]")).toContainText(`mentioned ${OWNER}`);
  await owner.goto("/deal/saved");
  await expect(owner.locator('tr[data-action="commented"] [data-by]')).toHaveText(VIEWER);
  await expect(owner.locator('tr[data-action="shared"] [data-by]')).toHaveText("You");
});
