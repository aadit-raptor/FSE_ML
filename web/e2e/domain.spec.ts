import { expect, test } from "@playwright/test";

/**
 * The site's own domain (PLAN.md 0.2): the old Vercel address and www send
 * every path, with its query string, to the same path on https://variater.com
 * (next.config.ts). The server decides by the Host header, so the built app
 * on localhost answers exactly as Vercel will; live.spec.ts checks the
 * deployed hosts. Anything else, preview addresses included, is served as is.
 */
const DOMAIN = "https://variater.com";

for (const host of ["fse-ml.vercel.app", "www.variater.com"]) {
  test(`${host} redirects every path to the domain`, async ({ request }) => {
    for (const path of ["/", "/deal/returns?tab=2", "/healthz", "/api/health"]) {
      const resp = await request.get(path, { headers: { host }, maxRedirects: 0 });
      expect(resp.status(), `${host}${path}`).toBe(308);
      // Next writes the root as the bare origin, which browsers read as "/"
      expect(resp.headers()["location"], `${host}${path}`).toBe(path === "/" ? DOMAIN : `${DOMAIN}${path}`);
    }
  });
}

test("other hosts are served, not redirected", async ({ request }) => {
  for (const host of ["localhost:3000", "fse-ml-git-staging-aadit7.vercel.app", "variater.com"]) {
    const resp = await request.get("/healthz", { headers: { host }, maxRedirects: 0 });
    expect(resp.status(), host).toBe(200);
  }
});
