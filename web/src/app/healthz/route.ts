/**
 * The web app's own health check, for the uptime monitor (ops/betterstack.py).
 *
 * Every page needs a signed-in user, so a signed-out monitor can't load one:
 * Clerk answers 404, or redirects to sign-in. This route is left out of
 * src/proxy.ts's matcher, so it works signed out and shows only that this
 * deployment is serving requests. It never calls the API: that would wake the
 * free Render service every few minutes (the API has its own monitor).
 */
export function GET() {
  return Response.json(
    { status: "ok", service: "FSE/ML web", commit: process.env.VERCEL_GIT_COMMIT_SHA ?? null },
    { headers: { "Cache-Control": "no-store" } },
  );
}
