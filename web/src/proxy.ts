/**
 * Runs before every page request (Next 16's proxy.ts, formerly middleware.ts).
 *
 * A signed-out visitor is sent to /sign-in before any screen renders, so the
 * app never flashes a workspace it can't fill. With Clerk that is Clerk's own
 * check; without it (local runs, browser tests) it is the development cookie.
 * Either way the real protection is on the API, which needs a valid token for
 * every call (api/auth.py).
 *
 * Every page also gets its content security policy here, with a fresh nonce
 * that Next and Clerk stamp on their scripts (lib/security/headers.ts). Both
 * sign-in modes use the same policy, so the browser tests check the one that
 * is deployed.
 *
 * `/api/*` is left alone: those requests carry a bearer token and are proxied
 * to the FastAPI app by next.config.ts, which sends its own headers.
 */
import { clerkMiddleware } from "@clerk/nextjs/server";
import { NextResponse, type NextRequest } from "next/server";

import { devUserFromCookies } from "@/lib/auth/dev";
import { AUTH_MODE, CLERK_PUBLISHABLE_KEY, isPublicRoute, SIGN_IN_URL, SIGN_UP_URL } from "@/lib/auth/mode";
import { contentSecurityPolicy, newNonce } from "@/lib/security/headers";

const CSP = "Content-Security-Policy";

/** Let the request through with a new nonce: Next reads it from the request's policy. */
function withPolicy(request: NextRequest): NextResponse {
  const policy = contentSecurityPolicy({
    nonce: newNonce(),
    development: process.env.NODE_ENV === "development",
    deployed: !!process.env.VERCEL_ENV,
    clerkPublishableKey: CLERK_PUBLISHABLE_KEY,
    sentryDsn: process.env.NEXT_PUBLIC_SENTRY_DSN ?? "",
  });
  const headers = new Headers(request.headers);
  headers.set(CSP, policy);
  const response = NextResponse.next({ request: { headers } });
  response.headers.set(CSP, policy);
  return response;
}

function development(request: NextRequest) {
  const { pathname } = request.nextUrl;
  if (isPublicRoute(pathname) || devUserFromCookies(request.headers.get("cookie"))) {
    return withPolicy(request);
  }
  return toSignIn(request);
}

/**
 * Sends a signed-out visitor to sign in, and back to the screen they asked
 * for afterwards. The site's root asks for no screen, so it carries no way
 * back and the sign-in lands on the launcher (lib/auth/mode.ts AFTER_SIGN_IN).
 */
function toSignIn(request: NextRequest): NextResponse {
  const { pathname } = request.nextUrl;
  const signIn = new URL(SIGN_IN_URL, request.url);
  if (pathname !== "/") signIn.searchParams.set("next", pathname);
  return NextResponse.redirect(signIn);
}

// clerkMiddleware() is only built when there is an instance for it to check
export default AUTH_MODE === "clerk"
  ? clerkMiddleware(
      async (auth, request) => {
        // protect() would send the root back to itself after sign-in, past the launcher
        if (request.nextUrl.pathname === "/" && !(await auth()).userId) return toSignIn(request);
        if (!isPublicRoute(request.nextUrl.pathname)) await auth.protect();
        return withPolicy(request);
      },
      // Signed out, protect() redirects here: our page, not Clerk's hosted portal
      { signInUrl: SIGN_IN_URL, signUpUrl: SIGN_UP_URL },
    )
  : development;

export const config = {
  // Everything except Next's own assets, files with an extension, /api, and
  // /healthz (the uptime monitor's signed-out check, app/healthz/route.ts)
  matcher: ["/((?!api|healthz|_next|favicon.ico|.*\\.[^/]+$).*)"],
};
