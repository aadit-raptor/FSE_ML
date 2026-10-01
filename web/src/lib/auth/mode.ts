/**
 * Which sign-in the app is running with (PLAN.md 1.4).
 *
 * - **clerk**: a real Clerk instance. `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` is
 *   set, so the sign-in and sign-up screens are Clerk's and the API gets a
 *   Clerk session token. Every deployed copy runs this way; next.config.ts
 *   fails a Vercel production build when the key is missing.
 * - **dev**: no Clerk instance (a local run, or the browser tests). The app
 *   signs in as a named development user and sends `dev:<name>` as the token.
 *   The API only accepts that outside production and only while no Clerk
 *   instance is configured (api/auth.py).
 *
 * The key is read at build time, so both modes are decided once, not per
 * request.
 */
export const CLERK_PUBLISHABLE_KEY = process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY ?? "";

export type AuthMode = "clerk" | "dev";

export const AUTH_MODE: AuthMode = CLERK_PUBLISHABLE_KEY ? "clerk" : "dev";

/** Pages a signed-out visitor may open. The privacy policy too: Google's consent screen links to it. */
export const PUBLIC_ROUTES = ["/sign-in", "/sign-up", "/privacy"];

export function isPublicRoute(pathname: string): boolean {
  return PUBLIC_ROUTES.some((route) => pathname === route || pathname.startsWith(`${route}/`));
}

/**
 * The app's own sign-in and sign-up pages, in the Variater design. Clerk is
 * told about them (proxy.ts, the root layout): left unsaid, a production
 * instance sends signed-out visitors to its hosted Account Portal
 * (accounts.<domain>), a different-looking page.
 */
export const SIGN_IN_URL = "/sign-in";
export const SIGN_UP_URL = "/sign-up";

/** Where a signed-in visitor lands once their account is complete. */
export const AFTER_SIGN_IN = "/deal/inputs";
/** Sign-up finishes on the account screen, which asks for country and currency. */
export const AFTER_SIGN_UP = "/account";
