/**
 * Security headers for the web app (PLAN.md 1.7).
 *
 * - `SECURITY_HEADERS` go on every response except `/api/*` (next.config.ts),
 *   which carries the API's own headers (api/security.py).
 * - `contentSecurityPolicy()` is built per page request in src/proxy.ts with a
 *   fresh nonce. Scripts run only with that nonce, or when a script that had
 *   it loads them (`'strict-dynamic'`: Clerk's sign-in scripts, Next's
 *   chunks). Every page is rendered per request so Next can stamp the nonce
 *   on its own scripts (app/layout.tsx).
 *
 * Imported by next.config.ts, so it uses relative imports and no Next APIs.
 */

export const SECURITY_HEADERS: { key: string; value: string }[] = [
  { key: "Strict-Transport-Security", value: "max-age=63072000; includeSubDomains" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=(), payment=(), usb=()" },
  // Popups keep a link back only if they open one of ours; Clerk signs in by redirect
  { key: "Cross-Origin-Opener-Policy", value: "same-origin-allow-popups" },
];

/** Clerk's Frontend API origin, from the base64 host inside a publishable key. */
export function clerkFrontendApi(publishableKey: string): string | null {
  const match = /^pk_(?:test|live)_([A-Za-z0-9+/=_-]+)$/.exec(publishableKey);
  if (!match) return null;
  let decoded: string;
  try {
    decoded = atob(match[1].replace(/-/g, "+").replace(/_/g, "/"));
  } catch {
    return null;
  }
  const host = decoded.replace(/\$$/, "");
  return /^[a-z0-9.-]+\.[a-z]{2,}$/i.test(host) ? `https://${host}` : null;
}

/** The origin a Sentry DSN sends events to. */
export function sentryOrigin(dsn: string): string | null {
  try {
    const url = new URL(dsn);
    return url.protocol === "https:" || url.protocol === "http:" ? url.origin : null;
  } catch {
    return null;
  }
}

export type CspOptions = {
  nonce: string;
  /** `next dev`: React needs eval for its error overlay */
  development: boolean;
  /** Deployed over HTTPS: upgrade any stray http:// request */
  deployed: boolean;
  clerkPublishableKey: string;
  sentryDsn: string;
};

/** One-time nonce: 16 random bytes, base64. */
export function newNonce(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return btoa(String.fromCharCode(...bytes));
}

export function contentSecurityPolicy(o: CspOptions): string {
  const clerk = o.clerkPublishableKey ? clerkFrontendApi(o.clerkPublishableKey) : null;
  const sentry = o.sentryDsn ? sentryOrigin(o.sentryDsn) : null;
  const directives: [string, (string | null | false)[]][] = [
    ["default-src", ["'self'"]],
    ["script-src", ["'self'", `'nonce-${o.nonce}'`, "'strict-dynamic'", o.development && "'unsafe-eval'"]],
    // React and Motion set style attributes, and Clerk injects its own styles
    ["style-src", ["'self'", "'unsafe-inline'"]],
    ["img-src", ["'self'", "data:", "blob:", clerk && "https://img.clerk.com"]],
    ["font-src", ["'self'", "data:"]],
    ["connect-src", ["'self'", clerk, clerk && "https://clerk-telemetry.com", clerk && "https://*.clerk-telemetry.com", sentry]],
    // Clerk's bot protection on sign-up (Cloudflare Turnstile)
    ["frame-src", [clerk ? "https://challenges.cloudflare.com" : "'none'"]],
    ["worker-src", ["'self'", "blob:"]],
    ["object-src", ["'none'"]],
    ["base-uri", ["'self'"]],
    ["form-action", ["'self'"]],
    ["frame-ancestors", ["'none'"]],
  ];
  const parts = directives.map(([name, values]) => `${name} ${values.filter(Boolean).join(" ")}`);
  if (o.deployed) parts.push("upgrade-insecure-requests");
  return parts.join("; ");
}
