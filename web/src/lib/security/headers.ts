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

/**
 * The Excel add-in's pages (PLAN.md R3): the task pane (`/excel`) and the
 * sign-in dialog it opens (`/excel/connect`), plus the files its manifest
 * names (`/excel/*.png`, `/excel/manifest*.xml`). Only these differ from
 * every other page:
 *
 * - Excel on the web shows the pane in a frame of Microsoft's page, so
 *   framing is allowed from Office's own hosts (`frame-ancestors`, below)
 *   and X-Frame-Options, which can't name hosts, is left off;
 * - Office's dialog keeps a link back to the pane through its opener, so
 *   the opener policy is `unsafe-none` (Microsoft's guidance for dialogs).
 */
export const EXCEL_PATH = "/excel";

export function isExcelPath(pathname: string): boolean {
  return pathname === EXCEL_PATH || pathname.startsWith(`${EXCEL_PATH}/`);
}

export const EXCEL_SECURITY_HEADERS: { key: string; value: string }[] = SECURITY_HEADERS
  .filter((h) => h.key !== "X-Frame-Options")
  .map((h) => (h.key === "Cross-Origin-Opener-Policy" ? { ...h, value: "unsafe-none" } : h));

/** Where Office.js is served from: Microsoft asks add-ins to load it from its CDN, never a copy. */
export const OFFICE_SCRIPT_ORIGIN = "https://appsforoffice.microsoft.com";
export const OFFICE_SCRIPT = `${OFFICE_SCRIPT_ORIGIN}/lib/1/hosted/office.js`;

/**
 * The pages Excel on the web puts the pane inside: Microsoft's own hosts
 * (Excel on the web on officeapps.live.com and cloud.microsoft, the
 * Microsoft 365 and OneDrive pages around it, SharePoint for work files).
 * Every page above the pane must be one of them.
 */
export const OFFICE_FRAME_ANCESTORS = [
  "https://*.officeapps.live.com",
  "https://*.office.com",
  "https://*.office365.com",
  "https://*.cloud.microsoft",
  "https://*.microsoft365.com",
  "https://*.sharepoint.com",
  "https://*.live.com",
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

/**
 * The API's own origin, which uploaded files are sent to (PLAN.md 6.1,
 * lib/files.ts): everything else reaches the API through this site's /api
 * proxy. One exact origin, never a wildcard. Plain http only for a server on
 * this machine, as in local runs and the browser tests.
 */
export function apiOrigin(raw: string): string | null {
  try {
    const url = new URL(raw.trim());
    const local = url.protocol === "http:" && ["127.0.0.1", "localhost", "[::1]"].includes(url.hostname);
    return (url.protocol === "https:" || local) && url.origin === raw.trim().replace(/\/+$/, "") ? url.origin : null;
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
  /** The API's own origin, for file uploads; empty leaves it out */
  apiOrigin?: string;
  /** One of the Excel add-in's pages (isExcelPath): Office.js, and framing by Office */
  excel?: boolean;
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
  const uploads = o.apiOrigin ? apiOrigin(o.apiOrigin) : null;
  const directives: [string, (string | null | false)[]][] = [
    ["default-src", ["'self'"]],
    // Office.js is loaded by our own (nonce'd) code, which 'strict-dynamic' allows; its host is named
    // too for webviews that predate 'strict-dynamic' (older Office on Windows)
    ["script-src", ["'self'", `'nonce-${o.nonce}'`, "'strict-dynamic'", !!o.excel && OFFICE_SCRIPT_ORIGIN, o.development && "'unsafe-eval'"]],
    // React and Motion set style attributes, and Clerk injects its own styles
    ["style-src", ["'self'", "'unsafe-inline'"]],
    ["img-src", ["'self'", "data:", "blob:", clerk && "https://img.clerk.com"]],
    ["font-src", ["'self'", "data:"]],
    ["connect-src", ["'self'", clerk, clerk && "https://clerk-telemetry.com", clerk && "https://*.clerk-telemetry.com", sentry, uploads]],
    // Clerk's bot protection on sign-up (Cloudflare Turnstile)
    ["frame-src", [clerk ? "https://challenges.cloudflare.com" : "'none'"]],
    ["worker-src", ["'self'", "blob:"]],
    ["object-src", ["'none'"]],
    ["base-uri", ["'self'"]],
    ["form-action", ["'self'"]],
    ["frame-ancestors", o.excel ? OFFICE_FRAME_ANCESTORS : ["'none'"]],
  ];
  const parts = directives.map(([name, values]) => `${name} ${values.filter(Boolean).join(" ")}`);
  if (o.deployed) parts.push("upgrade-insecure-requests");
  return parts.join("; ");
}
