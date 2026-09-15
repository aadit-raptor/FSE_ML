import * as Sentry from "@sentry/browser";

/**
 * Error tracking and request IDs for the web app (PLAN.md 1.2).
 *
 * Every API call carries a fresh X-Request-ID. The API logs it and tags its
 * own Sentry events with it, so a browser error and the API error behind it
 * share one ID.
 *
 * Sentry is on when the build had SENTRY_DSN (next.config.ts exposes it as
 * NEXT_PUBLIC_SENTRY_DSN). Events carry the error, the screen's path and the
 * request ID only: no request or response bodies, query strings, form
 * values, cookies, IP address or console output, so no deal contents leave
 * the browser.
 */
export const REQUEST_ID_HEADER = "X-Request-ID";

const dsn = process.env.NEXT_PUBLIC_SENTRY_DSN;
let enabled = false;

export function newRequestId(): string {
  return crypto.randomUUID().replace(/-/g, "");
}

const stripQuery = (url: string) => url.split(/[?#]/, 1)[0];

export function initMonitoring() {
  if (!dsn || enabled) return;
  Sentry.init({
    dsn,
    environment: process.env.NEXT_PUBLIC_FSE_ENV || "local",
    release: process.env.NEXT_PUBLIC_FSE_COMMIT || undefined,
    sendDefaultPii: false,
    tracesSampleRate: 0,
    // Navigation and API call breadcrumbs only (URLs without query strings);
    // no clicks, console output or input values
    integrations: (defaults) => defaults.filter((i) => i.name !== "Breadcrumbs").concat(
      Sentry.breadcrumbsIntegration({ console: false, dom: false, fetch: true, history: true, xhr: true, sentry: false }),
    ),
    beforeBreadcrumb(crumb) {
      if (crumb.data && typeof crumb.data.url === "string") crumb.data = { url: stripQuery(crumb.data.url), method: crumb.data.method, status_code: crumb.data.status_code };
      if (crumb.data && typeof crumb.data.to === "string") crumb.data = { from: stripQuery(String(crumb.data.from ?? "")), to: stripQuery(crumb.data.to) };
      return crumb;
    },
    beforeSend(event) {
      if (event.request) event.request = { url: event.request.url ? stripQuery(event.request.url) : undefined };
      delete event.user;
      delete event.extra;
      return event;
    },
  });
  enabled = true;
}

/** Record an API failure (5xx or no response) with the request ID the API logged. */
export function reportApiError(path: string, status: number | "network", requestId: string) {
  if (!enabled) return;
  const route = stripQuery(path);
  Sentry.captureMessage(`API ${status} on ${route}`, {
    level: "error",
    tags: { request_id: requestId, api_path: route, api_status: String(status) },
    fingerprint: ["api-error", route, String(status)],
  });
}

/** Record an error a screen couldn't render past; returns Sentry's event ID. */
export function reportRenderError(error: unknown): string | undefined {
  if (!enabled) return undefined;
  return Sentry.captureException(error);
}

/** A stored UTC timestamp in the viewer's own time zone and locale. */
export function formatForViewer(isoUtc: string): string {
  const date = new Date(isoUtc);
  if (Number.isNaN(date.getTime())) return isoUtc;
  return new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZoneName: "short" }).format(date);
}
