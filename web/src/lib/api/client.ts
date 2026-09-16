import createClient from "openapi-fetch";

import { newRequestId, reportApiError, REQUEST_ID_HEADER } from "@/lib/monitoring";

import type { components, paths } from "./schema";

/**
 * Typed client for the FastAPI app. Types are generated from web/openapi.json
 * (`npm run api:types`). Requests go to the same origin; next.config.ts
 * proxies /api to the Python server.
 */
export const api = createClient<paths>({ baseUrl: "" });

/**
 * Where the session token comes from. The auth provider registers it as it
 * mounts (components/auth/AuthProvider.tsx); until then only the public
 * health check works, which is why the screens are held behind the gate.
 */
type TokenSource = () => Promise<string | null>;
let tokenSource: TokenSource | null = null;

export function setTokenSource(source: TokenSource | null): void {
  tokenSource = source;
}

/** The Authorization header for calls made outside this client (file downloads). */
export async function authHeaders(): Promise<Record<string, string>> {
  const token = await tokenSource?.();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/** Called when the API says the session is over, so the app can sign out. */
type Unauthorized = () => void;
let onUnauthorized: Unauthorized | null = null;

export function setUnauthorizedHandler(handler: Unauthorized | null): void {
  onUnauthorized = handler;
}

// Each call gets its own request ID and the signed-in user's token; API
// failures go to error tracking with that ID
api.use({
  async onRequest({ request }) {
    request.headers.set(REQUEST_ID_HEADER, newRequestId());
    const token = await tokenSource?.();
    if (token) request.headers.set("Authorization", `Bearer ${token}`);
    return request;
  },
  onResponse({ request, response, schemaPath }) {
    if (response.status === 401) onUnauthorized?.();
    if (response.status >= 500) reportApiError(schemaPath, response.status, request.headers.get(REQUEST_ID_HEADER) ?? "");
    return response;
  },
  onError({ request, schemaPath }) {
    reportApiError(schemaPath, "network", request.headers.get(REQUEST_ID_HEADER) ?? "");
  },
});

export type Schemas = components["schemas"];
