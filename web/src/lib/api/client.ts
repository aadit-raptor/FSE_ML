import createClient from "openapi-fetch";

import { newRequestId, reportApiError, REQUEST_ID_HEADER } from "@/lib/monitoring";

import type { components, paths } from "./schema";

/**
 * Typed client for the FastAPI app. Types are generated from web/openapi.json
 * (`npm run api:types`). Requests go to the same origin; next.config.ts
 * proxies /api to the Python server.
 */
export const api = createClient<paths>({ baseUrl: "" });

// Each call gets its own request ID; API failures go to error tracking with it
api.use({
  onRequest({ request }) {
    request.headers.set(REQUEST_ID_HEADER, newRequestId());
    return request;
  },
  onResponse({ request, response, schemaPath }) {
    if (response.status >= 500) reportApiError(schemaPath, response.status, request.headers.get(REQUEST_ID_HEADER) ?? "");
    return response;
  },
  onError({ request, schemaPath }) {
    reportApiError(schemaPath, "network", request.headers.get(REQUEST_ID_HEADER) ?? "");
  },
});

export type Schemas = components["schemas"];
