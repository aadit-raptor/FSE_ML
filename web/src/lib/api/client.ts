import createClient from "openapi-fetch";

import type { components, paths } from "./schema";

/**
 * Typed client for the FastAPI app. Types are generated from web/openapi.json
 * (`npm run api:types`). Requests go to the same origin; next.config.ts
 * proxies /api to the Python server.
 */
export const api = createClient<paths>({ baseUrl: "" });

export type Schemas = components["schemas"];
