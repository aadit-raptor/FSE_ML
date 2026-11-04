"use client";

import { api, type Schemas } from "@/lib/api/client";

/**
 * The public API's keys and webhooks (PLAN.md 7.7), managed on Account -> Developer API. A new key and a new
 * webhook's signing secret are answered once and never again (only a key's hash is kept). A refusal is a code:
 * `developer.problem_<code>` says it in words.
 *
 * i18n-keys: developer.problem_*
 * i18n-keys: developer.state_*
 */
export type ApiKey = Schemas["ApiKeyOut"];
export type ApiKeyMade = Schemas["ApiKeyMade"];
export type ApiKeyList = Schemas["ApiKeyList"];
export type Webhook = Schemas["WebhookOut"];
export type WebhookMade = Schemas["WebhookMade"];
export type WebhookList = Schemas["WebhookList"];

/** Refusal codes with words of their own (`developer.problem_<code>`); anything else is `problem_failed`. */
const PROBLEMS = ["invalid_name", "full_keys", "full_webhooks", "invalid_url", "not_https", "private_address",
  "not_found", "no_account", "no_database", "site_only"];

export function problemKey(code: string): string {
  return `problem_${PROBLEMS.includes(code) ? code : "failed"}`;
}

type Result<T> = { ok: true; data: T } | { ok: false; code: string };

function codeOf(error: unknown): string {
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  if (detail && typeof detail === "object" && typeof (detail as { code?: unknown }).code === "string") {
    return (detail as { code: string }).code;
  }
  return "failed";
}

type Call<T> = Promise<{ data?: T; error?: unknown; response: Response }>;

async function attempt<T>(call: Call<T>): Promise<Result<T>> { // text-ok: generics, not words
  try {
    const { data, error, response } = await call;
    if (response.ok) return { ok: true, data: data as T };
    return { ok: false, code: codeOf(error) };
  } catch {
    return { ok: false, code: "failed" };
  }
}

export const listKeys = () => attempt(api.GET("/api/account/api-keys"));
export const createKey = (name: string) => attempt(api.POST("/api/account/api-keys", { body: { name } }));
export const revokeKey = (id: string) =>
  attempt(api.DELETE("/api/account/api-keys/{key_id}", { params: { path: { key_id: id } } }));
export const listWebhooks = () => attempt(api.GET("/api/account/webhooks"));
export const addWebhook = (url: string) => attempt(api.POST("/api/account/webhooks", { body: { url } }));
export const removeWebhook = (id: string) =>
  attempt(api.DELETE("/api/account/webhooks/{webhook_id}", { params: { path: { webhook_id: id } } }));
export const pingWebhook = (id: string) =>
  attempt(api.POST("/api/account/webhooks/{webhook_id}/ping", { params: { path: { webhook_id: id } } }));

/** Where the documentation lives: the API's own schema page and the guide in the repository. */
export const DOCS_URL = "https://github.com/aadit-raptor/FSE_ML/blob/main/docs/public-api.md";
