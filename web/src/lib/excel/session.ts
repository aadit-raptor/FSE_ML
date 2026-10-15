/**
 * The Excel add-in's sign-in, on the pane's side (PLAN.md R3).
 *
 * The pane can't use the site's session (on Excel on the web it is a frame of
 * Microsoft's page, which keeps the site's cookies from it), so it signs in
 * through Office's dialog: `/excel/connect` signs in on the site as anywhere
 * else, asks the API for an add-in pass and hands it over. The pane keeps the
 * pass in this browser's storage until it ends (`expires_at`, a working day)
 * or the person signs out, and sends it as the bearer token. The API lets a
 * pass reach the research endpoints only (api/auth.py `ADDIN_PATHS`).
 *
 * The dialog's message is `{"kind": "variater-addin-pass", token, expires_at, label}`;
 * anything else is ignored.
 */
import { api, setTokenSource, setUnauthorizedHandler } from "@/lib/api/client";

export const PASS_MESSAGE = "variater-addin-pass";
const STORAGE_KEY = "fse.excel.pass";

export type AddinPass = { token: string; expires_at: string; label: string | null };

export function parsePassMessage(message: string): AddinPass | null {
  try {
    const value = JSON.parse(message) as Record<string, unknown>;
    if (value.kind !== PASS_MESSAGE || typeof value.token !== "string" || !value.token.startsWith("addin:")) return null;
    if (typeof value.expires_at !== "string" || Number.isNaN(Date.parse(value.expires_at))) return null;
    return { token: value.token, expires_at: value.expires_at, label: typeof value.label === "string" ? value.label : null };
  } catch {
    return null;
  }
}

export function passMessage(pass: AddinPass): string {
  return JSON.stringify({ kind: PASS_MESSAGE, ...pass });
}

function live(pass: AddinPass | null): AddinPass | null {
  return pass && Date.parse(pass.expires_at) > Date.now() ? pass : null;
}

export function storedPass(): AddinPass | null {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw ? live(parsePassMessage(raw)) : null;
  } catch {
    return null;
  }
}

export function savePass(pass: AddinPass | null): void {
  try {
    if (pass) window.localStorage.setItem(STORAGE_KEY, passMessage(pass));
    else window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // storage refused (a private window): the pass lasts as long as the pane
  }
}

/**
 * Use `pass` for every API call from the pane; `onEnded` when the API says it is over. Set during render, as
 * the site's own sign-in sets its token source (AuthProvider), so the pane's first calls already carry it.
 */
export function applyPass(pass: AddinPass | null, onEnded: () => void): void {
  setTokenSource(pass ? async () => (live(pass) ? pass.token : null) : null);
  setUnauthorizedHandler(pass ? onEnded : null);
}

/** Sign the pane out: the pass ends on the server too. */
export async function endPass(): Promise<void> {
  await api.DELETE("/api/addin/session").catch(() => undefined);
  savePass(null);
  setTokenSource(null);
  setUnauthorizedHandler(null);
}

/** From the dialog, signed in on the site: a new pass for the pane. */
export async function newPass(): Promise<{ pass?: Omit<AddinPass, "label">; error?: string }> {
  const { data, error } = await api.POST("/api/addin/sessions");
  if (data) return { pass: { token: data.token, expires_at: data.expires_at } };
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  return { error: typeof detail === "string" && /^[a-z_]{3,40}$/.test(detail) ? detail : "failed" };
}
