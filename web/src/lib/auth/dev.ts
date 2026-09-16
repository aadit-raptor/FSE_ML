/**
 * Development sign-in: who is signed in when there is no Clerk instance.
 *
 * The name sits in a cookie so the proxy (src/proxy.ts) can send a signed-out
 * visitor to /sign-in exactly as Clerk does, and the browser tests can save
 * and reuse a signed-in state. It is not a credential: the API only accepts
 * `dev:<name>` outside production and only while Clerk is unconfigured
 * (api/auth.py).
 */
export const DEV_COOKIE = "fse_dev_user";
export const DEV_USER = "e2e";
/** Names the API accepts: letters, digits, hyphen and underscore. */
const NAME = /^[A-Za-z0-9_-]{1,64}$/;

export function isDevUserName(name: string): boolean {
  return NAME.test(name);
}

export function devToken(name: string): string {
  return `dev:${name}`;
}

/** The signed-in development user from a cookie string, or null. */
export function devUserFromCookies(cookies: string | undefined | null): string | null {
  if (!cookies) return null;
  for (const part of cookies.split(";")) {
    const [key, ...rest] = part.trim().split("=");
    if (key === DEV_COOKIE) {
      const value = decodeURIComponent(rest.join("="));
      return isDevUserName(value) ? value : null;
    }
  }
  return null;
}

export function readDevUser(): string | null {
  if (typeof document === "undefined") return null;
  return devUserFromCookies(document.cookie);
}

// The cookie is an external store: React subscribes to it rather than copying
// it into state, so signing in or out updates every screen at once.
const listeners = new Set<() => void>();

export function subscribeDevUser(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function changed(): void {
  for (const listener of listeners) listener();
}

export function writeDevUser(name: string): void {
  document.cookie = `${DEV_COOKIE}=${encodeURIComponent(name)}; path=/; max-age=${60 * 60 * 24}; SameSite=Lax`;
  changed();
}

export function clearDevUser(): void {
  document.cookie = `${DEV_COOKIE}=; path=/; max-age=0; SameSite=Lax`;
  changed();
}
