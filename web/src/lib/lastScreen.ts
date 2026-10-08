/**
 * The screen this browser last showed, so a visitor who is still signed in
 * and opens the site again carries on where they were (the root page reads
 * it). A sign-in always lands on the launcher instead.
 *
 * Only a mode/step address is kept: a convenience of this browser, never an
 * account's data, so it lives in localStorage and every read and write
 * survives storage being blocked.
 */
import { parsePath } from "./nav";

const KEY = "variater.lastScreen";

export function rememberScreen(pathname: string): void {
  const { mode, step } = parsePath(pathname);
  if (!mode || !step) return;
  try {
    window.localStorage.setItem(KEY, pathname);
  } catch {
    // Storage blocked (a private window, cleared site data): nothing to carry on from next time
  }
}

/** The last screen, if it is still a screen of the app. */
export function lastScreen(): string | null {
  let stored: string | null = null;
  try {
    stored = window.localStorage.getItem(KEY);
  } catch {
    return null;
  }
  if (!stored) return null;
  const { mode, step } = parsePath(stored);
  return mode && step ? stored : null;
}
