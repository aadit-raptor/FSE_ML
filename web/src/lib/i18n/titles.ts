/**
 * Browser-tab titles (PLAN.md 2.3b).
 *
 * A page's `metadata` is produced on the server, before anyone is known to be
 * signed in, so the account's language isn't available there: these come from
 * the English catalogue. `DocumentTitle` in the shell replaces the title with
 * the account's language as soon as the session is known, so the tab ends up
 * translated; the English one is only what a crawler or a still-loading tab
 * sees.
 *
 * The words are still in the translation file: only the choice of language is
 * fixed here.
 */
import { DEFAULT_MESSAGES } from "./messages";

type NavKey = keyof typeof DEFAULT_MESSAGES.nav;

/** "Deal · Deal inputs", from the same keys the step row uses. */
export function pageTitle(...keys: NavKey[]): string {
  return keys.map((key) => DEFAULT_MESSAGES.nav[key]).join(" · ");
}

// Screens outside the mode/step grid name themselves.
export const ACCOUNT_TITLE = DEFAULT_MESSAGES.account.title;
export const SIGN_IN_TITLE = DEFAULT_MESSAGES.auth.signIn;
export const SIGN_UP_TITLE = DEFAULT_MESSAGES.auth.signUp;
export const PRIVACY_TITLE = DEFAULT_MESSAGES.privacy.title;
export const APP_DESCRIPTION = DEFAULT_MESSAGES.app.description;
export const APP_BRAND = DEFAULT_MESSAGES.app.brand;
