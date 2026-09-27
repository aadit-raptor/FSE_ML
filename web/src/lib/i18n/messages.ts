/**
 * The translation catalogues (PLAN.md 2.3b).
 *
 * One file per language in `web/messages`, imported statically: English is the
 * only one, so there is nothing to fetch and no second render once the words
 * arrive. PLAN.md 7.8 adds languages here; when there are enough of them to
 * matter for the bundle, this is the one place that turns into a dynamic
 * `import()`.
 */
import en from "../../../messages/en.json";

import { DEFAULT_LANGUAGE, type Language } from "./config";

export type Catalogue = typeof en;

const CATALOGUES: Record<Language, Catalogue> = { en };

export function messagesFor(language: Language): Catalogue {
  return CATALOGUES[language] ?? CATALOGUES[DEFAULT_LANGUAGE];
}

/** English, for the few strings a server component must produce (page titles). */
export const DEFAULT_MESSAGES: Catalogue = en;
