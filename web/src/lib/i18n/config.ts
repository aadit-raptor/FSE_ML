/**
 * Interface language and text direction (PLAN.md 2.3b).
 *
 * Every word on screen comes from a translation file in `web/messages`; this
 * decides which file, and which way the layout runs. Both follow the
 * **account's locale** -- the same answer that chooses number and date
 * formats (PLAN.md 2.3a) -- and never a segment in the URL, so `src/proxy.ts`
 * and the content security policy it builds are untouched.
 *
 * English is the only complete catalogue; more languages are PLAN.md 7.8. A
 * locale with no catalogue of its own still gets its own numbers, dates and
 * text direction: only the words fall back to English. That is deliberate --
 * it is how the right-to-left layout is provable today (`e2e/text.spec.ts`
 * signs in as an Arabic-Egypt account), and the day an Arabic catalogue
 * lands, nothing about the layout has to change.
 */

export const DEFAULT_LANGUAGE = "en";

/** Languages with a catalogue in `web/messages`. PLAN.md 7.8 adds to this. */
export const LANGUAGES = [DEFAULT_LANGUAGE] as const;

export type Language = (typeof LANGUAGES)[number];

export type Direction = "ltr" | "rtl";

function isLanguage(tag: string): tag is Language {
  return (LANGUAGES as readonly string[]).includes(tag);
}

/**
 * Which catalogue an account's locale reads: the closest we have to its
 * language, else English: a German-Germany account and a plain German one both
 * look for the German catalogue.
 */
export function languageFor(locale: string | null | undefined): Language {
  if (!locale) return DEFAULT_LANGUAGE;
  let primary = locale.split(/[-_]/)[0].toLowerCase();
  try {
    primary = new Intl.Locale(locale).language.toLowerCase();
  } catch {
    // Not a tag the browser can parse: the split above is good enough
  }
  return isLanguage(primary) ? primary : DEFAULT_LANGUAGE;
}

/**
 * Languages written right to left, by their script and by their language
 * subtag, for browsers whose `Intl.Locale` has no text info yet.
 */
const RTL_SCRIPTS = new Set(["Adlm", "Arab", "Aran", "Hebr", "Mand", "Mend", "Nkoo", "Rohg", "Samr", "Syrc", "Thaa", "Yezi"]);
const RTL_LANGUAGES = new Set(["ar", "ckb", "dv", "fa", "he", "ku", "ps", "sd", "ug", "ur", "yi"]);

type WithTextInfo = Intl.Locale & {
  getTextInfo?: () => { direction?: string };
  textInfo?: { direction?: string };
};

/** Which way the interface runs for this locale: "rtl" for Arabic or Hebrew. */
export function directionFor(locale: string | null | undefined): Direction {
  try {
    const parsed = new Intl.Locale(locale || DEFAULT_LANGUAGE).maximize() as WithTextInfo;
    const direction = parsed.getTextInfo?.().direction ?? parsed.textInfo?.direction;
    if (direction === "rtl" || direction === "ltr") return direction;
    if (parsed.script && RTL_SCRIPTS.has(parsed.script)) return "rtl";
    if (RTL_LANGUAGES.has(parsed.language.toLowerCase())) return "rtl";
  } catch {
    // Not a tag the browser can parse: left to right, like the default
  }
  return "ltr";
}
