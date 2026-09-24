/**
 * Numbers and dates in the account's locale (PLAN.md 2.3a). The one place
 * that turns numbers into text: lib/format.ts builds on it, and
 * tests/test_no_hardcoded_locale.py fails on formatting anywhere else.
 *
 * The style is the account's locale plus its digit grouping: as the locale
 * does it, always in thousands (1,000,000), or in lakh and crore
 * (10,00,000). It is set during render by `LocaleScope` (components/shell),
 * before any screen draws, from the profile; pages never render on the
 * server before the session is known, so server and browser can't disagree.
 *
 * Figures always use Latin digits so the monospaced columns line up and
 * typed numbers parse the same way everywhere; native digits are part of
 * the translation work (PLAN.md 2.3b, 7.8).
 */

export type DigitGrouping = "locale" | "thousands" | "lakh";
export type NumberStyle = { locale: string; grouping: DigitGrouping };

/** Before the account answers, and whenever its locale can't be used. */
export const DEFAULT_STYLE: NumberStyle = { locale: "en-US", grouping: "locale" };

/** Formats the account screen suggests besides the browser's own languages; any BCP 47 tag works. */
export const LOCALE_SUGGESTIONS = ["en-GB", "en-US", "en-IN", "de-DE", "fr-FR", "pt-BR", "ja-JP", "hi-IN"];

export const DIGIT_GROUPINGS: { value: DigitGrouping; label: string }[] = [
  { value: "locale", label: "As the format does" },
  { value: "thousands", label: "Thousands" },
  { value: "lakh", label: "Lakh and crore" },
];

let current: NumberStyle = DEFAULT_STYLE;

function usable(locale: string): boolean {
  try {
    return Intl.NumberFormat.supportedLocalesOf([locale]).length > 0;
  } catch {
    return false;
  }
}

/** Use this style from now on; an unknown locale falls back to the default. */
export function setNumberStyle(style: Partial<NumberStyle> | null | undefined): void {
  const locale = style?.locale && usable(style.locale) ? style.locale : DEFAULT_STYLE.locale;
  const grouping = DIGIT_GROUPINGS.some((g) => g.value === style?.grouping) ? (style!.grouping as DigitGrouping) : "locale";
  if (locale !== current.locale || grouping !== current.grouping) current = { locale, grouping };
}

export function numberStyle(): NumberStyle {
  return current;
}

function latin(locale: string): string {
  try {
    return new Intl.Locale(locale, { numberingSystem: "latn" }).toString();
  } catch {
    return locale;
  }
}

const formatters = new Map<string, Intl.NumberFormat>();

function formatter(locale: string, options: Intl.NumberFormatOptions): Intl.NumberFormat {
  const key = `${locale}|${JSON.stringify(options)}`;
  let f = formatters.get(key);
  if (!f) {
    f = new Intl.NumberFormat(latin(locale), options);
    formatters.set(key, f);
  }
  return f;
}

type Separators = { decimal: string; group: string };
const separatorCache = new Map<string, Separators>();

/** The locale's decimal mark and thousands separator: "," and "." in de-DE. */
export function separators(locale = current.locale): Separators {
  let s = separatorCache.get(locale);
  if (!s) {
    const parts = formatter(locale, { useGrouping: true }).formatToParts(1234567.5);
    s = {
      decimal: parts.find((p) => p.type === "decimal")?.value ?? ".",
      group: parts.find((p) => p.type === "group")?.value ?? ",",
    };
    separatorCache.set(locale, s);
  }
  return s;
}

/** Digits grouped in threes, or lakh and crore: 3 then 2s. */
function regroup(digits: string, grouping: "thousands" | "lakh", group: string): string {
  if (digits.length <= 3) return digits;
  const head = digits.slice(0, -3);
  const size = grouping === "lakh" ? 2 : 3;
  const out: string[] = [];
  for (let end = head.length; end > 0; end -= size) out.unshift(head.slice(Math.max(0, end - size), end));
  return [...out, digits.slice(-3)].join(group);
}

export type NumberOptions = {
  /** Fixed decimals; or give minDecimals and maxDecimals */
  decimals?: number;
  minDecimals?: number;
  maxDecimals?: number;
  /** The value is a fraction shown as a percentage: 0.212 -> 21.2% */
  percent?: boolean;
  /** Always show the sign: +276.3 */
  signed?: boolean;
  /** Group long numbers (default); off for text in inputs */
  grouping?: boolean;
  /** Show as an amount of this ISO 4217 currency (the account screen's example) */
  currency?: string;
};

/** A number in the current style: 1,145.5 / 1.145,5 / 11,45,000.0 / 21,2 %. */
export function formatNumber(v: number, opts: NumberOptions = {}, style: NumberStyle = current): string {
  const min = opts.decimals ?? opts.minDecimals ?? 0;
  const max = Math.max(min, opts.decimals ?? opts.maxDecimals ?? min);
  const grouped = opts.grouping !== false;
  const custom = grouped && style.grouping !== "locale";
  const options: Intl.NumberFormatOptions = {
    minimumFractionDigits: min,
    maximumFractionDigits: max,
    useGrouping: grouped && !custom,
    ...(opts.percent ? { style: "percent" } : {}),
    ...(opts.currency ? { style: "currency", currency: opts.currency } : {}),
    ...(opts.signed ? { signDisplay: "always" } : {}),
  };
  const f = formatter(style.locale, options);
  if (!custom) return f.format(v);
  const { group } = separators(style.locale);
  return f
    .formatToParts(v)
    .map((p) => (p.type === "integer" ? regroup(p.value, style.grouping as "thousands" | "lakh", group) : p.value))
    .join("");
}

/** A whole count: 50,000 paths. */
export function formatCount(n: number, style: NumberStyle = current): string {
  return formatNumber(Math.round(n), { decimals: 0 }, style);
}

/** A number as an input shows it: no grouping, the locale's decimal mark, the fewest decimals that keep it exact (at least minDecimals, at most 4). */
export function formatInput(v: number, minDecimals: number, style: NumberStyle = current): string {
  let text = v.toFixed(4);
  for (let d = minDecimals; d <= 4; d++) {
    const s = v.toFixed(d);
    if (Math.abs(Number(s) - v) < 1e-9) {
      text = s;
      break;
    }
  }
  const { decimal } = separators(style.locale);
  return decimal === "." ? text : text.replace(".", decimal);
}

/**
 * A typed number, read the way the locale writes it: "1.234,5" in German,
 * "1,234.5" or "12,34,567" in English. Spaces and apostrophes are grouping
 * (fr-FR, de-CH). Where the decimal mark is a comma, a lone dot is also
 * taken as a decimal mark, since inputs never show grouping. NaN if it
 * isn't a number.
 */
export function parseNumber(text: string, style: NumberStyle = current): number {
  let s = text.trim().replace(/[\s\u00a0\u202f'\u2019]/g, "").replace(/\u2212/g, "-");
  if (separators(style.locale).decimal === ",") {
    if (s.includes(",")) s = s.replace(/\./g, "").replace(",", ".");
  } else {
    s = s.replace(/,/g, "");
  }
  return /^[+-]?(\d+\.?\d*|\.\d+)$/.test(s) ? Number(s) : NaN;
}

/** A date and time in the current locale: 24 Sept 2026, 14:05. */
export function formatDateTime(
  date: Date,
  options: Intl.DateTimeFormatOptions = { dateStyle: "medium", timeStyle: "short" },
  style: NumberStyle = current,
): string {
  return new Intl.DateTimeFormat(latin(style.locale), options).format(date);
}

/** A month's name in the current locale: "March", "März". */
export function monthName(month: number, style: NumberStyle = current): string {
  return new Intl.DateTimeFormat(latin(style.locale), { month: "long", timeZone: "UTC" }).format(Date.UTC(2000, month - 1, 1));
}
