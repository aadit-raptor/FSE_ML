/**
 * Number formatting. Money is a number in the screen's currency and unit (lib/money.ts), rates from the engine are
 * fractions (0.157 = 15.7%). Everything follows the account's locale and digit grouping (lib/locale.ts).
 *
 * The one word here -- what stands in for a figure the model didn't produce -- comes from the translation
 * files (PLAN.md 2.3b). Like the number style, it is a module value set during render by `I18nScope`, so the
 * formatters stay plain functions that any module can call.
 */
import { formatCount, formatInput, formatNumber } from "./locale";

type Maybe = number | null | undefined;

let missing = "n/a"; // text-ok: replaced by the account's own wording before any screen draws

/** What to show where the model has no figure ("n/a"). */
export function setMissingText(text: string): void {
  missing = text;
}

export const isNum = (v: Maybe): v is number => typeof v === "number" && Number.isFinite(v);

/** 1,145.5 */
export function fmtMoney(v: Maybe): string {
  return isNum(v) ? formatNumber(v, { decimals: 1 }) : missing;
}

/** Signed money: +276.3 / -38.6 */
export function fmtDelta(v: Maybe): string {
  return isNum(v) ? formatNumber(v, { decimals: 1, signed: true }) : missing;
}

/** Engine fraction to percent: 0.2117 -> 21.2% */
export function fmtRate(v: Maybe, digits = 1): string {
  return isNum(v) ? formatNumber(v, { decimals: digits, percent: true }) : missing;
}

/** Input percentage (already x100): 60 -> 60.0% */
export function fmtPct(v: Maybe, digits = 1): string {
  return isNum(v) ? formatNumber(v / 100, { decimals: digits, percent: true }) : missing;
}

/** 2.35x */
export function fmtMultiple(v: Maybe, digits = 2): string {
  return isNum(v) ? `${formatNumber(v, { decimals: digits })}x` : missing;
}

/** A plain number with fixed decimals: 0.87, -1.25 */
export function fmtNumber(v: Maybe, digits = 1, signed = false): string {
  return isNum(v) ? formatNumber(v, { decimals: digits, signed }) : missing;
}

/** A count: 50,000 */
export function fmtCount(v: Maybe): string {
  return isNum(v) ? formatCount(v) : missing;
}

/** Shortest representation with at least `minDecimals`, at most 4, as an input shows it. */
export function fmtInput(v: number, minDecimals: number): string {
  return formatInput(v, minDecimals);
}

/** A chart tick: 1,500 / 0.25 */
export function fmtAxis(v: number): string {
  return formatNumber(v, { minDecimals: 0, maxDecimals: 3 });
}
