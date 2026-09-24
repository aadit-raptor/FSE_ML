/**
 * Number formatting. Money is a number in the screen's currency and unit (lib/money.ts), rates from the engine are
 * fractions (0.157 = 15.7%). Everything follows the account's locale and digit grouping (lib/locale.ts).
 */
import { formatCount, formatInput, formatNumber } from "./locale";

type Maybe = number | null | undefined;

export const isNum = (v: Maybe): v is number => typeof v === "number" && Number.isFinite(v);

/** 1,145.5 */
export function fmtMoney(v: Maybe): string {
  return isNum(v) ? formatNumber(v, { decimals: 1 }) : "n/a";
}

/** Signed money: +276.3 / -38.6 */
export function fmtDelta(v: Maybe): string {
  return isNum(v) ? formatNumber(v, { decimals: 1, signed: true }) : "n/a";
}

/** Engine fraction to percent: 0.2117 -> 21.2% */
export function fmtRate(v: Maybe, digits = 1): string {
  return isNum(v) ? formatNumber(v, { decimals: digits, percent: true }) : "n/a";
}

/** Input percentage (already x100): 60 -> 60.0% */
export function fmtPct(v: Maybe, digits = 1): string {
  return isNum(v) ? formatNumber(v / 100, { decimals: digits, percent: true }) : "n/a";
}

/** 2.35x */
export function fmtMultiple(v: Maybe, digits = 2): string {
  return isNum(v) ? `${formatNumber(v, { decimals: digits })}x` : "n/a";
}

/** A plain number with fixed decimals: 0.87, -1.25 */
export function fmtNumber(v: Maybe, digits = 1, signed = false): string {
  return isNum(v) ? formatNumber(v, { decimals: digits, signed }) : "n/a";
}

/** A count: 50,000 */
export function fmtCount(v: Maybe): string {
  return isNum(v) ? formatCount(v) : "n/a";
}

/** Shortest representation with at least `minDecimals`, at most 4, as an input shows it. */
export function fmtInput(v: number, minDecimals: number): string {
  return formatInput(v, minDecimals);
}

/** A chart tick: 1,500 / 0.25 */
export function fmtAxis(v: number): string {
  return formatNumber(v, { minDecimals: 0, maxDecimals: 3 });
}
