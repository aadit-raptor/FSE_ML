/** Number formatting. Money is a number in the screen's currency and unit (lib/money.ts), rates from the engine are fractions (0.157 = 15.7%). */

const money = new Intl.NumberFormat("en-US", { minimumFractionDigits: 1, maximumFractionDigits: 1 });

type Maybe = number | null | undefined;

export const isNum = (v: Maybe): v is number => typeof v === "number" && Number.isFinite(v);

/** 1,145.5 */
export function fmtMoney(v: Maybe): string {
  return isNum(v) ? money.format(v) : "n/a";
}

/** Signed money: +276.3 / -38.6 */
export function fmtDelta(v: Maybe): string {
  if (!isNum(v)) return "n/a";
  return `${v < 0 ? "-" : "+"}${money.format(Math.abs(v))}`;
}

/** Engine fraction to percent: 0.2117 -> 21.2% */
export function fmtRate(v: Maybe, digits = 1): string {
  return isNum(v) ? `${(v * 100).toFixed(digits)}%` : "n/a";
}

/** Input percentage (already x100): 60 -> 60.0% */
export function fmtPct(v: Maybe, digits = 1): string {
  return isNum(v) ? `${v.toFixed(digits)}%` : "n/a";
}

export function fmtMultiple(v: Maybe, digits = 2): string {
  return isNum(v) ? `${v.toFixed(digits)}x` : "n/a";
}

/** Shortest representation with at least `minDecimals`, at most 4. */
export function fmtInput(v: number, minDecimals: number): string {
  for (let d = minDecimals; d <= 4; d++) {
    const s = v.toFixed(d);
    if (Math.abs(Number(s) - v) < 1e-9) return s;
  }
  return v.toFixed(4);
}
