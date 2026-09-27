/**
 * Honest labels for inception-era numbers (PLAN.md 2.1). Defaults, ranges,
 * correlations and presets were typed in without a source; the example deals
 * and the ML training sets are small. These labels stay until sourced data
 * replaces them (PLAN.md 4.3, 4.4, 2.7, 2.8, 5.2, 5.8).
 *
 * The wording is in the `provenance` namespace of the translation files
 * (PLAN.md 2.3b); this works out the counts and spans the wording needs.
 */

export type HistoricalSample = { deals: number; first_year: number | null; last_year: number | null };

/**
 * The counts behind "4 example deals from the 2006-2013 US market".
 * Counts deals with a deal year; the blank custom-deal template has none.
 */
export function backtestSample(years: (string | number)[]): { count: number; first: number | null; last: number | null } {
  const ys = years.map(Number).filter((y) => Number.isFinite(y) && y > 0);
  if (!ys.length) return { count: 0, first: null, last: null };
  return { count: ys.length, first: Math.min(...ys), last: Math.max(...ys) };
}
