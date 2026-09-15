/**
 * Honest labels for inception-era numbers (PLAN.md 2.1). Defaults, ranges,
 * correlations and presets were typed in without a source; the example deals
 * and the ML training sets are small. These labels stay until sourced data
 * replaces them (PLAN.md 4.3, 4.4, 2.7, 2.8, 5.2, 5.8).
 */

export const ILLUSTRATIVE = "Illustrative defaults — not market data";

export const ILLUSTRATIVE_DETAIL =
  "Typed in when the model was first built, with no source, sample or date. Replace them with your own figures for a real deal.";

/**
 * "4 example deals from the 2006–2013 US market; not a validation of the model".
 * Counts deals with a deal year; the blank custom-deal template has none.
 */
export function backtestSampleLabel(years: (string | number)[]): string {
  const ys = years.map(Number).filter((y) => Number.isFinite(y) && y > 0);
  const lo = Math.min(...ys);
  const hi = Math.max(...ys);
  const span = !ys.length ? "" : lo === hi ? `${lo} ` : `${lo}–${hi} `;
  return `${ys.length} example ${ys.length === 1 ? "deal" : "deals"} from the ${span}US market; not a validation of the model`;
}

export type HistoricalSample = { deals: number; first_year: number | null; last_year: number | null };

/** "Early estimate based on 30 historical deals (1989–2016)". */
export function riskSampleLabel(s: HistoricalSample): string {
  const span = s.first_year && s.last_year ? ` (${s.first_year}–${s.last_year})` : "";
  return `Early estimate based on ${s.deals} historical deals${span}`;
}
