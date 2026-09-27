/**
 * Fiscal years (PLAN.md 2.3a). A fiscal year is named by the calendar year it ends in, as filings and
 * EDGAR do: a December year-end's 2025 is "FY2025"; any other year-end spans two calendar years and says
 * so, "FY2024/25" for April 2024 to March 2025. Labels only: the model counts years from 1.
 *
 * The wording of the label -- "FY" and the shape of a spanning year -- lives in the translation files
 * (PLAN.md 2.3b), so this takes the formatter rather than writing English itself. `useFiscalLabels`
 * (lib/i18n/useFiscalLabels.ts) is what screens use.
 */

export type Fiscal = {
  /** Month the fiscal year ends, 1-12 */
  endMonth: number;
  /**
   * A fiscal year, named by the year it ends in: a deal's first projected year, a company's latest
   * historical year. Null when unknown, and the plain labels (Y1, LTM) stay.
   */
  year: number | null;
};

export const MONTHS = Array.from({ length: 12 }, (_, i) => i + 1);

/** How a language writes the two shapes of a fiscal year's name. */
export type FiscalFormat = {
  /** A year that starts and ends in the same calendar year: 2025 -> "FY2025" */
  single: (endYear: number) => string;
  /** A year that spans two: 2025 with a March end -> "FY2024/25" */
  span: (first: number, secondTwoDigits: string) => string;
};

/** "FY2025" (December year-end) or "FY2024/25", written the way this language does. */
export function fiscalYearLabel(endYear: number, endMonth: number, format: FiscalFormat): string {
  if (endMonth === 12) return format.single(endYear);
  return format.span(endYear - 1, String(endYear % 100).padStart(2, "0"));
}

/** n consecutive fiscal-year labels from `fiscal.year`, or `fallback(i)` for each when the year isn't known. */
export function fiscalYearLabels(n: number, fiscal: Fiscal, fallback: (i: number) => string, format: FiscalFormat): string[] {
  const { year, endMonth } = fiscal;
  return Array.from({ length: n }, (_, i) => (year === null ? fallback(i) : fiscalYearLabel(year + i, endMonth, format)));
}
