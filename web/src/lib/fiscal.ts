/**
 * Fiscal years (PLAN.md 2.3a). A fiscal year is named by the calendar year it ends in, as filings and
 * EDGAR do: a December year-end's 2025 is "FY2025"; any other year-end spans two calendar years and says
 * so, "FY2024/25" for April 2024 to March 2025. Labels only: the model counts years from 1.
 */
import { monthName } from "./locale";

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

/** "FY2025" (December year-end) or "FY2024/25". */
export function fiscalYearLabel(endYear: number, endMonth: number): string {
  if (endMonth === 12) return `FY${endYear}`;
  return `FY${endYear - 1}/${String(endYear % 100).padStart(2, "0")}`;
}

/** n consecutive fiscal-year labels from `fiscal.year`, or `fallback(i)` for each when the year isn't known. */
export function fiscalYearLabels(n: number, fiscal: Fiscal, fallback: (i: number) => string): string[] {
  const { year, endMonth } = fiscal;
  return Array.from({ length: n }, (_, i) => (year === null ? fallback(i) : fiscalYearLabel(year + i, endMonth)));
}

/** "Fiscal years end in March" */
export function yearEndNote(endMonth: number): string {
  return `Fiscal years end in ${monthName(endMonth)}`;
}
