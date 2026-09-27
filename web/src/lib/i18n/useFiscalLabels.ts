"use client";

import { useTranslations } from "next-intl";
import { useMemo } from "react";

import { type Fiscal, type FiscalFormat, fiscalYearLabels } from "@/lib/fiscal";
import { monthName } from "@/lib/locale";

/**
 * Fiscal-year column labels in the account's language (PLAN.md 2.3a, 2.3b).
 *
 * "FY2025", "FY2024/25", the plain "Y1" a deal falls back to, the "LTM" and
 * "F+1" a forecast falls back to, and the line under the year-end control all
 * come from the `fiscal` namespace of the translation files.
 */
export type FiscalLabels = {
  /** n projection years: "FY2027" ... or "Y1" ... when the deal has no first fiscal year. */
  deal: (n: number, fiscal: Fiscal) => string[];
  /** n historical years, oldest first: "FY2022/23" ... or "LTM-2" ... "LTM". */
  history: (n: number, fiscal: Fiscal) => string[];
  /** n forecast years: "FY2025/26" ... or "F+1" ... */
  forward: (n: number, fiscal: Fiscal) => string[];
  /** "Fiscal years end in March" */
  note: (endMonth: number) => string;
  /** "LTM", the last historical column when no fiscal year is set. */
  ltm: string;
};

export function useFiscalLabels(): FiscalLabels {
  const t = useTranslations("fiscal");
  return useMemo(() => {
    // Years and indexes go in as text: an ICU number argument would be grouped
    // ("FY2,025") by the account's own number format
    const format: FiscalFormat = {
      single: (endYear) => t("labelSingle", { year: String(endYear) }),
      span: (first, second) => t("labelSpan", { first: String(first), second }),
    };
    return {
      deal: (n, fiscal) => fiscalYearLabels(n, fiscal, (i) => t("yearShort", { number: String(i + 1) }), format),
      history: (n, fiscal) =>
        fiscalYearLabels(
          n,
          { ...fiscal, year: fiscal.year === null ? null : fiscal.year - n + 1 },
          (i) => (i === n - 1 ? t("ltm") : t("ltmMinus", { back: String(n - 1 - i) })),
          format,
        ),
      forward: (n, fiscal) =>
        fiscalYearLabels(n, { ...fiscal, year: fiscal.year === null ? null : fiscal.year + 1 }, (i) => t("forward", { number: String(i + 1) }), format),
      note: (endMonth) => t("yearEndNote", { month: monthName(endMonth) }),
      ltm: t("ltm"),
    };
  }, [t]);
}
