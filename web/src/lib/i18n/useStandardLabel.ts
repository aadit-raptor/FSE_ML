"use client";

import { useTranslations } from "next-intl";
import { useCallback } from "react";

/**
 * Statement labels that follow the accounting standard (PLAN.md 2.6). The
 * catalogue's ordinary words are US GAAP's ("Interest expense", "Net income");
 * a standard that names a line differently has its own word in the
 * `standards` namespace, keyed by the standard and the ordinary key
 * ("standards.ifrs.rowInterestExpense": "Finance costs").
 *
 * Call sites keep asking for the ordinary message, so it stays checked:
 * `std("rowNetIncome", t("rowNetIncome"))`.
 *
 * i18n-keys: standards.*
 */
export function useStandardLabel(standard: string) {
  const s = useTranslations("standards");
  return useCallback(
    (key: string, ordinary: string, values?: Record<string, string | number>): string => {
      const own = `${standard}.${key}`;
      // A message key built at run time: the declaration above covers it
      return standard && s.has(own as never) ? s(own as never, values as never) : ordinary;
    },
    [s, standard],
  );
}
