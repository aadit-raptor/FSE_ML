"use client";

import { useTranslations } from "next-intl";
import { useMemo } from "react";

import { backtestSample } from "@/lib/provenance";

/**
 * The honest labels of PLAN.md 2.1, in the account's language (PLAN.md 2.3b).
 * `web/e2e/provenance.spec.ts` checks the wording, so keep the keys and the
 * screens that show them in step.
 */
export function useProvenance() {
  const t = useTranslations("provenance");
  return useMemo(
    () => ({
      illustrative: t("illustrative"),
      illustrativeDetail: t("illustrativeDetail"),
      /** "4 example deals from the 2006–2013 US market; not a validation of the model" */
      backtest: (years: (string | number)[]) => {
        const { count, first, last } = backtestSample(years);
        const span =
          first === null || last === null
            ? ""
            : first === last
              ? t("backtestSpanOne", { year: String(first) })
              : t("backtestSpanRange", { first: String(first), last: String(last) });
        return t("backtestSample", { count, span });
      },
    }),
    [t],
  );
}
