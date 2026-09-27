"use client";

import { useTranslations } from "next-intl";
import { useCallback } from "react";

import { type FieldProblem } from "@/lib/fields";
import { fmtInput } from "@/lib/format";
import { MONEY, moneyLabel, type Money } from "@/lib/money";

/** Why a value was refused, in the account's language (PLAN.md 2.3b). */
export function useFieldProblem(): (problem: FieldProblem | null) => string | null {
  const t = useTranslations("validation");
  return useCallback(
    (problem) => (problem ? t(problem.key, { min: fmtInput(problem.bound ?? 0, 0), max: fmtInput(problem.bound ?? 0, 0) }) : null),
    [t],
  );
}

/**
 * The unit beside a numeric input. Either a symbol that is the same everywhere
 * ("%", "x"), the money label of whatever is on screen (PLAN.md 2.2), or a
 * short word with a key in the `units` namespace.
 *
 * i18n-keys: units.yr, units.d, units.n
 */
export function useUnitLabel(): (unit: string, money: Money) => string {
  const t = useTranslations("units");
  return useCallback(
    (unit, money) => {
      if (unit === MONEY) return moneyLabel(money);
      return t.has(unit) ? t(unit) : unit;
    },
    [t],
  );
}
