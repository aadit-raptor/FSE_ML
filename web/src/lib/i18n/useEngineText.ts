"use client";

import { useTranslations } from "next-intl";
import { useCallback } from "react";

/**
 * Labels the API sends, translated where the catalogue knows them (PLAN.md
 * 2.3b).
 *
 * Tranche names ("Senior Term Loan"), equity-bridge step names and Monte Carlo
 * driver names are produced by the engine (`lbo_engine/`, `core/deal.py`,
 * `simulation/`), which is model logic and stays as it is (CLAUDE.md "Keep
 * model logic as is"). So the web app translates them on the way in: the
 * engine's own words become a key, and anything the catalogue has no entry for
 * is shown exactly as the API sent it -- a new tranche type still reads
 * correctly in English before anyone translates it.
 *
 * i18n-keys: engine.*
 */
export type EngineLabelKind = "tranche" | "bridgeAxis" | "bridgeRow" | "driver";

/** "EBITDA\ngrowth" -> "EBITDAGrowth", so an engine label becomes one key. */
function camel(name: string): string {
  return name
    .split(/[^A-Za-z0-9]+/)
    .filter(Boolean)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join("");
}

export function useEngineLabel(kind: EngineLabelKind): (name: string) => string {
  const t = useTranslations("engine");
  return useCallback(
    (name) => {
      const key = `${kind}${camel(name)}`;
      return t.has(key) ? t(key) : name;
    },
    [t, kind],
  );
}
