"use client";

import { useTranslations } from "next-intl";
import { useCallback } from "react";

import type { StackedBars } from "@/components/charts/StackedBars";
import { type DealRun } from "@/lib/deal/fields";
import { fmtNumber, fmtRate, isNum } from "@/lib/format";
import { trancheTerm } from "@/lib/glossary";

/** The IRR against the hurdle as the screen writes it: "+1.2" points and "20%", or null without an IRR. */
export function hurdleFigures(irr: number | null | undefined, hurdle: number) {
  if (!isNum(irr)) return null;
  const diff = (irr - hurdle) * 100;
  return { diff: fmtNumber(diff, 1, true), hurdle: fmtRate(hurdle, 0), clears: diff >= 0 };
}

/** "+1.2 pts vs 20% hurdle" with gain / loss tone. */
export function useHurdleSub(): (irr: number | null | undefined, hurdle: number) => { sub?: string; tone?: "gain" | "loss" } {
  const t = useTranslations("deal");
  return useCallback(
    (irr, hurdle) => {
      const f = hurdleFigures(irr, hurdle);
      if (!f) return {};
      return { sub: t("hurdleSub", { diff: f.diff, hurdle: f.hurdle }), tone: f.clears ? "gain" : "loss" };
    },
    [t],
  );
}

const TRANCHE_COLORS = ["var(--color-accent)", "var(--color-neutral-bar)", "var(--color-soft)"];

/**
 * Debt balance per tranche: at close, then at each year end. The tranche names
 * come from the engine (lbo_engine/capital_structure.py); `trancheName`
 * translates the ones the catalogue knows and passes anything else through.
 */
export function debtSeries(
  result: DealRun,
  years: string[],
  labels: { close: string; chart: string; trancheName: (name: string) => string },
): Parameters<typeof StackedBars>[0] {
  const tranches = Object.entries(result.tranches);
  return {
    categories: [labels.close, ...years],
    categoryTerms: { [labels.close]: "close" },
    series: tranches.map(([name, rows], i) => ({
      name: labels.trancheName(name),
      term: facilityTerm(result, name),
      color: TRANCHE_COLORS[i % TRANCHE_COLORS.length],
      values: [rows[0]?.beginning_balance ?? 0, ...rows.map((r) => r.ending_balance ?? 0)],
    })),
    label: labels.chart,
  };
}

/** A facility's glossary entry: its kind's when the deal lists its facilities, else the engine's name's. */
export function facilityTerm(result: DealRun, name: string): string | null {
  return trancheTerm(name, result.capital_structure?.find((c) => c.name === name)?.kind);
}

/** How many year columns the debt schedule has. */
export function debtYears(result: DealRun): number {
  return Object.values(result.tranches)[0]?.length ?? 0;
}

/** Numeric array from the loosely typed debt_schedule totals. */
export function totals(result: DealRun, key: string): number[] {
  const v = result.debt_schedule[key];
  return Array.isArray(v) ? v.map((x) => (typeof x === "number" ? x : NaN)) : [];
}

/** Row and column of the base case in the exit sensitivity grid. */
export function baseCell(result: DealRun, exitMult: number, hold: number) {
  const s = result.exit_sensitivity;
  let row = -1;
  let best = Infinity;
  s.exit_multiples.forEach((m, i) => {
    const d = Math.abs(m - exitMult);
    if (d < best - 1e-9) {
      best = d;
      row = i;
    }
  });
  // Grid multiples are rounded to 0.1x, so allow that much drift
  return { row: best <= 0.05 + 1e-9 ? row : -1, col: s.holding_periods.indexOf(hold) };
}
