import type { StackedBars } from "@/components/charts/StackedBars";
import { type DealRun } from "@/lib/deal/fields";
import { type Fiscal, fiscalYearLabels } from "@/lib/fiscal";
import { fmtNumber, fmtRate, isNum } from "@/lib/format";

/** "+1.2 pts vs 20% hurdle" with gain / loss tone. */
export function hurdleSub(irr: number | null | undefined, hurdle: number): { sub?: string; tone?: "gain" | "loss" } {
  if (!isNum(irr)) return {};
  const diff = (irr - hurdle) * 100;
  return {
    sub: `${fmtNumber(diff, 1, true)} pts vs ${fmtRate(hurdle, 0)} hurdle`,
    tone: diff >= 0 ? "gain" : "loss",
  };
}

/** Projection years: "FY2027" or "FY2026/27" once the deal has a first fiscal year (PLAN.md 2.3a), else "Y1". */
export const yearLabels = (n: number, fiscal: Fiscal) => fiscalYearLabels(n, fiscal, (i) => `Y${i + 1}`);

const TRANCHE_COLORS = ["var(--color-accent)", "var(--color-neutral-bar)", "var(--color-soft)"];

/** Debt balance per tranche: at close, then at each year end. */
export function debtSeries(result: DealRun, fiscal: Fiscal): Parameters<typeof StackedBars>[0] {
  const tranches = Object.entries(result.tranches);
  const years = tranches[0]?.[1].length ?? 0;
  return {
    categories: ["Close", ...yearLabels(years, fiscal)],
    series: tranches.map(([name, rows], i) => ({
      name,
      color: TRANCHE_COLORS[i % TRANCHE_COLORS.length],
      values: [rows[0]?.beginning_balance ?? 0, ...rows.map((r) => r.ending_balance ?? 0)],
    })),
    label: "Debt balance by tranche",
  };
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
