import type { Schemas } from "@/lib/api/client";
import { changedKeys, type FieldSpec, validate } from "@/lib/fields";
import { DEFAULT_MONEY, MONEY } from "@/lib/money";

/** One facility in the deal's debt (PLAN.md 2.4). The editor is 2.4b; until then a
 *  deal gets these through the API, and the screens have to carry them faithfully. */
export type Tranche = Schemas["TrancheIn"];

export type DealInputs = Required<Schemas["DealInputsIn"]>;
export type DealRun = Schemas["DealRunResponse"];
/** Fiscal year labels (PLAN.md 2.3a): not model inputs, set in their own rail group */
export type FiscalDealKey = "fiscal_year_end_month" | "first_fiscal_year";
export type NumericDealKey = Exclude<{ [K in keyof DealInputs]: DealInputs[K] extends number ? K : never }[keyof DealInputs], FiscalDealKey>;

/** Matches the API's DealInputsIn defaults (api/schemas.py). */
export const DEFAULT_INPUTS: DealInputs = {
  ebitda: 100,
  entry_mult: 10,
  exit_mult: 11,
  hold: 5,
  growth: 5,
  gross_margin: 40,
  opex: 18,
  tax: 25,
  da: 4,
  debt_pct: 60,
  senior_pct: 70,
  base_rate: 6.5,
  mezz_spread: 4,
  capex: 4,
  nwc: 1,
  mincash: 0,
  wsp_mode: false,
  ar_days: 45,
  inv_days: 30,
  ap_days: 60,
  currency: DEFAULT_MONEY.currency,
  unit: DEFAULT_MONEY.unit,
  fiscal_year_end_month: 12,
  first_fiscal_year: null,
  tranches: [],
};

/**
 * The inputs as the API gets them: the fiscal year labels only when set. The API stores them the same
 * way (db/deals.py), and an API from before them (a deploy still rolling out, or a rollback) keeps
 * answering every deal that doesn't use them.
 */
export function apiInputs(inputs: DealInputs): Schemas["DealInputsIn"] {
  const { fiscal_year_end_month, first_fiscal_year, tranches, ...rest } = inputs;
  // The generated type lists every defaulted field as present; the API fills in the ones left out
  return {
    ...rest,
    ...(fiscal_year_end_month !== DEFAULT_INPUTS.fiscal_year_end_month ? { fiscal_year_end_month } : {}),
    ...(first_fiscal_year !== null ? { first_fiscal_year } : {}),
    ...(tranches.length ? { tranches } : {}),
  } as Schemas["DealInputsIn"];
}

/** The tranche list with every amount in a new unit, so a deal keeps its size. */
export function tranchesInUnit(tranches: Tranche[], k: number): Tranche[] {
  if (k === 1) return tranches;
  return tranches.map((t) => ({
    ...t,
    amount: (t.amount ?? 0) * k,
    ...(t.amort_schedule?.length ? { amort_schedule: t.amort_schedule.map((x) => x * k) } : {}),
  }));
}

/** The deal's fiscal years, for labels. */
export function dealFiscal(inputs: DealInputs): { endMonth: number; year: number | null } {
  return { endMonth: inputs.fiscal_year_end_month, year: inputs.first_fiscal_year };
}

export type { FieldSpec };

/**
 * Bounds mirror DealInputsIn in api/schemas.py. The label of each field is in
 * the `fields` namespace of the translation files, under the same key
 * (PLAN.md 2.3b).
 */
export const FIELDS: Record<NumericDealKey, FieldSpec> = {
  ebitda: { unit: MONEY, step: 5, decimals: 1, min: 0, exclusiveMin: true },
  entry_mult: { unit: "x", step: 0.5, decimals: 1, min: 0, exclusiveMin: true },
  exit_mult: { unit: "x", step: 0.5, decimals: 1, min: 0, exclusiveMin: true },
  hold: { unit: "yr", step: 1, decimals: 0, min: 1, max: 15, integer: true },
  growth: { unit: "%", step: 0.5, decimals: 1, min: -50, max: 100 },
  gross_margin: { unit: "%", step: 1, decimals: 1, min: 0, max: 100 },
  opex: { unit: "%", step: 1, decimals: 1, min: 0, max: 100 },
  tax: { unit: "%", step: 1, decimals: 1, min: 0, max: 100 },
  da: { unit: "%", step: 0.5, decimals: 1, min: 0, max: 100 },
  debt_pct: { unit: "%", step: 1, decimals: 1, min: 0, max: 99 },
  senior_pct: { unit: "%", step: 1, decimals: 1, min: 0, max: 100 },
  base_rate: { unit: "%", step: 0.25, decimals: 2, min: 0, max: 50 },
  mezz_spread: { unit: "%", step: 0.25, decimals: 2, min: 0, max: 50 },
  capex: { unit: "%", step: 0.5, decimals: 1, min: 0, max: 100 },
  nwc: { unit: "%", step: 0.25, decimals: 2, min: -100, max: 100 },
  mincash: { unit: MONEY, step: 5, decimals: 1, min: 0 },
  ar_days: { unit: "d", step: 1, decimals: 0, min: 0, max: 365 },
  inv_days: { unit: "d", step: 1, decimals: 0, min: 0, max: 365 },
  ap_days: { unit: "d", step: 1, decimals: 0, min: 0, max: 365 },
};

export { changedKeys, validate };
