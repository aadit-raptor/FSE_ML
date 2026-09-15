import type { Schemas } from "@/lib/api/client";
import { changedKeys, type FieldSpec, validate } from "@/lib/fields";

export type DealInputs = Required<Schemas["DealInputsIn"]>;
export type DealRun = Schemas["DealRunResponse"];
export type NumericDealKey = { [K in keyof DealInputs]: DealInputs[K] extends number ? K : never }[keyof DealInputs];

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
};

export type { FieldSpec };

/** Bounds mirror DealInputsIn in api/schemas.py. */
export const FIELDS: Record<NumericDealKey, FieldSpec> = {
  ebitda: { label: "EBITDA", unit: "$M", step: 5, decimals: 1, min: 0, exclusiveMin: true },
  entry_mult: { label: "Entry multiple", unit: "x", step: 0.5, decimals: 1, min: 0, exclusiveMin: true },
  exit_mult: { label: "Exit multiple", unit: "x", step: 0.5, decimals: 1, min: 0, exclusiveMin: true },
  hold: { label: "Hold", unit: "yr", step: 1, decimals: 0, min: 1, max: 15, integer: true },
  growth: { label: "Revenue growth", unit: "%", step: 0.5, decimals: 1, min: -50, max: 100 },
  gross_margin: { label: "Gross margin", unit: "%", step: 1, decimals: 1, min: 0, max: 100 },
  opex: { label: "Opex", unit: "%", step: 1, decimals: 1, min: 0, max: 100 },
  tax: { label: "Tax rate", unit: "%", step: 1, decimals: 1, min: 0, max: 100 },
  da: { label: "D&A", unit: "%", step: 0.5, decimals: 1, min: 0, max: 100 },
  debt_pct: { label: "Debt / EV", unit: "%", step: 1, decimals: 1, min: 0, max: 99 },
  senior_pct: { label: "Senior share", unit: "%", step: 1, decimals: 1, min: 0, max: 100 },
  base_rate: { label: "Senior rate", unit: "%", step: 0.25, decimals: 2, min: 0, max: 50 },
  mezz_spread: { label: "Mezz spread", unit: "%", step: 0.25, decimals: 2, min: 0, max: 50 },
  capex: { label: "Capex", unit: "%", step: 0.5, decimals: 1, min: 0, max: 100 },
  nwc: { label: "NWC change", unit: "%", step: 0.25, decimals: 2, min: -100, max: 100 },
  mincash: { label: "Minimum cash", unit: "$M", step: 5, decimals: 1, min: 0 },
  ar_days: { label: "Receivable days", unit: "d", step: 1, decimals: 0, min: 0, max: 365 },
  inv_days: { label: "Inventory days", unit: "d", step: 1, decimals: 0, min: 0, max: 365 },
  ap_days: { label: "Payable days", unit: "d", step: 1, decimals: 0, min: 0, max: 365 },
};

export { changedKeys, validate };
