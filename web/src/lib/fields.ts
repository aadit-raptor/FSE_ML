/**
 * A numeric input's unit and API bounds. The label is not here: it lives in
 * the translation files and is passed to the input (PLAN.md 2.3b).
 */
export type FieldSpec = {
  /** "%", "x", MONEY, or a key in the `units` namespace ("yr", "d", "n") */
  unit: string;
  step: number;
  decimals: number;
  min?: number;
  max?: number;
  /** The API requires strictly greater than min */
  exclusiveMin?: boolean;
  integer?: boolean;
};

/** Why a value is rejected: a key in the `validation` namespace, and the bound it broke. i18n-keys: validation.* */
export type FieldProblem = { key: "notANumber" | "wholeNumbers" | "above" | "atLeast" | "atMost"; bound?: number };

/** Why a value is rejected, or null if the API will accept it. */
export function validate(spec: FieldSpec, v: number): FieldProblem | null {
  if (!Number.isFinite(v)) return { key: "notANumber" };
  if (spec.integer && !Number.isInteger(v)) return { key: "wholeNumbers" };
  if (spec.min !== undefined && (spec.exclusiveMin ? v <= spec.min : v < spec.min)) {
    return spec.exclusiveMin ? { key: "above", bound: spec.min } : { key: "atLeast", bound: spec.min };
  }
  if (spec.max !== undefined && v > spec.max) return { key: "atMost", bound: spec.max };
  return null;
}

/** Keys whose values differ between two flat records. */
export function changedKeys<T extends object>(a: T, b: T): (keyof T)[] {
  return (Object.keys(a) as (keyof T)[]).filter((k) => a[k] !== b[k]);
}
