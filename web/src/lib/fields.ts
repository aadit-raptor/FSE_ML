/** A numeric input's label, unit and API bounds. */
export type FieldSpec = {
  label: string;
  unit: string;
  step: number;
  decimals: number;
  min?: number;
  max?: number;
  /** The API requires strictly greater than min */
  exclusiveMin?: boolean;
  integer?: boolean;
};

/** Why a value is rejected, or null if the API will accept it. */
export function validate(spec: FieldSpec, v: number): string | null {
  if (!Number.isFinite(v)) return "Enter a number";
  if (spec.integer && !Number.isInteger(v)) return "Whole numbers only";
  if (spec.min !== undefined && (spec.exclusiveMin ? v <= spec.min : v < spec.min)) {
    return spec.exclusiveMin ? `Must be above ${spec.min}` : `At least ${spec.min}`;
  }
  if (spec.max !== undefined && v > spec.max) return `At most ${spec.max}`;
  return null;
}

/** Keys whose values differ between two flat records. */
export function changedKeys<T extends object>(a: T, b: T): (keyof T)[] {
  return (Object.keys(a) as (keyof T)[]).filter((k) => a[k] !== b[k]);
}
