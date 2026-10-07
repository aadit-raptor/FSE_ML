/**
 * The optional reference library (PLAN.md 4.5): whether it is shown, its base rates and its coverage.
 * The screens are under Library; the state lives in `LibraryProvider`.
 */
import { api, type Schemas } from "@/lib/api/client";

export type LibraryState = Schemas["LibraryState"];
export type BaseRates = Schemas["BaseRatesResponse"];
export type BaseRateSource = Schemas["BaseRateSource"];
export type Coverage = Schemas["CoverageResponse"];
export type SpRegion = Schemas["SpeculativeByRegion"]["regions"][number];
export type CumulativeRegion = Schemas["CumulativeRates"]["region"];

/** The mode the switch hides. */
export const LIBRARY_MODE = "library";

/** Shown in the tabs: on for everyone, or switchable by this caller (an administrator turning it back on). */
export function libraryVisible(state: LibraryState | null): boolean {
  return !!state && (state.enabled || state.can_switch);
}

export async function fetchLibraryState(): Promise<LibraryState | null> {
  try {
    const { data } = await api.GET("/api/library");
    return data ?? null;
  } catch {
    return null;
  }
}

export async function putLibrarySwitch(enabled: boolean): Promise<LibraryState | null> {
  const { data } = await api.PUT("/api/library/switch", { body: { enabled } });
  return data ?? null;
}

export async function fetchBaseRates(country: string | null): Promise<BaseRates | null> {
  try {
    const { data } = await api.GET("/api/library/base-rates", { params: { query: country ? { country } : {} } });
    return data ?? null;
  } catch {
    return null;
  }
}

export async function fetchCoverage(): Promise<Coverage | null> {
  try {
    const { data } = await api.GET("/api/library/coverage");
    return data ?? null;
  } catch {
    return null;
  }
}

/** The cumulative table to open on: the deal's region when S&P publishes one for it, else global. */
export function cumulativeRegionFor(region: SpRegion | null | undefined, available: CumulativeRegion[]): CumulativeRegion {
  return region && (available as string[]).includes(region) ? (region as CumulativeRegion) : "global";
}

/** The first year from which every region has a figure (emerging markets start late). */
export function firstCompleteYear(rates: Record<string, (number | null)[]>, years: number[]): number {
  const series = Object.values(rates);
  let start = 0;
  years.forEach((_, i) => {
    if (series.some((s) => s[i] === null || s[i] === undefined)) start = i + 1;
  });
  return start;
}
