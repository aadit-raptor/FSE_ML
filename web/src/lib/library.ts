/**
 * The optional reference library (PLAN.md 4.5): whether it is shown, its base rates, its coverage,
 * the sourced reference transactions and their review (4.5b), and the fees they give.
 * The screens are under Library; the state lives in `LibraryProvider`.
 */
import { api, type Schemas } from "@/lib/api/client";

export type LibraryState = Schemas["LibraryState"];
export type BaseRates = Schemas["BaseRatesResponse"];
export type BaseRateSource = Schemas["BaseRateSource"];
export type Coverage = Schemas["CoverageResponse"];
export type SpRegion = Schemas["SpeculativeByRegion"]["regions"][number];
export type CumulativeRegion = Schemas["CumulativeRates"]["region"];
export type ReferenceDeal = Schemas["ReferenceDealView"];
export type ReferenceDeals = Schemas["ReferenceDealsResponse"];
export type ReviewQueue = Schemas["ReviewQueue"];
export type ReviewProposal = Schemas["ReviewProposal"];
export type RejectReason = NonNullable<Schemas["ReviewVerdict"]["reason"]>;
export type SourcedFees = Schemas["SourcedFeesResponse"];
export type FeeSetting = Extract<keyof SourcedFees["settings"], string>;

/** The reasons a rejection can give, in the order offered. */
export const REJECT_REASONS: RejectReason[] = ["figure_wrong", "source_wrong", "not_a_buyout", "duplicate", "other"];
/** The figures a transaction may carry, in the order shown. */
export const REFERENCE_FIGURES = [
  "transaction_value",
  "transaction_value_usd",
  "ebitda",
  "debt",
  "equity",
  "transaction_fees",
  "financing_fees",
  "senior_amort_pct",
] as const;
export type ReferenceFigureName = (typeof REFERENCE_FIGURES)[number];
/** The Settings the reference transactions source, as Settings -> Fees applies them. */
export const FEE_SETTINGS: FeeSetting[] = ["tx_fee_pct", "fin_fee_pct", "def_senior_amort"];

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

export async function fetchReferences(): Promise<ReferenceDeals | null> {
  try {
    const { data } = await api.GET("/api/library/references");
    return data ?? null;
  } catch {
    return null;
  }
}

/** The review queue, or the API's refusal (403 for non-administrators, 503 without a database). */
export async function fetchReview(): Promise<{ queue: ReviewQueue | null; status: number }> {
  try {
    const { data, response } = await api.GET("/api/library/review");
    return { queue: data ?? null, status: response.status };
  } catch {
    return { queue: null, status: 0 };
  }
}

export type ReviewAnswer = { proposal: ReviewProposal | null; status: number; problems: Schemas["RuleProblem"][] };

function refusal(error: unknown): Schemas["RuleProblem"][] {
  const detail = (error as { detail?: { problems?: Schemas["RuleProblem"][] } } | undefined)?.detail;
  return typeof detail === "object" && detail?.problems ? detail.problems : [];
}

export async function postVerdict(id: string, verdict: "approve" | "reject", reason?: RejectReason): Promise<ReviewAnswer> {
  try {
    const { data, error, response } = await api.POST("/api/library/review/{reference_id}", {
      params: { path: { reference_id: id } },
      body: { verdict, ...(reason ? { reason } : {}) },
    });
    return { proposal: data ?? null, status: response.status, problems: refusal(error) };
  } catch {
    return { proposal: null, status: 0, problems: [] };
  }
}

export async function postProposal(deal: unknown): Promise<ReviewAnswer> {
  try {
    const { data, error, response } = await api.POST("/api/library/review", { body: deal as Schemas["ReferenceDealIn"] });
    return { proposal: data ?? null, status: response.status, problems: refusal(error) };
  } catch {
    return { proposal: null, status: 0, problems: [] };
  }
}

export async function fetchSourcedFees(): Promise<SourcedFees | null> {
  try {
    const { data } = await api.GET("/api/library/fees");
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
