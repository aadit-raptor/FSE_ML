import { api, type Schemas } from "@/lib/api/client";

/**
 * The results reader (PLAN.md R1): a clean, uploaded PDF is read twice by two
 * AI models and checked, as a background job. While the reader runs on a free
 * AI tier the person must confirm the file is public (a listed company's
 * published results): the API refuses a read without it, and the screen keeps
 * the button off until the box is ticked.
 */
export type ReaderInfo = Schemas["ReaderInfo"];
export type ReadJob = Schemas["ReadJobOut"];
export type ReadResult = Schemas["ReadResult"];
export type ReadTable = Schemas["ReadTable"];

/** The read's refusal and failure codes the screens have words for (`reader.error_*`); others read "failed". */
export const READER_ERRORS: string[] = [
  "reader_off", "reader_not_measured", "confirm_public", "not_pdf", "not_clean", "no_file", "page_limit", "too_long",
  "store_unreachable", "allowance_spent", "provider_busy", "key_refused", "unreadable_answer", "provider_unreachable",
  "not_recorded", "answer_blocked", "unreadable_pdf", "upload_failed", "no_company", "too_many_companies", "bad_name",
];

/** Accepting a review's refusal codes the screens have words for (`coverage.error_*`). */
export const ACCEPT_ERRORS: string[] = [
  "amber_unreviewed", "bad_review", "quarter_unknown", "unit_unknown", "no_figures", "no_read", "read_not_done",
  "read_expired", "not_found", "too_many_periods", "coverage_full", "no_account", "no_database",
];

/** What the API answers in `detail` when it refuses or a read fails: a code the screen words. */
export function errorCode(error: unknown): string {
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  return typeof detail === "string" && /^[a-z_]{3,40}$/.test(detail) ? detail : "failed";
}

export async function startRead(
  fileId: string,
  confirmed: boolean,
  quarterEnd: string | null,
  companyId: string | null = null,
): Promise<{ job: ReadJob; error?: undefined } | { job?: undefined; error: string }> {
  const { data, error } = await api.POST("/api/research/reads", {
    body: { file_id: fileId, public: confirmed, quarter_end: quarterEnd || null, company_id: companyId || null },
  });
  return data ? { job: data } : { error: errorCode(error) };
}

export async function readJob(id: string): Promise<ReadJob | null> {
  const { data } = await api.GET("/api/research/reads/{job_id}", { params: { path: { job_id: id } } });
  return data ?? null;
}

/** How many decimals a table prints: two when any figure has a fraction, else none. */
export function decimalsOf(table: ReadTable): number {
  return table.rows.some((row) => row.cells.some((c) => c.value !== null && c.value !== undefined && !Number.isInteger(c.value))) ? 2 : 0;
}

/* ------------------------------------------------------------------ coverage (PLAN.md R2) */

export type CoverageList = Schemas["CoverageList"];
export type CoverageCompany = Schemas["CoverageCompanyOut"];
export type PeriodSummary = Schemas["CoveragePeriodSummary"];
export type StoredPeriod = Schemas["CoveragePeriodOut"];
export type StoredTable = Schemas["StoredTable"];
export type Correction = Schemas["FigureCorrection"];

/** A figure of a read: its table, row and column, as the review names it. */
export type CellRef = [number, number, number];
export const cellKey = (ref: CellRef) => ref.join(":");

/** Every amber figure the analyst must answer before the quarter can be accepted. */
export function amberCells(result: ReadResult): CellRef[] {
  const out: CellRef[] = [];
  result.tables.forEach((table, t) =>
    table.rows.forEach((row, r) =>
      row.cells.forEach((cell, c) => {
        if (cell.amber && cell.value !== null && cell.value !== undefined) out.push([t, r, c]);
      }),
    ),
  );
  return out;
}

export async function listCoverage(): Promise<CoverageList | null> {
  const { data } = await api.GET("/api/research/companies");
  return data ?? null;
}

export async function addCompany(name: string): Promise<{ company?: CoverageCompany; error?: string }> {
  const { data, error } = await api.POST("/api/research/companies", { body: { name } });
  return data ? { company: data } : { error: errorCode(error) };
}

export async function removeCompany(id: string): Promise<boolean> {
  const { response } = await api.DELETE("/api/research/companies/{company_id}", { params: { path: { company_id: id } } });
  return response.ok;
}

export async function acceptRead(
  companyId: string,
  readId: string,
  accepted: CellRef[],
  corrected: Correction[],
): Promise<{ periods?: PeriodSummary[]; error?: string }> {
  const { data, error } = await api.POST("/api/research/companies/{company_id}/periods", {
    params: { path: { company_id: companyId } },
    body: { read_id: readId, accepted, corrected },
  });
  return data ? { periods: data.periods } : { error: errorCode(error) };
}

export async function storedPeriod(id: string): Promise<StoredPeriod | null> {
  const { data } = await api.GET("/api/research/periods/{period_id}", { params: { path: { period_id: id } } });
  return data ?? null;
}

export async function removePeriod(id: string): Promise<boolean> {
  const { response } = await api.DELETE("/api/research/periods/{period_id}", { params: { path: { period_id: id } } });
  return response.ok;
}

export const workbookPath = (periodId: string) => `/api/research/periods/${periodId}/workbook`;

/** "DMart_2026-09-30_consolidated.xlsx": the name the API gives the workbook too. */
export function workbookName(company: string, period: PeriodSummary): string {
  const stem = `${company}_${period.period_end}_${period.scope}`.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^[._]+|[._]+$/g, "");
  return `${stem || "results"}.xlsx`;
}
