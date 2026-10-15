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

/** What the API answers in `detail` when it refuses or a read fails: a code the screen words. */
export function errorCode(error: unknown): string {
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  return typeof detail === "string" && /^[a-z_]{3,40}$/.test(detail) ? detail : "failed";
}

export async function startRead(
  fileId: string,
  confirmed: boolean,
  quarterEnd: string | null,
): Promise<{ job: ReadJob; error?: undefined } | { job?: undefined; error: string }> {
  const { data, error } = await api.POST("/api/research/reads", {
    body: { file_id: fileId, public: confirmed, quarter_end: quarterEnd || null },
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
