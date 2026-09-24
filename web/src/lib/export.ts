import type { Row } from "@/components/charts/DataTable";
import { authHeaders, type Schemas } from "@/lib/api/client";
import { numberStyle } from "@/lib/locale";
import type { Money } from "@/lib/money";
import { newRequestId, reportApiError, REQUEST_ID_HEADER } from "@/lib/monitoring";

export type Sheet = Schemas["WorkbookSheet"];
/** How a cell's number shows in Excel; percent cells hold fractions (0.212 shows 21.2%). */
export type CellFormat = NonNullable<NonNullable<Sheet["column_formats"]>[number]>;
type Cell = string | number | boolean | null;

/**
 * A DataTable's rows as a sheet, each row with its Excel number format (PLAN.md 2.3a): money as numbers in
 * the screen's unit, rates as fractions shown as percentages. Excel shows them with the reader's own
 * separators, so the file needs no locale.
 */
export function tableSheet(name: string, columns: string[], rows: Row[]): Sheet {
  return {
    name,
    columns: ["", ...columns],
    rows: rows.map((r) => [
      r.label,
      ...r.values.map((v) => (typeof v === "number" && Number.isFinite(v) ? (r.kind === "outflow" ? -Math.abs(v) : v) : null)),
    ]),
    row_formats: rows.map((r) => (r.kind === "rate" ? "percent" : "money")),
  };
}

/** Plain sheet from a header row and cells, with a number format per column (and per row, which wins). */
export function sheet(
  name: string,
  columns: string[],
  rows: Cell[][],
  formats: { columns?: (CellFormat | null)[]; rows?: (CellFormat | null)[] } = {},
): Sheet {
  return {
    name,
    columns,
    rows: rows.map((r) => r.map((c) => (typeof c === "number" && !Number.isFinite(c) ? null : c))),
    ...(formats.columns ? { column_formats: formats.columns } : {}),
    ...(formats.rows ? { row_formats: formats.rows } : {}),
  };
}

/** An engine rate kept as a fraction for a percent cell; null when missing. */
export const fraction = (v: number | null | undefined, per = 1): number | null => (typeof v === "number" && Number.isFinite(v) ? v / per : null);

function save(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function postForFile(path: string, body: unknown, filename: string) {
  const requestId = newRequestId();
  // A plain fetch, so the signed-in user's token has to be added by hand
  const headers = { "Content-Type": "application/json", [REQUEST_ID_HEADER]: requestId, ...(await authHeaders()) };
  const res = await fetch(path, { method: "POST", headers, body: JSON.stringify(body) }).catch((e: unknown) => {
    reportApiError(path, "network", requestId);
    throw e;
  });
  if (res.status >= 500) reportApiError(path, res.status, requestId);
  if (!res.ok) {
    const detail = await res.json().then((j) => (typeof j?.detail === "string" ? j.detail : null)).catch(() => null);
    throw new Error(detail ?? `Download failed (${res.status})`);
  }
  save(await res.blob(), filename);
}

/** Ask the API to write the sheets to .xlsx and save it; an About sheet says what the money is in. */
export function downloadWorkbook(filename: string, sheets: Sheet[], money: Money) {
  // Lakh and crore need patterns of their own; the other groupings are Excel's standard formats
  const { grouping } = numberStyle();
  return postForFile("/api/export/workbook", { filename, sheets, money, ...(grouping === "locale" ? {} : { grouping }) }, filename);
}

/** Up to 10,000 simulated paths for the given Monte Carlo request. */
export function downloadMonteCarloSample(body: unknown) {
  return postForFile("/api/export/montecarlo-sample", body, "mc_simulation.xlsx");
}
