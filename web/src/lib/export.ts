import type { Row } from "@/components/charts/DataTable";
import { authHeaders, type Schemas } from "@/lib/api/client";
import type { Money } from "@/lib/money";
import { newRequestId, reportApiError, REQUEST_ID_HEADER } from "@/lib/monitoring";

export type Sheet = Schemas["WorkbookSheet"];
type Cell = string | number | boolean | null;

/** A DataTable's rows as a sheet: money as numbers in the screen's unit, rates as percentages. */
export function tableSheet(name: string, columns: string[], rows: Row[]): Sheet {
  return {
    name,
    columns: ["", ...columns],
    rows: rows.map((r) => [
      r.kind === "rate" ? `${r.label} (%)` : r.label,
      ...r.values.map((v) => (typeof v === "number" && Number.isFinite(v) ? (r.kind === "rate" ? v * 100 : r.kind === "outflow" ? -Math.abs(v) : v) : null)),
    ]),
  };
}

/** Plain sheet from a header row and cells. */
export function sheet(name: string, columns: string[], rows: Cell[][]): Sheet {
  return { name, columns, rows: rows.map((r) => r.map((c) => (typeof c === "number" && !Number.isFinite(c) ? null : c))) };
}

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
  return postForFile("/api/export/workbook", { filename, sheets, money }, filename);
}

/** Up to 10,000 simulated paths for the given Monte Carlo request. */
export function downloadMonteCarloSample(body: unknown) {
  return postForFile("/api/export/montecarlo-sample", body, "mc_simulation.xlsx");
}
