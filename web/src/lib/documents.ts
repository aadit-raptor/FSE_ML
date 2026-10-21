import { api, type Schemas } from "@/lib/api/client";

import { errorCode } from "./research";

/**
 * A document read for a deal (PLAN.md 6.2): a clean, uploaded PDF (an annual report, an information pack) is read
 * twice by two AI models, every value with its page and a verbatim quote, as a background job. The person accepts
 * each value; the deal inputs are then worked out by the API from the read's own result and only the accepted
 * values, so nothing else can reach the deal. On a free AI tier the person must confirm the document is public:
 * the API refuses a read without it, and the screen keeps the button off until the box is ticked.
 */
export type DocumentReaderInfo = Schemas["DocumentReaderInfo"];
export type DocumentJob = Schemas["DocumentReadJobOut"];
export type DocumentResult = Schemas["DocumentReadResult"];
export type DocumentFigure = Schemas["DocumentFigure"];
export type DocumentLabel = Schemas["DocumentLabel"];
export type DocumentDeal = Schemas["DocumentDealResponse"];

/** The refusal and failure codes the screen has words for (`documents.error_*`); others read "failed". */
export const DOCUMENT_ERRORS: string[] = [
  "reader_off", "reader_not_measured", "confirm_public", "not_pdf", "not_clean", "no_file", "page_limit", "too_long",
  "store_unreachable", "allowance_spent", "provider_busy", "key_refused", "unreadable_answer", "provider_unreachable",
  "not_recorded", "answer_blocked", "unreadable_pdf", "read_not_done", "read_expired", "no_read",
];

export async function readerInfo(): Promise<DocumentReaderInfo | null> {
  const { data } = await api.GET("/api/documents/reader");
  return data ?? null;
}

export async function startDocumentRead(fileId: string, confirmed: boolean): Promise<{ job?: DocumentJob; error?: string }> {
  const { data, error } = await api.POST("/api/documents/reads", { body: { file_id: fileId, public: confirmed } });
  return data ? { job: data } : { error: errorCode(error) };
}

export async function documentJob(id: string): Promise<DocumentJob | null> {
  const { data } = await api.GET("/api/documents/reads/{job_id}", { params: { path: { job_id: id } } });
  return data ?? null;
}

/** The deal inputs the accepted values give, from the API: never worked out from the values in the browser. */
export async function dealFromAccepted(jobId: string, accepted: string[]): Promise<{ deal?: DocumentDeal; error?: string }> {
  const { data, error } = await api.POST("/api/documents/reads/{job_id}/deal", {
    params: { path: { job_id: jobId } },
    body: { accepted },
  });
  return data ? { deal: data } : { error: errorCode(error) };
}

/** The money and plain fields of a deal answer, apart: money changes the deal's currency and unit first. */
export function splitDeal(inputs: DocumentDeal["inputs"]) {
  const { currency, unit, ...rest } = inputs;
  const fields = Object.fromEntries(Object.entries(rest).filter(([, value]) => value !== null && value !== undefined));
  return { money: currency && unit ? { currency, unit } : null, fields };
}
