"use client";

import { api, type Schemas } from "@/lib/api/client";
import { downloadFile } from "@/lib/export";

/**
 * Investment memos (PLAN.md 6.3): a saved deal version's memo, drafted by the API as a background job that runs the
 * version through the model, then drawn from that run with every figure checked against it. The browser sends
 * only which version; it never sends a figure. The memo's words are in the account's language (`memo.doc` in
 * web/messages, read by the API).
 *
 * i18n-keys: memo.doc.*
 */
export type Memo = Schemas["MemoOut"];
export type MemoSummary = Schemas["MemoSummary"];
export type MemoJob = Schemas["MemoJobOut"];

/** Refusals with words of their own (`memo.refused_<code>`); anything else is `refused_failed`. */
const REFUSALS = ["figure_differs", "label_differs", "number_not_from_model", "call_in_text", "no_run"];

export function refusedKey(code: string): string {
  return `refused_${REFUSALS.includes(code) ? code : "failed"}`;
}

function codeOf(error: unknown, status: number): string {
  if (status === 429) return "limit";
  if (status === 409) return "full";
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  if (detail && typeof detail === "object" && typeof (detail as { code?: unknown }).code === "string") {
    return (detail as { code: string }).code;
  }
  return typeof detail === "string" ? detail : "failed";
}

const at = (dealId: string) => ({ params: { path: { deal_id: dealId } } });
const atVersion = (dealId: string, version: number) => ({ params: { path: { deal_id: dealId, version } } });

export type Answer = { memo?: Memo; refused?: string };

export async function listMemos(dealId: string): Promise<{ memos: MemoSummary[]; max: number } | null> {
  const { data } = await api.GET("/api/deals/{deal_id}/memos", at(dealId));
  return data ? { memos: data.memos, max: data.max_memos } : null;
}

export async function getMemo(dealId: string, version: number): Promise<Answer> {
  const { data, error, response } = await api.GET("/api/deals/{deal_id}/memos/{version}", atVersion(dealId, version));
  return data ? { memo: data } : { refused: codeOf(error, response.status) };
}

export async function writeSentences(dealId: string, version: number): Promise<Answer> {
  const { data, error, response } = await api.POST("/api/deals/{deal_id}/memos/{version}/sentences",
    atVersion(dealId, version));
  return data ? { memo: data } : { refused: codeOf(error, response.status) };
}

export async function deleteMemo(dealId: string, version: number): Promise<boolean> {
  const { response } = await api.DELETE("/api/deals/{deal_id}/memos/{version}", atVersion(dealId, version));
  return response.ok;
}

const FIRST_POLL_MS = 500;
const MAX_POLL_MS = 3000;

function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("aborted", "AbortError"));
    }, { once: true });
  });
}

/** Draft a version's memo: the job runs the model; answers the memo, or why not. `onStage` gets the job's stage. */
export async function draftMemo(dealId: string, version: number, signal: AbortSignal,
  onStage?: (job: MemoJob) => void): Promise<Answer> {
  const started = await api.POST("/api/deals/{deal_id}/memos", { ...at(dealId), body: { version }, signal });
  if (!started.data) return { refused: codeOf(started.error, started.response.status) };
  const id = started.data.id;
  let delay = FIRST_POLL_MS;
  for (;;) {
    await sleep(delay, signal);
    delay = Math.min(MAX_POLL_MS, delay * 1.4);
    const { data, error, response } = await api.GET("/api/deals/{deal_id}/memos/jobs/{job_id}", {
      params: { path: { deal_id: dealId, job_id: id } }, signal,
    });
    if (!data) {
      if (response.status >= 500) continue;
      return { refused: codeOf(error, response.status) };
    }
    if (data.status === "succeeded") return getMemo(dealId, version);
    if (data.status === "failed" || data.status === "cancelled") return { refused: data.error_status === 409 ? "full" : "failed" };
    onStage?.(data);
  }
}

function fileName(deal: string, version: number, extension: string): string {
  const base = deal.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 80) || "deal";
  return `${base}_v${version}_memo.${extension}`;
}

export function downloadMemo(dealId: string, deal: string, version: number, kind: "word" | "pdf") {
  return downloadFile(`/api/deals/${dealId}/memos/${version}/${kind}`,
    fileName(deal, version, kind === "word" ? "docx" : "pdf"));
}
