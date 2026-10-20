"use client";

import { api, type Schemas } from "@/lib/api/client";
import { downloadFile } from "@/lib/export";

/**
 * Results notes (PLAN.md R8): a covered company's note drawn by the API from its saved sector model, every figure
 * checked against its cell there. The browser sends only what the analyst types (the rating, the target, the
 * argument, the date, the version to compare with) and the account's firm name and disclosures; it never sends a
 * figure.
 */
export type Report = Schemas["ReportOut"];
export type ReportFields = Schemas["ReportFieldsIn"];
export type ReportDetails = Schemas["ReportDetailsOut"];
export type ReportBlock = Schemas["ReportBlock"];
export type ReportSegment = Schemas["ReportSegment"];

/** Refusals with words of their own (`report.refused_<code>`); anything else is `refused_failed`. */
export const REPORT_REFUSALS = [
  "no_model", "no_version", "not_found", "figure_differs", "number_not_from_model", "label_differs", "typed_differs",
  "call_in_text", "no_model_values",
];

export function refusedKey(code: string): string {
  return `refused_${REPORT_REFUSALS.includes(code) ? code : "failed"}`;
}

function codeOf(error: unknown, status: number): string {
  if (status === 429) return "limit";
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  if (detail && typeof detail === "object" && typeof (detail as { code?: unknown }).code === "string") {
    return (detail as { code: string }).code;
  }
  return typeof detail === "string" ? detail : "failed";
}

const at = (companyId: string) => ({ params: { path: { company_id: companyId } } });

export type Answer = { report?: Report; refused?: string };

export async function getReport(companyId: string): Promise<Answer> {
  const { data, error, response } = await api.GET("/api/research/companies/{company_id}/report", at(companyId));
  return data ? { report: data } : { refused: codeOf(error, response.status) };
}

export async function putReport(companyId: string, body: ReportFields): Promise<Answer> {
  const { data, error, response } = await api.PUT("/api/research/companies/{company_id}/report", {
    ...at(companyId), body,
  });
  return data ? { report: data } : { refused: codeOf(error, response.status) };
}

export async function writeSentences(companyId: string): Promise<Answer> {
  const { data, error, response } = await api.POST("/api/research/companies/{company_id}/report/sentences",
    at(companyId));
  return data ? { report: data } : { refused: codeOf(error, response.status) };
}

export async function getDetails(): Promise<ReportDetails | null> {
  const { data } = await api.GET("/api/account/report-details");
  return data ?? null;
}

export async function putDetails(body: ReportDetails): Promise<ReportDetails | null> {
  const { data } = await api.PUT("/api/account/report-details", { body });
  return data ?? null;
}

function fileName(company: string, extension: string): string {
  const base = company.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 80) || "company";
  return `${base}_results_note.${extension}`;
}

export function downloadNote(companyId: string, company: string, kind: "word" | "pdf") {
  return downloadFile(`/api/research/companies/${companyId}/report/${kind}`,
    fileName(company, kind === "word" ? "docx" : "pdf"));
}

/** A cell's or a paragraph's text, as the note writes it. */
export const plain = (parts: ReportSegment[] | null | undefined) => (parts ?? []).map((p) => p.text).join("");

/** The fields as the API keeps them, from what the analyst typed. */
export function fieldsFrom(report: Report): ReportFields {
  const f = report.fields;
  return {
    argument: f.argument ?? "", rating: f.rating ?? "", target: f.target ?? null,
    note_date: f.note_date ?? null, compare_version: f.compare_version ?? null,
  };
}
