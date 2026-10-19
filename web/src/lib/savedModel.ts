"use client";

import { api, type Schemas } from "@/lib/api/client";
import { downloadFile } from "@/lib/export";

/**
 * Saved sector models (PLAN.md R5): one per covered company, autosaved, with versions; rolled forward on newer
 * accepted results, the estimate made before them kept beside them; saved back from Excel by its named cells.
 * The model on screen (lib/sectorModel.ts) carries a `Saved` while it is a company's saved model.
 */
export type SavedModel = Schemas["SavedModelOut"];
export type AsOf = Schemas["ResearchAsOf"];
export type ModelVersion = Schemas["ModelVersionOut"];
export type RollComparison = Schemas["RollComparison"];
export type RollAnswer = Schemas["ModelRollOut"];
export type SheetAnswer = Schemas["ModelSheetOut"];
export type Scope = SavedModel["scope"];

export type Saved = {
  companyId: string;
  scope: Scope;
  asOf: AsOf | null;
  version: number;
  updatedAt: string | null;
  newer: AsOf | null;
};

/** Refusals the screen and the pane have words for (`models.saved_*`): the API's message, if any, beside it. */
export type SavedRefusal = { code: string; message: string };

function refusal(error: unknown): SavedRefusal {
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  if (detail && typeof detail === "object" && typeof (detail as { code?: unknown }).code === "string") {
    const { code, message } = detail as { code: string; message?: unknown };
    return { code, message: typeof message === "string" ? message : "" };
  }
  return { code: typeof detail === "string" ? detail : "failed", message: "" };
}

export function savedFrom(model: SavedModel): Saved {
  return {
    companyId: model.company_id, scope: model.scope, asOf: model.as_of ?? null, version: model.latest_version,
    updatedAt: model.updated_at, newer: model.newer_results ?? null,
  };
}

const at = (companyId: string) => ({ params: { path: { company_id: companyId } } });

/** The company's saved model; `missing` when it has none yet; null when the API couldn't answer. */
export async function getSaved(companyId: string): Promise<{ model?: SavedModel; missing?: boolean } | null> {
  const { data, error, response } = await api.GET("/api/research/companies/{company_id}/model", at(companyId));
  if (data) return { model: data };
  if (response.status === 404 && refusal(error).code === "no_model") return { missing: true };
  return null;
}

export async function putSaved(companyId: string, body: Schemas["SavedModelIn"]): Promise<{ model?: SavedModel; refused?: SavedRefusal }> {
  const { data, error } = await api.PUT("/api/research/companies/{company_id}/model", { ...at(companyId), body });
  return data ? { model: data } : { refused: refusal(error) };
}

export async function listVersions(companyId: string): Promise<ModelVersion[]> {
  const { data } = await api.GET("/api/research/companies/{company_id}/model/versions", at(companyId));
  return data?.versions ?? [];
}

export async function keepVersion(companyId: string, label: string | null): Promise<boolean> {
  const { response } = await api.POST("/api/research/companies/{company_id}/model/versions", {
    ...at(companyId), body: { label: label || null },
  });
  return response.ok;
}

export async function restoreVersion(companyId: string, number: number): Promise<SavedModel | null> {
  const { data } = await api.POST("/api/research/companies/{company_id}/model/versions/{number}/restore", {
    params: { path: { company_id: companyId, number } },
  });
  return data ?? null;
}

export async function rollForward(companyId: string): Promise<{ rolled?: RollAnswer; refused?: SavedRefusal }> {
  const { data, error } = await api.POST("/api/research/companies/{company_id}/model/roll", at(companyId));
  return data ? { rolled: data } : { refused: refusal(error) };
}

export async function savedCells(companyId: string): Promise<Schemas["ModelCellsOut"] | null> {
  const { data } = await api.GET("/api/research/companies/{company_id}/model/cells", at(companyId));
  return data ?? null;
}

export function downloadSaved(companyId: string, filename: string) {
  return downloadFile(`/api/research/companies/${companyId}/model/workbook`, filename);
}

export async function saveSheet(
  companyId: string,
  grid: (number | string | boolean | null)[][],
  names: Record<string, string>,
): Promise<{ saved?: SheetAnswer; refused?: SavedRefusal }> {
  const { data, error } = await api.POST("/api/research/companies/{company_id}/model/sheet", {
    ...at(companyId), body: { grid, names },
  });
  return data ? { saved: data } : { refused: refusal(error) };
}

/** The company id a saved model's sheet carries beside its title ("Saved in Variater: <id>"), or null. */
export function savedReference(text: unknown): string | null {
  const match = /([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\s*$/i.exec(String(text ?? ""));
  return match ? match[1].toLowerCase() : null;
}

/** Codes with words of their own (`models.saved_<code>`); anything else is `saved_failed`. */
export const SAVED_REFUSALS = [
  "no_as_of", "nothing_newer", "older_results", "too_far", "no_results", "changed_meanwhile", "model_too_large",
  "too_many_versions", "coverage_full", "restructured", "not_a_number", "other_model", "not_found",
];

export function refusalKey(code: string): string {
  return `saved_${SAVED_REFUSALS.includes(code) ? code : "failed"}`;
}
