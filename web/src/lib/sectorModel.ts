"use client";

import { useSyncExternalStore } from "react";

import { api, type Schemas } from "@/lib/api/client";
import { downloadPost } from "@/lib/export";
import { putSaved, savedFrom, type Saved, type SavedRefusal } from "@/lib/savedModel";

/**
 * Sector models (PLAN.md R4): the analyst picks a sector (which picks the model and its words), makes the last
 * changes for the company -- add, drop or swap a line of revenue -- types the drivers' history and the
 * assumptions, and sees the projection at once. The server works the model out (`/api/research/models/run`) and
 * writes it as live Excel formulas (`/workbook`, and `/cells` for the add-in's "Load model"), from one
 * definition.
 *
 * A covered company's model is saved (R5, lib/savedModel.ts): while the model on screen carries a `saved`, every
 * change that works out is autosaved a moment after the typing rests. A model with none (an example, a company
 * not covered) lives in this page's memory until it is kept in coverage.
 */
export type Catalogue = Schemas["ModelCatalogueOut"];
export type ModelSpec = Schemas["ModelSpecOut"];
export type Structure = Schemas["ResearchStructure"];
export type ModelCompany = Schemas["ResearchCompanyIn"];
export type ModelRun = Schemas["ModelRunOut"];
export type ModelRow = Schemas["ModelRowOut"];
export type ModelCell = Schemas["ModelCellOut"];
export type ModelFill = Schemas["ModelFillOut"];
export type ModelCells = Schemas["ModelCellsOut"];
export type Block = Schemas["ResearchSegment"]["block"];
export type Change = Schemas["ModelChangeIn"]["action"];

/** Refusals the screen has words for (`models.error_*`); the API's message, naming the row, is shown beside. */
export type Refusal = { code: string; message: string };

function refusal(error: unknown): Refusal {
  const detail = (error as { detail?: unknown } | undefined)?.detail as { code?: unknown; message?: unknown } | string | undefined;
  if (detail && typeof detail === "object" && typeof detail.code === "string") {
    return { code: detail.code, message: typeof detail.message === "string" ? detail.message : "" };
  }
  return { code: "failed", message: "" };
}

export async function loadCatalogue(): Promise<Catalogue | null> {
  const { data } = await api.GET("/api/research/models");
  return data ?? null;
}

export async function runModel(structure: Structure, company: ModelCompany): Promise<{ run?: ModelRun; refused?: Refusal }> {
  const { data, error } = await api.POST("/api/research/models/run", { body: { structure, company } });
  return data ? { run: data } : { refused: refusal(error) };
}

export async function changeModel(
  structure: Structure,
  action: Change,
  key: string | null,
  block: Block | null,
): Promise<{ structure?: Structure; refused?: Refusal }> {
  const { data, error } = await api.POST("/api/research/models/change", {
    body: { structure, action, key: key ?? undefined, block: block ?? undefined },
  });
  return data ? { structure: data.structure } : { refused: refusal(error) };
}

export async function fillFromCoverage(companyId: string, scope: "consolidated" | "standalone"): Promise<ModelFill | null> {
  const { data } = await api.GET("/api/research/companies/{company_id}/fill", {
    params: { path: { company_id: companyId }, query: { scope } },
  });
  return data ?? null;
}

export async function modelCells(structure: Structure, company: ModelCompany): Promise<ModelCells | null> {
  const { data } = await api.POST("/api/research/models/cells", { body: { structure, company } });
  return data ?? null;
}

export function workbookFilename(company: ModelCompany, model: string): string {
  const stem = `${company.name || "company"}_${model}_model`.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^[._]+|[._]+$/g, "");
  return `${stem || "model"}.xlsx`;
}

export function downloadModel(structure: Structure, company: ModelCompany) {
  return downloadPost("/api/research/models/workbook", { structure, company }, workbookFilename(company, structure.model));
}

/** The statements' unit as the model's words say it ("INR lakh") and the currency units in one of it. */
export const UNIT_RUPEES: Record<string, number> = {
  units: 1, thousands: 1e3, lakhs: 1e5, millions: 1e6, crores: 1e7, billions: 1e9,
};

/**
 * A company with nothing typed but its sector's model: its words come from the sector, every choice takes its
 * first option, and the past is filled from accepted quarters when `fill` is given.
 */
export function blankCompany(name: string, sector: string, unitName: string, unitRupees: number, fill: ModelFill | null): ModelCompany {
  return {
    name,
    sector,
    unit_name: unitName,
    basis: "",
    words: {},
    period_labels: fill ? { ...fill.labels } : {},
    constants: { unit_rupees: fill?.unit_rupees ?? unitRupees },
    choices: {},
    values: fill ? Object.fromEntries(Object.entries(fill.values).map(([row, by]) => [row, { ...by }])) : {},
    off: [],
    marks: {},
  };
}

/** The company with one typed cell changed (null clears it). */
export function withValue(company: ModelCompany, row: string, period: string, value: number | null): ModelCompany {
  const values = { ...company.values, [row]: { ...(company.values?.[row] ?? {}), [period]: value } };
  return { ...company, values };
}

/** Keeps only figures for cells the run says are typed: after a change to the lines, what no longer fits goes. */
export function prunedTo(company: ModelCompany, run: ModelRun): ModelCompany {
  if (!run.dropped.length) return { ...company, choices: { ...run.choices_made } };
  const gone = new Set(run.dropped.map((d) => `${d.row}|${d.period}`));
  const values: ModelCompany["values"] = {};
  for (const [row, by] of Object.entries(company.values ?? {})) {
    const kept = Object.fromEntries(Object.entries(by ?? {}).filter(([p]) => !gone.has(`${row}|${p}`)));
    if (Object.keys(kept).length) values[row] = kept;
  }
  return { ...company, values, choices: { ...run.choices_made } };
}

/* ------------------------------------------------------------------ the model on screen, kept across steps */

export type SaveState = "idle" | "saving" | "saved" | "failed";

export type ModelState = {
  owner: string | null;
  structure: Structure | null;
  company: ModelCompany | null;
  run: ModelRun | null;
  refused: Refusal | null;
  fillNotes: string[];
  busy: boolean;
  saved: Saved | null;
  saving: SaveState;
  saveRefused: SavedRefusal | null;
};

const EMPTY: ModelState = {
  owner: null, structure: null, company: null, run: null, refused: null, fillNotes: [], busy: false,
  saved: null, saving: "idle", saveRefused: null,
};
let state: ModelState = EMPTY;
let asked = 0;
const listeners = new Set<() => void>();

function publish(next: Partial<ModelState>) {
  state = { ...state, ...next };
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

const snapshot = () => state;

/** The model on screen; another account signed in on this tab starts from nothing. */
export function useModelState(owner: string | null): ModelState {
  const current = useSyncExternalStore(subscribe, snapshot, () => EMPTY);
  return current.owner === owner ? current : EMPTY;
}

/**
 * Work the model out for `structure` and `company` and show it; a later edit's answer wins over an earlier one's.
 * `saved` makes it a company's saved model (null: no longer one; left out: as it was). A saved model that works out
 * is autosaved.
 */
export async function setModel(
  owner: string | null, structure: Structure, company: ModelCompany, fillNotes?: string[], saved?: Saved | null,
) {
  const mine = ++asked;
  const keeping = saved === undefined ? {} : { saved, saving: "idle" as SaveState, saveRefused: null };
  publish({ owner, structure, company, busy: true, ...(fillNotes ? { fillNotes } : {}), ...keeping });
  const { run, refused } = await runModel(structure, company);
  if (mine !== asked) return;
  if (run) {
    publish({ run, refused: null, busy: false, structure: run.structure, company: prunedTo(company, run) });
    if (state.saved) scheduleSave();
  } else publish({ refused: refused ?? null, busy: false });
}

let saveTimer: ReturnType<typeof setTimeout> | null = null;
let saving: Promise<void> = Promise.resolve();

/** Autosave a moment after the last change that worked out; saves go one after another, newest content last. */
function scheduleSave(wait = 600) {
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    saveTimer = null;
    saving = saving.then(saveNow);
  }, wait);
}

async function saveNow() {
  const { saved, structure, company, owner } = state;
  if (!saved || !structure || !company) return;
  publish({ saving: "saving" });
  const { model, refused } = await putSaved(saved.companyId, { structure, company, scope: saved.scope, as_of: saved.asOf });
  if (state.owner !== owner || state.saved?.companyId !== saved.companyId) return;
  if (model) publish({ saved: savedFrom(model), saving: "saved", saveRefused: null });
  else publish({ saving: "failed", saveRefused: refused ?? null });
}

/** Save what is pending now and wait for every save under way: before a roll forward, which reads the saved copy. */
export async function flushSave() {
  if (saveTimer) {
    clearTimeout(saveTimer);
    saveTimer = null;
    saving = saving.then(saveNow);
  }
  if (pending) {
    clearTimeout(pending);
    pending = null;
    if (state.structure && state.company) await setModel(state.owner, state.structure, state.company);
    if (saveTimer) {
      clearTimeout(saveTimer);
      saveTimer = null;
      saving = saving.then(saveNow);
    }
  }
  await saving;
}

/** Make the model on screen a company's saved model (kept in coverage): saved at once. */
export function keepAs(saved: Saved) {
  publish({ saved, saving: "idle", saveRefused: null });
  scheduleSave(0);
}


let pending: ReturnType<typeof setTimeout> | null = null;

/** A typed change to the company: shown at once, worked out once the typing rests (300 ms). */
export function editCompany(owner: string | null, company: ModelCompany) {
  asked++;                // a run already on its way was for the company before this edit: its answer is stale
  publish({ company });
  if (pending) clearTimeout(pending);
  pending = setTimeout(() => {
    pending = null;
    if (state.structure && state.company) void setModel(owner, state.structure, state.company);
  }, 300);
}

/* ------------------------------------------------------------------ valuation (PLAN.md R6) */

export type RiskFree = Schemas["RiskFreeOut"];

/** The country's 10-year government yield for a covered company, the discount rate's starting point. */
export async function riskFree(companyId: string): Promise<RiskFree | null> {
  const { data } = await api.GET("/api/research/companies/{company_id}/risk-free", {
    params: { path: { company_id: companyId } },
  });
  return data ?? null;
}

const EXCEL_DAY_ZERO = Date.UTC(1899, 11, 30);
const DAY_MS = 86_400_000;

/** A date as Excel holds it (a day number), so the price's date is a cell the workbook and the add-in read alike. */
export function excelDay(iso: string): number | null {
  const ms = Date.parse(`${iso}T00:00:00Z`);
  return Number.isFinite(ms) ? Math.round((ms - EXCEL_DAY_ZERO) / DAY_MS) : null;
}

/** An Excel day number as a calendar date, "2026-10-16". */
export function isoOfExcelDay(day: number | null | undefined): string {
  if (day === null || day === undefined || !Number.isFinite(day)) return "";
  return new Date(EXCEL_DAY_ZERO + Math.round(day) * DAY_MS).toISOString().slice(0, 10);
}

/** The value in last year's column of a valuation row (the model values the company as at last year's end). */
export function valuationValue(run: ModelRun, row: string): number | null {
  const found = run.rows.find((r) => r.key === row);
  return found?.cells?.ly?.value ?? null;
}
