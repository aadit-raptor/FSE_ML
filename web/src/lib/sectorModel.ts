"use client";

import { useSyncExternalStore } from "react";

import { api, type Schemas } from "@/lib/api/client";
import { downloadPost } from "@/lib/export";

/**
 * Sector models (PLAN.md R4): the analyst picks a sector (which picks the model and its words), makes the last
 * changes for the company -- add, drop or swap a line of revenue -- types the drivers' history and the
 * assumptions, and sees the projection at once. The server works the model out (`/api/research/models/run`) and
 * writes it as live Excel formulas (`/workbook`, and `/cells` for the add-in's "Load model"), from one
 * definition.
 *
 * Nothing is saved yet (R5 keeps a model per company with versions): the model on screen lives in this page's
 * memory, kept across steps, and the workbook is how to keep it.
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

export type ModelState = {
  owner: string | null;
  structure: Structure | null;
  company: ModelCompany | null;
  run: ModelRun | null;
  refused: Refusal | null;
  fillNotes: string[];
  busy: boolean;
};

const EMPTY: ModelState = { owner: null, structure: null, company: null, run: null, refused: null, fillNotes: [], busy: false };
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

/** Work the model out for `structure` and `company` and show it; a later edit's answer wins over an earlier one's. */
export async function setModel(owner: string | null, structure: Structure, company: ModelCompany, fillNotes?: string[]) {
  const mine = ++asked;
  publish({ owner, structure, company, busy: true, ...(fillNotes ? { fillNotes } : {}) });
  const { run, refused } = await runModel(structure, company);
  if (mine !== asked) return;
  if (run) publish({ run, refused: null, busy: false, structure: run.structure, company: prunedTo(company, run) });
  else publish({ refused: refused ?? null, busy: false });
}

export function clearModel() {
  asked++;
  publish({ ...EMPTY });
}

let pending: ReturnType<typeof setTimeout> | null = null;

/** A typed change to the company: shown at once, worked out once the typing rests (300 ms). */
export function editCompany(owner: string | null, company: ModelCompany) {
  publish({ company });
  if (pending) clearTimeout(pending);
  pending = setTimeout(() => {
    pending = null;
    if (state.structure && state.company) void setModel(owner, state.structure, state.company);
  }, 300);
}
