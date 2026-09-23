"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

import { api, type Schemas } from "@/lib/api/client";
import { useProfile } from "@/components/auth/ProfileProvider";
import { num, type Settings, useSettings } from "@/components/settings/SettingsProvider";
import { MoneyScope } from "@/components/ui/MoneyScope";
import { changedKeys, DEFAULT_INPUTS, type DealInputs, type DealRun } from "@/lib/deal/fields";
import { type Money, unitFactor } from "@/lib/money";

/** Debounce between the last edit and an automatic rerun. */
const AUTO_RUN_DELAY_MS = 300;
/** Debounce between the last edit and autosaving an open deal. */
const AUTOSAVE_DELAY_MS = 800;
/** The deal this browser had open last, reopened on the next visit (a convenience only). */
const LAST_DEAL_KEY = "fse.lastDeal.v1";

type RunState = {
  status: "idle" | "running" | "ok" | "error";
  result?: DealRun;
  /** Inputs the current result was computed from */
  ranFor?: DealInputs;
  /** JSON of the settings overrides it was computed with */
  ranSettings?: string;
  ms?: number;
  error?: string;
};

/** The saved deal being edited. */
export type OpenDeal = { id: string; name: string; latestVersion: number; archived: boolean; updatedAt: string };
/** Whether the open deal's latest edits are in the database. */
export type DealSaveState = "unsaved" | "saving" | "saved" | "error";
type DealDetail = Schemas["DealDetail"];
type Outcome = { ok: boolean; error?: string };

type DealContext = {
  inputs: DealInputs;
  setField: <K extends keyof DealInputs>(key: K, value: DealInputs[K]) => void;
  setFields: (patch: Partial<DealInputs>) => void;
  /** The deal's currency and unit. A new unit keeps the deal's size: 100 (millions) becomes 100,000 (thousands). */
  money: Money;
  setMoney: (money: Money) => void;
  autoUpdate: boolean;
  setAutoUpdate: (on: boolean) => void;
  run: RunState;
  /** Inputs edited since the result was computed */
  pending: (keyof DealInputs)[];
  /** Settings changed since the result was computed */
  settingsChanged: boolean;
  runNow: () => void;
  discard: () => void;
  /** Hurdle IRR from settings, as a fraction */
  hurdle: number;
  /** The saved deal these inputs belong to; null until the deal is saved */
  current: OpenDeal | null;
  saveState: DealSaveState;
  /** Open a saved deal: its inputs and settings replace what's on screen */
  openDeal: (id: string) => Promise<Outcome>;
  /** Save what's on screen as a new deal and carry on editing it */
  saveAs: (name: string) => Promise<Outcome>;
  /** Start again from the default inputs, not attached to a saved deal */
  newDeal: () => void;
  /** Show a deal the API returned (after a restore) as the open, saved deal */
  adopt: (deal: DealDetail) => void;
  /** Keep the open deal's name, version number or archive flag in step after a change */
  patchCurrent: (patch: Partial<OpenDeal>) => void;
  /** Send a pending autosave now (before keeping a version); false if it failed */
  flush: () => Promise<boolean>;
};

const Ctx = createContext<DealContext | null>(null);

type ValidationError = { detail?: { loc?: (string | number)[]; msg?: string }[] };

function describeError(err: unknown): string {
  const detail = (err as ValidationError | { detail?: string })?.detail;
  // A refusal such as a usage limit is one sentence
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => `${String(d.loc?.at(-1) ?? "input")}: ${d.msg}`).join("; ");
  }
  return "The model couldn't run with these inputs.";
}

/** A message from an API refusal, for the saved-deal screens. */
export function apiMessage(err: unknown, fallback: string): string {
  const detail = (err as { detail?: unknown } | undefined)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) return String((detail[0] as { msg?: string }).msg ?? fallback);
  return fallback;
}

/** Inputs and settings as one comparable string, whatever order the keys arrive in. */
function contentKey(inputs: DealInputs, settings: Settings): string {
  const sorted = (o: object) => Object.fromEntries(Object.entries(o).sort(([a], [b]) => a.localeCompare(b)));
  return JSON.stringify([sorted(inputs), sorted(settings)]);
}

function toOpenDeal(d: Schemas["DealSummary"]): OpenDeal {
  return { id: d.id, name: d.name, latestVersion: d.latest_version, archived: d.archived, updatedAt: d.updated_at };
}

function rememberDeal(id: string | null) {
  try {
    if (id) window.localStorage.setItem(LAST_DEAL_KEY, id);
    else window.localStorage.removeItem(LAST_DEAL_KEY);
  } catch {
    // Blocked storage: the deal just won't reopen by itself next visit
  }
}

function lastDeal(): string | null {
  try {
    return window.localStorage.getItem(LAST_DEAL_KEY);
  } catch {
    return null;
  }
}

export function DealProvider({ children }: { children: React.ReactNode }) {
  const { overrides, effective, defaults, loaded: settingsLoaded, replace: replaceSettings, set: setSetting } = useSettings();
  const { profile } = useProfile();
  const [inputs, setInputs] = useState<DealInputs>(DEFAULT_INPUTS);
  // A new deal starts in the account's currency (PLAN.md 2.2)
  const accountCurrency = profile?.preferred_currency || DEFAULT_INPUTS.currency;
  const startInputs = useMemo(() => ({ ...DEFAULT_INPUTS, currency: accountCurrency }), [accountCurrency]);
  const [autoUpdate, setAutoUpdate] = useState(true);
  const [run, setRun] = useState<RunState>({ status: "idle" });
  const inflight = useRef<AbortController | null>(null);
  const settingsKey = JSON.stringify(overrides);
  // Hurdle comes from settings so Returns and Monte Carlo agree
  const hurdle = num(effective, "mc_hurdle", 20) / 100;

  const execute = useCallback(async (snapshot: DealInputs, settings: Settings) => {
    inflight.current?.abort();
    const ctrl = new AbortController();
    inflight.current = ctrl;
    setRun((r) => ({ ...r, status: "running" }));
    const t0 = performance.now();
    try {
      const { data, error } = await api.POST("/api/deal/run", {
        body: { inputs: snapshot, settings },
        signal: ctrl.signal,
      });
      if (ctrl.signal.aborted) return;
      if (data) {
        setRun({ status: "ok", result: data, ranFor: snapshot, ranSettings: JSON.stringify(settings), ms: performance.now() - t0 });
      } else {
        setRun((r) => ({ ...r, status: "error", error: describeError(error) }));
      }
    } catch (e) {
      if (ctrl.signal.aborted || (e as Error)?.name === "AbortError") return;
      setRun((r) => ({ ...r, status: "error", error: "Can't reach the API. If it was idle it may still be starting; try again shortly." }));
    }
  }, []);

  // Automatic rerun after edits or settings changes (and the first run once settings load)
  useEffect(() => {
    if (!autoUpdate || !defaults) return;
    const id = setTimeout(() => void execute(inputs, overrides), AUTO_RUN_DELAY_MS);
    return () => clearTimeout(id);
  }, [inputs, overrides, defaults, autoUpdate, execute]);

  const setField = useCallback(<K extends keyof DealInputs>(key: K, value: DealInputs[K]) => {
    setInputs((prev) => (prev[key] === value ? prev : { ...prev, [key]: value }));
  }, []);
  const setFields = useCallback((patch: Partial<DealInputs>) => {
    setInputs((prev) => ({ ...prev, ...patch }));
  }, []);

  const money = useMemo<Money>(() => ({ currency: inputs.currency, unit: inputs.unit }), [inputs.currency, inputs.unit]);
  const otherUses = num(effective, "other_uses", 0);
  const setMoney = useCallback(
    (next: Money) => {
      setInputs((prev) => {
        const k = unitFactor(prev.unit, next.unit);
        // Keep the deal the same size in its new unit
        return k === 1
          ? { ...prev, currency: next.currency }
          : { ...prev, currency: next.currency, unit: next.unit, ebitda: prev.ebitda * k, mincash: prev.mincash * k };
      });
      const k = unitFactor(inputs.unit, next.unit);
      if (k !== 1 && otherUses !== 0) setSetting("other_uses", otherUses * k);
    },
    [inputs.unit, otherUses, setSetting],
  );
  // ---- Saved deals (PLAN.md 1.5) ----
  const [current, setCurrent] = useState<OpenDeal | null>(null);
  const [saveState, setSaveState] = useState<DealSaveState>("unsaved");
  // What the database holds for the open deal, as a contentKey
  const savedKey = useRef<string | null>(null);
  const liveKey = contentKey(inputs, overrides);

  const adopt = useCallback(
    (deal: DealDetail) => {
      const dealInputs = { ...DEFAULT_INPUTS, ...deal.inputs } as DealInputs;
      const dealSettings = deal.settings as Settings;
      savedKey.current = contentKey(dealInputs, dealSettings);
      setInputs(dealInputs);
      replaceSettings(dealSettings);
      setCurrent(toOpenDeal(deal));
      setSaveState("saved");
      rememberDeal(deal.id);
    },
    [replaceSettings],
  );

  const openDeal = useCallback(
    async (id: string): Promise<Outcome> => {
      const { data, error } = await api.GET("/api/deals/{deal_id}", { params: { path: { deal_id: id } } });
      if (!data) return { ok: false, error: apiMessage(error, "That deal couldn't be opened.") };
      adopt(data);
      return { ok: true };
    },
    [adopt],
  );

  const saveAs = useCallback(
    async (name: string): Promise<Outcome> => {
      const { data, error } = await api.POST("/api/deals", { body: { name, inputs, settings: overrides } });
      if (!data) return { ok: false, error: apiMessage(error, "The deal couldn't be saved.") };
      adopt(data);
      return { ok: true };
    },
    [adopt, inputs, overrides],
  );

  const newDeal = useCallback(() => {
    savedKey.current = null;
    setCurrent(null);
    setSaveState("unsaved");
    setInputs(startInputs);
    rememberDeal(null);
  }, [startInputs]);

  // The start deal takes the account's currency once the profile loads
  // (adjusted during render, as React recommends for derived state), unless a
  // saved deal is open or the currency was already changed by hand
  const [appliedCurrency, setAppliedCurrency] = useState(DEFAULT_INPUTS.currency);
  if (appliedCurrency !== accountCurrency) {
    setAppliedCurrency(accountCurrency);
    if (current === null && inputs.currency === appliedCurrency) setInputs({ ...inputs, currency: accountCurrency });
  }

  const patchCurrent = useCallback((patch: Partial<OpenDeal>) => {
    setCurrent((c) => (c ? { ...c, ...patch } : c));
  }, []);

  const writeDraft = useCallback(async (id: string, key: string, body: { inputs: DealInputs; settings: Settings }) => {
    setSaveState("saving");
    try {
      const { data } = await api.PUT("/api/deals/{deal_id}/draft", { params: { path: { deal_id: id } }, body });
      if (!data) {
        setSaveState("error");
        return false;
      }
      savedKey.current = key;
      setCurrent((c) => (c && c.id === id ? { ...c, latestVersion: data.latest_version, updatedAt: data.updated_at } : c));
      return true;
    } catch {
      setSaveState("error");
      return false;
    }
  }, []);

  // Reopen the deal this browser had open, once settings have loaded (the deal's settings win)
  const reopened = useRef(false);
  useEffect(() => {
    if (!settingsLoaded || reopened.current) return;
    reopened.current = true;
    const id = lastDeal();
    if (!id) return;
    api
      .GET("/api/deals/{deal_id}", { params: { path: { deal_id: id } } })
      .then(({ data }) => (data ? adopt(data) : rememberDeal(null)))
      .catch(() => {});
  }, [settingsLoaded, adopt]);

  // Autosave the open deal shortly after each edit to its inputs or settings
  const currentId = current?.id;
  const latest = useRef({ inputs, overrides });
  useEffect(() => {
    latest.current = { inputs, overrides };
  }, [inputs, overrides]);
  useEffect(() => {
    if (!currentId || savedKey.current === null || liveKey === savedKey.current) {
      if (currentId && liveKey === savedKey.current) setSaveState((s) => (s === "error" ? s : "saved"));
      return;
    }
    setSaveState("saving");
    const id = setTimeout(() => {
      const body = { inputs: latest.current.inputs, settings: latest.current.overrides };
      void writeDraft(currentId, liveKey, body).then((ok) => {
        if (ok && savedKey.current === contentKey(latest.current.inputs, latest.current.overrides)) setSaveState("saved");
      });
    }, AUTOSAVE_DELAY_MS);
    return () => clearTimeout(id);
  }, [liveKey, currentId, writeDraft]);

  const flush = useCallback(async () => {
    if (!currentId) return false;
    if (liveKey === savedKey.current) return true;
    const ok = await writeDraft(currentId, liveKey, { inputs, settings: overrides });
    if (ok) setSaveState("saved");
    return ok;
  }, [currentId, liveKey, inputs, overrides, writeDraft]);

  const pending = useMemo(() => (run.ranFor ? changedKeys(run.ranFor, inputs) : []), [run.ranFor, inputs]);
  const settingsChanged = run.ranSettings !== undefined && run.ranSettings !== settingsKey;
  const runNow = useCallback(() => void execute(inputs, overrides), [execute, inputs, overrides]);
  const discard = useCallback(() => {
    if (run.ranFor) setInputs(run.ranFor);
  }, [run.ranFor]);

  const value = useMemo(
    () => ({
      inputs, setField, setFields, money, setMoney, autoUpdate, setAutoUpdate, run, pending, settingsChanged, runNow, discard, hurdle,
      current, saveState, openDeal, saveAs, newDeal, adopt, patchCurrent, flush,
    }),
    [inputs, setField, setFields, money, setMoney, autoUpdate, run, pending, settingsChanged, runNow, discard, hurdle,
      current, saveState, openDeal, saveAs, newDeal, adopt, patchCurrent, flush],
  );
  return (
    <Ctx.Provider value={value}>
      <MoneyScope money={money}>{children}</MoneyScope>
    </Ctx.Provider>
  );
}

export function useDeal(): DealContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useDeal must be used inside DealProvider");
  return ctx;
}
