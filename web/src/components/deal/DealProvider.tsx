"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api/client";
import { num, type Settings, useSettings } from "@/components/settings/SettingsProvider";
import { changedKeys, DEFAULT_INPUTS, type DealInputs, type DealRun } from "@/lib/deal/fields";

/** Debounce between the last edit and an automatic rerun. */
const AUTO_RUN_DELAY_MS = 300;

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

type DealContext = {
  inputs: DealInputs;
  setField: <K extends keyof DealInputs>(key: K, value: DealInputs[K]) => void;
  setFields: (patch: Partial<DealInputs>) => void;
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
};

const Ctx = createContext<DealContext | null>(null);

type ValidationError = { detail?: { loc?: (string | number)[]; msg?: string }[] };

function describeError(err: unknown): string {
  const detail = (err as ValidationError)?.detail;
  if (Array.isArray(detail) && detail.length) {
    return detail.map((d) => `${String(d.loc?.at(-1) ?? "input")}: ${d.msg}`).join("; ");
  }
  return "The model couldn't run with these inputs.";
}

export function DealProvider({ children }: { children: React.ReactNode }) {
  const { overrides, effective, defaults } = useSettings();
  const [inputs, setInputs] = useState<DealInputs>(DEFAULT_INPUTS);
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
      setRun((r) => ({ ...r, status: "error", error: "Can't reach the API. Is uvicorn running on port 8000?" }));
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

  const pending = useMemo(() => (run.ranFor ? changedKeys(run.ranFor, inputs) : []), [run.ranFor, inputs]);
  const settingsChanged = run.ranSettings !== undefined && run.ranSettings !== settingsKey;
  const runNow = useCallback(() => void execute(inputs, overrides), [execute, inputs, overrides]);
  const discard = useCallback(() => {
    if (run.ranFor) setInputs(run.ranFor);
  }, [run.ranFor]);

  const value = useMemo(
    () => ({ inputs, setField, setFields, autoUpdate, setAutoUpdate, run, pending, settingsChanged, runNow, discard, hurdle }),
    [inputs, setField, setFields, autoUpdate, run, pending, settingsChanged, runNow, discard, hurdle],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useDeal(): DealContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useDeal must be used inside DealProvider");
  return ctx;
}
