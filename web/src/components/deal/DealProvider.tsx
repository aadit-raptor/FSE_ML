"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api/client";
import { changedKeys, DEFAULT_INPUTS, type DealInputs, type DealRun } from "@/lib/deal/fields";

/** Debounce between the last edit and an automatic rerun. */
const AUTO_RUN_DELAY_MS = 300;

type RunState = {
  status: "idle" | "running" | "ok" | "error";
  result?: DealRun;
  /** Inputs the current result was computed from */
  ranFor?: DealInputs;
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
  const [inputs, setInputs] = useState<DealInputs>(DEFAULT_INPUTS);
  const [autoUpdate, setAutoUpdate] = useState(true);
  const [run, setRun] = useState<RunState>({ status: "idle" });
  const [hurdle, setHurdle] = useState(0.2);
  const inflight = useRef<AbortController | null>(null);

  const execute = useCallback(async (snapshot: DealInputs) => {
    inflight.current?.abort();
    const ctrl = new AbortController();
    inflight.current = ctrl;
    setRun((r) => ({ ...r, status: "running" }));
    const t0 = performance.now();
    try {
      const { data, error } = await api.POST("/api/deal/run", {
        body: { inputs: snapshot, settings: {} },
        signal: ctrl.signal,
      });
      if (ctrl.signal.aborted) return;
      if (data) {
        setRun({ status: "ok", result: data, ranFor: snapshot, ms: performance.now() - t0 });
      } else {
        setRun((r) => ({ ...r, status: "error", error: describeError(error) }));
      }
    } catch (e) {
      if (ctrl.signal.aborted || (e as Error)?.name === "AbortError") return;
      setRun((r) => ({ ...r, status: "error", error: "Can't reach the API. Is uvicorn running on port 8000?" }));
    }
  }, []);

  // Automatic rerun after edits (and the first run on load)
  useEffect(() => {
    if (!autoUpdate) return;
    const id = setTimeout(() => void execute(inputs), AUTO_RUN_DELAY_MS);
    return () => clearTimeout(id);
  }, [inputs, autoUpdate, execute]);

  // Hurdle rate comes from settings so the returns screen matches Monte Carlo
  useEffect(() => {
    let cancelled = false;
    api
      .GET("/api/settings/defaults")
      .then(({ data }) => {
        const h = data?.defaults?.mc_hurdle;
        if (!cancelled && typeof h === "number") setHurdle(h / 100);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  const setField = useCallback(<K extends keyof DealInputs>(key: K, value: DealInputs[K]) => {
    setInputs((prev) => (prev[key] === value ? prev : { ...prev, [key]: value }));
  }, []);
  const setFields = useCallback((patch: Partial<DealInputs>) => {
    setInputs((prev) => ({ ...prev, ...patch }));
  }, []);

  const pending = useMemo(() => (run.ranFor ? changedKeys(run.ranFor, inputs) : []), [run.ranFor, inputs]);
  const runNow = useCallback(() => void execute(inputs), [execute, inputs]);
  const discard = useCallback(() => {
    if (run.ranFor) setInputs(run.ranFor);
  }, [run.ranFor]);

  const value = useMemo(
    () => ({ inputs, setField, setFields, autoUpdate, setAutoUpdate, run, pending, runNow, discard, hurdle }),
    [inputs, setField, setFields, autoUpdate, run, pending, runNow, discard, hurdle],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useDeal(): DealContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useDeal must be used inside DealProvider");
  return ctx;
}
