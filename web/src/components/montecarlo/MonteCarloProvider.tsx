"use client";

import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { dealLabel, describeDealValue } from "@/components/deal/DealScreen";
import { num, type Settings, useSettings } from "@/components/settings/SettingsProvider";
import type { Schemas } from "@/lib/api/client";
import type { DealInputs } from "@/lib/deal/fields";
import type { FieldSpec } from "@/lib/fields";
import { fmtInput } from "@/lib/format";
import { describeJob, type Job, JobFailed, runJob } from "@/lib/jobs";
import { MAX_SIMULATION_PATHS } from "@/lib/limits";

export type Scenario = "recession" | "base" | "bull" | "stagflation";
export const SCENARIOS: { id: Scenario; label: string }[] = [
  { id: "recession", label: "Recession" },
  { id: "stagflation", label: "Stagflation" },
  { id: "base", label: "Base" },
  { id: "bull", label: "Bull" },
];

/** Simulation settings owned by this screen; EBITDA, entry multiple and hold come from the deal. */
export type SimInputs = {
  n: number;
  hurdle: number;
  growth_mean: number;
  growth_std: number;
  exit_mean: number;
  exit_std: number;
  rate_mean: number;
  rate_std: number;
  gm_mean: number;
  gm_std: number;
};
export type SimKey = keyof SimInputs;

/** Bounds mirror MCInputsIn in api/schemas.py. */
export const SIM_FIELDS: Record<SimKey, FieldSpec> = {
  n: { label: "Paths", unit: "n", step: 5000, decimals: 0, min: 1000, max: MAX_SIMULATION_PATHS, integer: true },
  hurdle: { label: "Hurdle IRR", unit: "%", step: 1, decimals: 1, min: 0 },
  growth_mean: { label: "Mean", unit: "%", step: 0.5, decimals: 1 },
  growth_std: { label: "Std dev", unit: "%", step: 0.5, decimals: 1, min: 0.1 },
  exit_mean: { label: "Mean", unit: "x", step: 0.5, decimals: 1, min: 1 },
  exit_std: { label: "Std dev", unit: "x", step: 0.25, decimals: 2, min: 0.1 },
  rate_mean: { label: "Mean", unit: "%", step: 0.25, decimals: 2, min: 0 },
  rate_std: { label: "Std dev", unit: "%", step: 0.25, decimals: 2, min: 0.1 },
  gm_mean: { label: "Mean", unit: "%", step: 1, decimals: 1, min: 1, max: 99 },
  gm_std: { label: "Std dev", unit: "%", step: 0.5, decimals: 1, min: 0.1 },
};

const SIM_LABELS: Record<SimKey, string> = {
  n: "Paths",
  hurdle: "Hurdle IRR",
  growth_mean: "Growth mean",
  growth_std: "Growth std dev",
  exit_mean: "Exit multiple mean",
  exit_std: "Exit multiple std dev",
  rate_mean: "Rate mean",
  rate_std: "Rate std dev",
  gm_mean: "Gross margin mean",
  gm_std: "Gross margin std dev",
};

/** Starting values: the Monte Carlo defaults in Settings. */
function simFromSettings(s: Settings): SimInputs {
  return {
    n: num(s, "mc_n", 50000),
    hurdle: num(s, "mc_hurdle", 20),
    growth_mean: num(s, "mc_growth_mean", 5),
    growth_std: num(s, "mc_growth_std", 3),
    exit_mean: num(s, "mc_exit_mean", 10),
    exit_std: num(s, "mc_exit_std", 1.5),
    rate_mean: num(s, "mc_rate_mean", 6.5),
    rate_std: num(s, "mc_rate_std", 1.5),
    gm_mean: num(s, "mc_gm_mean", 40),
    gm_std: num(s, "mc_gm_std", 3),
  };
}

/** Deal inputs the simulation reads (core/montecarlo.py::build_sim_params); others don't make it stale. */
const SIM_DEAL_KEYS: (keyof DealInputs)[] = ["ebitda", "entry_mult", "hold", "opex", "da", "tax", "capex", "nwc", "debt_pct", "senior_pct", "mezz_spread"];

type Snapshot ={ sim: SimInputs; deal: DealInputs; settings: Settings; scenario: Scenario | null; seed: number | null };

type RunState = {
  status: "idle" | "running" | "ok" | "error" | "cancelled";
  result?: Schemas["MonteCarloResponse"];
  scenarios?: Schemas["ScenariosResponse"];
  ranFor?: Snapshot;
  error?: string;
  /** While running: what the server is doing, and how far along (0-1) */
  progress?: { stage: string; fraction: number };
};

type MonteCarloContext = {
  sim: SimInputs;
  setSim: (key: SimKey, value: number) => void;
  scenario: Scenario | null;
  setScenario: (s: Scenario | null) => void;
  seed: number | null;
  setSeed: (s: number | null) => void;
  run: RunState;
  runNow: () => void;
  /** Stop the run in progress (the server drops it too) */
  cancel: () => void;
  /** A result exists but inputs, the deal or settings changed since */
  stale: boolean;
  /** Human-readable differences from the last run */
  changes: string[];
  /** Hurdle as a fraction */
  hurdle: number;
};

const Ctx = createContext<MonteCarloContext | null>(null);

/** Both runs' progress as one: the main simulation, then the four scenarios. */
function combined(main: Job | undefined, scen: Job | undefined): { stage: string; fraction: number } {
  const done = (j: Job | undefined) => (j?.status === "succeeded" ? 1 : (j?.progress ?? 0));
  const fraction = (done(main) + done(scen)) / 2;
  const mainDone = main?.status === "succeeded";
  return { stage: mainDone ? `Scenarios: ${describeJob(scen)}` : describeJob(main), fraction };
}

export function MonteCarloProvider({ children }: { children: React.ReactNode }) {
  const { effective, overrides } = useSettings();
  const { inputs: deal } = useDeal();
  const [edits, setEdits] = useState<Partial<SimInputs>>({});
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [seed, setSeed] = useState<number | null>(42);
  const [run, setRun] = useState<RunState>({ status: "idle" });
  const inflight = useRef<AbortController | null>(null);

  const sim = useMemo(() => ({ ...simFromSettings(effective), ...edits }), [effective, edits]);
  const setSim = useCallback((key: SimKey, value: number) => setEdits((e) => ({ ...e, [key]: value })), []);

  const snapshot: Snapshot = useMemo(() => ({ sim, deal, settings: overrides, scenario, seed }), [sim, deal, overrides, scenario, seed]);

  const runNow = useCallback(async () => {
    inflight.current?.abort();
    const ctrl = new AbortController();
    inflight.current = ctrl;
    const snap = snapshot;
    setRun((r) => ({ ...r, status: "running", error: undefined, progress: { stage: "Starting", fraction: 0 } }));
    const mc = { ...snap.sim, ebitda: snap.deal.ebitda, entry_mult: snap.deal.entry_mult, hold: snap.deal.hold };
    // Background jobs (PLAN.md 1.9): the server queues both and runs them one
    // at a time, so the rest of the app stays usable while they run
    let main: Job | undefined;
    let scen: Job | undefined;
    const update = () => {
      if (!ctrl.signal.aborted) setRun((r) => ({ ...r, progress: combined(main, scen) }));
    };
    try {
      const [result, scenarios] = await Promise.all([
        runJob<Schemas["MonteCarloResponse"]>(
          { kind: "montecarlo.run", input: { mc, deal: snap.deal, settings: snap.settings, scenario: snap.scenario, seed: snap.seed, histogram_bins: 80, scatter_points: 2000 } },
          { signal: ctrl.signal, onUpdate: (j) => { main = j; update(); } },
        ).then((r) => {
          if (main) main = { ...main, status: "succeeded" };
          update();
          return r;
        }),
        runJob<Schemas["ScenariosResponse"]>(
          { kind: "montecarlo.scenarios", input: { mc, deal: snap.deal, settings: snap.settings, seed: snap.seed } },
          { signal: ctrl.signal, onUpdate: (j) => { scen = j; update(); } },
        ),
      ]);
      if (ctrl.signal.aborted) return;
      setRun({ status: "ok", result, scenarios, ranFor: snap });
    } catch (e) {
      if (ctrl.signal.aborted || (e as Error)?.name === "AbortError") return;
      ctrl.abort(); // cancels the other job on the server
      const error = e instanceof JobFailed ? e.message : "Can't reach the API. If it was idle it may still be starting; try again shortly.";
      setRun((r) => ({ ...r, status: "error", error, progress: undefined }));
    }
  }, [snapshot]);

  const cancel = useCallback(() => {
    inflight.current?.abort();
    inflight.current = null;
    setRun((r) => ({ ...r, status: r.result ? "ok" : "cancelled", progress: undefined }));
  }, []);

  const changes = useMemo(() => {
    const prev = run.ranFor;
    if (!prev) return [];
    const out: string[] = [];
    (Object.keys(sim) as SimKey[]).forEach((k) => {
      if (prev.sim[k] !== sim[k]) {
        const d = SIM_FIELDS[k].decimals;
        out.push(`${SIM_LABELS[k]} ${fmtInput(prev.sim[k], d)} → ${fmtInput(sim[k], d)}`);
      }
    });
    SIM_DEAL_KEYS.forEach((k) => {
      if (prev.deal[k] !== deal[k]) out.push(`${dealLabel(k)} ${describeDealValue(k, prev.deal[k])} → ${describeDealValue(k, deal[k])}`);
    });
    if (prev.scenario !== scenario) out.push(`Scenario ${prev.scenario ?? "none"} → ${scenario ?? "none"}`);
    if (prev.seed !== seed) out.push(`Seed ${prev.seed ?? "random"} → ${seed ?? "random"}`);
    if (JSON.stringify(prev.settings) !== JSON.stringify(overrides)) out.push("Settings changed");
    return out;
  }, [run.ranFor, sim, deal, scenario, seed, overrides]);

  const value = useMemo(
    () => ({
      sim,
      setSim,
      scenario,
      setScenario,
      seed,
      setSeed,
      run,
      runNow: () => void runNow(),
      cancel,
      stale: !!run.result && changes.length > 0,
      changes,
      hurdle: sim.hurdle / 100,
    }),
    [sim, setSim, scenario, seed, run, runNow, cancel, changes],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useMonteCarlo(): MonteCarloContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useMonteCarlo must be used inside MonteCarloProvider");
  return ctx;
}
