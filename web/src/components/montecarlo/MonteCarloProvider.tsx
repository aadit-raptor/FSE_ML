"use client";

import { useTranslations } from "next-intl";
import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { useDealChange, useDealLabel } from "@/components/deal/DealScreen";
import { num, type Settings, useSettings } from "@/components/settings/SettingsProvider";
import type { Schemas } from "@/lib/api/client";
import { apiInputs, changedKeys, type DealInputs } from "@/lib/deal/fields";
import type { FieldSpec } from "@/lib/fields";
import { fmtInput } from "@/lib/format";
import { describeJob, type Job, JobFailed, type JobMessages, runJob } from "@/lib/jobs";
import { MAX_SIMULATION_PATHS } from "@/lib/limits";

export type Scenario = "recession" | "base" | "bull" | "stagflation";

/** `labelKey` names the preset in the `montecarlo` namespace (PLAN.md 2.3b). i18n-keys: montecarlo.scenario* */
export const SCENARIOS: { id: Scenario; labelKey: string }[] = [
  { id: "recession", labelKey: "scenarioRecession" },
  { id: "stagflation", labelKey: "scenarioStagflation" },
  { id: "base", labelKey: "scenarioBase" },
  { id: "bull", labelKey: "scenarioBull" },
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
  n: { unit: "n", step: 5000, decimals: 0, min: 1000, max: MAX_SIMULATION_PATHS, integer: true },
  hurdle: { unit: "%", step: 1, decimals: 1, min: 0 },
  growth_mean: { unit: "%", step: 0.5, decimals: 1 },
  growth_std: { unit: "%", step: 0.5, decimals: 1, min: 0.1 },
  exit_mean: { unit: "x", step: 0.5, decimals: 1, min: 1 },
  exit_std: { unit: "x", step: 0.25, decimals: 2, min: 0.1 },
  rate_mean: { unit: "%", step: 0.25, decimals: 2, min: 0 },
  rate_std: { unit: "%", step: 0.25, decimals: 2, min: 0.1 },
  gm_mean: { unit: "%", step: 1, decimals: 1, min: 1, max: 99 },
  gm_std: { unit: "%", step: 0.5, decimals: 1, min: 0.1 },
};

/** The short label each field takes in the rail ("Mean"). i18n-keys: montecarlo.sim*, montecarlo.change* */
export const SIM_LABEL_KEY: Record<SimKey, string> = {
  n: "simPaths",
  hurdle: "simHurdle",
  growth_mean: "simMean",
  growth_std: "simStdDev",
  exit_mean: "simMean",
  exit_std: "simStdDev",
  rate_mean: "simMean",
  rate_std: "simStdDev",
  gm_mean: "simMean",
  gm_std: "simStdDev",
};

const CHANGE_LABEL_KEY: Record<SimKey, string> = {
  n: "simPaths",
  hurdle: "simHurdle",
  growth_mean: "changeGrowthMean",
  growth_std: "changeGrowthStd",
  exit_mean: "changeExitMean",
  exit_std: "changeExitStd",
  rate_mean: "changeRateMean",
  rate_std: "changeRateStd",
  gm_mean: "changeGmMean",
  gm_std: "changeGmStd",
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
const SIM_DEAL_KEYS: (keyof DealInputs)[] = [
  "ebitda", "entry_mult", "hold", "opex", "da", "tax", "capex", "nwc", "debt_pct", "senior_pct", "mezz_spread", "tranches",
  "tax_interest_limit", "tax_interest_limit_pct", "tax_interest_limit_amount", "tax_loss_carryforward",
  "tax_loss_limit_pct", "tax_loss_limit_amount", "tax_minimum_pct",
];

/** What a run was for; `dealId` is the saved deal it ran on (null if unsaved), so its downloads go in that deal's history. */
type Snapshot = { sim: SimInputs; deal: DealInputs; settings: Settings; scenario: Scenario | null; seed: number | null; dealId: string | null };

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
  /** Drop the rail's own edits of these fields, so each shows its Setting again (sourced figures were applied) */
  clearSimEdits: (keys: SimKey[]) => void;
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

export function MonteCarloProvider({ children }: { children: React.ReactNode }) {
  const { effective, overrides } = useSettings();
  const { inputs: deal, current } = useDeal();
  const dealId = current?.id ?? null;
  const t = useTranslations("montecarlo");
  const e = useTranslations("errors");
  const jobText = useTranslations("jobs");
  const dealLabel = useDealLabel();
  const dealChange = useDealChange();
  const [edits, setEdits] = useState<Partial<SimInputs>>({});
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [seed, setSeed] = useState<number | null>(42);
  const [run, setRun] = useState<RunState>({ status: "idle" });
  const inflight = useRef<AbortController | null>(null);

  const sim = useMemo(() => ({ ...simFromSettings(effective), ...edits }), [effective, edits]);
  const setSim = useCallback((key: SimKey, value: number) => setEdits((prev) => ({ ...prev, [key]: value })), []);
  const clearSimEdits = useCallback(
    (keys: SimKey[]) =>
      setEdits((prev) => {
        const next = { ...prev };
        keys.forEach((k) => delete next[k]);
        return next;
      }),
    [],
  );

  const snapshot: Snapshot = useMemo(() => ({ sim, deal, settings: overrides, scenario, seed, dealId }), [sim, deal, overrides, scenario, seed, dealId]);

  const jobMessages = useMemo<JobMessages>(
    () => ({
      startFailed: jobText("startFailed"),
      lostTrack: jobText("lostTrack"),
      resultExpired: jobText("resultExpired"),
      didntFinish: jobText("didntFinish"),
    }),
    [jobText],
  );

  /** Both runs' progress as one: the main simulation, then the four scenarios. */
  const combined = useCallback(
    (main: Job | undefined, scen: Job | undefined): { stage: string; fraction: number } => {
      const done = (j: Job | undefined) => (j?.status === "succeeded" ? 1 : (j?.progress ?? 0));
      const fraction = (done(main) + done(scen)) / 2;
      const describe = (j: Job | undefined) => {
        const s = describeJob(j);
        return s.key === "running" ? (s.stage ?? jobText("running")) : jobText(s.key, { count: s.count ?? 0 });
      };
      return { stage: main?.status === "succeeded" ? t("scenariosStage", { stage: describe(scen) }) : describe(main), fraction };
    },
    [jobText, t],
  );

  const runNow = useCallback(async () => {
    inflight.current?.abort();
    const ctrl = new AbortController();
    inflight.current = ctrl;
    const snap = snapshot;
    setRun((r) => ({ ...r, status: "running", error: undefined, progress: { stage: t("starting"), fraction: 0 } }));
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
          {
            kind: "montecarlo.run",
            input: {
              mc,
              deal: apiInputs(snap.deal),
              settings: snap.settings,
              scenario: snap.scenario,
              seed: snap.seed,
              histogram_bins: 80,
              scatter_points: 2000,
            },
          },
          {
            signal: ctrl.signal,
            messages: jobMessages,
            onUpdate: (j) => {
              main = j;
              update();
            },
          },
        ).then((r) => {
          if (main) main = { ...main, status: "succeeded" };
          update();
          return r;
        }),
        runJob<Schemas["ScenariosResponse"]>(
          { kind: "montecarlo.scenarios", input: { mc, deal: apiInputs(snap.deal), settings: snap.settings, seed: snap.seed } },
          {
            signal: ctrl.signal,
            messages: jobMessages,
            onUpdate: (j) => {
              scen = j;
              update();
            },
          },
        ),
      ]);
      if (ctrl.signal.aborted) return;
      setRun({ status: "ok", result, scenarios, ranFor: snap });
    } catch (err) {
      if (ctrl.signal.aborted || (err as Error)?.name === "AbortError") return;
      ctrl.abort(); // cancels the other job on the server
      const error = err instanceof JobFailed ? err.message : e("apiWaking");
      setRun((r) => ({ ...r, status: "error", error, progress: undefined }));
    }
  }, [snapshot, combined, jobMessages, t, e]);

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
        out.push(`${t(CHANGE_LABEL_KEY[k])} ${fmtInput(prev.sim[k], d)} → ${fmtInput(sim[k], d)}`);
      }
    });
    // changedKeys compares the facility list by content, not by identity
    const moved = changedKeys(prev.deal, deal);
    SIM_DEAL_KEYS.forEach((k) => {
      if (moved.includes(k)) out.push(`${dealLabel(k)} ${dealChange(k, prev.deal[k])} → ${dealChange(k, deal[k])}`);
    });
    const presetName = (s: Scenario | null) => (s ? t(SCENARIOS.find((x) => x.id === s)!.labelKey) : t("noPreset"));
    if (prev.scenario !== scenario) out.push(t("changeScenario", { from: presetName(prev.scenario), to: presetName(scenario) }));
    if (prev.seed !== seed)
      out.push(t("changeSeed", { from: prev.seed === null ? t("randomSeed") : String(prev.seed), to: seed === null ? t("randomSeed") : String(seed) }));
    if (JSON.stringify(prev.settings) !== JSON.stringify(overrides)) out.push(t("changeSettings"));
    return out;
  }, [run.ranFor, sim, deal, scenario, seed, overrides, t, dealLabel, dealChange]);

  const value = useMemo(
    () => ({
      sim,
      setSim,
      clearSimEdits,
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
    [sim, setSim, clearSimEdits, scenario, seed, run, runNow, cancel, changes],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useMonteCarlo(): MonteCarloContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useMonteCarlo must be used inside MonteCarloProvider");
  return ctx;
}
