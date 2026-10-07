"use client";

import { useTranslations } from "next-intl";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { useLibrary } from "@/components/library/LibraryProvider";
import { type Settings, useSettings } from "@/components/settings/SettingsProvider";
import { api, type Schemas } from "@/lib/api/client";
import { type ActualsDraft, apiActuals, blankDraft, draftFrom, hasFigures, resizeYears, scaleDraft } from "@/lib/backtest/actuals";
import { apiInputs } from "@/lib/deal/fields";
import { type Money, unitFactor } from "@/lib/money";

export type PlanActualResult = Schemas["PlanActualResponse"];
export type ExampleDeal = Schemas["ExampleDeal"];
type DealSummary = Schemas["DealSummary"];
type DealInputsIn = Schemas["DealInputsIn"];

/** Which plan: a saved deal by id, or an example from the library by name. */
export type PlanRef = { kind: "deal"; id: string } | { kind: "example"; name: string };

/** The plan as it runs: a saved deal's inputs and Settings, or an example's. */
export type Plan = { ref: PlanRef; name: string; inputs: DealInputsIn; settings: Settings; money: Money; hold: number; example?: ExampleDeal };

/** Whether the actuals on screen are stored with the deal. Examples are never stored. */
export type ActualsSaveState = "saved" | "saving" | "unsaved" | "error" | "example";

type BacktestContext = {
  /** The account's saved deals; null while loading or when there is no database */
  deals: DealSummary[] | null;
  /** The example library; empty when it is switched off */
  examples: ExampleDeal[];
  libraryEnabled: boolean;
  /** The plan list has loaded (so no selection means there is no plan to pick) */
  ready: boolean;
  selected: PlanRef | null;
  select: (ref: PlanRef) => void;
  plan: Plan | null;
  actuals: ActualsDraft | null;
  setActuals: (next: ActualsDraft) => void;
  saveState: ActualsSaveState;
  status: "idle" | "loading" | "running" | "ok" | "error";
  result?: PlanActualResult;
  error?: string;
  /**
   * Called by each Backtest screen while it is mounted: nothing is fetched until the mode is opened,
   * and nothing reruns while it isn't on screen (the open deal is edited on the Deal screens). Returns
   * the cleanup.
   */
  activate: () => () => void;
};

const Ctx = createContext<BacktestContext | null>(null);
const PATHS = 30000;
const RUN_DELAY_MS = 300;
const SAVE_DELAY_MS = 800;

export const sameRef = (a: PlanRef | null, b: PlanRef | null) =>
  !!a && !!b && a.kind === b.kind && (a.kind === "deal" ? a.id === (b as typeof a).id : a.name === (b as typeof a).name);

/** A refusal (a usage limit, actuals that don't fit the plan) is already a sentence; validation errors are listed. */
function describe(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) return detail.map((d) => (d as { msg?: string }).msg ?? fallback).join("; ");
  return fallback;
}

export function BacktestProvider({ children }: { children: React.ReactNode }) {
  const t = useTranslations("backtest");
  const e = useTranslations("errors");
  const { overrides } = useSettings();
  const { inputs: openInputs, current } = useDeal();
  const [enabled, setEnabled] = useState(false);
  const [visible, setVisible] = useState(0);
  const [deals, setDeals] = useState<DealSummary[] | null>(null);
  const [library, setLibrary] = useState<{ enabled: boolean; examples: ExampleDeal[] }>({ enabled: false, examples: [] });
  // The plan list is fetched again when an administrator shows or hides the example library
  const libraryOn = useLibrary().state?.enabled ?? null;
  const [listedFor, setListedFor] = useState<boolean | null | undefined>(undefined);
  const listed = listedFor === libraryOn;
  const [selected, setSelected] = useState<PlanRef | null>(null);
  const [stored, setStored] = useState<Omit<Plan, "settings" | "inputs"> & { inputs?: DealInputsIn; settings?: Settings } | null>(null);
  const [actuals, setActualsState] = useState<ActualsDraft | null>(null);
  const [saveState, setSaveState] = useState<ActualsSaveState>("saved");
  const [state, setState] = useState<{ status: BacktestContext["status"]; result?: PlanActualResult; error?: string }>({ status: "idle" });
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  /** An example is its own plan and actuals, already in hand: nothing to fetch, nothing to save. */
  const showExample = useCallback((ex: ExampleDeal) => {
    const money = { currency: ex.plan.currency ?? "USD", unit: ex.plan.unit ?? "millions" };
    setStored({ ref: { kind: "example", name: ex.name }, name: ex.name, inputs: ex.plan, settings: {}, money, hold: ex.plan.hold ?? 5, example: ex });
    setActualsState(draftFrom(ex.actuals));
    setSaveState("example");
  }, []);

  // The plan list: saved deals and the example library, fetched once the mode opens
  useEffect(() => {
    if (!enabled || listed) return;
    let cancelled = false;
    Promise.all([
      api.GET("/api/deals").then(({ data }) => data?.deals ?? null).catch(() => null),
      api.GET("/api/backtesting/examples").then(({ data }) => data ?? { enabled: false, examples: [] }).catch(() => ({ enabled: false, examples: [] })),
    ]).then(([saved, lib]) => {
      if (cancelled) return;
      setDeals(saved);
      setLibrary(lib);
      setListedFor(libraryOn);
      // The deal open on the Deal screens first, else the newest saved deal, else an example
      const open = current && saved?.find((d) => d.id === current.id);
      const first = open ?? saved?.[0];
      if (first) setSelected({ kind: "deal", id: first.id });
      else if (lib.examples.length) {
        setSelected({ kind: "example", name: lib.examples[0].name });
        showExample(lib.examples[0]);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [enabled, listed, libraryOn, current, showExample]);

  // Load the selected saved deal and its stored actuals
  useEffect(() => {
    if (selected?.kind !== "deal") return;
    let cancelled = false;
    const path = { params: { path: { deal_id: selected.id } } };
    Promise.all([api.GET("/api/deals/{deal_id}", path), api.GET("/api/deals/{deal_id}/actuals", path)])
      .then(([deal, saved]) => {
        if (cancelled) return;
        if (!deal.data) return setState({ status: "error", error: describe((deal.error as { detail?: unknown })?.detail, t("loadFailed")) });
        const d = deal.data;
        const money = { currency: d.inputs.currency ?? "USD", unit: d.inputs.unit ?? "millions" };
        const hold = d.inputs.hold ?? 5;
        setStored({ ref: selected, name: d.name, inputs: d.inputs, settings: d.settings, money, hold });
        let draft = saved.data?.actuals ? draftFrom(saved.data.actuals) : blankDraft(money, hold);
        // A deal whose unit changed since: the same actuals, counted in its new unit
        if (draft.money.unit !== money.unit && draft.money.currency === money.currency) {
          draft = scaleDraft(draft, unitFactor(draft.money.unit, money.unit), money);
        }
        setActualsState(draft.years.length > hold ? resizeYears(draft, hold) : draft);
        setSaveState("saved");
      })
      .catch(() => !cancelled && setState({ status: "error", error: e("apiUnreachable") }));
    return () => {
      cancelled = true;
    };
  }, [selected, t, e]);

  // The open deal is the plan as it is on screen now (autosave stores the same thing)
  const plan: Plan | null = useMemo(() => {
    if (!stored) return null;
    const live = stored.ref.kind === "deal" && current?.id === stored.ref.id;
    const inputs = live ? apiInputs(openInputs) : stored.inputs!;
    const settings = live ? overrides : (stored.settings ?? {});
    return { ...stored, inputs, settings, hold: inputs.hold ?? stored.hold, money: live ? { currency: openInputs.currency, unit: openInputs.unit } : stored.money };
  }, [stored, current, openInputs, overrides]);

  const select = useCallback(
    (ref: PlanRef) => {
      if (sameRef(selected, ref)) return;
      setSelected(ref);
      setStored(null);
      setActualsState(null);
      setState({ status: "loading" });
      const ex = ref.kind === "example" ? library.examples.find((x) => x.name === ref.name) : undefined;
      if (ex) showExample(ex);
    },
    [selected, library, showExample],
  );

  const save = useCallback(async (id: string, draft: ActualsDraft) => {
    setSaveState("saving");
    try {
      const { data } = await api.PUT("/api/deals/{deal_id}/actuals", { params: { path: { deal_id: id } }, body: apiActuals(draft) });
      setSaveState(data ? "saved" : "error");
    } catch {
      setSaveState("error");
    }
  }, []);

  const setActuals = useCallback(
    (next: ActualsDraft) => {
      setActualsState(next);
      if (selected?.kind !== "deal") return;
      setSaveState("unsaved");
      if (saveTimer.current) clearTimeout(saveTimer.current);
      const id = selected.id;
      saveTimer.current = setTimeout(() => void save(id, next), SAVE_DELAY_MS);
    },
    [selected, save],
  );

  // Run whenever the plan or the actuals change and there is something to compare
  const body = useMemo(() => {
    if (!visible || !plan || !actuals || !hasFigures(actuals)) return null;
    const trimmed = actuals.years.length > plan.hold ? resizeYears(actuals, plan.hold) : actuals;
    return { plan: plan.inputs, settings: plan.settings, actuals: apiActuals(trimmed), n: PATHS, histogram_bins: 60 };
  }, [visible, plan, actuals]);

  useEffect(() => {
    if (!body) return;
    const ctrl = new AbortController();
    const id = setTimeout(async () => {
      setState((s) => ({ ...s, status: "running" }));
      try {
        const { data, error } = await api.POST("/api/backtesting/plan-vs-actual", { body, signal: ctrl.signal });
        if (ctrl.signal.aborted) return;
        setState(data ? { status: "ok", result: data } : { status: "error", error: describe((error as { detail?: unknown })?.detail, t("runFailed")) });
      } catch (err) {
        if (!ctrl.signal.aborted && (err as Error)?.name !== "AbortError") setState({ status: "error", error: e("apiUnreachable") });
      }
    }, RUN_DELAY_MS);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [body, t, e]);

  const activate = useCallback(() => {
    setEnabled(true);
    setVisible((v) => v + 1);
    return () => setVisible((v) => v - 1);
  }, []);
  const value = useMemo<BacktestContext>(
    () => ({
      deals,
      examples: library.examples,
      libraryEnabled: library.enabled,
      ready: listed,
      selected,
      select,
      plan,
      actuals,
      setActuals,
      saveState,
      // Nothing to compare yet (no figures entered): idle, and no result from before
      status: !listed && enabled ? "loading" : !body && visible && plan && actuals ? "idle" : state.status,
      result: body || !visible ? state.result : undefined,
      error: state.error,
      activate,
    }),
    [deals, library, selected, select, plan, actuals, setActuals, saveState, listed, enabled, state, activate, body, visible],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useBacktest(): BacktestContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useBacktest must be used inside BacktestProvider");
  return ctx;
}

export const BACKTEST_PATHS = PATHS;
