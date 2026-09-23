"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { api, type Schemas } from "@/lib/api/client";
import { DEFAULT_MONEY, type Money, unitFactor } from "@/lib/money";

type Series = Record<string, number[]>;
export type ForecastRun = Schemas["ForecastRunResponse"];
export type Metrics = Schemas["HistoricalMetrics"];

const SIM_PATHS = 20000;
/** Assumptions that are money amounts (the rest are rates, shares and days) */
const MONEY_ASSUMPTIONS = ["other_inc", "divs", "buybacks", "ltd_chg", "min_cash"];

type Source = { kind: "sample" } | { kind: "edgar"; ticker: string; company: string; years: number[]; warnings: string[] };

type ForecastContext = {
  ready: boolean;
  nHist: number;
  nFwd: number;
  history: Series;
  assumptions: Series;
  setHistory: (key: string, year: number, value: number) => void;
  setAssumption: (key: string, year: number, value: number) => void;
  /** Same value for every forecast year */
  fillAssumption: (key: string, value: number) => void;
  /** Replace assumptions with values derived from the historicals */
  reseed: () => void;
  resetSample: () => void;
  source: Source;
  /** The company's reporting currency and the unit its figures are in */
  money: Money;
  /** A new unit keeps the company's size: every figure is converted */
  setMoney: (money: Money) => void;
  fetchEdgar: (ticker: string) => Promise<void>;
  edgar: { status: "idle" | "loading" | "error"; error?: string };
  metrics: Metrics[] | null;
  seeded: Record<string, number> | null;
  status: "idle" | "running" | "ok" | "error";
  result?: ForecastRun;
  error?: string;
  simPaths: number;
  activate: () => void;
};

const Ctx = createContext<ForecastContext | null>(null);

function spread(seeded: Record<string, number>, n: number): Series {
  return Object.fromEntries(Object.entries(seeded).map(([k, v]) => [k, Array(n).fill(v)]));
}

function detail(err: unknown): string {
  const d = (err as { detail?: unknown } | undefined)?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x: { loc?: unknown[]; msg?: string }) => `${x.loc?.slice(1).join(".")}: ${x.msg}`).join("; ");
  return "The forecast couldn't run with these inputs.";
}

export function ForecastProvider({ children }: { children: React.ReactNode }) {
  const [enabled, setEnabled] = useState(false);
  const [defaults, setDefaults] = useState<Schemas["ForecastDefaultsResponse"] | null>(null);
  const [history, setHistoryState] = useState<Series>({});
  const [assumptions, setAssumptions] = useState<Series>({});
  const [source, setSource] = useState<Source>({ kind: "sample" });
  const [money, setMoneyState] = useState<Money>(DEFAULT_MONEY);
  const [edgar, setEdgar] = useState<ForecastContext["edgar"]>({ status: "idle" });
  const [seedInfo, setSeedInfo] = useState<{ metrics: Metrics[]; seeded: Record<string, number> } | null>(null);
  const [run, setRun] = useState<{ status: ForecastContext["status"]; result?: ForecastRun; error?: string }>({ status: "idle" });

  const nHist = defaults?.n_hist ?? 3;
  const nFwd = defaults?.n_fwd ?? 5;

  useEffect(() => {
    if (!enabled || defaults) return;
    let cancelled = false;
    api
      .GET("/api/forecasting/defaults")
      .then(({ data }) => {
        if (cancelled || !data) return;
        setDefaults(data);
        setMoneyState(data.money);
        setHistoryState(data.history);
        setAssumptions(spread(data.seeded_assumptions, data.n_fwd));
      })
      .catch(() => !cancelled && setRun({ status: "error", error: "Can't load forecasting defaults from the API." }));
    return () => {
      cancelled = true;
    };
  }, [enabled, defaults]);

  // Historical ratios (and suggested assumptions) for the current historicals
  useEffect(() => {
    if (!defaults) return;
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      api
        .POST("/api/forecasting/seed", { body: { history, money }, signal: ctrl.signal })
        .then(({ data }) => data && setSeedInfo({ metrics: data.historical_metrics, seeded: data.seeded_assumptions }))
        .catch(() => {});
    }, 300);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [history, money, defaults]);

  // The forecast reruns automatically, simulation included
  useEffect(() => {
    if (!defaults || !Object.keys(assumptions).length) return;
    const ctrl = new AbortController();
    const id = setTimeout(async () => {
      setRun((r) => ({ ...r, status: "running" }));
      try {
        const { data, error } = await api.POST("/api/forecasting/run", {
          body: { history, assumptions, simulate: true, n_sim: SIM_PATHS, money },
          signal: ctrl.signal,
        });
        if (ctrl.signal.aborted) return;
        setRun(data ? { status: "ok", result: data } : (r) => ({ ...r, status: "error", error: detail(error) }));
      } catch (e) {
        if (!ctrl.signal.aborted && (e as Error)?.name !== "AbortError") setRun((r) => ({ ...r, status: "error", error: "Can't reach the API." }));
      }
    }, 400);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [history, assumptions, money, defaults]);

  const setHistory = useCallback((key: string, year: number, value: number) => {
    setHistoryState((h) => ({ ...h, [key]: (h[key] ?? []).map((v, i) => (i === year ? value : v)) }));
  }, []);
  const setAssumption = useCallback((key: string, year: number, value: number) => {
    setAssumptions((a) => ({ ...a, [key]: (a[key] ?? []).map((v, i) => (i === year ? value : v)) }));
  }, []);
  const fillAssumption = useCallback((key: string, value: number) => {
    setAssumptions((a) => ({ ...a, [key]: (a[key] ?? []).map(() => value) }));
  }, []);
  const reseed = useCallback(() => {
    if (seedInfo) setAssumptions(spread(seedInfo.seeded, nFwd));
  }, [seedInfo, nFwd]);
  const setMoney = useCallback(
    (next: Money) => {
      const k = unitFactor(money.unit, next.unit);
      if (k !== 1) {
        const scale = (s: Series, keys?: string[]) =>
          Object.fromEntries(Object.entries(s).map(([key, v]) => [key, !keys || keys.includes(key) ? v.map((x) => x * k) : v]));
        setHistoryState((h) => scale(h));
        setAssumptions((a) => scale(a, MONEY_ASSUMPTIONS));
      }
      setMoneyState(next);
    },
    [money.unit],
  );
  const resetSample = useCallback(() => {
    if (!defaults) return;
    setMoneyState(defaults.money);
    setHistoryState(defaults.history);
    setAssumptions(spread(defaults.seeded_assumptions, defaults.n_fwd));
    setSource({ kind: "sample" });
    setEdgar({ status: "idle" });
  }, [defaults]);

  const fetchEdgar = useCallback(
    async (ticker: string) => {
      const t = ticker.trim().toUpperCase();
      if (!t) return;
      setEdgar({ status: "loading" });
      try {
        const { data, error, response } = await api.GET("/api/edgar/{ticker}", { params: { path: { ticker: t } } });
        if (!data) {
          setEdgar({ status: "error", error: response.status === 503 ? "EDGAR autofill isn't available on this server." : detail(error) });
          return;
        }
        // Fields EDGAR didn't return keep their current values, in EDGAR's unit
        // SEC filings come in US dollar millions (the API says so)
        const scaled = Object.fromEntries(Object.entries(history).map(([k, v]) => [k, v.map((x) => x * unitFactor(money.unit, data.money.unit))]));
        const merged = { ...scaled, ...Object.fromEntries(Object.entries(data.history).filter(([, v]) => v.length === nHist)) };
        setMoneyState(data.money);
        setHistoryState(merged);
        setSource({ kind: "edgar", ticker: data.ticker, company: data.company_name, years: data.years, warnings: data.warnings });
        const seeded = await api.POST("/api/forecasting/seed", { body: { history: merged, money: data.money } });
        if (seeded.data) setAssumptions(spread(seeded.data.seeded_assumptions, nFwd));
        setEdgar({ status: "idle" });
      } catch {
        setEdgar({ status: "error", error: "SEC EDGAR request failed. Check the connection and try again." });
      }
    },
    [history, money.unit, nHist, nFwd],
  );

  const activate = useCallback(() => setEnabled(true), []);

  const value = useMemo(
    () => ({
      ready: !!defaults,
      nHist,
      nFwd,
      history,
      assumptions,
      setHistory,
      setAssumption,
      fillAssumption,
      reseed,
      resetSample,
      source,
      money,
      setMoney,
      fetchEdgar,
      edgar,
      metrics: seedInfo?.metrics ?? null,
      seeded: seedInfo?.seeded ?? null,
      status: run.status,
      result: run.result,
      error: run.error,
      simPaths: SIM_PATHS,
      activate,
    }),
    [defaults, nHist, nFwd, history, assumptions, setHistory, setAssumption, fillAssumption, reseed, resetSample, source, money, setMoney, fetchEdgar, edgar, seedInfo, run, activate],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useForecast(): ForecastContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useForecast must be used inside ForecastProvider");
  return ctx;
}
