"use client";

import { useTranslations } from "next-intl";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { api, type Schemas } from "@/lib/api/client";
import { type Fiscal } from "@/lib/fiscal";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";
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
  /** The company's fiscal year-end and its latest historical fiscal year (PLAN.md 2.3a); labels only */
  fiscal: Fiscal;
  setFiscal: (fiscal: Fiscal) => void;
  /** Column labels: history oldest first ("FY2023/24" ... or "LTM-2" ... "LTM"), then forecast years */
  histLabels: string[];
  fwdLabels: string[];
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

/** A company with no fiscal year set: history labelled LTM, forecast F+1 ... */
const NO_FISCAL_YEAR: Fiscal = { endMonth: 12, year: null };

function spread(seeded: Record<string, number>, n: number): Series {
  return Object.fromEntries(Object.entries(seeded).map(([k, v]) => [k, Array(n).fill(v)]));
}

function detail(err: unknown, fallback: string): string {
  const d = (err as { detail?: unknown } | undefined)?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map((x: { loc?: unknown[]; msg?: string }) => `${x.loc?.slice(1).join(".")}: ${x.msg}`).join("; ");
  return fallback;
}

type Standard = Schemas["HistoryRequest"]["accounting_standard"];

/**
 * A request body with the company's accounting standard (PLAN.md 2.6), only when a filing gave one:
 * an API from before it (a deploy rolling out, a rollback) refuses fields it doesn't know. The
 * generated types list every defaulted field as present; the API fills in the one left out.
 */
function withStandard<T extends object>(body: T, standard: Standard): T & { accounting_standard: Standard } {
  return (standard ? { ...body, accounting_standard: standard } : body) as T & { accounting_standard: Standard };
}

export function ForecastProvider({ children }: { children: React.ReactNode }) {
  const t = useTranslations("forecast");
  const e = useTranslations("errors");
  const labels = useFiscalLabels();
  const [enabled, setEnabled] = useState(false);
  const [defaults, setDefaults] = useState<Schemas["ForecastDefaultsResponse"] | null>(null);
  const [history, setHistoryState] = useState<Series>({});
  const [assumptions, setAssumptions] = useState<Series>({});
  const [source, setSource] = useState<Source>({ kind: "sample" });
  const [fiscal, setFiscal] = useState<Fiscal>(NO_FISCAL_YEAR);
  const [money, setMoneyState] = useState<Money>(DEFAULT_MONEY);
  // The standard the company reports under (PLAN.md 2.6): a filing says, the sample says nothing
  const [standard, setStandard] = useState<Standard>("");
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
      .catch(() => !cancelled && setRun({ status: "error", error: t("defaultsFailed") }));
    return () => {
      cancelled = true;
    };
  }, [enabled, defaults, t]);

  // Historical ratios (and suggested assumptions) for the current historicals
  useEffect(() => {
    if (!defaults) return;
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      api
        .POST("/api/forecasting/seed", { body: withStandard({ history, money }, standard), signal: ctrl.signal })
        .then(({ data }) => data && setSeedInfo({ metrics: data.historical_metrics, seeded: data.seeded_assumptions }))
        .catch(() => {});
    }, 300);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [history, money, standard, defaults]);

  // The forecast reruns automatically, simulation included
  useEffect(() => {
    if (!defaults || !Object.keys(assumptions).length) return;
    const ctrl = new AbortController();
    const id = setTimeout(async () => {
      setRun((r) => ({ ...r, status: "running" }));
      try {
        const { data, error } = await api.POST("/api/forecasting/run", {
          body: withStandard({ history, assumptions, simulate: true, n_sim: SIM_PATHS, money }, standard),
          signal: ctrl.signal,
        });
        if (ctrl.signal.aborted) return;
        setRun(data ? { status: "ok", result: data } : (r) => ({ ...r, status: "error", error: detail(error, t("runFailed")) }));
      } catch (err) {
        if (!ctrl.signal.aborted && (err as Error)?.name !== "AbortError") setRun((r) => ({ ...r, status: "error", error: e("apiUnreachable") }));
      }
    }, 400);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [history, assumptions, money, standard, defaults, t, e]);

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
    setFiscal(NO_FISCAL_YEAR);
    setEdgar({ status: "idle" });
  }, [defaults]);

  const fetchEdgar = useCallback(
    async (ticker: string) => {
      const code = ticker.trim().toUpperCase();
      if (!code) return;
      setEdgar({ status: "loading" });
      try {
        const { data, error, response } = await api.GET("/api/edgar/{ticker}", { params: { path: { ticker: code } } });
        if (!data) {
          setEdgar({ status: "error", error: response.status === 503 ? t("edgarUnavailable") : detail(error, t("runFailed")) });
          return;
        }
        // Fields EDGAR didn't return keep their current values, in EDGAR's unit
        // SEC filings come in US dollar millions (the API says so)
        const scaled = Object.fromEntries(Object.entries(history).map(([k, v]) => [k, v.map((x) => x * unitFactor(money.unit, data.money.unit))]));
        const merged = { ...scaled, ...Object.fromEntries(Object.entries(data.history).filter(([, v]) => v.length === nHist)) };
        setMoneyState(data.money);
        setStandard(data.accounting_standard);
        setHistoryState(merged);
        setSource({ kind: "edgar", ticker: data.ticker, company: data.company_name, years: data.years, warnings: data.warnings });
        // Filings name each fiscal year by the year it ends in (api/routers/integrations.py)
        setFiscal({ endMonth: data.fiscal_year_end_month ?? 12, year: data.years.at(-1) ?? null });
        const seeded = await api.POST("/api/forecasting/seed", {
          body: withStandard({ history: merged, money: data.money }, data.accounting_standard),
        });
        if (seeded.data) setAssumptions(spread(seeded.data.seeded_assumptions, nFwd));
        setEdgar({ status: "idle" });
      } catch {
        setEdgar({ status: "error", error: t("edgarFailed") });
      }
    },
    [history, money.unit, nHist, nFwd, t],
  );

  const activate = useCallback(() => setEnabled(true), []);

  const histLabels = useMemo(() => labels.history(nHist, fiscal), [labels, fiscal, nHist]);
  const fwdLabels = useMemo(() => labels.forward(nFwd, fiscal), [labels, fiscal, nFwd]);

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
      fiscal,
      setFiscal,
      histLabels,
      fwdLabels,
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
    [defaults, nHist, nFwd, history, assumptions, setHistory, setAssumption, fillAssumption, reseed, resetSample, source, money, setMoney, fiscal, histLabels, fwdLabels, fetchEdgar, edgar, seedInfo, run, activate],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useForecast(): ForecastContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useForecast must be used inside ForecastProvider");
  return ctx;
}
