"use client";

import { useTranslations } from "next-intl";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { useSettings } from "@/components/settings/SettingsProvider";
import { api, type Schemas } from "@/lib/api/client";

export type PreloadedDeal = Schemas["PreloadedDeal"];
export type BacktestResult = Schemas["BacktestResponse"];

type BacktestContext = {
  deals: PreloadedDeal[] | null;
  selected: string | null;
  select: (name: string) => void;
  deal: PreloadedDeal | null;
  status: "idle" | "running" | "ok" | "error";
  result?: BacktestResult;
  error?: string;
  /** Called by Backtest screens so nothing is fetched until the mode is opened */
  activate: () => void;
};

const Ctx = createContext<BacktestContext | null>(null);
const PATHS = 30000;

/** A refusal (a usage limit, a busy server) is already a sentence; validation errors are listed. */
function backtestError(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  return JSON.stringify(detail ?? fallback);
}

/** "Burger King (3G Capital, 2010)" -> name, sponsor, year */
export function splitDealName(full: string) {
  const m = full.match(/^(.*?)\s*\((.*),\s*(\d{4})\)$/);
  return m ? { name: m[1], sponsor: m[2], year: m[3] } : { name: full, sponsor: "", year: "" };
}

export function BacktestProvider({ children }: { children: React.ReactNode }) {
  const t = useTranslations("backtest");
  const e = useTranslations("errors");
  const { overrides, defaults } = useSettings();
  const [enabled, setEnabled] = useState(false);
  const [deals, setDeals] = useState<PreloadedDeal[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [state, setState] = useState<{ status: BacktestContext["status"]; result?: BacktestResult; error?: string; key?: string }>({ status: "idle" });

  useEffect(() => {
    if (!enabled || deals) return;
    let cancelled = false;
    api
      .GET("/api/backtesting/deals")
      .then(({ data }) => {
        if (cancelled || !data) return;
        setDeals(data);
        setSelected((s) => s ?? data[0]?.name ?? null);
      })
      .catch(() => !cancelled && setState({ status: "error", error: t("loadFailed") }));
    return () => {
      cancelled = true;
    };
  }, [enabled, deals, t]);

  const deal = useMemo(() => deals?.find((d) => d.name === selected) ?? null, [deals, selected]);

  useEffect(() => {
    if (!deal || !defaults) return;
    const ctrl = new AbortController();
    const entry = deal.entry;
    const body = {
      entry: {
        entry_ebitda: entry.entry_ebitda,
        entry_multiple: entry.entry_multiple,
        exit_multiple: entry.exit_multiple,
        holding_period: Math.round(entry.holding_period),
        debt_pct: entry.debt_pct,
        senior_pct: entry.senior_pct,
        base_rate: entry.base_rate,
        mezz_spread: entry.mezz_spread,
        revenue_growth: entry.revenue_growth,
        gross_margin: entry.gross_margin,
        opex_pct: entry.opex_pct,
        da_pct: entry.da_pct,
        tax_rate: entry.tax_rate,
        capex_pct: entry.capex_pct,
        nwc_pct: entry.nwc_pct,
      },
      actual: deal.actual as Schemas["BacktestActuals"],
      actual_exit: deal.actual_exit as Schemas["BacktestActualExit"],
      settings: overrides,
      n: PATHS,
      histogram_bins: 60,
      // The example deals are US dollar millions; the API says so with each one
      money: deal.money,
    };
    const id = setTimeout(async () => {
      setState((s) => ({ ...s, status: "running" }));
      try {
        const { data, error } = await api.POST("/api/backtesting/run", { body, signal: ctrl.signal });
        if (ctrl.signal.aborted) return;
        setState(data ? { status: "ok", result: data } : { status: "error", error: backtestError((error as { detail?: unknown })?.detail, t("runFailed")) });
      } catch (err) {
        if (!ctrl.signal.aborted && (err as Error)?.name !== "AbortError") setState({ status: "error", error: e("apiUnreachable") });
      }
    }, 150);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [deal, overrides, defaults, t, e]);

  const activate = useCallback(() => setEnabled(true), []);
  const value = useMemo(
    () => ({ deals, selected, select: setSelected, deal, status: state.status, result: state.result, error: state.error, activate }),
    [deals, selected, deal, state, activate],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useBacktest(): BacktestContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useBacktest must be used inside BacktestProvider");
  return ctx;
}

export const BACKTEST_PATHS = PATHS;
