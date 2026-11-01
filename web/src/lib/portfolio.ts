"use client";

import { api, type Schemas } from "@/lib/api/client";
import type { Money, MoneyUnit } from "@/lib/money";

/**
 * Portfolio tracking (PLAN.md 7.4): deals marked as owned, their quarterly results against the plan, drift
 * alerts, the reforecast job and the dashboard in a reporting currency. Every figure is the API's; the browser
 * sends what the person typed and nothing worked out.
 */
export type Holding = Schemas["HoldingOut"];
export type Portfolio = Schemas["PortfolioResponse"];
export type PortfolioHolding = Schemas["PortfolioHolding"];
export type Drift = Schemas["Drift"];
export type DriftAlert = Schemas["DriftAlert"];
export type Reforecast = Schemas["ReforecastResult"];
export type QuarterRow = Schemas["QuarterIn"];

/** A quarter's figures, plan vs actual's lines (PLAN.md 2.7). i18n-keys: portfolio.line* */
export const QUARTER_LINES = [
  { key: "revenue", labelKey: "lineRevenue" },
  { key: "ebitda", labelKey: "lineEbitda" },
  { key: "net_income", labelKey: "lineNetIncome" },
  { key: "fcf", labelKey: "lineFcf" },
  { key: "total_debt", labelKey: "lineTotalDebt" },
] as const;
export type QuarterLine = (typeof QUARTER_LINES)[number]["key"];

/** The dashboard's money columns, in the order shown. i18n-keys: portfolio.col_* */
export const BOARD_COLUMNS = [
  "entry_ev", "entry_equity", "plan_exit_equity", "reforecast_exit_equity", "ltm_revenue", "ltm_ebitda", "latest_debt",
] as const;
export type BoardColumn = (typeof BOARD_COLUMNS)[number];

/** Codes the screens word (`portfolio.problem_<code>`); anything else is `problem_failed`. */
const PROBLEMS = ["no_ltm", "ebitda_not_positive", "beyond_hold", "other_currency", "portfolio_full", "duplicate_quarter",
  "limit", "forbidden"];

export function problemKey(code: string): string {
  return `problem_${PROBLEMS.includes(code) ? code : "failed"}`;
}

function codeOf(error: unknown, status: number): string {
  if (status === 429) return "limit";
  if (status === 403) return "forbidden";
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  if (detail && typeof detail === "object" && typeof (detail as { code?: unknown }).code === "string") {
    return (detail as { code: string }).code;
  }
  return "failed";
}

const at = (dealId: string) => ({ params: { path: { deal_id: dealId } } });

export type Answer = { holding?: Holding; problem?: string };

async function answered(call: Promise<{ data?: Holding; error?: unknown; response: Response }>): Promise<Answer> { // text-ok
  const { data, error, response } = await call;
  return data ? { holding: data } : { problem: codeOf(error, response.status) };
}

export function getHolding(dealId: string): Promise<Answer> {
  return answered(api.GET("/api/deals/{deal_id}/portfolio", at(dealId)));
}

export function setHolding(dealId: string, since: string | null, tolerance: number): Promise<Answer> {
  return answered(api.PUT("/api/deals/{deal_id}/portfolio", { ...at(dealId), body: { since, tolerance } }));
}

export function removeHolding(dealId: string): Promise<Answer> {
  return answered(api.DELETE("/api/deals/{deal_id}/portfolio", at(dealId)));
}

export function saveQuarters(dealId: string, money: Money, quarters: QuarterRow[]): Promise<Answer> {
  if (quarters.length === 0) return answered(api.DELETE("/api/deals/{deal_id}/quarters", at(dealId)));
  return answered(api.PUT("/api/deals/{deal_id}/quarters", {
    ...at(dealId), body: { currency: money.currency, unit: money.unit, quarters },
  }));
}

const FIRST_POLL_MS = 500;
const MAX_POLL_MS = 3000;

function sleep(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(new DOMException("aborted", "AbortError"));
    }, { once: true });
  });
}

/** Reforecast the deal from its last four quarters: the job runs the model; answers the holding as it then is. */
export async function reforecast(dealId: string, signal: AbortSignal, onStage?: (stage: string | null) => void)
  : Promise<Answer> {
  const started = await api.POST("/api/deals/{deal_id}/reforecast", { ...at(dealId), signal });
  if (!started.data) return { problem: codeOf(started.error, started.response.status) };
  const id = started.data.id;
  let delay = FIRST_POLL_MS;
  for (;;) {
    await sleep(delay, signal);
    delay = Math.min(MAX_POLL_MS, delay * 1.4);
    const { data, error, response } = await api.GET("/api/deals/{deal_id}/reforecast/jobs/{job_id}", {
      params: { path: { deal_id: dealId, job_id: id } }, signal,
    });
    if (!data) {
      if (response.status >= 500) continue;
      return { problem: codeOf(error, response.status) };
    }
    if (data.status === "succeeded") return getHolding(dealId);
    if (data.status === "failed" || data.status === "cancelled") return { problem: "failed" };
    onStage?.(data.stage ?? null);
  }
}

export async function getPortfolio(currency: string | null, unit: MoneyUnit, on: string | null)
  : Promise<{ portfolio?: Portfolio; problem?: string }> {
  const query: { currency?: string; unit: MoneyUnit; on?: string } = { unit };
  if (currency) query.currency = currency;
  if (on) query.on = on;
  const { data, error, response } = await api.GET("/api/portfolio", { params: { query } });
  return data ? { portfolio: data } : { problem: codeOf(error, response.status) };
}

/** The editor's rows: one per quarter, oldest first, each line a number or null (not known). */
export type QuarterDraft = { year: number; quarter: number; values: Record<QuarterLine, number | null> };

export function draftsOf(holding: Holding | null): QuarterDraft[] {
  return (holding?.quarters?.quarters ?? []).map((q) => ({
    year: q.year, quarter: q.quarter,
    values: Object.fromEntries(QUARTER_LINES.map((l) => [l.key, q[l.key] ?? null])) as Record<QuarterLine, number | null>,
  }));
}

/** The quarter after the last one (year 1's first when there is none), within the hold; null past it. */
export function nextQuarter(drafts: QuarterDraft[], hold: number): { year: number; quarter: number } | null {
  const last = drafts[drafts.length - 1];
  const next = !last ? { year: 1, quarter: 1 } : last.quarter === 4
    ? { year: last.year + 1, quarter: 1 } : { year: last.year, quarter: last.quarter + 1 };
  return next.year > hold ? null : next;
}

export function rowsOf(drafts: QuarterDraft[]): QuarterRow[] {
  return drafts.map((d) => {
    const row: QuarterRow = { year: d.year, quarter: d.quarter };
    for (const l of QUARTER_LINES) {
      const v = d.values[l.key];
      if (v !== null && Number.isFinite(v)) row[l.key] = v;
    }
    return row;
  });
}
