"use client";

import { useEffect, useState, type ReactNode } from "react";

import { DataTable, type Row } from "@/components/charts/DataTable";
import { LineChart } from "@/components/charts/LineChart";
import { CellInput } from "@/components/ui/CellInput";
import { EmptyState, LoadingTiles, Notice, RailGroup, Screen, SecondaryButton } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { ASSUMPTION_GROUPS, HISTORY_GROUPS } from "@/lib/forecast";
import { fmtInput, fmtMoney, fmtRate, isNum } from "@/lib/format";

import { type ForecastRun, useForecast } from "./ForecastProvider";

const FWD = (n: number) => Array.from({ length: n }, (_, i) => `F+${i + 1}`);

function Rail() {
  const { source, fetchEdgar, edgar, resetSample, metrics, status, result } = useForecast();
  const [ticker, setTicker] = useState("");
  const latest = metrics?.at(-1);
  return (
    <>
      <div className="flex items-center justify-between gap-2 border-b border-line px-3.5 py-2">
        <span className="type-control">{source.kind === "edgar" ? source.ticker : "Sample company"}</span>
        <span role="status" className={`font-mono text-[10px] ${status === "error" ? "text-loss" : status === "running" ? "text-attention" : "text-dim"}`}>
          {status === "running" ? "Updating" : status === "error" ? "Error" : result ? "Up to date" : "Loading"}
        </span>
      </div>
      <RailGroup title="Autofill from SEC EDGAR">
        <form
          className="grid grid-cols-[1fr_auto] gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void fetchEdgar(ticker);
          }}
        >
          <label className="sr-only" htmlFor="edgar-ticker">
            Ticker
          </label>
          <input
            id="edgar-ticker"
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            placeholder="Ticker, e.g. DELL"
            autoComplete="off"
            className="border border-line bg-field px-2 py-1 font-mono text-[11.5px] text-ink uppercase outline-none placeholder:text-dim placeholder:normal-case focus:border-accent"
          />
          <button type="submit" disabled={edgar.status === "loading" || !ticker.trim()} className="type-action-secondary px-2.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)] disabled:opacity-50">
            {edgar.status === "loading" ? "Fetching" : "Fetch"}
          </button>
        </form>
        {edgar.status === "error" && <p className="font-mono text-[10px] text-loss">{edgar.error}</p>}
        {source.kind === "edgar" && (
          <div className="grid gap-1 pt-1">
            <p className="type-input-label">{source.company}</p>
            <p className="font-mono text-[10px] text-muted">Fiscal years {source.years.join(", ")}</p>
            {source.warnings.map((w) => (
              <p key={w} className="font-mono text-[10px] text-attention">
                {w}
              </p>
            ))}
          </div>
        )}
        <div className="pt-1.5">
          <SecondaryButton onClick={resetSample}>Reset to sample</SecondaryButton>
        </div>
      </RailGroup>
      {latest && (
        <RailGroup title="Latest year ratios">
          <dl className="grid grid-cols-[1fr_auto] gap-x-2 gap-y-1">
            {(
              [
                ["Revenue", fmtMoney(latest.revenue)],
                ["Revenue growth", fmtRate(latest.revenue_growth)],
                ["Gross margin", fmtRate(latest.gross_margin)],
                ["R&D, % sales", fmtRate(latest.rd_pct)],
                ["SG&A, % sales", fmtRate(latest.sga_pct)],
                ["EBITDA margin", fmtRate(latest.ebitda_margin)],
                ["Adj. EBITDA margin", fmtRate(latest.adj_ebitda_margin)],
              ] as const
            ).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="type-input-label">{k}</dt>
                <dd className="text-right font-mono text-[11.5px] text-ink">{v}</dd>
              </div>
            ))}
          </dl>
        </RailGroup>
      )}
    </>
  );
}

function ForecastScreen({ children, needsResult = true }: { children: ReactNode; needsResult?: boolean }) {
  const { activate, ready, result, status, error } = useForecast();
  useEffect(() => activate(), [activate]);
  const bar =
    status === "error" ? (
      <Notice tone="loss" title="Forecast didn't run" role="alert">
        {error}
      </Notice>
    ) : undefined;
  const body = !ready ? (
    status === "error" ? <EmptyState title="No forecast yet">Check that the API is running, then reload.</EmptyState> : <LoadingTiles />
  ) : needsResult && !result ? (
    <LoadingTiles />
  ) : (
    children
  );
  return (
    <Screen rail={<Rail />} bar={bar}>
      {body}
    </Screen>
  );
}

function yearLabels(n: number, edgarYears?: number[]) {
  if (edgarYears?.length === n) return edgarYears.map((y) => `FY${y}`);
  return Array.from({ length: n }, (_, i) => (i === n - 1 ? "LTM" : `LTM-${n - 1 - i}`));
}

export function HistoricalsStep() {
  return (
    <ForecastScreen needsResult={false}>
      <Historicals />
    </ForecastScreen>
  );
}

function Historicals() {
  const { history, setHistory, nHist, source } = useForecast();
  const cols = yearLabels(nHist, source.kind === "edgar" ? source.years : undefined);
  return (
    <Tiles>
      {HISTORY_GROUPS.map((g) => (
        <Tile key={g.title} span={g.rows.length > 10 ? 6 : 6} title={g.title}>
          <EditableGrid rows={g.rows} columns={cols} values={history} onCommit={setHistory} />
        </Tile>
      ))}
    </Tiles>
  );
}

function EditableGrid({
  rows,
  columns,
  values,
  onCommit,
  extra,
}: {
  rows: { key: string; label: string }[];
  columns: string[];
  values: Record<string, number[]>;
  onCommit: (key: string, col: number, value: number) => void;
  extra?: (key: string) => ReactNode;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th />
            {columns.map((c) => (
              <th key={c} scope="col" className="px-1 py-1 text-right font-mono text-[10.5px] font-normal text-muted">
                {c}
              </th>
            ))}
            {extra && <th />}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key}>
              <th scope="row" className="type-input-label py-0.5 pr-2 text-left text-[9px] font-normal whitespace-nowrap">
                {r.label}
              </th>
              {columns.map((c, i) => (
                <td key={c} className="px-1 py-0.5">
                  <CellInput label={`${r.label}, ${c}`} value={values[r.key]?.[i] ?? NaN} onCommit={(v) => onCommit(r.key, i, v)} />
                </td>
              ))}
              {extra && <td className="py-0.5 pl-2 whitespace-nowrap">{extra(r.key)}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function AssumptionsStep() {
  return (
    <ForecastScreen needsResult={false}>
      <Assumptions />
    </ForecastScreen>
  );
}

function Assumptions() {
  const { assumptions, setAssumption, fillAssumption, nFwd, seeded, reseed } = useForecast();
  const cols = FWD(nFwd);
  return (
    <Tiles>
      <Notice
        tone="info"
        title="Suggested values"
        className="col-span-12"
        actions={<SecondaryButton onClick={reseed} disabled={!seeded}>Use all suggestions</SecondaryButton>}
      >
        Suggestions come from the historicals. Click one to use it for every year, or edit any year directly.
      </Notice>
      {ASSUMPTION_GROUPS.map((g) => (
        <Tile key={g.title} span={6} title={g.title}>
          <EditableGrid
            rows={g.rows}
            columns={cols}
            values={assumptions}
            onCommit={setAssumption}
            extra={(key) =>
              seeded && isNum(seeded[key]) ? (
                <button
                  type="button"
                  onClick={() => fillAssumption(key, seeded[key])}
                  title="Use for every year"
                  className="font-mono text-[10px] text-accent hover:underline"
                >
                  {fmtInput(seeded[key], 1)}
                </button>
              ) : null
            }
          />
        </Tile>
      ))}
    </Tiles>
  );
}

function statementRows(res: ForecastRun, keys: [string, string, Row["kind"]?, boolean?][]): Row[] {
  const ltm = res.ltm as Record<string, unknown>;
  return keys.map(([key, label, kind, total]) => ({
    label,
    kind,
    total,
    values: [typeof ltm[key] === "number" ? (ltm[key] as number) : null, ...res.years.map((y) => (y as Record<string, unknown>)[key] as number | null)],
  }));
}

export function StatementsStep() {
  return (
    <ForecastScreen>
      <Statements />
    </ForecastScreen>
  );
}

function Statements() {
  const { result: res, nFwd } = useForecast();
  if (!res) return null;
  const cols = ["LTM", ...FWD(nFwd)];
  const last = res.years.at(-1);
  const maxGap = Math.max(0, ...res.forecast_balance_gaps.map(Math.abs));
  return (
    <Tiles>
      <Kpi title={`Revenue ${cols.at(-1)}`} value={fmtMoney(last?.revenue)} sub={`CAGR ${fmtRate(res.revenue_cagr)}`} lead />
      <Kpi title={`EBITDA ${cols.at(-1)}`} value={fmtMoney(last?.ebitda)} sub={`margin ${fmtRate(last?.ebitda_margin)}`} />
      <Kpi title={`Net income ${cols.at(-1)}`} value={fmtMoney(last?.net_income)} sub={`margin ${fmtRate(last?.net_margin)}`} />
      <Kpi title={`Cash ${cols.at(-1)}`} value={fmtMoney(last?.cash)} sub="$M" />
      <Kpi title="Balance sheet" value={res.balanced ? "Balances" : "Doesn't balance"} sub={`largest gap ${fmtMoney(maxGap)} $M`} tone={res.balanced ? "gain" : "loss"} />
      <Kpi title="Opening gap" value={fmtMoney(res.opening_balance_gap)} sub="in the historicals, $M" tone={Math.abs(res.opening_balance_gap) > 0.5 ? "attention" : undefined} />
      <Tile span={12} title="Income statement" unit="$M">
        <DataTable
          caption="Forecast income statement"
          columns={cols}
          rows={statementRows(res, [
            ["revenue", "Revenue"],
            ["gross_profit", "Gross profit"],
            ["rd", "R&D"],
            ["sga", "SG&A"],
            ["ebit", "EBIT", undefined, true],
            ["interest_inc", "Interest income"],
            ["interest_exp", "Interest expense"],
            ["pretax", "Pre-tax income"],
            ["taxes", "Taxes"],
            ["net_income", "Net income", undefined, true],
            ["ebitda", "EBITDA"],
            ["ebitda_margin", "EBITDA margin", "rate"],
          ])}
        />
      </Tile>
      <Tile span={6} title="Balance sheet" unit="$M">
        <DataTable
          caption="Forecast balance sheet"
          columns={cols}
          rows={statementRows(res, [
            ["cash", "Cash"],
            ["ar", "Receivables"],
            ["inventory", "Inventory"],
            ["ppe_net", "PP&E, net"],
            ["total_assets", "Total assets", undefined, true],
            ["ap", "Payables"],
            ["revolver", "Revolver"],
            ["ltd", "Long-term debt"],
            ["total_liab", "Total liabilities"],
            ["total_equity", "Total equity"],
            ["balance_check", "Balance check"],
          ])}
        />
      </Tile>
      <Tile span={6} title="Cash flow" unit="$M">
        <DataTable
          caption="Forecast cash flow"
          columns={cols}
          rows={statementRows(res, [
            ["cfo", "Operating cash flow"],
            ["cfi", "Investing cash flow"],
            ["cff", "Financing cash flow"],
            ["net_cash_chg", "Net change in cash", undefined, true],
            ["delta_nwc", "Change in NWC"],
            ["revolver_draw", "Revolver draw"],
          ])}
        />
      </Tile>
    </Tiles>
  );
}

export function SimulationStep() {
  return (
    <ForecastScreen>
      <Simulation />
    </ForecastScreen>
  );
}

function Simulation() {
  const { result: res, nFwd, simPaths } = useForecast();
  const sim = res?.simulation;
  if (!res || !sim) return <EmptyState title="No simulation">The forecast ran without a simulation.</EmptyState>;
  const cols = FWD(nFwd);
  const fan = (b: typeof sim.revenue_bands, det: number[], name: string) => (
    <LineChart
      label={`${name} simulation fan`}
      xLabels={cols}
      yFormat={(v) => v.toFixed(0)}
      bands={[
        { lower: b.p5, upper: b.p95, color: "var(--color-accent)", opacity: 0.14 },
        { lower: b.p25, upper: b.p75, color: "var(--color-accent)", opacity: 0.28 },
      ]}
      lines={[
        { name: "P50", values: b.p50, color: "var(--color-accent)" },
        { name: "Plan", values: det, color: "var(--color-muted)", dashed: true, width: 1.5 },
      ]}
    />
  );
  return (
    <Tiles>
      <Kpi title={`Revenue ${cols.at(-1)}`} value={fmtMoney(sim.revenue_final.median)} sub={`plan ${fmtMoney(sim.revenue_final.deterministic)} $M`} lead />
      <Kpi title="Revenue P5 / P95" value={fmtMoney(sim.revenue_final.p5)} sub={`to ${fmtMoney(sim.revenue_final.p95)} $M`} />
      <Kpi title={`EBITDA ${cols.at(-1)}`} value={fmtMoney(sim.ebitda_final.median)} sub={`plan ${fmtMoney(sim.ebitda_final.deterministic)} $M`} />
      <Kpi title="EBITDA P5 / P95" value={fmtMoney(sim.ebitda_final.p5)} sub={`to ${fmtMoney(sim.ebitda_final.p95)} $M`} />
      <Kpi title="Simulated growth" value={fmtRate(sim.growth_final_mean)} sub="mean, final year" />
      <Kpi title="Paths" value={sim.n.toLocaleString("en-US")} sub={`requested ${simPaths.toLocaleString("en-US")}`} />
      <Tile span={6} title="Revenue fan" unit="$M · P5-P95, P25-P75, median, plan">
        {fan(sim.revenue_bands, res.years.map((y) => y.revenue ?? NaN), "Revenue")}
      </Tile>
      <Tile span={6} title="EBITDA fan" unit="$M · P5-P95, P25-P75, median, plan">
        {fan(sim.ebitda_bands, res.years.map((y) => y.ebitda ?? NaN), "EBITDA")}
      </Tile>
      <Tile span={12} title={`Chance of reaching EBITDA in ${cols.at(-1)}`} unit="share of paths at or above target">
        <table className="w-full border-collapse font-mono text-[11.5px]">
          <tbody>
            {sim.target_probabilities.map((t) => (
              <tr key={t.target}>
                <th scope="row" className="w-40 border-b border-grid py-1.5 text-left font-normal text-ink">
                  {fmtMoney(t.target)} $M
                </th>
                <td className="border-b border-grid py-1.5">
                  <div className="h-3 bg-accent" style={{ width: `${Math.max(0.5, t.probability * 100)}%`, opacity: 0.35 + 0.65 * t.probability }} />
                </td>
                <td className="w-20 border-b border-grid py-1.5 text-right text-bright">{fmtRate(t.probability)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="type-body text-[9px]">
          <span className="chip mr-2 text-attention">finding 4</span>
          The API labels these targets Bull, Base and Bear, but the labels look inverted, so they&apos;re left off until that&apos;s fixed.
        </p>
      </Tile>
    </Tiles>
  );
}
