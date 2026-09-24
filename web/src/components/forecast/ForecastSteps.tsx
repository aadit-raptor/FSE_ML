"use client";

import { useEffect, useState, type ReactNode } from "react";

import { DataTable, type Row } from "@/components/charts/DataTable";
import { LineChart } from "@/components/charts/LineChart";
import { Waterfall } from "@/components/charts/Waterfall";
import { CellInput } from "@/components/ui/CellInput";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { EmptyState, LoadingTiles, Notice, RailGroup, Screen, SecondaryButton } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { MoneyScope, useMoney } from "@/components/ui/MoneyScope";
import { FiscalSelects } from "@/components/ui/FiscalSelects";
import { MoneySelects } from "@/components/ui/MoneySelects";
import { downloadWorkbook, sheet, tableSheet } from "@/lib/export";
import { ASSUMPTION_GROUPS, HISTORY_GROUPS } from "@/lib/forecast";
import { fmtCount, fmtInput, fmtMoney, fmtNumber, fmtRate, isNum } from "@/lib/format";
import { withMoney } from "@/lib/money";

import { type ForecastRun, useForecast } from "./ForecastProvider";

function Rail() {
  const { source, fetchEdgar, edgar, resetSample, metrics, status, result, money, setMoney, fiscal, setFiscal, histLabels } = useForecast();
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
      <RailGroup title="Reporting currency">
        <MoneySelects money={money} onChange={setMoney} of="Company" />
      </RailGroup>
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
            <p className="font-mono text-[10px] text-muted">Fiscal years {histLabels.join(", ")}</p>
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
      <RailGroup title="Fiscal years">
        <FiscalSelects fiscal={fiscal} onChange={setFiscal} of="Company" yearLabel="Latest fiscal year" />
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
  const { activate, ready, result, status, error, money } = useForecast();
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
  // The company's reporting currency, not the open deal's
  return (
    <MoneyScope money={money}>
      <Screen rail={<Rail />} bar={bar}>
        {body}
      </Screen>
    </MoneyScope>
  );
}

export function HistoricalsStep() {
  return (
    <ForecastScreen needsResult={false}>
      <Historicals />
    </ForecastScreen>
  );
}

function Historicals() {
  const { label: mu } = useMoney();
  const { history, setHistory, histLabels: cols } = useForecast();
  return (
    <Tiles>
      {HISTORY_GROUPS.map((g) => (
        <Tile key={g.title} span={g.rows.length > 10 ? 6 : 6} title={withMoney(g.title, mu)}>
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
  const { label: mu } = useMoney();
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
                {withMoney(r.label, mu)}
              </th>
              {columns.map((c, i) => (
                <td key={c} className="px-1 py-0.5">
                  <CellInput label={`${withMoney(r.label, mu)}, ${c}`} value={values[r.key]?.[i] ?? NaN} onCommit={(v) => onCommit(r.key, i, v)} />
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
  const { label: mu } = useMoney();
  const { assumptions, setAssumption, fillAssumption, seeded, reseed, fwdLabels: cols } = useForecast();
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
        <Tile key={g.title} span={6} title={withMoney(g.title, mu)}>
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

/** Rows for a forecast-years-only schedule. */
const fwdRow = (label: string, values: number[], kind?: Row["kind"], total?: boolean): Row => ({ label, values, kind, total });

function statementTables(res: ForecastRun) {
  return {
    income: statementRows(res, [
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
    ]),
    balance: statementRows(res, [
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
    ]),
    cash: statementRows(res, [
      ["cfo", "Operating cash flow"],
      ["cfi", "Investing cash flow"],
      ["cff", "Financing cash flow"],
      ["net_cash_chg", "Net change in cash", undefined, true],
      ["delta_nwc", "Change in NWC"],
      ["revolver_draw", "Revolver draw"],
    ]),
  };
}

/** Supporting schedules, derived exactly from the three statements. */
function scheduleTables(res: ForecastRun, assumptions: Record<string, number[]>) {
  const y = res.years;
  const ltm = res.ltm;
  const at = (key: string, i: number) => assumptions[key]?.[i] ?? NaN;
  const prevCash = y.map((_, i) => (i === 0 ? ltm.cash : y[i - 1].cash) ?? 0);
  const prevDebt = y.map((_, i) => (i === 0 ? ltm.ltd : y[i - 1].ltd) ?? 0);
  return {
    ppe: [
      fwdRow("Opening PP&E", y.map((r) => r.ppe_beg ?? 0)),
      // Capex isn't reported separately: the roll-forward implies it exactly
      fwdRow("Capex", y.map((r) => (r.ppe_end ?? 0) - (r.ppe_beg ?? 0) + (r.da ?? 0))),
      fwdRow("Depreciation", y.map((r) => r.da ?? 0), "outflow"),
      fwdRow("Closing PP&E", y.map((r) => r.ppe_end ?? 0), undefined, true),
    ],
    retained: [
      fwdRow("Opening retained earnings", y.map((r) => r.re_beg ?? 0)),
      fwdRow("Net income", y.map((r) => r.net_income ?? 0)),
      fwdRow("Dividends and buybacks", y.map((r) => (r.re_beg ?? 0) + (r.net_income ?? 0) - (r.re_end ?? 0)), "outflow"),
      fwdRow("Closing retained earnings", y.map((r) => r.re_end ?? 0), undefined, true),
    ],
    workingCapital: [
      fwdRow("Receivables", y.map((r) => r.ar ?? 0)),
      fwdRow("Inventory", y.map((r) => r.inventory ?? 0)),
      fwdRow("Payables", y.map((r) => r.ap ?? 0), "outflow"),
      fwdRow("Net working capital", y.map((r) => r.nwc ?? 0), undefined, true),
      fwdRow("Change in NWC", y.map((r) => r.delta_nwc ?? 0)),
    ],
    cycleDays: y.map((_, i) => at("ar_d", i) + at("inv_d", i) - at("ap_d", i)),
    interest: [
      fwdRow("Opening cash", prevCash),
      fwdRow("Rate on cash", y.map((_, i) => at("r_cash", i) / 100), "rate"),
      fwdRow("Interest income", y.map((r) => r.interest_inc ?? 0)),
      fwdRow("Opening debt", prevDebt),
      fwdRow("Rate on debt", y.map((_, i) => at("r_debt", i) / 100), "rate"),
      fwdRow("Interest expense", y.map((r) => r.interest_exp ?? 0), "outflow", true),
    ],
    revolver: [
      fwdRow("Draw / (repay)", y.map((r) => r.revolver_draw ?? 0)),
      fwdRow("Closing revolver", y.map((r) => r.revolver ?? 0), undefined, true),
      fwdRow("Closing cash", y.map((r) => r.cash ?? 0)),
    ],
  };
}

export function StatementsStep() {
  return (
    <ForecastScreen>
      <Statements />
    </ForecastScreen>
  );
}

function Statements() {
  const { label: mu, money } = useMoney();
  const { result: res, assumptions, source, histLabels, fwdLabels: fwd } = useForecast();
  if (!res) return null;
  const cols = [histLabels.at(-1) ?? "LTM", ...fwd];
  const last = res.years.at(-1);
  const maxGap = Math.max(0, ...res.forecast_balance_gaps.map(Math.abs));
  const t = statementTables(res);
  const s = scheduleTables(res, assumptions);
  const company = source.kind === "edgar" ? source.ticker : "sample";
  const everything = () => [
    tableSheet("Income statement", cols, t.income),
    tableSheet("Balance sheet", cols, t.balance),
    tableSheet("Cash flow", cols, t.cash),
    tableSheet("PP&E", fwd, s.ppe),
    tableSheet("Retained earnings", fwd, s.retained),
    tableSheet("Working capital", fwd, s.workingCapital),
    tableSheet("Interest", fwd, s.interest),
    tableSheet("Revolver", fwd, s.revolver),
  ];
  return (
    <Tiles>
      <Kpi title={`Revenue ${cols.at(-1)}`} value={fmtMoney(last?.revenue)} sub={`CAGR ${fmtRate(res.revenue_cagr)}`} lead />
      <Kpi title={`EBITDA ${cols.at(-1)}`} value={fmtMoney(last?.ebitda)} sub={`margin ${fmtRate(last?.ebitda_margin)}`} />
      <Kpi title={`Net income ${cols.at(-1)}`} value={fmtMoney(last?.net_income)} sub={`margin ${fmtRate(last?.net_margin)}`} />
      <Kpi title={`Cash ${cols.at(-1)}`} value={fmtMoney(last?.cash)} sub={mu} />
      <Kpi title="Balance sheet" value={res.balanced ? "Balances" : "Doesn't balance"} sub={`largest gap ${fmtMoney(maxGap)} ${mu}`} tone={res.balanced ? "gain" : "loss"} />
      <Kpi title="Opening gap" value={fmtMoney(res.opening_balance_gap)} sub={`in the historicals, ${mu}`} tone={Math.abs(res.opening_balance_gap) > 0.5 ? "attention" : undefined} />
      <div className="col-span-12 flex items-center justify-between gap-4 bg-canvas px-3 py-2">
        <p className="type-body">The three statements and every supporting schedule in one workbook.</p>
        <DownloadButton label="3-statement model" onDownload={() => downloadWorkbook(`${company}_3statement_model.xlsx`, everything(), money)} />
      </div>
      <Tile span={12} title="Income statement" unit={mu} action={<DownloadButton onDownload={() => downloadWorkbook("income_statement.xlsx", [tableSheet("Income statement", cols, t.income)], money)} />}>
        <DataTable caption="Forecast income statement" columns={cols} rows={t.income} />
      </Tile>
      <Tile span={6} title="Balance sheet" unit={mu} action={<DownloadButton onDownload={() => downloadWorkbook("balance_sheet.xlsx", [tableSheet("Balance sheet", cols, t.balance)], money)} />}>
        <DataTable caption="Forecast balance sheet" columns={cols} rows={t.balance} />
      </Tile>
      <Tile span={6} title="Cash flow" unit={mu} action={<DownloadButton onDownload={() => downloadWorkbook("cash_flow.xlsx", [tableSheet("Cash flow", cols, t.cash)], money)} />}>
        <DataTable caption="Forecast cash flow" columns={cols} rows={t.cash} />
      </Tile>
    </Tiles>
  );
}

export function SchedulesStep() {
  return (
    <ForecastScreen>
      <Schedules />
    </ForecastScreen>
  );
}

function Schedules() {
  const { label: mu, money } = useMoney();
  const { result: res, assumptions, fwdLabels: fwd } = useForecast();
  if (!res) return null;
  const s = scheduleTables(res, assumptions);
  const y = res.years.at(-1);
  const bridge = y
    ? [
        { label: "EBITDA", value: y.ebitda ?? 0, isTotal: true },
        { label: "D&A", value: (y.ebit ?? 0) - (y.ebitda ?? 0), isTotal: false },
        { label: "Int. income", value: y.interest_inc ?? 0, isTotal: false },
        { label: "Int. expense", value: y.interest_exp ?? 0, isTotal: false },
        { label: "Other", value: (y.pretax ?? 0) - (y.ebit ?? 0) - (y.interest_inc ?? 0) - (y.interest_exp ?? 0), isTotal: false },
        { label: "Taxes", value: (y.net_income ?? 0) - (y.pretax ?? 0), isTotal: false },
        { label: "Net income", value: y.net_income ?? 0, isTotal: true },
      ]
    : [];
  const all = () => [
    tableSheet("PP&E", fwd, s.ppe),
    tableSheet("Retained earnings", fwd, s.retained),
    tableSheet("Working capital", fwd, s.workingCapital),
    tableSheet("Interest", fwd, s.interest),
    tableSheet("Revolver", fwd, s.revolver),
  ];
  return (
    <Tiles>
      <div className="col-span-12 flex items-center justify-between gap-4 bg-canvas px-3 py-2">
        <p className="type-body">Each schedule is derived from the statements, so it ties to them exactly.</p>
        <DownloadButton label="All schedules" onDownload={() => downloadWorkbook("supporting_schedules.xlsx", all(), money)} />
      </div>
      <Tile span={6} title="PP&E roll-forward" unit={mu}>
        <DataTable caption="PP&E roll-forward" columns={fwd} rows={s.ppe} />
      </Tile>
      <Tile span={6} title="Retained earnings" unit={mu}>
        <DataTable caption="Retained earnings roll-forward" columns={fwd} rows={s.retained} />
      </Tile>
      <Tile span={6} title="Working capital" unit={mu}>
        <DataTable caption="Working capital schedule" columns={fwd} rows={s.workingCapital} />
        <p className="font-mono text-[10.5px] text-muted">
          Cash conversion cycle {s.cycleDays.map((d) => (Number.isFinite(d) ? `${fmtNumber(d, 0)}d` : "n/a")).join(" · ")}
        </p>
      </Tile>
      <Tile span={6} title="Interest" unit={mu}>
        <DataTable caption="Interest schedule" columns={fwd} rows={s.interest} />
      </Tile>
      <Tile span={6} title="Revolver" unit={`model plug, ${mu}`}>
        <DataTable caption="Revolver schedule" columns={fwd} rows={s.revolver} />
        <p className="type-body text-[9px]">The revolver draws when closing cash would fall below the minimum cash balance.</p>
      </Tile>
      <Tile span={6} title={`EBITDA to net income, ${fwd.at(-1)}`} unit={mu}>
        <Waterfall label={`EBITDA to net income bridge for ${fwd.at(-1)}`} steps={bridge} />
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
  const { label: mu, money } = useMoney();
  const { result: res, simPaths, fwdLabels: cols } = useForecast();
  const sim = res?.simulation;
  if (!res || !sim) return <EmptyState title="No simulation">The forecast ran without a simulation.</EmptyState>;
  const fan = (b: typeof sim.revenue_bands, det: number[], name: string) => (
    <LineChart
      label={`${name} simulation fan`}
      xLabels={cols}
      yFormat={(v) => fmtNumber(v, 0)}
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
      <Kpi title={`Revenue ${cols.at(-1)}`} value={fmtMoney(sim.revenue_final.median)} sub={`plan ${fmtMoney(sim.revenue_final.deterministic)} ${mu}`} lead />
      <Kpi title="Revenue P5 / P95" value={fmtMoney(sim.revenue_final.p5)} sub={`to ${fmtMoney(sim.revenue_final.p95)} ${mu}`} />
      <Kpi title={`EBITDA ${cols.at(-1)}`} value={fmtMoney(sim.ebitda_final.median)} sub={`plan ${fmtMoney(sim.ebitda_final.deterministic)} ${mu}`} />
      <Kpi title="EBITDA P5 / P95" value={fmtMoney(sim.ebitda_final.p5)} sub={`to ${fmtMoney(sim.ebitda_final.p95)} ${mu}`} />
      <Kpi title="Simulated growth" value={fmtRate(sim.growth_final_mean)} sub="mean, final year" />
      <Kpi title="Paths" value={fmtCount(sim.n)} sub={`requested ${fmtCount(simPaths)}`} />
      <div className="col-span-12 flex items-center justify-between gap-4 bg-canvas px-3 py-2">
        <p className="type-body">Percentile bands for revenue and EBITDA each year, and the target probabilities.</p>
        <DownloadButton
          onDownload={() =>
            downloadWorkbook("simulation_results.xlsx", [
              sheet("Revenue bands", ["Percentile", ...cols], (["p5", "p25", "p50", "p75", "p95"] as const).map((q) => [q.toUpperCase(), ...sim.revenue_bands[q]]), {
                columns: ["text", ...cols.map(() => "money" as const)],
              }),
              sheet("EBITDA bands", ["Percentile", ...cols], (["p5", "p25", "p50", "p75", "p95"] as const).map((q) => [q.toUpperCase(), ...sim.ebitda_bands[q]]), {
                columns: ["text", ...cols.map(() => "money" as const)],
              }),
              sheet("Targets", [`EBITDA target (${mu})`, "Probability", "Case"], sim.target_probabilities.map((t) => [t.target, t.probability, t.scenario]), {
                columns: ["money", "percent", "text"],
              }),
            ], money)
          }
        />
      </div>
      <Tile span={6} title="Revenue fan" unit={`${mu} · P5-P95, P25-P75, median, plan`}>
        {fan(sim.revenue_bands, res.years.map((y) => y.revenue ?? NaN), "Revenue")}
      </Tile>
      <Tile span={6} title="EBITDA fan" unit={`${mu} · P5-P95, P25-P75, median, plan`}>
        {fan(sim.ebitda_bands, res.years.map((y) => y.ebitda ?? NaN), "EBITDA")}
      </Tile>
      <Tile span={12} title={`Chance of reaching EBITDA in ${cols.at(-1)}`} unit="share of paths at or above target">
        <table className="w-full border-collapse font-mono text-[11.5px]">
          <tbody>
            {sim.target_probabilities.map((t) => (
              <tr key={t.target}>
                <th scope="row" className="w-48 border-b border-grid py-1.5 text-left font-normal text-ink">
                  {fmtMoney(t.target)} {mu}{" "}
                  <span className={`chip ml-1 ${t.scenario === "Bull" ? "text-gain" : t.scenario === "Bear" ? "text-loss" : "text-dim"}`}>
                    {t.scenario.toLowerCase()}
                  </span>
                </th>
                <td className="border-b border-grid py-1.5">
                  <div className="h-3 bg-accent" style={{ width: `${Math.max(0.5, t.probability * 100)}%`, opacity: 0.35 + 0.65 * t.probability }} />
                </td>
                <td className="w-20 border-b border-grid py-1.5 text-right text-bright">{fmtRate(t.probability)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="type-body text-[9px]">Targets run from 80% to 120% of plan EBITDA: above plan is the bull case, below plan the bear case.</p>
      </Tile>
    </Tiles>
  );
}
