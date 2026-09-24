"use client";

import { useEffect, type ReactNode } from "react";

import { DivergingBars, GroupedBars } from "@/components/charts/Bars";
import { DataTable } from "@/components/charts/DataTable";
import { Histogram } from "@/components/charts/Histogram";
import { LineChart } from "@/components/charts/LineChart";
import { EmptyState, LoadingTiles, Notice, RailGroup, Screen } from "@/components/ui/Screen";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { MoneyScope, useMoney } from "@/components/ui/MoneyScope";
import { downloadWorkbook, fraction, sheet } from "@/lib/export";
import { fmtCount, fmtDelta, fmtMoney, fmtMultiple, fmtNumber, fmtPct, isNum } from "@/lib/format";
import { DEFAULT_MONEY, fieldUnit, MONEY } from "@/lib/money";
import { backtestSampleLabel } from "@/lib/provenance";

import { BACKTEST_PATHS, splitDealName, useBacktest } from "./BacktestProvider";

const pctNum = (v: number | null | undefined, d = 1) => fmtPct(v, d);

const ENTRY_LABELS: [string, string, string][] = [
  ["entry_ebitda", "EBITDA", MONEY],
  ["entry_multiple", "Entry multiple", "x"],
  ["exit_multiple", "Exit multiple", "x"],
  ["holding_period", "Hold", "yr"],
  ["debt_pct", "Debt / EV", "%"],
  ["senior_pct", "Senior share", "%"],
  ["base_rate", "Senior rate", "%"],
  ["mezz_spread", "Mezz spread", "%"],
  ["revenue_growth", "Revenue growth", "%"],
  ["gross_margin", "Gross margin", "%"],
  ["opex_pct", "Opex", "%"],
  ["da_pct", "D&A", "%"],
  ["tax_rate", "Tax rate", "%"],
  ["capex_pct", "Capex", "%"],
  ["nwc_pct", "NWC change", "%"],
];

function Rail() {
  const { deals, selected, select, deal } = useBacktest();
  const { money } = useMoney();
  return (
    <>
      <RailGroup title="Historical deal">
        <div className="grid gap-px bg-line" role="radiogroup" aria-label="Historical deal">
          {(deals ?? []).map((d) => {
            const n = splitDealName(d.name);
            const on = d.name === selected;
            return (
              <button
                key={d.name}
                type="button"
                role="radio"
                aria-checked={on}
                onClick={() => select(d.name)}
                className={`grid px-2.5 py-2 text-left ${on ? "bg-raised shadow-[inset_2px_0_0_var(--color-accent)]" : "bg-panel hover:bg-raised"}`}
              >
                <span className={on ? "type-control-value" : "type-control"}>{n.name}</span>
                <span className="font-mono text-[10px] text-muted">
                  {n.sponsor} · {n.year}
                </span>
              </button>
            );
          })}
        </div>
      </RailGroup>
      {deal && (
        <>
          <RailGroup
            title="About"
            aside={deal.outcome && <span className={`chip ${deal.outcome === "SUCCESS" ? "text-gain" : "text-loss"}`}>{deal.outcome.toLowerCase()}</span>}
          >
            <p className="type-body text-[9px]">{deal.description}</p>
            <p className="font-mono text-[10px] text-muted">
              {deal.sector} · {deal.geography}
            </p>
          </RailGroup>
          <RailGroup title="Entry assumptions">
            <dl className="grid grid-cols-[1fr_auto] gap-x-2 gap-y-1">
              {ENTRY_LABELS.map(([k, label, unit]) => (
                <div key={k} className="contents">
                  <dt className="type-input-label">{label}</dt>
                  <dd className="text-right font-mono text-[11.5px] text-ink">
                    {deal.entry[k]} <span className="text-[10px] text-[#56636a]">{fieldUnit(unit, money)}</span>
                  </dd>
                </div>
              ))}
            </dl>
          </RailGroup>
        </>
      )}
    </>
  );
}

function BacktestScreen({ children }: { children: ReactNode }) {
  const { activate, status, result, error, deals, deal } = useBacktest();
  useEffect(() => activate(), [activate]);
  // The example deal's own currency and unit, not the open deal's
  return (
    <MoneyScope money={result?.money ?? deal?.money ?? DEFAULT_MONEY}>
    <Screen
      rail={<Rail />}
      bar={
        <>
          {status === "error" && (
            <Notice tone="loss" title="Backtest didn't run" role="alert">
              {error}
            </Notice>
          )}
          {deals && (
            <Notice title="Examples, not evidence" role="note">
              {backtestSampleLabel(deals.map((d) => splitDealName(d.name).year))}. Actuals are approximate and unsourced.
            </Notice>
          )}
        </>
      }
    >
      {result ? (
        <div className={status === "running" ? "opacity-60 transition-opacity" : ""}>{children}</div>
      ) : status === "error" ? (
        <EmptyState title="No backtest yet">Check that the API is running, then reload.</EmptyState>
      ) : (
        <LoadingTiles />
      )}
    </Screen>
    </MoneyScope>
  );
}

function Headline() {
  const { label: mu } = useMoney();
  const { result: r } = useBacktest();
  if (!r) return null;
  const lastPred = r.predicted_ebitda.at(-1);
  const lastAct = r.years.at(-1)?.actual_ebitda;
  return (
    <>
      <Kpi title="Actual IRR" value={pctNum(r.actual_irr)} sub={`MOIC ${fmtMultiple(r.actual_moic)}`} lead />
      <Kpi title="Predicted IRR" value={pctNum(r.predicted_irr_mean)} sub={`mean of ${fmtCount(BACKTEST_PATHS)} paths`} />
      <Kpi title="Predicted range" value={pctNum(r.predicted_irr_p5)} sub={`to ${pctNum(r.predicted_irr_p95)} (P5 to P95)`} />
      <Kpi title="Actual percentile" value={fmtNumber(r.actual_percentile, 1)} sub="of predicted paths" />
      <Kpi title="Exit EBITDA" value={fmtMoney(lastAct)} sub={`predicted ${fmtMoney(lastPred)} ${mu}`} tone={isNum(lastAct) && isNum(lastPred) ? (lastAct >= lastPred ? "gain" : "loss") : undefined} />
      <Kpi title="Exit equity" value={fmtMoney(r.actual_exit_equity)} sub={`predicted ${fmtMoney(r.predicted_exit_equity)} ${mu}`} />
    </>
  );
}

export function PredictedStep() {
  return (
    <BacktestScreen>
      <Predicted />
    </BacktestScreen>
  );
}

function Predicted() {
  const { label: mu } = useMoney();
  const { result: r, deal } = useBacktest();
  if (!r || !deal) return null;
  const years = deal.actual_years.length ? deal.actual_years.map(String) : r.years.map((y) => `Y${y.year_index}`);
  return (
    <Tiles>
      <Headline />
      <Tile span={7} title="EBITDA, predicted vs actual" unit={mu}>
        <GroupedBars
          label={`Predicted and actual EBITDA for ${splitDealName(deal.name).name}`}
          categories={years}
          format={fmtMoney}
          series={[
            { name: "Predicted", values: r.predicted_ebitda, color: "var(--color-neutral-bar)" },
            { name: "Actual", values: r.years.map((y) => y.actual_ebitda), color: "var(--color-accent)", labelled: true },
          ]}
        />
      </Tile>
      <Tile span={5} title="Where the actual IRR landed" unit="predicted distribution">
        <Histogram
          label="Predicted IRR distribution with the actual IRR marked"
          edges={r.irr_histogram.edges}
          density={r.irr_histogram.density}
          format={(v) => fmtPct(v, 0)}
          tickCount={5}
          markers={[
            { value: r.predicted_irr_p5, label: `P5 ${pctNum(r.predicted_irr_p5)}` },
            { value: r.actual_irr, label: `Actual ${pctNum(r.actual_irr)}`, tone: "accent" },
            { value: r.predicted_irr_p95, label: `P95 ${pctNum(r.predicted_irr_p95)}` },
          ]}
        />
      </Tile>
    </Tiles>
  );
}

const ATTRIBUTION: Record<string, string> = {
  exit_ebitda: "Exit EBITDA",
  exit_multiple: "Exit multiple",
  net_debt: "Net debt at exit",
};

export function AttributionStep() {
  return (
    <BacktestScreen>
      <Attribution />
    </BacktestScreen>
  );
}

function Attribution() {
  const { label: mu } = useMoney();
  const { result: r, deal } = useBacktest();
  if (!r || !deal) return null;
  const years = deal.actual_years.length ? deal.actual_years.map(String) : r.years.map((y) => `Y${y.year_index}`);
  return (
    <Tiles>
      <Headline />
      <Tile span={6} title="Where the prediction missed" unit={`${mu} of exit equity`}>
        <DivergingBars
          label="Error attribution"
          format={fmtDelta}
          rows={Object.entries(r.attribution).map(([k, v]) => ({ label: ATTRIBUTION[k] ?? k, value: v }))}
        />
        <p className="type-body text-[9px]">
          The parts add up to actual minus predicted exit equity ({fmtDelta(Object.values(r.attribution).reduce((a, b) => a + b, 0))} {mu}). Exit
          multiple {fmtMultiple(deal.entry.exit_multiple, 1)} predicted, {fmtMultiple(r.actual_exit_multiple, 1)} actual; predicted net debt at exit{" "}
          {fmtMoney(r.predicted_net_debt_at_exit)} {mu}.
        </p>
      </Tile>
      <Tile span={6} title="EBITDA margin" unit="actual vs predicted">
        <LineChart
          label="Actual EBITDA margin by year against the predicted margin"
          xLabels={years}
          yFormat={(v) => fmtPct(v, 0)}
          lines={[
            { name: "Actual", values: r.actual_ebitda_margin, color: "var(--color-accent)" },
            { name: "Predicted", values: years.map(() => r.predicted_ebitda_margin), color: "var(--color-muted)", dashed: true },
          ]}
        />
      </Tile>
    </Tiles>
  );
}

export function YearsStep() {
  return (
    <BacktestScreen>
      <Years />
    </BacktestScreen>
  );
}

function Years() {
  const { label: mu, money } = useMoney();
  const { result: r, deal } = useBacktest();
  if (!r || !deal) return null;
  const years = deal.actual_years.length ? deal.actual_years.map(String) : r.years.map((y) => `Y${y.year_index}`);
  return (
    <Tiles>
      <Headline />
      <Tile
        span={12}
        title="Year by year"
        unit={mu}
        action={
          <DownloadButton
            onDownload={() =>
              downloadWorkbook("backtest.xlsx", [
                sheet(
                  "Year by year",
                  ["Year", "Predicted EBITDA", "Actual EBITDA", "Variance", "Actual revenue", "Actual FCF", "Actual total debt"],
                  r.years.map((y, i) => [years[i] ?? y.year_index, y.predicted_ebitda, y.actual_ebitda, y.ebitda_variance, y.actual_revenue, y.actual_fcf, y.actual_total_debt]),
                  { columns: ["text", "money", "money", "money", "money", "money", "money"] },
                ),
                sheet(
                  "Returns",
                  ["Metric", "Predicted", "Actual"],
                  [
                    ["IRR", fraction(r.predicted_irr_mean, 100), fraction(r.actual_irr, 100)],
                    ["MOIC", r.predicted_moic, r.actual_moic],
                    [`Entry equity (${mu})`, r.predicted_equity_entry, r.actual_equity_entry],
                    [`Exit equity (${mu})`, r.predicted_exit_equity, r.actual_exit_equity],
                  ],
                  { rows: ["percent", "multiple", "money", "money"] },
                ),
                sheet("Attribution", ["Part", `${mu}`], Object.entries(r.attribution).map(([k, v]) => [k, v]), { columns: ["text", "money"] }),
              ], money)
            }
          />
        }
      >
        <DataTable
          caption="Predicted and actual results by year"
          columns={years}
          rows={[
            { label: "EBITDA, predicted", values: r.years.map((y) => y.predicted_ebitda) },
            { label: "EBITDA, actual", values: r.years.map((y) => y.actual_ebitda), total: true },
            { label: "EBITDA variance", values: r.years.map((y) => y.ebitda_variance) },
            { label: "Revenue, actual", values: r.years.map((y) => y.actual_revenue) },
            { label: "Free cash flow, actual", values: r.years.map((y) => y.actual_fcf) },
            { label: "Total debt, actual", values: r.years.map((y) => y.actual_total_debt) },
          ]}
        />
      </Tile>
      <Tile span={6} title="Equity" unit={mu}>
        <DataTable
          caption="Predicted and actual equity"
          columns={["Predicted", "Actual"]}
          rows={[
            { label: "Equity at entry", values: [r.predicted_equity_entry, r.actual_equity_entry] },
            { label: "Equity at exit", values: [r.predicted_exit_equity, r.actual_exit_equity], total: true },
          ]}
        />
      </Tile>
      <Tile span={6} title="Returns" unit="IRR % and MOIC">
        <table className="w-full border-collapse font-mono text-[11px]">
          <thead>
            <tr className="text-muted">
              <th />
              <th scope="col" className="border-b border-grid px-2 py-1 text-right font-normal">
                Predicted
              </th>
              <th scope="col" className="border-b border-grid px-2 py-1 text-right font-normal">
                Actual
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row" className="type-input-label border-b border-grid px-2 py-1 text-left text-[9px] font-normal text-soft">
                IRR
              </th>
              <td className="border-b border-grid px-2 py-1 text-right">{pctNum(r.predicted_irr_mean)}</td>
              <td className="border-b border-grid px-2 py-1 text-right text-bright">{pctNum(r.actual_irr)}</td>
            </tr>
            <tr>
              <th scope="row" className="type-input-label border-b border-grid px-2 py-1 text-left text-[9px] font-normal text-soft">
                MOIC
              </th>
              <td className="border-b border-grid px-2 py-1 text-right">{fmtMultiple(r.predicted_moic)}</td>
              <td className="border-b border-grid px-2 py-1 text-right text-bright">{fmtMultiple(r.actual_moic)}</td>
            </tr>
          </tbody>
        </table>
      </Tile>
    </Tiles>
  );
}
