"use client";

import { useEffect, type ReactNode } from "react";

import { DivergingBars, GroupedBars } from "@/components/charts/Bars";
import { DataTable } from "@/components/charts/DataTable";
import { Histogram } from "@/components/charts/Histogram";
import { LineChart } from "@/components/charts/LineChart";
import { EmptyState, LoadingTiles, Notice, RailGroup, Screen } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { fmtDelta, fmtMoney, fmtMultiple, isNum } from "@/lib/format";

import { BACKTEST_PATHS, splitDealName, useBacktest } from "./BacktestProvider";

const pctNum = (v: number | null | undefined, d = 1) => (isNum(v) ? `${v.toFixed(d)}%` : "n/a");

const ENTRY_LABELS: [string, string, string][] = [
  ["entry_ebitda", "EBITDA", "$M"],
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
                    {deal.entry[k]} <span className="text-[10px] text-[#56636a]">{unit}</span>
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
  const { activate, status, result, error } = useBacktest();
  useEffect(() => activate(), [activate]);
  return (
    <Screen
      rail={<Rail />}
      bar={
        status === "error" ? (
          <Notice tone="loss" title="Backtest didn't run" role="alert">
            {error}
          </Notice>
        ) : undefined
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
  );
}

function Headline() {
  const { result: r } = useBacktest();
  if (!r) return null;
  const lastPred = r.predicted_ebitda.at(-1);
  const lastAct = r.years.at(-1)?.actual_ebitda;
  return (
    <>
      <Kpi title="Actual IRR" value={pctNum(r.actual_irr)} sub={`MOIC ${fmtMultiple(r.actual_moic)}`} lead />
      <Kpi title="Predicted IRR" value={pctNum(r.predicted_irr_mean)} sub={`mean of ${BACKTEST_PATHS.toLocaleString("en-US")} paths`} />
      <Kpi title="Predicted range" value={pctNum(r.predicted_irr_p5)} sub={`to ${pctNum(r.predicted_irr_p95)} (P5 to P95)`} />
      <Kpi title="Actual percentile" value={isNum(r.actual_percentile) ? r.actual_percentile.toFixed(1) : "n/a"} sub="of predicted paths" />
      <Kpi title="Exit EBITDA" value={fmtMoney(lastAct)} sub={`predicted ${fmtMoney(lastPred)} $M`} tone={isNum(lastAct) && isNum(lastPred) ? (lastAct >= lastPred ? "gain" : "loss") : undefined} />
      <Kpi title="Exit equity" value={fmtMoney(r.actual_exit_equity)} sub={`predicted ${fmtMoney(r.predicted_exit_equity)} $M`} />
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
  const { result: r, deal } = useBacktest();
  if (!r || !deal) return null;
  const years = deal.actual_years.length ? deal.actual_years.map(String) : r.years.map((y) => `Y${y.year_index}`);
  return (
    <Tiles>
      <Headline />
      <Tile span={7} title="EBITDA, predicted vs actual" unit="$M">
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
          format={(v) => `${v.toFixed(0)}%`}
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
  ebitda_growth_miss: "EBITDA growth",
  margin_difference: "Margin",
  fcf_conversion: "FCF conversion",
  debt_paydown: "Debt paydown",
};

export function AttributionStep() {
  return (
    <BacktestScreen>
      <Attribution />
    </BacktestScreen>
  );
}

function Attribution() {
  const { result: r, deal } = useBacktest();
  if (!r || !deal) return null;
  const years = deal.actual_years.length ? deal.actual_years.map(String) : r.years.map((y) => `Y${y.year_index}`);
  return (
    <Tiles>
      <Notice title="Rough split" role="note" className="col-span-12">
        The attribution uses fixed weights (0.3, 0.1, 0.05, 0.75) rather than a decomposition of the model. Open finding 5: read the direction, not the size.
      </Notice>
      <Headline />
      <Tile span={6} title="Where the prediction missed" unit="$M">
        <DivergingBars
          label="Error attribution"
          format={fmtDelta}
          rows={Object.entries(r.attribution).map(([k, v]) => ({ label: ATTRIBUTION[k] ?? k, value: v }))}
        />
      </Tile>
      <Tile span={6} title="EBITDA margin" unit="actual vs predicted">
        <LineChart
          label="Actual EBITDA margin by year against the predicted margin"
          xLabels={years}
          yFormat={(v) => `${v.toFixed(0)}%`}
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
  const { result: r, deal } = useBacktest();
  if (!r || !deal) return null;
  const years = deal.actual_years.length ? deal.actual_years.map(String) : r.years.map((y) => `Y${y.year_index}`);
  return (
    <Tiles>
      <Headline />
      <Tile span={12} title="Year by year" unit="$M">
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
      <Tile span={6} title="Equity" unit="$M">
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
