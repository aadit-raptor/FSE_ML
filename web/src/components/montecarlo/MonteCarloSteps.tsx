"use client";

import { useState } from "react";

import { DivergingBars, RangeRows, Scatter } from "@/components/charts/Bars";
import { heat } from "@/components/charts/HeatTable";
import { Histogram } from "@/components/charts/Histogram";
import { LineChart } from "@/components/charts/LineChart";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { downloadMonteCarloSample, downloadWorkbook, sheet } from "@/lib/export";
import { fmtMultiple, fmtRate, isNum } from "@/lib/format";

import { MacroRegime } from "./MonteCarloML";
import { SCENARIOS, useMonteCarlo } from "./MonteCarloProvider";
import { MonteCarloScreen, useStaleClass } from "./MonteCarloScreen";

const pct0 = (v: number) => `${(v * 100).toFixed(0)}%`;
const count = (share: number | null | undefined, n: number) => (isNum(share) ? Math.round(share * n).toLocaleString("en-US") : "n/a");

function useResult() {
  const { run, hurdle } = useMonteCarlo();
  return { r: run.result!, scen: run.scenarios!, hurdle, ranFor: run.ranFor! };
}

export function DistributionStep() {
  return (
    <MonteCarloScreen>
      <Distribution />
    </MonteCarloScreen>
  );
}

function Distribution() {
  const { r, hurdle, ranFor } = useResult();
  const s = r.summary;
  const staleClass = useStaleClass();
  const p = r.params as Record<string, number>;
  const assumptions: [string, string][] = [
    ["Revenue growth", `${fmtRate(p.growth_mean)} ± ${fmtRate(p.growth_std)}`],
    ["Exit multiple", `${fmtMultiple(p.exit_mean, 1)} ± ${fmtMultiple(p.exit_std, 2)}`],
    ["Interest rate", `${fmtRate(p.interest_mean, 2)} ± ${fmtRate(p.interest_std, 2)}`],
    ["Gross margin", `${fmtRate(p.gross_margin_mean)} ± ${fmtRate(p.gross_margin_std)}`],
    ["Debt / EV, senior share", `${fmtRate(p.debt_pct)}, ${fmtRate(p.senior_pct)}`],
    ["Fees: transaction, financing", `${fmtRate(p.transaction_fees_pct, 2)}, ${fmtRate(p.financing_fees_pct, 2)}`],
    ["Opex, D&A, capex, NWC", `${fmtRate(p.opex_pct)}, ${fmtRate(p.da_pct)}, ${fmtRate(p.capex_pct)}, ${fmtRate(p.nwc_pct)}`],
    ["Tax rate", fmtRate(p.tax_rate)],
  ];

  return (
    <div className={staleClass}>
      <Tiles>
        <Kpi title="Mean IRR" value={fmtRate(s.mean_irr)} sub={`median ${fmtRate(s.median_irr)}`} lead />
        <Kpi title={`P(IRR > ${pct0(hurdle)})`} value={fmtRate(s.p_above_hurdle)} sub={`${count(s.p_above_hurdle, r.n)} of ${r.n.toLocaleString("en-US")}`} />
        <Kpi title="P5" value={fmtRate(s.p5_irr)} sub="1 in 20 below" />
        <Kpi title="P95" value={fmtRate(s.p95_irr)} sub="1 in 20 above" />
        <Kpi title="Wipeout" value={fmtRate(s.wipeout_rate, 3)} sub={`${count(s.wipeout_rate, r.n)} paths`} tone={(s.wipeout_rate ?? 0) > 0.05 ? "loss" : undefined} />
        <Kpi title="Run time" value={`${Math.round(r.elapsed_ms)} ms`} sub={`${ranFor.seed === null ? "random seed" : `seed ${ranFor.seed}`} · ${r.scenario ?? "no preset"}`} />

        <Tile
          span={7}
          title="IRR distribution"
          unit={`${r.n.toLocaleString("en-US")} paths`}
          action={
            <DownloadButton
              label="10k paths"
              title={ranFor.seed === null ? "Random seed: the file is a fresh draw with the same inputs" : "The paths behind these results"}
              onDownload={() =>
                downloadMonteCarloSample({
                  mc: { ...ranFor.sim, ebitda: ranFor.deal.ebitda, entry_mult: ranFor.deal.entry_mult, hold: ranFor.deal.hold },
                  deal: ranFor.deal,
                  settings: ranFor.settings,
                  scenario: ranFor.scenario,
                  seed: ranFor.seed,
                })
              }
            />
          }
        >
          <Histogram
            label="Simulated IRR distribution"
            edges={r.irr_histogram.edges}
            density={r.irr_histogram.density}
            highlightFrom={hurdle}
            format={pct0}
            markers={[
              ...(isNum(s.p5_irr) ? [{ value: s.p5_irr, label: `P5 ${fmtRate(s.p5_irr)}` }] : []),
              { value: hurdle, label: `Hurdle ${pct0(hurdle)}`, tone: "attention" as const, dashed: true },
              ...(isNum(s.p95_irr) ? [{ value: s.p95_irr, label: `P95 ${fmtRate(s.p95_irr)}` }] : []),
            ]}
          />
        </Tile>
        <Tile span={5} title="IRR by percentile" unit="share of paths below">
          <LineChart
            label="IRR at each percentile"
            xLabels={r.irr_cdf.percentiles.map((q) => `P${q.toFixed(0)}`)}
            xTickEvery={25}
            lines={[{ name: "", values: r.irr_cdf.values, color: "var(--color-accent)" }]}
            hlines={[{ value: hurdle, label: `Hurdle ${pct0(hurdle)}`, tone: "attention" }]}
            yFormat={pct0}
            endLabels={false}
          />
        </Tile>
        <Tile span={6} title="MOIC distribution" unit="x">
          <Histogram
            label="Simulated MOIC distribution"
            edges={r.moic_histogram.edges}
            density={r.moic_histogram.density}
            highlightFrom={1}
            format={(v) => `${v.toFixed(1)}x`}
            markers={[{ value: 1, label: "1.0x money back", tone: "attention", dashed: true }]}
          />
        </Tile>
        <Tile span={6} title="What was simulated" unit="mean ± std dev">
          <table className="w-full border-collapse font-mono text-[11px]">
            <tbody>
              {assumptions.map(([k, v]) => (
                <tr key={k}>
                  <th scope="row" className="type-input-label border-b border-grid py-1.5 text-left text-[9px] font-normal text-soft">
                    {k}
                  </th>
                  <td className="border-b border-grid py-1.5 text-right text-ink">{v}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Tile>
      </Tiles>
    </div>
  );
}

export function ScenariosStep() {
  return (
    <MonteCarloScreen>
      <Scenarios />
    </MonteCarloScreen>
  );
}

function Scenarios() {
  const { scen, hurdle, r } = useResult();
  const staleClass = useStaleClass();
  const rows = SCENARIOS.map((sc) => ({ ...sc, st: scen.scenarios[sc.id] })).filter((x) => x.st);
  return (
    <div className={staleClass}>
      <Tiles>
        {rows.map(({ id, label, st }) => (
          <Kpi key={id} title={label} value={fmtRate(st.mean_irr)} sub={`${fmtRate(st.p_above_hurdle)} clear the hurdle`} lead={id === "base"} tone={(st.mean_irr ?? 0) < 0 ? "loss" : undefined} />
        ))}
        <Kpi title="Paths each" value={r.n.toLocaleString("en-US")} sub="same seed for all four" />
        <Kpi title="Hurdle" value={pct0(scen.hurdle)} sub="from the rail" />
        <MacroRegime />

        <Tile span={7} title="IRR range by scenario" unit="P5 to P95, mean · share above hurdle">
          <RangeRows
            label="IRR range for each scenario"
            format={pct0}
            threshold={{ value: hurdle, label: "Hurdle" }}
            rows={rows.map(({ label, st }) => ({
              label,
              low: st.p5_irr ?? 0,
              high: st.p95_irr ?? 0,
              mid: st.mean_irr ?? 0,
              note: fmtRate(st.p_above_hurdle),
            }))}
          />
        </Tile>
        <Tile
          span={5}
          title="Scenario table"
          unit="IRR unless noted"
          action={
            <DownloadButton
              onDownload={() =>
                downloadWorkbook("scenario_comparison.xlsx", [
                  sheet(
                    "Scenarios (%)",
                    ["Scenario", "Mean IRR", "Median IRR", "P5 IRR", "P95 IRR", "P(IRR > hurdle)", "Wipeout", "MOIC P50 (x)"],
                    rows.map(({ label, st }) => [
                      label,
                      ...[st.mean_irr, st.median_irr, st.p5_irr, st.p95_irr, st.p_above_hurdle, st.wipeout_rate].map((v) => (v == null ? null : v * 100)),
                      st.moic_box.p50,
                    ]),
                  ),
                ])
              }
            />
          }
        >
          <table className="w-full border-collapse font-mono text-[11px]">
            <thead>
              <tr className="text-muted">
                <th />
                {["Mean", "P5", "P95", "Wipeout", "MOIC P50"].map((h) => (
                  <th key={h} scope="col" className="border-b border-grid px-1.5 py-1 text-right font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map(({ id, label, st }) => (
                <tr key={id} className={id === "base" ? "text-bright" : "text-ink"}>
                  <th scope="row" className="type-input-label border-b border-grid py-1.5 text-left text-[9px] font-normal text-soft">
                    {label}
                  </th>
                  {[st.mean_irr, st.p5_irr, st.p95_irr].map((v, i) => (
                    <td key={i} className={`border-b border-grid px-1.5 py-1.5 text-right ${(v ?? 0) < 0 ? "text-loss" : ""}`}>
                      {fmtRate(v)}
                    </td>
                  ))}
                  <td className="border-b border-grid px-1.5 py-1.5 text-right">{fmtRate(st.wipeout_rate, 2)}</td>
                  <td className="border-b border-grid px-1.5 py-1.5 text-right">{fmtMultiple(st.moic_box.p50)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="type-body text-[9px]">Multipliers for each preset are in Settings, Scenario presets.</p>
        </Tile>
      </Tiles>
    </div>
  );
}

const DRIVERS = ["Growth", "Exit Multiple", "Interest", "Gross Margin"] as const;

export function DriversStep() {
  return (
    <MonteCarloScreen>
      <Drivers />
    </MonteCarloScreen>
  );
}

function Drivers() {
  const { r } = useResult();
  const staleClass = useStaleClass();
  const [driver, setDriver] = useState<(typeof DRIVERS)[number]>("Growth");
  const xs = (r.scatter[driver] ?? []).map((v) => v ?? NaN);
  const ys = (r.scatter.IRR ?? []).map((v) => (v ?? NaN) * 100);
  const fit = r.driver_fits[driver];
  const corr = r.correlations as { labels: string[]; matrix: number[][] };
  const xFormat = driver === "Exit Multiple" ? (v: number) => `${v.toFixed(1)}x` : (v: number) => `${(v * 100).toFixed(1)}%`;

  return (
    <div className={staleClass}>
      <Tiles>
        <Tile span={5} title="What moves IRR" unit="rank correlation">
          <DivergingBars
            label="Spearman rank correlation of each driver with IRR"
            domain={1}
            format={(v) => `${v > 0 ? "+" : ""}${v.toFixed(2)}`}
            rows={r.drivers.map((d) => ({ label: d.driver, value: d.spearman_rho ?? 0 }))}
          />
          <p className="type-body text-[9px]">+1 means IRR rises whenever the driver does; 0 means no link.</p>
        </Tile>
        <Tile
          span={7}
          title={`IRR against ${driver.toLowerCase()}`}
          aside={
            <span className="flex gap-px bg-line" role="radiogroup" aria-label="Driver">
              {DRIVERS.map((d) => (
                <button
                  key={d}
                  type="button"
                  role="radio"
                  aria-checked={d === driver}
                  onClick={() => setDriver(d)}
                  className={`type-action-secondary px-2 py-1 text-[8.5px] ${d === driver ? "bg-raised text-bright" : "bg-canvas text-muted hover:text-ink"}`}
                >
                  {d}
                </button>
              ))}
            </span>
          }
        >
          <Scatter
            label={`Sample of simulated IRR against ${driver}`}
            xs={xs}
            ys={ys}
            fit={isNum(fit?.slope) && isNum(fit?.intercept) ? { slope: fit.slope, intercept: fit.intercept } : undefined}
            xFormat={xFormat}
            yFormat={(v) => `${v.toFixed(0)}%`}
          />
          <p className="type-body text-[9px]">
            {xs.length.toLocaleString("en-US")} sampled paths; amber line is the least-squares fit
            {isNum(fit?.r) ? ` (r = ${fit.r.toFixed(2)})` : ""}.
          </p>
        </Tile>
        <Tile span={12} title="Correlations in the simulated paths" unit="Pearson">
          <table className="w-full border-separate border-spacing-px font-mono text-[11px]">
            <thead>
              <tr>
                <th />
                {corr.labels.map((l) => (
                  <th key={l} scope="col" className="px-2 py-1 text-right font-normal text-muted">
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {corr.matrix.map((row, i) => (
                <tr key={corr.labels[i]}>
                  <th scope="row" className="type-input-label px-2 py-1 text-left text-[9px] font-normal text-soft">
                    {corr.labels[i]}
                  </th>
                  {row.map((v, j) => (
                    <td key={j} className="px-2 py-1 text-right" style={i === j ? { color: "var(--color-dim)" } : { background: heat(v, 0, 1) }}>
                      {v.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </Tile>
      </Tiles>
    </div>
  );
}

export function HeatmapStep() {
  return (
    <MonteCarloScreen>
      <Heatmap />
    </MonteCarloScreen>
  );
}

function Heatmap() {
  const { r, hurdle } = useResult();
  const staleClass = useStaleClass();
  const h = r.heatmap;
  return (
    <div className={staleClass}>
      <Tiles>
        <p className="type-body col-span-12 bg-canvas px-3 py-2 text-[9.5px]">{h.note}</p>
        <Tile span={12} title="IRR by growth and exit multiple" unit="rows exit multiple, columns revenue growth">
          <div className="overflow-x-auto">
            <table className="w-full border-separate border-spacing-px font-mono text-[11px]">
              <thead>
                <tr>
                  <th className="px-2 py-1 text-right font-normal text-muted">Exit</th>
                  {h.growth.map((g) => (
                    <th key={g} scope="col" className="px-2 py-1 text-right font-normal text-muted">
                      {(g * 100).toFixed(1)}%
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {h.irr.map((row, i) => (
                  <tr key={h.exit_multiple[i]}>
                    <th scope="row" className="px-2 py-1 text-right font-normal text-muted">
                      {h.exit_multiple[i].toFixed(1)}x
                    </th>
                    {row.map((v, j) => (
                      <td key={j} className="px-2 py-1.5 text-right" style={isNum(v) ? { background: heat(v, hurdle, 0.25) } : undefined}>
                        {fmtRate(v)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Tile>
      </Tiles>
    </div>
  );
}
