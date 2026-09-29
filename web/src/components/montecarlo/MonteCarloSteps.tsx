"use client";

import { useTranslations } from "next-intl";
import { useState } from "react";

import { DivergingBars, RangeRows, Scatter } from "@/components/charts/Bars";
import { heat } from "@/components/charts/HeatTable";
import { Histogram } from "@/components/charts/Histogram";
import { LineChart } from "@/components/charts/LineChart";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { useMoney } from "@/components/ui/MoneyScope";
import { apiInputs } from "@/lib/deal/fields";
import { downloadMonteCarloSample, downloadWorkbook, fraction, sheet } from "@/lib/export";
import { fmtCount, fmtMultiple, fmtNumber, fmtRate, isNum } from "@/lib/format";
import { useEngineLabel } from "@/lib/i18n/useEngineText";

import { MacroRegime } from "./MonteCarloML";
import { SCENARIOS, useMonteCarlo } from "./MonteCarloProvider";
import { MonteCarloScreen, useStaleClass } from "./MonteCarloScreen";

const pct0 = (v: number) => fmtRate(v, 0);

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
  const t = useTranslations("montecarlo");
  const s = r.summary;
  const staleClass = useStaleClass();
  const p = r.params as Record<string, number>;
  const count = (share: number | null | undefined, n: number) => (isNum(share) ? fmtCount(share * n) : fmtCount(null));
  const assumptions: [string, string][] = [
    [t("assumptionGrowth"), t("plusMinus", { mean: fmtRate(p.growth_mean), std: fmtRate(p.growth_std) })],
    [t("assumptionExit"), t("plusMinus", { mean: fmtMultiple(p.exit_mean, 1), std: fmtMultiple(p.exit_std, 2) })],
    [t("assumptionRate"), t("plusMinus", { mean: fmtRate(p.interest_mean, 2), std: fmtRate(p.interest_std, 2) })],
    [t("assumptionMargin"), t("plusMinus", { mean: fmtRate(p.gross_margin_mean), std: fmtRate(p.gross_margin_std) })],
    [t("assumptionDebt"), t("twoValues", { first: fmtRate(p.debt_pct), second: fmtRate(p.senior_pct) })],
    [t("assumptionFees"), t("twoValues", { first: fmtRate(p.transaction_fees_pct, 2), second: fmtRate(p.financing_fees_pct, 2) })],
    [
      t("assumptionCosts"),
      t("fourValues", { first: fmtRate(p.opex_pct), second: fmtRate(p.da_pct), third: fmtRate(p.capex_pct), fourth: fmtRate(p.nwc_pct) }),
    ],
    [t("assumptionTax"), fmtRate(p.tax_rate)],
  ];
  const presetName = r.scenario ? t(SCENARIOS.find((x) => x.id === r.scenario)?.labelKey ?? "noPreset") : t("noPreset");

  return (
    <div className={staleClass}>
      <Tiles>
        <Kpi title={t("kpiMeanIrr")} value={fmtRate(s.mean_irr)} sub={t("kpiMedianSub", { median: fmtRate(s.median_irr) })} lead />
        <Kpi
          title={t("kpiAboveHurdle", { hurdle: pct0(hurdle) })}
          value={fmtRate(s.p_above_hurdle)}
          sub={t("kpiAboveHurdleSub", { count: count(s.p_above_hurdle, r.n), total: fmtCount(r.n) })}
        />
        <Kpi title={t("kpiP5")} value={fmtRate(s.p5_irr)} sub={t("oneInTwentyBelow")} />
        <Kpi title={t("kpiP95")} value={fmtRate(s.p95_irr)} sub={t("oneInTwentyAbove")} />
        <Kpi
          title={t("kpiWipeout")}
          value={fmtRate(s.wipeout_rate, 3)}
          sub={t("wipeoutSub", { count: count(s.wipeout_rate, r.n) })}
          tone={(s.wipeout_rate ?? 0) > 0.05 ? "loss" : undefined}
        />
        <Kpi
          title={t("kpiRunTime")}
          value={t("runTimeValue", { ms: Math.round(r.elapsed_ms) })}
          sub={t("runTimeSub", {
            seed: ranFor.seed === null ? t("seedRandom") : t("seedFixed", { seed: String(ranFor.seed) }),
            preset: presetName,
          })}
        />

        <Tile
          span={7}
          title={t("tileIrrDistribution")}
          unit={t("pathsUnit", { count: fmtCount(r.n) })}
          action={
            <DownloadButton
              label={t("sample10k")}
              title={ranFor.seed === null ? t("sampleRandomTitle") : t("sampleSeededTitle")}
              onDownload={() =>
                downloadMonteCarloSample({
                  mc: { ...ranFor.sim, ebitda: ranFor.deal.ebitda, entry_mult: ranFor.deal.entry_mult, hold: ranFor.deal.hold },
                  deal: apiInputs(ranFor.deal),
                  settings: ranFor.settings,
                  scenario: ranFor.scenario,
                  seed: ranFor.seed,
                })
              }
            />
          }
        >
          <Histogram
            label={t("chartIrrDistribution")}
            edges={r.irr_histogram.edges}
            density={r.irr_histogram.density}
            highlightFrom={hurdle}
            format={pct0}
            markers={[
              ...(isNum(s.p5_irr) ? [{ value: s.p5_irr, label: t("markerP5", { value: fmtRate(s.p5_irr) }) }] : []),
              { value: hurdle, label: t("markerHurdle", { value: pct0(hurdle) }), tone: "attention" as const, dashed: true },
              ...(isNum(s.p95_irr) ? [{ value: s.p95_irr, label: t("markerP95", { value: fmtRate(s.p95_irr) }) }] : []),
            ]}
          />
        </Tile>
        <Tile span={5} title={t("tileIrrByPercentile")} unit={t("shareBelow")}>
          <LineChart
            label={t("chartIrrPercentile")}
            xLabels={r.irr_cdf.percentiles.map((q) => t("percentile", { value: fmtNumber(q, 0) }))}
            xTickEvery={25}
            lines={[{ name: "", values: r.irr_cdf.values, color: "var(--color-accent)" }]}
            hlines={[{ value: hurdle, label: t("markerHurdle", { value: pct0(hurdle) }), tone: "attention" }]}
            yFormat={pct0}
            endLabels={false}
          />
        </Tile>
        <Tile span={6} title={t("tileMoicDistribution")} unit="x">
          <Histogram
            label={t("chartMoicDistribution")}
            edges={r.moic_histogram.edges}
            density={r.moic_histogram.density}
            highlightFrom={1}
            format={(v) => fmtMultiple(v, 1)}
            markers={[{ value: 1, label: t("markerMoneyBack"), tone: "attention", dashed: true }]}
          />
        </Tile>
        <Tile span={6} title={t("tileWhatSimulated")} unit={t("meanPlusMinus")}>
          <table className="w-full border-collapse font-mono text-[11px]">
            <tbody>
              {assumptions.map(([k, v]) => (
                <tr key={k}>
                  <th scope="row" className="type-input-label border-b border-grid py-1.5 text-start text-[9px] font-normal text-soft">
                    {k}
                  </th>
                  <td className="border-b border-grid py-1.5 text-end text-ink">{v}</td>
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
  const { money } = useMoney();
  const { scen, hurdle, r } = useResult();
  const t = useTranslations("montecarlo");
  const x = useTranslations("export");
  const staleClass = useStaleClass();
  const rows = SCENARIOS.map((sc) => ({ ...sc, label: t(sc.labelKey), st: scen.scenarios[sc.id] })).filter((y) => y.st);
  return (
    <div className={staleClass}>
      <Tiles>
        {rows.map(({ id, label, st }) => (
          <Kpi
            key={id}
            title={label}
            value={fmtRate(st.mean_irr)}
            sub={t("clearHurdleSub", { share: fmtRate(st.p_above_hurdle) })}
            lead={id === "base"}
            tone={(st.mean_irr ?? 0) < 0 ? "loss" : undefined}
          />
        ))}
        <Kpi title={t("kpiPathsEach")} value={fmtCount(r.n)} sub={t("sameSeed")} />
        <Kpi title={t("kpiHurdle")} value={pct0(scen.hurdle)} sub={t("fromTheRail")} />
        <MacroRegime />

        <Tile span={7} title={t("tileIrrRange")} unit={t("irrRangeUnit")}>
          <RangeRows
            label={t("chartIrrRange")}
            format={pct0}
            threshold={{ value: hurdle, label: t("kpiHurdle") }}
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
          title={t("tileScenarioTable")}
          unit={t("irrUnlessNoted")}
          action={
            <DownloadButton
              onDownload={() =>
                downloadWorkbook(
                  x("fileScenarios"),
                  [
                    sheet(
                      x("sheetScenarios"),
                      [
                        x("colScenario"),
                        x("colMeanIrr"),
                        x("colMedianIrr"),
                        x("colP5Irr"),
                        x("colP95Irr"),
                        x("colAboveHurdle"),
                        x("colWipeout"),
                        x("colMoicP50"),
                      ],
                      rows.map(({ label, st }) => [
                        label,
                        ...[st.mean_irr, st.median_irr, st.p5_irr, st.p95_irr, st.p_above_hurdle, st.wipeout_rate].map((v) => fraction(v)),
                        st.moic_box.p50,
                      ]),
                      { columns: ["text", "percent", "percent", "percent", "percent", "percent", "percent", "multiple"] },
                    ),
                  ],
                  money,
                )
              }
            />
          }
        >
          <table className="w-full border-collapse font-mono text-[11px]">
            <thead>
              <tr className="text-muted">
                <th />
                {[t("colMean"), t("kpiP5"), t("kpiP95"), t("colWipeout"), t("colMoicP50")].map((h) => (
                  <th key={h} scope="col" className="border-b border-grid px-1.5 py-1 text-end font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map(({ id, label, st }) => (
                <tr key={id} className={id === "base" ? "text-bright" : "text-ink"}>
                  <th scope="row" className="type-input-label border-b border-grid py-1.5 text-start text-[9px] font-normal text-soft">
                    {label}
                  </th>
                  {[st.mean_irr, st.p5_irr, st.p95_irr].map((v, i) => (
                    <td key={i} className={`border-b border-grid px-1.5 py-1.5 text-end ${(v ?? 0) < 0 ? "text-loss" : ""}`}>
                      {fmtRate(v)}
                    </td>
                  ))}
                  <td className="border-b border-grid px-1.5 py-1.5 text-end">{fmtRate(st.wipeout_rate, 2)}</td>
                  <td className="border-b border-grid px-1.5 py-1.5 text-end">{fmtMultiple(st.moic_box.p50)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="type-body text-[9px]">{t("presetsNote")}</p>
        </Tile>
      </Tiles>
    </div>
  );
}

/** The simulation's own column names (simulation/vectorized_simulation.py), shown translated. */
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
  const t = useTranslations("montecarlo");
  const driverName = useEngineLabel("driver");
  const staleClass = useStaleClass();
  const [driver, setDriver] = useState<(typeof DRIVERS)[number]>("Growth");
  const xs = (r.scatter[driver] ?? []).map((v) => v ?? NaN);
  const ys = (r.scatter.IRR ?? []).map((v) => (v ?? NaN) * 100);
  const fit = r.driver_fits[driver];
  const corr = r.correlations as { labels: string[]; matrix: number[][] };
  const xFormat = driver === "Exit Multiple" ? (v: number) => fmtMultiple(v, 1) : (v: number) => fmtRate(v, 1);

  return (
    <div className={staleClass}>
      <Tiles>
        <Tile span={5} title={t("tileWhatMovesIrr")} unit={t("rankCorrelation")}>
          <DivergingBars
            label={t("chartSpearman")}
            domain={1}
            format={(v) => fmtNumber(v, 2, v > 0)}
            rows={r.drivers.map((d) => ({ label: driverName(d.driver), value: d.spearman_rho ?? 0 }))}
          />
          <p className="type-body text-[9px]">{t("spearmanNote")}</p>
        </Tile>
        <Tile
          span={7}
          title={t("tileIrrAgainst", { driver: driverName(driver).toLowerCase() })}
          aside={
            <span className="flex gap-px bg-line" role="radiogroup" aria-label={t("driver")}>
              {DRIVERS.map((d) => (
                <button
                  key={d}
                  type="button"
                  role="radio"
                  aria-checked={d === driver}
                  onClick={() => setDriver(d)}
                  className={`type-action-secondary px-2 py-1 text-[8.5px] ${d === driver ? "bg-raised text-bright" : "bg-canvas text-muted hover:text-ink"}`}
                >
                  {driverName(d)}
                </button>
              ))}
            </span>
          }
        >
          <Scatter
            label={t("chartScatter", { driver: driverName(driver) })}
            xs={xs}
            ys={ys}
            fit={isNum(fit?.slope) && isNum(fit?.intercept) ? { slope: fit.slope, intercept: fit.intercept } : undefined}
            xFormat={xFormat}
            yFormat={(v) => fmtRate(v / 100, 0)}
          />
          <p className="type-body text-[9px]">
            {t("scatterNote", {
              count: fmtCount(xs.length),
              fit: isNum(fit?.r) ? t("scatterFit", { r: fmtNumber(fit.r, 2) }) : "",
            })}
          </p>
        </Tile>
        <Tile span={12} title={t("tileCorrelations")} unit={t("pearson")}>
          <table className="w-full border-separate border-spacing-px font-mono text-[11px]">
            <thead>
              <tr>
                <th />
                {corr.labels.map((l) => (
                  <th key={l} scope="col" className="px-2 py-1 text-end font-normal text-muted">
                    {driverName(l)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {corr.matrix.map((row, i) => (
                <tr key={corr.labels[i]}>
                  <th scope="row" className="type-input-label px-2 py-1 text-start text-[9px] font-normal text-soft">
                    {driverName(corr.labels[i])}
                  </th>
                  {row.map((v, j) => (
                    <td key={j} className="px-2 py-1 text-end" style={i === j ? { color: "var(--color-dim)" } : { background: heat(v, 0, 1) }}>
                      {fmtNumber(v, 2)}
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
  const t = useTranslations("montecarlo");
  const staleClass = useStaleClass();
  const h = r.heatmap;
  return (
    <div className={staleClass}>
      <Tiles>
        <p className="type-body col-span-12 bg-canvas px-3 py-2 text-[9.5px]">{h.note}</p>
        <Tile span={12} title={t("tileHeatmap")} unit={t("heatmapUnit")}>
          <div className="overflow-x-auto">
            <table className="w-full border-separate border-spacing-px font-mono text-[11px]">
              <thead>
                <tr>
                  <th className="px-2 py-1 text-end font-normal text-muted">{t("colExit")}</th>
                  {h.growth.map((g) => (
                    <th key={g} scope="col" className="px-2 py-1 text-end font-normal text-muted">
                      {fmtRate(g, 1)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {h.irr.map((row, i) => (
                  <tr key={h.exit_multiple[i]}>
                    <th scope="row" className="px-2 py-1 text-end font-normal text-muted">
                      {fmtMultiple(h.exit_multiple[i], 1)}
                    </th>
                    {row.map((v, j) => (
                      <td key={j} className="px-2 py-1.5 text-end" style={isNum(v) ? { background: heat(v, hurdle, 0.25) } : undefined}>
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
