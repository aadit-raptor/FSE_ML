"use client";

import { useTranslations } from "next-intl";
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
import { useUnitLabel } from "@/lib/i18n/useFieldText";
import { useProvenance } from "@/lib/i18n/useProvenance";
import { DEFAULT_MONEY, MONEY } from "@/lib/money";

import { BACKTEST_PATHS, splitDealName, useBacktest } from "./BacktestProvider";

const pctNum = (v: number | null | undefined, d = 1) => fmtPct(v, d);

/**
 * Each entry assumption: the API's own key, the key of its label, and the unit
 * it is shown in. One row per assumption, so a label can't drift away from the
 * figure it names.
 *
 * i18n-keys: backtest.entry*, backtest.exitMultiple, backtest.holdingPeriod
 * i18n-keys: backtest.debtPct, backtest.seniorPct, backtest.baseRate
 * i18n-keys: backtest.mezzSpread, backtest.revenueGrowth, backtest.grossMargin
 * i18n-keys: backtest.opexPct, backtest.daPct, backtest.taxRate
 * i18n-keys: backtest.capexPct, backtest.nwcPct
 */
const ENTRY_FIELDS: { key: string; labelKey: string; unit: string }[] = [
  { key: "entry_ebitda", labelKey: "entryEbitda", unit: MONEY },
  { key: "entry_multiple", labelKey: "entryMultiple", unit: "x" },
  { key: "exit_multiple", labelKey: "exitMultiple", unit: "x" },
  { key: "holding_period", labelKey: "holdingPeriod", unit: "yr" },
  { key: "debt_pct", labelKey: "debtPct", unit: "%" },
  { key: "senior_pct", labelKey: "seniorPct", unit: "%" },
  { key: "base_rate", labelKey: "baseRate", unit: "%" },
  { key: "mezz_spread", labelKey: "mezzSpread", unit: "%" },
  { key: "revenue_growth", labelKey: "revenueGrowth", unit: "%" },
  { key: "gross_margin", labelKey: "grossMargin", unit: "%" },
  { key: "opex_pct", labelKey: "opexPct", unit: "%" },
  { key: "da_pct", labelKey: "daPct", unit: "%" },
  { key: "tax_rate", labelKey: "taxRate", unit: "%" },
  { key: "capex_pct", labelKey: "capexPct", unit: "%" },
  { key: "nwc_pct", labelKey: "nwcPct", unit: "%" },
];

function Rail() {
  const { deals, selected, select, deal } = useBacktest();
  const t = useTranslations("backtest");
  const { money } = useMoney();
  const unitLabel = useUnitLabel();
  return (
    <>
      <RailGroup title={t("groupHistoricalDeal")}>
        <div className="grid gap-px bg-line" role="radiogroup" aria-label={t("groupHistoricalDeal")}>
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
                className={`grid px-2.5 py-2 text-start ${on ? "bg-raised shadow-[inset_2px_0_0_var(--color-accent)]" : "bg-panel hover:bg-raised"}`}
              >
                <span className={on ? "type-control-value" : "type-control"}>{n.name}</span>
                <span className="font-mono text-[10px] text-muted">{t("sponsorYear", { sponsor: n.sponsor, year: n.year })}</span>
              </button>
            );
          })}
        </div>
      </RailGroup>
      {deal && (
        <>
          <RailGroup
            title={t("groupAbout")}
            aside={deal.outcome && <span className={`chip ${deal.outcome === "SUCCESS" ? "text-gain" : "text-loss"}`}>{deal.outcome.toLowerCase()}</span>}
          >
            <p className="type-body text-[9px]">{deal.description}</p>
            <p className="font-mono text-[10px] text-muted">{t("sectorGeography", { sector: deal.sector ?? "", geography: deal.geography ?? "" })}</p>
          </RailGroup>
          <RailGroup title={t("groupEntry")}>
            <dl className="grid grid-cols-[1fr_auto] gap-x-2 gap-y-1">
              {ENTRY_FIELDS.map((f) => (
                <div key={f.key} className="contents">
                  <dt className="type-input-label">{t(f.labelKey)}</dt>
                  <dd className="text-end font-mono text-[11.5px] text-ink">
                    {deal.entry[f.key]} <span className="text-[10px] text-[#56636a]">{unitLabel(f.unit, money)}</span>
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
  const t = useTranslations("backtest");
  const provenance = useProvenance();
  useEffect(() => activate(), [activate]);
  // The example deal's own currency and unit, not the open deal's
  return (
    <MoneyScope money={result?.money ?? deal?.money ?? DEFAULT_MONEY}>
      <Screen
        rail={<Rail />}
        bar={
          <>
            {status === "error" && (
              <Notice tone="loss" title={t("didntRun")} role="alert">
                {error}
              </Notice>
            )}
            {deals && (
              <Notice title={t("examplesTitle")} role="note">
                {provenance.backtest(deals.map((d) => splitDealName(d.name).year))}
                {t("examplesAfter")}
              </Notice>
            )}
          </>
        }
      >
        {result ? (
          <div className={status === "running" ? "opacity-60 transition-opacity" : ""}>{children}</div>
        ) : status === "error" ? (
          <EmptyState title={t("noBacktest")}>{t("noBacktestBody")}</EmptyState>
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
  const t = useTranslations("backtest");
  if (!r) return null;
  const lastPred = r.predicted_ebitda.at(-1);
  const lastAct = r.years.at(-1)?.actual_ebitda;
  return (
    <>
      <Kpi title={t("kpiActualIrr")} value={pctNum(r.actual_irr)} sub={t("moicSub", { moic: fmtMultiple(r.actual_moic) })} lead />
      <Kpi title={t("kpiPredictedIrr")} value={pctNum(r.predicted_irr_mean)} sub={t("meanOfPaths", { count: fmtCount(BACKTEST_PATHS) })} />
      <Kpi title={t("kpiPredictedRange")} value={pctNum(r.predicted_irr_p5)} sub={t("toP95", { p95: pctNum(r.predicted_irr_p95) })} />
      <Kpi title={t("kpiActualPercentile")} value={fmtNumber(r.actual_percentile, 1)} sub={t("ofPredicted")} />
      <Kpi
        title={t("kpiExitEbitda")}
        value={fmtMoney(lastAct)}
        sub={t("predictedSub", { value: fmtMoney(lastPred), money: mu })}
        tone={isNum(lastAct) && isNum(lastPred) ? (lastAct >= lastPred ? "gain" : "loss") : undefined}
      />
      <Kpi
        title={t("kpiExitEquity")}
        value={fmtMoney(r.actual_exit_equity)}
        sub={t("predictedSub", { value: fmtMoney(r.predicted_exit_equity), money: mu })}
      />
    </>
  );
}

/** Historical years if the deal has them, else the model's own year numbers. */
function useYears(): string[] {
  const { result: r, deal } = useBacktest();
  const fiscal = useTranslations("fiscal");
  if (!r || !deal) return [];
  return deal.actual_years.length ? deal.actual_years.map(String) : r.years.map((y) => fiscal("yearShort", { number: String(y.year_index) }));
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
  const t = useTranslations("backtest");
  const years = useYears();
  if (!r || !deal) return null;
  return (
    <Tiles>
      <Headline />
      <Tile span={7} title={t("tileEbitda")} unit={mu}>
        <GroupedBars
          label={t("chartEbitda", { deal: splitDealName(deal.name).name })}
          categories={years}
          format={fmtMoney}
          series={[
            { name: t("seriesPredicted"), values: r.predicted_ebitda, color: "var(--color-neutral-bar)" },
            { name: t("seriesActual"), values: r.years.map((y) => y.actual_ebitda), color: "var(--color-accent)", labelled: true },
          ]}
        />
      </Tile>
      <Tile span={5} title={t("tileWhereLanded")} unit={t("predictedDistribution")}>
        <Histogram
          label={t("chartPredictedDistribution")}
          edges={r.irr_histogram.edges}
          density={r.irr_histogram.density}
          format={(v) => fmtPct(v, 0)}
          tickCount={5}
          markers={[
            { value: r.predicted_irr_p5, label: t("markerP5", { value: pctNum(r.predicted_irr_p5) }) },
            { value: r.actual_irr, label: t("markerActual", { value: pctNum(r.actual_irr) }), tone: "accent" },
            { value: r.predicted_irr_p95, label: t("markerP95", { value: pctNum(r.predicted_irr_p95) }) },
          ]}
        />
      </Tile>
    </Tiles>
  );
}

/** i18n-keys: backtest.attribution* */
const ATTRIBUTION_KEY: Record<string, string> = {
  exit_ebitda: "attributionExitEbitda",
  exit_multiple: "attributionExitMultiple",
  net_debt: "attributionNetDebt",
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
  const t = useTranslations("backtest");
  const years = useYears();
  if (!r || !deal) return null;
  return (
    <Tiles>
      <Headline />
      <Tile span={6} title={t("tileWhereMissed")} unit={t("ofExitEquity", { money: mu })}>
        <DivergingBars
          label={t("chartAttribution")}
          format={fmtDelta}
          rows={Object.entries(r.attribution).map(([k, v]) => ({ label: ATTRIBUTION_KEY[k] ? t(ATTRIBUTION_KEY[k]) : k, value: v }))}
        />
        <p className="type-body text-[9px]">
          {t("attributionNote", {
            total: fmtDelta(Object.values(r.attribution).reduce((a, b) => a + b, 0)),
            money: mu,
            predicted: fmtMultiple(deal.entry.exit_multiple, 1),
            actual: fmtMultiple(r.actual_exit_multiple, 1),
            netDebt: fmtMoney(r.predicted_net_debt_at_exit),
          })}
        </p>
      </Tile>
      <Tile span={6} title={t("tileEbitdaMargin")} unit={t("actualVsPredicted")}>
        <LineChart
          label={t("chartEbitdaMargin")}
          xLabels={years}
          yFormat={(v) => fmtPct(v, 0)}
          lines={[
            { name: t("seriesActual"), values: r.actual_ebitda_margin, color: "var(--color-accent)" },
            { name: t("seriesPredicted"), values: years.map(() => r.predicted_ebitda_margin), color: "var(--color-muted)", dashed: true },
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
  const t = useTranslations("backtest");
  const x = useTranslations("export");
  const years = useYears();
  if (!r || !deal) return null;
  return (
    <Tiles>
      <Headline />
      <Tile
        span={12}
        title={t("tileYearByYear")}
        unit={mu}
        action={
          <DownloadButton
            onDownload={() =>
              downloadWorkbook(
                x("fileBacktest"),
                [
                  sheet(
                    x("sheetYearByYear"),
                    [
                      x("colYear"),
                      x("colPredictedEbitda"),
                      x("colActualEbitda"),
                      x("colVariance"),
                      x("colActualRevenue"),
                      x("colActualFcf"),
                      x("colActualDebt"),
                    ],
                    r.years.map((y, i) => [
                      years[i] ?? y.year_index,
                      y.predicted_ebitda,
                      y.actual_ebitda,
                      y.ebitda_variance,
                      y.actual_revenue,
                      y.actual_fcf,
                      y.actual_total_debt,
                    ]),
                    { columns: ["text", "money", "money", "money", "money", "money", "money"] },
                  ),
                  sheet(
                    x("sheetReturns"),
                    [x("colMetric"), x("colPredicted"), x("colActual")],
                    [
                      [x("rowIrr"), fraction(r.predicted_irr_mean, 100), fraction(r.actual_irr, 100)],
                      [x("rowMoic"), r.predicted_moic, r.actual_moic],
                      [x("colEntryEquity", { money: mu }), r.predicted_equity_entry, r.actual_equity_entry],
                      [x("colExitEquity", { money: mu }), r.predicted_exit_equity, r.actual_exit_equity],
                    ],
                    { rows: ["percent", "multiple", "money", "money"] },
                  ),
                  sheet(
                    x("sheetAttribution"),
                    [x("colPart"), mu],
                    Object.entries(r.attribution).map(([k, v]) => [ATTRIBUTION_KEY[k] ? t(ATTRIBUTION_KEY[k]) : k, v]),
                    { columns: ["text", "money"] },
                  ),
                ],
                money,
              )
            }
          />
        }
      >
        <DataTable
          caption={t("chartYearByYear")}
          columns={years}
          rows={[
            { label: t("rowEbitdaPredicted"), values: r.years.map((y) => y.predicted_ebitda) },
            { label: t("rowEbitdaActual"), values: r.years.map((y) => y.actual_ebitda), total: true },
            { label: t("rowEbitdaVariance"), values: r.years.map((y) => y.ebitda_variance) },
            { label: t("rowRevenueActual"), values: r.years.map((y) => y.actual_revenue) },
            { label: t("rowFcfActual"), values: r.years.map((y) => y.actual_fcf) },
            { label: t("rowDebtActual"), values: r.years.map((y) => y.actual_total_debt) },
          ]}
        />
      </Tile>
      <Tile span={6} title={t("tileEquity")} unit={mu}>
        <DataTable
          caption={t("chartEquity")}
          columns={[t("colPredicted"), t("colActual")]}
          rows={[
            { label: t("rowEquityEntry"), values: [r.predicted_equity_entry, r.actual_equity_entry] },
            { label: t("rowEquityExit"), values: [r.predicted_exit_equity, r.actual_exit_equity], total: true },
          ]}
        />
      </Tile>
      <Tile span={6} title={t("tileReturns")} unit={t("returnsUnit")}>
        <table className="w-full border-collapse font-mono text-[11px]">
          <thead>
            <tr className="text-muted">
              <th />
              <th scope="col" className="border-b border-grid px-2 py-1 text-end font-normal">
                {t("colPredicted")}
              </th>
              <th scope="col" className="border-b border-grid px-2 py-1 text-end font-normal">
                {t("colActual")}
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row" className="type-input-label border-b border-grid px-2 py-1 text-start text-[9px] font-normal text-soft">
                {t("rowIrr")}
              </th>
              <td className="border-b border-grid px-2 py-1 text-end">{pctNum(r.predicted_irr_mean)}</td>
              <td className="border-b border-grid px-2 py-1 text-end text-bright">{pctNum(r.actual_irr)}</td>
            </tr>
            <tr>
              <th scope="row" className="type-input-label border-b border-grid px-2 py-1 text-start text-[9px] font-normal text-soft">
                {t("rowMoic")}
              </th>
              <td className="border-b border-grid px-2 py-1 text-end">{fmtMultiple(r.predicted_moic)}</td>
              <td className="border-b border-grid px-2 py-1 text-end text-bright">{fmtMultiple(r.actual_moic)}</td>
            </tr>
          </tbody>
        </table>
      </Tile>
    </Tiles>
  );
}
