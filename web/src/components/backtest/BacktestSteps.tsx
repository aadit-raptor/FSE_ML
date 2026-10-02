"use client";

import Link from "next/link";
import { useTranslations } from "next-intl";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { DivergingBars, GroupedBars } from "@/components/charts/Bars";
import { DataTable } from "@/components/charts/DataTable";
import { Histogram } from "@/components/charts/Histogram";
import { LineChart } from "@/components/charts/LineChart";
import { CellInput } from "@/components/ui/CellInput";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { MoneyScope, useMoney } from "@/components/ui/MoneyScope";
import { EmptyState, LoadingTiles, Notice, RailGroup, Screen, SecondaryButton, Switch } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import {
  ACTUAL_LINES, type ActualsDraft, actualsCsv, blankExit, completeExit, EXIT_FIELDS, type ExitKey, type LineKey,
  parseActualsCsv, resizeYears,
} from "@/lib/backtest/actuals";
import { downloadText, downloadWorkbook, fraction, sheet } from "@/lib/export";
import { fmtCount, fmtDelta, fmtMoney, fmtMultiple, fmtNumber, fmtRate, isNum } from "@/lib/format";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";
import { useProvenance } from "@/lib/i18n/useProvenance";
import { DEFAULT_MONEY } from "@/lib/money";
import { stepHref } from "@/lib/nav";

import { BACKTEST_PATHS, type PlanActualResult, sameRef, useBacktest } from "./BacktestProvider";

/** "Burger King (3G Capital, 2010)" -> name, sponsor, year */
export function splitDealName(full: string) {
  const m = full.match(/^(.*?)\s*\((.*),\s*(\d{4})\)$/);
  return m ? { name: m[1], sponsor: m[2], year: m[3] } : { name: full, sponsor: "", year: "" };
}

const RADIO = "grid px-2.5 py-2 text-start";
const radioClass = (on: boolean) => `${RADIO} ${on ? "bg-raised shadow-[inset_2px_0_0_var(--color-accent)]" : "bg-panel hover:bg-raised"}`;

function Rail() {
  const { deals, examples, selected, select, plan, actuals, setActuals, saveState, status } = useBacktest();
  const t = useTranslations("backtest");
  const { label: mu } = useMoney();
  return (
    <>
      <RailGroup title={t("groupSavedDeals")}>
        {deals?.length ? (
          <div className="grid gap-px bg-line" role="radiogroup" aria-label={t("groupSavedDeals")}>
            {deals.map((d) => {
              const on = sameRef(selected, { kind: "deal", id: d.id });
              return (
                <button key={d.id} type="button" role="radio" aria-checked={on} onClick={() => select({ kind: "deal", id: d.id })} className={radioClass(on)}>
                  <span className={`truncate ${on ? "type-control-value" : "type-control"}`}>{d.name}</span>
                </button>
              );
            })}
          </div>
        ) : (
          <p className="type-body text-[9px]">
            {deals === null && status === "loading" ? t("loadingDeals") : t("noSavedDeals")}{" "}
            <Link href={stepHref("deal", "saved")} className="text-accent underline">
              {t("toSavedDeals")}
            </Link>
          </p>
        )}
      </RailGroup>
      {examples.length > 0 && (
        <RailGroup title={t("groupExamples")}>
          <div className="grid gap-px bg-line" role="radiogroup" aria-label={t("groupExamples")}>
            {examples.map((x) => {
              const n = splitDealName(x.name);
              const on = sameRef(selected, { kind: "example", name: x.name });
              return (
                <button key={x.name} type="button" role="radio" aria-checked={on} onClick={() => select({ kind: "example", name: x.name })} className={radioClass(on)}>
                  <span className={on ? "type-control-value" : "type-control"}>{n.name}</span>
                  <span className="font-mono text-[10px] text-muted">{t("sponsorYear", { sponsor: n.sponsor, year: n.year })}</span>
                </button>
              );
            })}
          </div>
        </RailGroup>
      )}
      {plan?.example && (
        <RailGroup
          title={t("groupAbout")}
          aside={<span className={`chip ${plan.example.outcome === "SUCCESS" ? "text-gain" : "text-loss"}`}>{plan.example.outcome.toLowerCase()}</span>}
        >
          <p className="type-body text-[9px]">{plan.example.description}</p>
          <p className="font-mono text-[10px] text-muted">{t("sectorGeography", { sector: plan.example.sector, geography: plan.example.geography })}</p>
        </RailGroup>
      )}
      {plan && actuals && (
        <RailGroup title={t("groupActuals")} aside={<SaveChip state={saveState} />}>
          <label className="grid gap-1">
            <span className="type-input-label">{t("yearsWithResults")}</span>
            <select
              value={actuals.years.length}
              onChange={(e) => setActuals(resizeYears(actuals, Number(e.target.value)))}
              className="border border-line bg-field px-1.5 py-1 font-mono text-[11px] text-ink"
            >
              {Array.from({ length: plan.hold }, (_, i) => i + 1).map((n) => (
                <option key={n} value={n}>
                  {t("yearsOf", { count: n, hold: String(plan.hold) })}
                </option>
              ))}
            </select>
          </label>
          <p className="type-body text-[9px]">{t("planIn", { money: mu, hold: String(plan.hold) })}</p>
        </RailGroup>
      )}
    </>
  );
}

/** i18n-keys: backtest.save* */
function SaveChip({ state }: { state: string }) {
  const t = useTranslations("backtest");
  const key = `save${state[0].toUpperCase()}${state.slice(1)}`;
  return (
    <span className={`chip ${state === "error" ? "text-loss" : "text-muted"}`} data-save-state={state}>
      {t(key)}
    </span>
  );
}

/** The plan's year labels: its fiscal years when it names them, else Y1, Y2 ... */
function useYears(n?: number): string[] {
  const { plan, actuals } = useBacktest();
  const fiscal = useFiscalLabels();
  if (!plan) return [];
  return fiscal.deal(n ?? actuals?.years.length ?? plan.hold, { endMonth: plan.inputs.fiscal_year_end_month ?? 12, year: plan.inputs.first_fiscal_year ?? null });
}

function BacktestScreen({ children, needsResult = true }: { children: ReactNode; needsResult?: boolean }) {
  const { activate, status, result, error, plan, selected, ready, examples } = useBacktest();
  const t = useTranslations("backtest");
  const provenance = useProvenance();
  useEffect(() => activate(), [activate]);
  const noPlan = ready && !selected;
  let body: ReactNode;
  if (noPlan) {
    body = (
      <EmptyState
        title={t("noPlan")}
        action={
          <Link href={stepHref("deal", "saved")} className="type-action-secondary px-2.5 py-1.5 text-muted shadow-[inset_0_0_0_1px_#2a343a] hover:text-ink">
            {t("toSavedDeals")}
          </Link>
        }
      >
        {t("noPlanBody")}
      </EmptyState>
    );
  } else if (!plan) {
    body = status === "error" ? <EmptyState title={t("noBacktest")}>{t("noBacktestBody")}</EmptyState> : <LoadingTiles />;
  } else if (!needsResult) {
    body = children;
  } else if (result) {
    body = <div className={status === "running" ? "opacity-60 transition-opacity" : ""}>{children}</div>;
  } else if (status === "idle") {
    body = (
      <EmptyState
        title={t("noActuals")}
        action={
          <Link href={stepHref("backtest", "actuals")} className="type-action-secondary px-2.5 py-1.5 text-muted shadow-[inset_0_0_0_1px_#2a343a] hover:text-ink">
            {t("toActuals")}
          </Link>
        }
      >
        {t("noActualsBody")}
      </EmptyState>
    );
  } else if (status === "error") {
    body = <EmptyState title={t("noBacktest")}>{error}</EmptyState>;
  } else {
    body = <LoadingTiles />;
  }
  return (
    <MoneyScope money={result?.money ?? plan?.money ?? DEFAULT_MONEY}>
      <Screen
        rail={<Rail />}
        bar={
          <>
            {status === "error" && (
              <Notice tone="loss" title={t("didntRun")} role="alert">
                {error}
              </Notice>
            )}
            {plan?.example && (
              <Notice title={t("examplesTitle")} role="note">
                {provenance.backtest(examples.map((d) => splitDealName(d.name).year))}
                {t("examplesAfter")}
              </Notice>
            )}
          </>
        }
      >
        {body}
      </Screen>
    </MoneyScope>
  );
}

// ---------------------------------------------------------------------------
// Plan and actuals: the editor
// ---------------------------------------------------------------------------
export function ActualsStep() {
  return (
    <BacktestScreen needsResult={false}>
      <ActualsEditor />
    </BacktestScreen>
  );
}

/** i18n-keys: backtest.csv* */
function ActualsEditor() {
  const { plan, actuals, setActuals, saveState } = useBacktest();
  const { label: mu } = useMoney();
  const t = useTranslations("backtest");
  const years = useYears();
  const file = useRef<HTMLInputElement>(null);
  const [upload, setUpload] = useState<{ tone: "info" | "loss"; text: string } | null>(null);
  if (!plan || !actuals) return <LoadingTiles />;

  const setYear = (i: number, key: LineKey, v: number | null) =>
    setActuals({ ...actuals, years: actuals.years.map((y, j) => (j === i ? { ...y, [key]: v } : y)) });
  const setExit = (key: ExitKey, v: number | null) => setActuals({ ...actuals, exit: { ...(actuals.exit ?? blankExit()), [key]: v } });
  const exitMissing = actuals.exit !== null && !completeExit(actuals.exit);

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    const parsed = parseActualsCsv(await f.text());
    if (file.current) file.current.value = "";
    if (!parsed.ok) return setUpload({ tone: "loss", text: t(`csv${parsed.problem[0].toUpperCase()}${parsed.problem.slice(1)}`, { line: parsed.where ?? "", max: String(plan.hold) }) });
    if (parsed.years.length > plan.hold) return setUpload({ tone: "loss", text: t("csvTooManyYears", { max: String(plan.hold) }) });
    const next: ActualsDraft = { ...actuals, years: parsed.years, exit: parsed.exit };
    setActuals(next);
    setUpload({
      tone: "info",
      text: parsed.ignored.length
        ? t("csvLoadedIgnored", { count: parsed.years.length, ignored: parsed.ignored.join(", ") })
        : t("csvLoaded", { count: parsed.years.length }),
    });
  };

  return (
    <>
      {actuals.money.currency !== plan.money.currency && (
        <Notice
          title={t("currencyTitle")}
          role="alert"
          actions={<SecondaryButton onClick={() => setActuals({ ...actuals, money: plan.money })}>{t("currencyAdopt", { currency: plan.money.currency })}</SecondaryButton>}
        >
          {t("currencyBody", { actual: actuals.money.currency, deal: plan.money.currency })}
        </Notice>
      )}
      {saveState === "error" && <Notice tone="loss" title={t("saveFailedTitle")} role="alert">{t("saveFailedBody")}</Notice>}
      {upload && (
        <Notice tone={upload.tone} title={t("csvTitle")} role={upload.tone === "loss" ? "alert" : "status"}>
          {upload.text}
        </Notice>
      )}
      <Tiles>
        <Tile
          span={12}
          title={t("tileActuals", { plan: plan.name })}
          unit={mu}
          action={
            <span className="flex items-center gap-2">
              <input ref={file} type="file" accept=".csv,text/csv,text/plain" className="sr-only" aria-label={t("csvUpload")} onChange={(e) => void onFile(e.target.files?.[0])} />
              <SecondaryButton onClick={() => file.current?.click()}>{t("csvUpload")}</SecondaryButton>
              <DownloadButton label={t("csvDownload")} onDownload={async () => downloadText(t("csvFile"), actualsCsv(actuals, years))} />
            </span>
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full border-collapse font-mono text-[11px]">
              <caption className="sr-only">{t("tileActuals", { plan: plan.name })}</caption>
              <thead>
                <tr>
                  <th scope="col" className="w-[22%]" />
                  {years.map((y) => (
                    <th key={y} scope="col" className="border-b border-grid px-1 py-1 text-end font-normal text-muted">
                      {y}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ACTUAL_LINES.map((line) => (
                  <tr key={line.key}>
                    <th scope="row" className="type-input-label border-b border-grid px-2 py-1 text-start text-[9px] font-normal text-soft">
                      {t(line.labelKey)}
                    </th>
                    {actuals.years.map((y, i) => (
                      <td key={i} className="border-b border-grid px-1 py-0.5">
                        <CellInput
                          label={t("cellLabel", { line: t(line.labelKey), year: years[i] ?? String(i + 1) })}
                          value={y[line.key] ?? NaN}
                          onCommit={(v) => setYear(i, line.key, v)}
                          onClear={() => setYear(i, line.key, null)}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="type-body text-[9px]">{t("actualsNote")}</p>
        </Tile>
        <Tile span={6} title={t("tileExit")} unit={mu}>
          <Switch
            checked={actuals.exit !== null}
            onChange={(on) => setActuals({ ...actuals, exit: on ? blankExit() : null })}
            label={t("exitSold", { year: years.at(-1) ?? "" })}
          />
          {actuals.exit && (
            <dl className="grid grid-cols-[1fr_120px] items-center gap-x-3 gap-y-1">
              {EXIT_FIELDS.map((f) => (
                <div key={f.key} className="contents">
                  <dt className="type-input-label">{t(f.labelKey)}</dt>
                  <dd>
                    <CellInput
                      label={t(f.labelKey)}
                      decimals={f.key === "moic" ? 2 : 1}
                      value={actuals.exit![f.key] ?? NaN}
                      onCommit={(v) => setExit(f.key, v)}
                      onClear={() => setExit(f.key, null)}
                    />
                  </dd>
                </div>
              ))}
            </dl>
          )}
          {exitMissing && <p className="type-body text-[9px] text-attention">{t("exitIncomplete")}</p>}
          <p className="type-body text-[9px]">{t("exitNote")}</p>
        </Tile>
        <Tile span={6} title={t("tileCsv")}>
          <p className="type-body text-[9px]">{t("csvHelp")}</p>
          <pre className="overflow-x-auto bg-panel px-2 py-1.5 font-mono text-[10px] text-soft">{actualsCsv({ ...actuals, years: actuals.years.slice(0, 2).map(() => ({ revenue: 100, ebitda: 20, net_income: 8, fcf: 9, total_debt: 60 })), exit: null }, years.slice(0, 2))}</pre>
        </Tile>
      </Tiles>
    </>
  );
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------
function Headline() {
  const { label: mu } = useMoney();
  const { result: r } = useBacktest();
  const t = useTranslations("backtest");
  if (!r) return null;
  const last = r.years.at(-1);
  const a = r.actual;
  return (
    <>
      <Kpi
        title={t("kpiActualIrr")}
        value={fmtRate(a?.irr)}
        sub={!a ? t("stillHeld") : a.moic_given ? t("moicSubGiven", { moic: fmtMultiple(a.moic) }) : t("moicSub", { moic: fmtMultiple(a.moic) })}
        lead
      />
      <Kpi title={t("kpiPlanIrr")} value={fmtRate(r.plan.irr)} sub={t("moicSub", { moic: fmtMultiple(r.plan.moic) })} />
      <Kpi title={t("kpiPlanRange")} value={fmtRate(r.plan.irr_p5)} sub={t("toP95", { p95: fmtRate(r.plan.irr_p95) })} />
      <Kpi title={t("kpiActualPercentile")} value={fmtNumber(a?.percentile, 1)} sub={t("ofPlanPaths", { count: fmtCount(BACKTEST_PATHS) })} />
      <Kpi
        title={t("kpiLastEbitda")}
        value={fmtMoney(last?.actual_ebitda)}
        sub={t("planSub", { value: fmtMoney(last?.plan_ebitda), money: mu })}
        tone={isNum(last?.actual_ebitda) && isNum(last?.plan_ebitda) ? (last.actual_ebitda >= last.plan_ebitda ? "gain" : "loss") : undefined}
      />
      <Kpi title={t("kpiExitEquity")} value={fmtMoney(a?.exit_equity)} sub={t("planSub", { value: fmtMoney(r.plan.exit_equity), money: mu })} />
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
  const { result: r, plan } = useBacktest();
  const t = useTranslations("backtest");
  const years = useYears(r?.years.length);
  if (!r || !plan) return null;
  const markers = [
    { value: r.plan.irr_p5, label: t("markerP5", { value: fmtRate(r.plan.irr_p5) }) },
    { value: r.plan.irr_p95, label: t("markerP95", { value: fmtRate(r.plan.irr_p95) }) },
    ...(isNum(r.actual?.irr) ? [{ value: r.actual.irr, label: t("markerActual", { value: fmtRate(r.actual.irr) }), tone: "accent" as const }] : []),
  ];
  return (
    <Tiles>
      <Headline />
      <Tile span={7} title={t("tileEbitda")} unit={mu}>
        <GroupedBars
          label={t("chartEbitda", { deal: plan.name })}
          categories={years}
          format={fmtMoney}
          series={[
            { name: t("seriesPlan"), values: r.years.map((y) => y.plan_ebitda), color: "var(--color-neutral-bar)" },
            { name: t("seriesActual"), values: r.years.map((y) => y.actual_ebitda ?? 0), color: "var(--color-accent)", labelled: true },
          ]}
        />
      </Tile>
      <Tile span={5} title={t("tileWhereLanded")} unit={t("planDistribution", { year: String(r.exit_year ?? r.hold) })}>
        <Histogram
          label={t("chartPlanDistribution")}
          edges={r.irr_histogram.edges}
          density={r.irr_histogram.density}
          format={(v) => fmtRate(v, 0)}
          tickCount={5}
          markers={markers}
        />
        {!r.actual && <p className="type-body text-[9px]">{t("heldNoExit")}</p>}
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
  const { result: r } = useBacktest();
  const t = useTranslations("backtest");
  const years = useYears(r?.years.length);
  if (!r) return null;
  const actualMargins = r.actual_ebitda_margin.every(isNum) ? r.actual_ebitda_margin : null;
  return (
    <Tiles>
      <Headline />
      <Tile span={6} title={t("tileWhereMissed")} unit={t("ofExitEquity", { money: mu })}>
        {r.attribution && r.actual ? (
          <>
            <DivergingBars
              label={t("chartAttribution")}
              format={fmtDelta}
              rows={Object.entries(r.attribution).map(([k, v]) => ({ label: ATTRIBUTION_KEY[k] ? t(ATTRIBUTION_KEY[k]) : k, value: v }))}
            />
            <p className="type-body text-[9px]">
              {t("attributionNote", {
                total: fmtDelta(Object.values(r.attribution).reduce((a, b) => a + b, 0)),
                money: mu,
                plan: fmtMultiple(r.plan.exit_multiple, 1),
                actual: fmtMultiple(r.actual.exit_multiple, 1),
                netDebt: fmtMoney(r.plan.net_debt_at_exit),
              })}
            </p>
            {r.lease_addback > 0 && <p className="type-body text-[9px]">{t("attributionLeases", { addback: fmtMoney(r.lease_addback), money: mu })}</p>}
          </>
        ) : (
          <p className="type-body text-[9px]">{t("attributionNeedsExit")}</p>
        )}
      </Tile>
      <Tile span={6} title={t("tileEbitdaMargin")} unit={t("actualVsPlan")}>
        <LineChart
          label={t("chartEbitdaMargin")}
          xLabels={years}
          yFormat={(v) => fmtRate(v, 0)}
          lines={[
            ...(actualMargins ? [{ name: t("seriesActual"), values: actualMargins, color: "var(--color-accent)" }] : []),
            { name: t("seriesPlan"), values: r.plan_ebitda_margin.map((v) => v ?? NaN), color: "var(--color-muted)", dashed: true },
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

/** Plan, actual and variance rows for every line. i18n-keys: backtest.row* */
function lineRows(r: PlanActualResult, t: (key: string) => string) {
  return ACTUAL_LINES.flatMap(({ key, labelKey }) => {
    const k = key as LineKey;
    return [
      { label: `${t(labelKey)} · ${t("rowPlan")}`, values: r.years.map((y) => y[`plan_${k}`]) },
      { label: `${t(labelKey)} · ${t("rowActual")}`, values: r.years.map((y) => y[`actual_${k}`]), total: true },
      { label: `${t(labelKey)} · ${t("rowVariance")}`, values: r.years.map((y) => y[`variance_${k}`]) },
    ];
  });
}

function Years() {
  const { label: mu, money } = useMoney();
  const { result: r } = useBacktest();
  const t = useTranslations("backtest");
  const x = useTranslations("export");
  const years = useYears(r?.years.length);
  if (!r) return null;
  const rows = lineRows(r, t);
  const a = r.actual;
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
                    [x("colLine"), ...years],
                    rows.map((row) => [row.label, ...row.values]),
                    { columns: ["text", ...years.map(() => "money" as const)] },
                  ),
                  sheet(
                    x("sheetReturns"),
                    [x("colMetric"), x("colPlan"), x("colActual")],
                    [
                      [x("rowIrr"), fraction(r.plan.irr), fraction(a?.irr)],
                      [x("rowMoic"), r.plan.moic, a?.moic ?? null],
                      [x("colEntryEquity", { money: mu }), r.plan.entry_equity, a?.entry_equity ?? null],
                      [x("colExitEquity", { money: mu }), r.plan.exit_equity, a?.exit_equity ?? null],
                    ],
                    { rows: ["percent", "multiple", "money", "money"] },
                  ),
                  ...(r.attribution
                    ? [
                        sheet(
                          x("sheetAttribution"),
                          [x("colPart"), mu],
                          Object.entries(r.attribution).map(([k, v]) => [ATTRIBUTION_KEY[k] ? t(ATTRIBUTION_KEY[k]) : k, v]),
                          { columns: ["text", "money"] },
                        ),
                      ]
                    : []),
                ],
                money,
              )
            }
          />
        }
      >
        <DataTable caption={t("chartYearByYear")} columns={years} rows={rows} />
      </Tile>
      <Tile span={6} title={t("tileEquity")} unit={mu}>
        <DataTable
          caption={t("chartEquity")}
          columns={[t("colPlan"), t("colActual")]}
          rows={[
            { label: t("rowEquityEntry"), values: [r.plan.entry_equity, a?.entry_equity] },
            { label: t("rowExitEv"), values: [r.plan.exit_ev, a?.exit_ev] },
            { label: t("rowNetDebtExit"), values: [r.plan.net_debt_at_exit, a?.net_debt_at_exit] },
            { label: t("rowEquityExit"), values: [r.plan.exit_equity, a?.exit_equity], total: true },
          ]}
        />
      </Tile>
      <Tile span={6} title={t("tileReturns")} unit={t("returnsUnit")}>
        <table className="w-full border-collapse font-mono text-[11px]">
          <thead>
            <tr className="text-muted">
              <th />
              <th scope="col" className="border-b border-grid px-2 py-1 text-end font-normal">
                {t("colPlan")}
              </th>
              <th scope="col" className="border-b border-grid px-2 py-1 text-end font-normal">
                {t("colActual")}
              </th>
            </tr>
          </thead>
          <tbody>
            {[
              { key: "rowIrr", plan: fmtRate(r.plan.irr), actual: fmtRate(a?.irr) },
              { key: "rowMoic", plan: fmtMultiple(r.plan.moic), actual: fmtMultiple(a?.moic) },
              { key: "rowExitMultiple", plan: fmtMultiple(r.plan.exit_multiple, 1), actual: fmtMultiple(a?.exit_multiple, 1) },
            ].map((row) => (
              <tr key={row.key}>
                <th scope="row" className="type-input-label border-b border-grid px-2 py-1 text-start text-[9px] font-normal text-soft">
                  {t(row.key)}
                </th>
                <td className="border-b border-grid px-2 py-1 text-end">{row.plan}</td>
                <td className="border-b border-grid px-2 py-1 text-end text-bright">{row.actual}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Tile>
    </Tiles>
  );
}
