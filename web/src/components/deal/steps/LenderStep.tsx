"use client";

import { useTranslations } from "next-intl";
import { useEffect, useRef } from "react";

import { useMonteCarlo } from "@/components/montecarlo/MonteCarloProvider";
import { CellInput } from "@/components/ui/CellInput";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { SELECT_CLASS } from "@/components/ui/MoneySelects";
import { useMoney } from "@/components/ui/MoneyScope";
import { Notice, SecondaryButton } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { TipLabel, TipText, useTip } from "@/components/ui/Tip";
import { type Schemas } from "@/lib/api/client";
import { type Covenant, dealFiscal, type DealInputs, type DealRun } from "@/lib/deal/fields";
import { downloadWorkbook, sheet } from "@/lib/export";
import { fmtMoney, fmtMultiple, fmtPct, fmtRate, isNum } from "@/lib/format";
import { useEngineLabel } from "@/lib/i18n/useEngineText";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";

import { useDeal } from "../DealProvider";
import { DealDistressTile } from "../Distress";
import { DealScreen, LoadingTiles, RailGroup } from "../DealScreen";

type Lender = DealRun["lender"];
type Metric = Covenant["metric"];
type Convention = Lender["convention"];
type Simulated = NonNullable<Schemas["MonteCarloResponse"]["covenants"]>;

const METRICS: Metric[] = ["leverage", "interest_cover", "fixed_charge_cover", "liquidity"];
const CONVENTIONS: Convention[] = ["us", "lma", "india"];
/** A new covenant starts this far from the deal's own year-one figure: room for the lender's level to be typed */
const HEADROOM = 0.3;

/**
 * Where a new covenant starts: the deal's own year-one figure with 30% headroom, rounded, so it
 * neither breaches at once nor means nothing. The facility agreement's level replaces it.
 */
function startingLevel(metric: Metric, scope: string | null | undefined, lender: Lender | undefined): number {
  const one = lender?.year_one;
  if (metric === "liquidity") return Math.max(Math.round((one?.liquidity ?? 0) * (1 - HEADROOM) * 10) / 10, 0);
  const v = metric === "leverage" ? (scope === "senior" ? one?.senior_leverage : one?.leverage) : one?.[metric];
  if (!isNum(v)) return metric === "leverage" ? 6 : 1.5;
  const level = metric === "leverage" ? v * (1 + HEADROOM) : v * (1 - HEADROOM);
  return Math.round(level * 4) / 4;
}

/** The rail: how covenants are defined, and each one's facility, test and levels by year. */
function CovenantRail() {
  const { inputs, setFields, run } = useDeal();
  const t = useTranslations("lender");
  const trancheName = useEngineLabel("tranche");
  const fiscalLabels = useFiscalLabels();
  const facilities = Object.keys(run.result?.tranches ?? {});
  const covenants = inputs.covenants;
  const years = fiscalLabels.deal(inputs.hold, dealFiscal(inputs));
  const conventionTip = useTip("covenantConvention");
  const addTip = useTip("covenantAdd");
  const write = (next: Covenant[]) => setFields({ covenants: next });
  const update = (i: number, patch: Partial<Covenant>) => write(covenants.map((c, j) => (j === i ? { ...c, ...patch } : c)));
  const country = run.result?.lender.chosen === false ? run.result.lender.convention : null;

  return (
    <>
      <RailGroup title={t("groupConvention")}>
        <label className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
          <span className="type-input-label"><TipLabel tip={conventionTip}>{t("convention")}</TipLabel></span>
          <select aria-describedby={conventionTip.id} value={inputs.covenant_convention}
            onChange={(e) => setFields({ covenant_convention: e.target.value as DealInputs["covenant_convention"] })}
            className={SELECT_CLASS}>
            {/* i18n-keys: lender.convention_* */}
            <option value="">{country ? t("conventionCountry", { convention: t(`convention_${country}`) }) : t("conventionFromCountry")}</option>
            {CONVENTIONS.map((c) => <option key={c} value={c}>{t(`convention_${c}`)}</option>)}
          </select>
        </label>
        <TipText tip={conventionTip} />
        <p className="type-body pt-1.5 text-[9px]" role="note">{t("conventionNote")}</p>
      </RailGroup>
      <RailGroup title={t("groupCovenants")}>
        {covenants.length === 0 && <p className="type-body text-[9px]">{t("noCovenants")}</p>}
        <ol className="grid grid-cols-[minmax(0,1fr)] border-t border-line">
          {covenants.map((c, i) => (
            <CovenantFields key={i} index={i} covenant={c} facilities={facilities} years={years}
              trancheName={trancheName} onChange={(patch) => update(i, patch)}
              onRemove={() => write(covenants.filter((_, j) => j !== i))} />
          ))}
        </ol>
        <div className="flex pt-2">
          <SecondaryButton describedBy={addTip.id} disabled={!facilities.length || covenants.length >= 24}
            onClick={() => {
              const facility = facilities[0];
              write([...covenants, { facility, metric: "leverage", scope: "total", levels: [startingLevel("leverage", "total", run.result?.lender)] }]);
            }}>
            {t("addCovenant")}
          </SecondaryButton>
          <TipText tip={addTip} />
        </div>
      </RailGroup>
    </>
  );
}

/** i18n-keys: lender.metric_*, lender.scope_* */
function CovenantFields({ index, covenant: c, facilities, years, trancheName, onChange, onRemove }: {
  index: number;
  covenant: Covenant;
  facilities: string[];
  years: string[];
  trancheName: (n: string) => string;
  onChange: (patch: Partial<Covenant>) => void;
  onRemove: () => void;
}) {
  const t = useTranslations("lender");
  const { run } = useDeal();
  const facilityTip = useTip("covenantFacility");
  const metricTip = useTip("covenantMetric");
  const scopeTip = useTip("covenantScope");
  const missing = !facilities.includes(c.facility);
  const levels = c.levels;
  // One level a year of the hold; a shorter list repeats its last year, as the API reads it
  const level = (y: number) => levels[Math.min(y, levels.length - 1)];
  const setLevel = (y: number, v: number) => {
    const full = years.map((_, i) => level(i));
    full[y] = v;
    // Store only as many years as differ from the last: an unstepped covenant stays one number
    let n = full.length;
    while (n > 1 && full[n - 1] === full[n - 2]) n -= 1;
    onChange({ levels: full.slice(0, n) });
  };
  return (
    <li className="grid grid-cols-[minmax(0,1fr)] gap-0.5 border-b border-line py-1.5" data-covenant={index}>
      <label className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
        <span className="type-input-label"><TipLabel tip={facilityTip}>{t("facility")}</TipLabel></span>
        <select aria-describedby={facilityTip.id} value={c.facility} onChange={(e) => onChange({ facility: e.target.value })} className={SELECT_CLASS}>
          {missing && <option value={c.facility}>{t("missingFacility", { name: c.facility })}</option>}
          {facilities.map((f) => <option key={f} value={f}>{trancheName(f)}</option>)}
        </select>
      </label>
      <TipText tip={facilityTip} />
      <label className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
        <span className="type-input-label"><TipLabel tip={metricTip}>{t("metric")}</TipLabel></span>
        <select aria-describedby={metricTip.id} value={c.metric}
          onChange={(e) => {
            const metric = e.target.value as Metric;
            onChange({ metric, levels: [startingLevel(metric, c.scope, run.result?.lender)] });
          }}
          className={SELECT_CLASS}>
          {METRICS.map((m) => <option key={m} value={m}>{t(`metric_${m}`)}</option>)}
        </select>
      </label>
      <TipText tip={metricTip} />
      {c.metric === "leverage" && (
        <>
          <label className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
            <span className="type-input-label"><TipLabel tip={scopeTip}>{t("scope")}</TipLabel></span>
            <select aria-describedby={scopeTip.id} value={c.scope ?? "total"} onChange={(e) => onChange({ scope: e.target.value as Covenant["scope"] })} className={SELECT_CLASS}>
              <option value="total">{t("scope_total")}</option>
              <option value="senior">{t("scope_senior")}</option>
            </select>
          </label>
          <TipText tip={scopeTip} />
        </>
      )}
      <p className="type-input-label pt-1">{c.metric === "leverage" ? t("levelsMaximum") : t("levelsMinimum")}</p>
      <div className="grid grid-cols-[repeat(auto-fill,minmax(52px,1fr))] gap-1">
        {years.map((y, i) => (
          <label key={i} className="grid gap-0.5">
            <span className="font-mono text-[9px] text-dim">{y}</span>
            <CellInput value={level(i)} decimals={c.metric === "liquidity" ? 1 : 2} tip="covenantLevel"
              label={t("levelFor", { year: y })} onCommit={(v) => setLevel(i, v)} />
          </label>
        ))}
      </div>
      {missing && <p className="type-body text-[9px] text-attention" role="note">{t("missingFacilityNote")}</p>}
      <div className="flex pt-1.5">
        <SecondaryButton onClick={onRemove}>{t("removeCovenant")}</SecondaryButton>
      </div>
    </li>
  );
}

export function LenderStep() {
  const { run } = useDeal();
  return (
    <DealScreen rail={<CovenantRail />}>
      {!run.result ? <LoadingTiles /> : <LenderResults />}
    </DealScreen>
  );
}

/** A covenant's figure in a year, as its kind reads: a multiple, or money. */
function shownFigure(metric: Metric, y: Lender["covenants"][number]["years"][number]): string {
  return metric === "liquidity" ? fmtMoney(y.liquidity) : isNum(y.value) ? fmtMultiple(y.value, 2) : "∞";
}

function shownLevel(metric: Metric, y: Lender["covenants"][number]["years"][number]): string {
  return metric === "liquidity" ? fmtMoney(y.minimum_liquidity) : fmtMultiple(y.level, 2);
}

function LenderResults() {
  const { inputs, run, current } = useDeal();
  const { label: mu, money } = useMoney();
  const t = useTranslations("lender");
  const x = useTranslations("export");
  const fiscalLabels = useFiscalLabels();
  const trancheName = useEngineLabel("tranche");
  const res = run.result!;
  const lender = res.lender;
  const years = fiscalLabels.deal(res.cash_flow.levered_fcf?.length ?? inputs.hold, dealFiscal(inputs));
  const mc = useMonteCarlo();
  const simulated = mc.run.result && !mc.stale && mc.run.status !== "running" ? mc.run.result.covenants ?? null : null;
  const metricName = (c: { metric: Metric; scope?: string | null }) =>
    c.metric === "leverage" && c.scope === "senior" ? t("metric_senior_leverage") : t(`metric_${c.metric}`);
  const covenantName = (c: { facility: string; metric: Metric; scope?: string | null }) =>
    t("covenantName", { facility: trancheName(c.facility), metric: metricName(c) });

  const download = () =>
    downloadWorkbook(
      x("fileLender"),
      [
        sheet(x("sheetCovenants"), [x("colCovenant"), x("colYear"), x("colFigure"), x("colLevel"), x("colHeadroom"), x("colBreach")],
          lender.covenants.flatMap((c) => c.years.map((y) => [
            covenantName(c), years[y.year - 1] ?? y.year,
            c.metric === "liquidity" ? y.liquidity ?? null : y.value ?? null,
            c.metric === "liquidity" ? y.minimum_liquidity ?? null : y.level ?? null,
            y.headroom ?? null, !y.tested ? x("notTested") : y.breach ? x("yes") : x("no"),
          ])),
          { columns: ["text", "text", "number", "number", "percent", "text"] }),
        ...(simulated ? [sheet(x("sheetBreachOdds"), [x("colCovenant"), x("colYear"), x("colBreachOdds"), x("colBreachedBy")],
          [...simulated.covenants.map((c) => ({ name: covenantName(c), years: c.years })), { name: t("anyCovenant"), years: simulated.any }]
            .flatMap((c) => c.years.map((y) => [c.name, years[y.year - 1] ?? y.year, y.breach, y.breached_by])),
          { columns: ["text", "text", "percent", "percent"] })] : []),
        sheet(x("sheetExpectedLoss"), [x("colFacility"), x("colExposure"), x("colDefault"), x("colLgd"), x("colExpectedLoss"), x("colRecovery")],
          lender.facilities.map((f) => [trancheName(f.name), f.exposure, isNum(f.default_pct) ? f.default_pct / 100 : null,
            isNum(f.lgd_pct) ? f.lgd_pct / 100 : null, f.expected_loss ?? null, f.expected_recovery ?? null]),
          { columns: ["text", "money", "percent", "percent", "money", "money"] }),
      ],
      money, res.model, current?.id ?? null,
    );

  return (
    <Tiles>
      {/* i18n-keys: lender.convention_* */}
      <Kpi title={t("kpiConvention")} value={t(`convention_${lender.convention}`)} sub={lender.chosen ? t("chosen") : t("fromCountry")} lead />
      <Kpi title={t("kpiFirstBreach")} value={lender.first_breach ? years[lender.first_breach - 1] ?? String(lender.first_breach) : t("none")}
        sub={t("covenantCount", { count: lender.covenants.length })} tone={lender.first_breach ? "loss" : undefined} />
      <Kpi title={t("kpiBreachOdds")} value={simulated ? fmtRate(simulated.any.at(-1)?.breached_by, 1) : t("notRun")}
        sub={simulated ? t("breachOddsSub", { paths: simulated.paths }) : t("notRunSub")} />
      <Kpi title={t("kpiDefault")} value={fmtPct(lender.default_pct, 2)} sub={t("defaultSub", { rating: lender.rating ?? t("none"), years: lender.years })} />
      <Kpi title={t("kpiExpectedLoss")} value={fmtMoney(lender.expected_loss)} sub={mu} />
      <Kpi title={t("kpiLossRate")} value={fmtRate(lender.loss_rate, 2)} sub={t("lossRateSub")} />

      <Tile span={12} title={t("tileTests")} unit={t("testsUnit", { money: mu })} action={<DownloadButton onDownload={download} />}>
        {lender.covenants.length === 0 ? (
          <p className="type-body p-3">{t("noCovenantsResult")}</p>
        ) : (
          <CovenantTable lender={lender} years={years} name={covenantName} />
        )}
      </Tile>

      <Tile span={12} title={t("tileBreachOdds")} unit={simulated ? t("breachOddsUnit", { paths: simulated.paths }) : undefined}>
        <BreachOdds simulated={simulated} hasCovenants={lender.covenants.length > 0} years={years} name={covenantName} />
      </Tile>

      <Tile span={6} title={t("tileExpectedLoss")} unit={mu}>
        <ExpectedLoss lender={lender} trancheName={trancheName} />
      </Tile>
      <DealDistressTile d={res.distress} years={years} />

      <Tile span={12} title={t("tileSources")}>
        <ul className="grid gap-1 p-3">
          {lender.sources.map((s) => (
            <li key={s.id} className="type-body text-[10px]">
              {s.url ? <a href={s.url} target="_blank" rel="noreferrer" className="text-accent underline">{s.title}</a> : s.title}
              <span className="text-dim"> · {s.publisher} · {s.detail}</span>
            </li>
          ))}
          <li className="type-body text-[10px] text-dim">{t("definitionsNote")}</li>
        </ul>
      </Tile>
    </Tiles>
  );
}

function CovenantTable({ lender, years, name }: {
  lender: Lender;
  years: string[];
  name: (c: Lender["covenants"][number]) => string;
}) {
  const t = useTranslations("lender");
  const cols = years.slice(0, lender.covenants[0]?.years.length ?? 0);
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse font-mono text-[11px]">
        <caption className="sr-only">{t("tileTests")}</caption>
        <thead>
          <tr>
            <th scope="col" className="border-b border-grid px-2 py-1 text-start font-normal text-muted">{t("colCovenant")}</th>
            {cols.map((y) => <th key={y} scope="col" className="border-b border-grid px-2 py-1 text-end font-normal text-muted">{y}</th>)}
          </tr>
        </thead>
        <tbody>
          {lender.covenants.map((c) => (
            <tr key={c.index} data-covenant-test={c.index}>
              <th scope="row" className="border-b border-grid px-2 py-1 text-start font-normal">
                <span className="type-input-label block text-soft">{name(c)}</span>
                <span className="block text-[9.5px] text-dim">{c.metric === "leverage" ? t("maximum") : t("minimum")}</span>
              </th>
              {c.years.map((y) => (
                <td key={y.year} data-year={y.year} data-breach={y.breach ? "" : undefined} data-tested={y.tested ? "" : undefined}
                  className={`border-b border-grid px-2 py-1 text-end ${!y.tested ? "text-dim" : y.breach ? "bg-loss/15 text-loss" : "text-ink"}`}>
                  {y.tested ? shownFigure(c.metric, y) : t("notTested")}
                  <span className="block text-[9.5px] text-dim">
                    {y.tested ? (y.breach ? t("breachAgainst", { level: shownLevel(c.metric, y) }) : t("levelShort", { level: shownLevel(c.metric, y) })) : ""}
                  </span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** The simulation's breach odds; run once for this screen when no current result has them. */
function BreachOdds({ simulated, hasCovenants, years, name }: {
  simulated: Simulated | null;
  hasCovenants: boolean;
  years: string[];
  name: (c: Simulated["covenants"][number]) => string;
}) {
  const t = useTranslations("lender");
  const mc = useMonteCarlo();
  const asked = useRef(false);
  const current = !!mc.run.result && !mc.stale && mc.run.status !== "running";
  // Like the summary's metrics, the simulation runs once per visit when no current result exists
  useEffect(() => {
    if (!hasCovenants || asked.current || mc.run.status === "running" || current) return;
    asked.current = true;
    mc.runNow();
  }, [hasCovenants, current, mc]);
  if (!hasCovenants) return <p className="type-body p-3">{t("noCovenantsOdds")}</p>;
  if (!simulated) {
    return (
      <Notice tone={mc.run.status === "error" ? "loss" : "attention"} title={mc.run.status === "running" ? t("simulating") : t("notRun")}
        actions={mc.run.status === "running" ? undefined : <SecondaryButton onClick={mc.runNow}>{t("runSimulation")}</SecondaryButton>}>
        {t("oddsFromSimulation")}
      </Notice>
    );
  }
  const rows = [...simulated.covenants.map((c) => ({ key: String(c.index), name: name(c), years: c.years })),
    { key: "any", name: t("anyCovenant"), years: simulated.any }];
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse font-mono text-[11px]">
        <caption className="sr-only">{t("tileBreachOdds")}</caption>
        <thead>
          <tr>
            <th scope="col" className="border-b border-grid px-2 py-1 text-start font-normal text-muted">{t("colCovenant")}</th>
            {years.slice(0, simulated.any.length).map((y) => <th key={y} scope="col" className="border-b border-grid px-2 py-1 text-end font-normal text-muted">{y}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.key} data-breach-odds={r.key}>
              <th scope="row" className="border-b border-grid px-2 py-1 text-start font-normal">
                <span className="type-input-label block text-soft">{r.name}</span>
              </th>
              {r.years.map((y) => (
                <td key={y.year} data-year={y.year} data-odds={y.breach} className="border-b border-grid px-2 py-1 text-end text-ink">
                  {fmtRate(y.breach, 1)}
                  <span className="block text-[9.5px] text-dim">{t("breachedBy", { share: fmtRate(y.breached_by, 1) })}</span>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** i18n-keys: lender.seniority_* */
function ExpectedLoss({ lender, trancheName }: { lender: Lender; trancheName: (n: string) => string }) {
  const t = useTranslations("lender");
  const head = [t("colFacility"), t("colExposure"), t("colDefault"), t("colLgd"), t("colExpectedLoss"), t("colRecovery")];
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse font-mono text-[11px]">
        <caption className="sr-only">{t("tileExpectedLoss")}</caption>
        <thead>
          <tr>{head.map((h, i) => <th key={h} scope="col" className={`border-b border-grid px-2 py-1 font-normal text-muted ${i === 0 ? "text-start" : "text-end"}`}>{h}</th>)}</tr>
        </thead>
        <tbody>
          {lender.facilities.map((f) => (
            <tr key={f.name} data-loss-facility={f.name} className="text-ink">
              <th scope="row" className="border-b border-grid px-2 py-1 text-start font-normal">
                <span className="type-input-label block text-soft">{trancheName(f.name)}</span>
                <span className="block text-[9.5px] text-dim">{t(`seniority_${f.seniority}`)}</span>
              </th>
              <td className="border-b border-grid px-2 py-1 text-end">{fmtMoney(f.exposure)}</td>
              <td className="border-b border-grid px-2 py-1 text-end">{f.lender ? fmtPct(f.default_pct, 2) : t("sponsor")}</td>
              <td className="border-b border-grid px-2 py-1 text-end">{f.lender ? fmtPct(f.lgd_pct, 0) : ""}</td>
              <td className="border-b border-grid px-2 py-1 text-end" data-expected-loss={f.expected_loss ?? undefined}>{f.lender ? fmtMoney(f.expected_loss) : ""}</td>
              <td className="border-b border-grid px-2 py-1 text-end">{f.lender ? fmtMoney(f.expected_recovery) : ""}</td>
            </tr>
          ))}
          <tr className="text-bright">
            <th scope="row" className="px-2 py-1 text-start font-normal">{t("total")}</th>
            <td className="px-2 py-1 text-end">{fmtMoney(lender.exposure)}</td>
            <td /><td />
            <td className="px-2 py-1 text-end">{fmtMoney(lender.expected_loss)}</td>
            <td />
          </tr>
        </tbody>
      </table>
      <p className="type-body p-2 text-[9px]" role="note">{t("lossNote")}</p>
    </div>
  );
}
