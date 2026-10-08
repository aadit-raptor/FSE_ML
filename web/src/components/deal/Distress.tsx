"use client";

import { useTranslations } from "next-intl";

import { SELECT_CLASS } from "@/components/ui/MoneySelects";
import { Tile } from "@/components/ui/Tile";
import type { components } from "@/lib/api/schema";
import { fmtMultiple, fmtNumber, fmtRate } from "@/lib/format";

import { useDeal } from "./DealProvider";

type DealDistress = components["schemas"]["DealDistress"];
type SimulatedDistress = components["schemas"]["SimulatedDistress"];
type Context = DealDistress | SimulatedDistress;
type Band = components["schemas"]["DealDistressYear"]["band"];

const BUSINESS_RISKS = [1, 2, 3, 4, 5, 6] as const;
const BANDS: Band[] = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C"];
const CELL = "border-b border-grid px-2 py-1 text-end whitespace-nowrap";
const HEAD = "type-input-label border-b border-grid px-2 py-1 text-start text-[9px] font-normal text-soft";

/** A band's colour: investment grade calm, BB attention, B and below loss. */
function bandTone(band: Band): string {
  const i = BANDS.indexOf(band);
  return i >= 5 ? "text-loss" : i === 4 ? "text-attention" : "text-ink";
}

/**
 * S&P's business risk profile (PLAN.md 5.3), the one input only the distress
 * predictor reads: with leverage it sets each year's rating band (Corporate
 * Methodology, Table 3). A choice, 4 (fair) until the user picks another.
 *
 * i18n-keys: distress.businessRisk_*
 */
export function BusinessRiskSelect() {
  const { inputs, setFields } = useDeal();
  const t = useTranslations("distress");
  const fields = useTranslations("fields");
  return (
    <>
      <label className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
        <span className="type-input-label">{fields("business_risk")}</span>
        <select
          value={inputs.business_risk}
          onChange={(e) => setFields({ business_risk: Number(e.target.value) })}
          className={SELECT_CLASS}
        >
          {BUSINESS_RISKS.map((b) => (
            <option key={b} value={b}>
              {t(`businessRisk_${b}`)}
            </option>
          ))}
        </select>
      </label>
      <p className="type-body pt-1.5 text-[9px]" role="note">
        {t("businessRiskNote")}
      </p>
    </>
  );
}

/**
 * Why the chances of default are (or are not) shown, what the bands read and
 * where every table comes from.
 *
 * i18n-keys: distress.table_*, library.region_*
 */
function DistressNotes({ d }: { d: Context }) {
  const t = useTranslations("distress");
  const lib = useTranslations("library");
  const region = d.region ? lib(`region_${d.region}`) : t("noRegion");
  const c = d.card;
  const verdict = d.shown
    ? t("shown", { cases: c.cases, region, model: fmtNumber(c.model, 2), baseline: fmtNumber(c.baseline, 2) })
    : c.verdict === "does_not_beat_baseline"
      ? t("notBetter", { cases: c.cases, region, model: fmtNumber(c.model, 2), baseline: fmtNumber(c.baseline, 2) })
      : t("tooFew", { cases: c.cases, region });
  return (
    <div className="grid gap-1.5 pt-2">
      <p className="type-body text-[9px]" data-distress-shown={String(d.shown)}>
        {!d.shown && <span className="type-alert me-1 text-[9px] text-attention">{t("notEnoughData")}</span>}
        {verdict}
      </p>
      <p className="type-body text-[9px]">
        {t("reads", { business: t(`businessRisk_${d.business_risk}`), table: t(`table_${d.table}`) })}
      </p>
      <p className="type-body text-[9px]">
        <span className="me-1">{t("sources")}</span>
        {d.sources
          .filter((s) => s.url)
          .map((s, i) => (
            <span key={s.id}>
              {i > 0 && " · "}
              <a href={s.url ?? undefined} target="_blank" rel="noopener noreferrer" className="text-accent underline">
                {t("sourceLink", { publisher: s.publisher, title: s.title })}
              </a>
            </span>
          ))}
      </p>
    </div>
  );
}

function Header({ years }: { years: string[] }) {
  return (
    <thead>
      <tr>
        <th scope="col" className="w-[30%]" />
        {years.map((y) => (
          <th key={y} scope="col" className="border-b border-grid px-2 py-1 text-end font-normal text-muted">
            {y}
          </th>
        ))}
      </tr>
    </thead>
  );
}

/** Deal -> Debt: each year's coverage, leverage, rating band and, where its card allows, chance of default. */
export function DealDistressTile({ d, years }: { d: DealDistress; years: string[] }) {
  const t = useTranslations("distress");
  const rows = d.years;
  const cols = years.slice(0, rows.length);
  const shown = (v: number | null | undefined) => (d.shown ? fmtRate(v, 2) : t("notShown"));
  return (
    <Tile span={12} title={t("tile")} unit={t("unitDeal")}>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse font-mono text-[11px]" data-distress="deal">
          <caption className="sr-only">{t("tile")}</caption>
          <Header years={cols} />
          <tbody>
            <tr className="text-ink">
              <th scope="row" className={HEAD}>{t("rowCoverage")}</th>
              {rows.map((r) => <td key={r.year} className={CELL}>{fmtMultiple(r.coverage, 2)}</td>)}
            </tr>
            <tr className="text-ink">
              <th scope="row" className={HEAD}>{t("rowLeverage")}</th>
              {rows.map((r) => <td key={r.year} className={CELL}>{fmtMultiple(r.leverage, 2)}</td>)}
            </tr>
            <tr className="text-ink">
              <th scope="row" className={HEAD}>{t("rowCoverageBand")}</th>
              {rows.map((r) => <td key={r.year} className={CELL}>{r.coverage_band}</td>)}
            </tr>
            <tr className="text-ink">
              <th scope="row" className={HEAD}>{t("rowLeverageBand")}</th>
              {rows.map((r) => <td key={r.year} className={CELL}>{r.leverage_band}</td>)}
            </tr>
            <tr className="text-bright">
              <th scope="row" className={HEAD}>{t("rowBand")}</th>
              {rows.map((r) => (
                <td key={r.year} className={`${CELL} font-semibold ${bandTone(r.band)}`} data-band-year={r.year}>
                  {r.band}
                </td>
              ))}
            </tr>
            <tr className="text-ink">
              <th scope="row" className={HEAD}>{t("rowProbability")}</th>
              {rows.map((r) => <td key={r.year} className={CELL}>{shown(r.probability)}</td>)}
            </tr>
            <tr className="text-ink">
              <th scope="row" className={HEAD}>{t("rowCumulative")}</th>
              {rows.map((r) => <td key={r.year} className={CELL}>{shown(r.cumulative)}</td>)}
            </tr>
          </tbody>
        </table>
      </div>
      <DistressNotes d={d} />
    </Tile>
  );
}

/** Monte Carlo: the share of simulated paths in each band each year and, where allowed, the mean chances. */
export function SimulatedDistressTile({ d, years }: { d: SimulatedDistress; years: string[] }) {
  const t = useTranslations("distress");
  const rows = d.years;
  const cols = years.slice(0, rows.length);
  // Only the bands some path reaches, strongest first
  const bands = BANDS.filter((b) => rows.some((r) => (r.band_shares[b] ?? 0) > 0));
  const shown = (v: number | null | undefined) => (d.shown ? fmtRate(v, 2) : t("notShown"));
  return (
    <Tile span={12} title={t("tile")} unit={t("unitSimulated")}>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse font-mono text-[11px]" data-distress="simulated">
          <caption className="sr-only">{t("tile")}</caption>
          <Header years={cols} />
          <tbody>
            {bands.map((b) => (
              <tr key={b} className="text-ink" data-band={b}>
                <th scope="row" className={HEAD}>{t("rowPathsAt", { band: b })}</th>
                {rows.map((r) => (
                  <td key={r.year} className={`${CELL} ${bandTone(b)}`}>
                    {fmtRate(r.band_shares[b] ?? 0, 1)}
                  </td>
                ))}
              </tr>
            ))}
            <tr className="text-ink">
              <th scope="row" className={HEAD}>{t("rowMeanProbability")}</th>
              {rows.map((r) => <td key={r.year} className={CELL}>{shown(r.probability)}</td>)}
            </tr>
            <tr className="text-ink">
              <th scope="row" className={HEAD}>{t("rowMeanCumulative")}</th>
              {rows.map((r) => <td key={r.year} className={CELL}>{shown(r.cumulative)}</td>)}
            </tr>
          </tbody>
        </table>
      </div>
      <DistressNotes d={d} />
    </Tile>
  );
}
