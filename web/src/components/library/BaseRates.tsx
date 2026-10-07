"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { LineChart } from "@/components/charts/LineChart";
import { useDeal } from "@/components/deal/DealProvider";
import { LoadingTiles, Notice, RailGroup, Screen } from "@/components/ui/Screen";
import { Tile, Tiles } from "@/components/ui/Tile";
import { fmtAxis, fmtCount, fmtPct } from "@/lib/format";
import {
  type BaseRates, type BaseRateSource, cumulativeRegionFor, type CumulativeRegion, fetchBaseRates, firstCompleteYear,
} from "@/lib/library";
import { periodLabel, regionName } from "@/lib/locale";

import { LibraryGate } from "./LibraryGate";

const RATING_LINES = [
  { key: "BBB", color: "var(--color-muted)" },
  { key: "BB", color: "var(--color-attention)" },
  { key: "B", color: "var(--color-accent)" },
  { key: "CCC/C", color: "var(--color-loss)" },
] as const;
const REGION_COLORS: Record<string, string> = {
  us: "var(--color-accent)",
  europe: "var(--color-attention)",
  emerging: "var(--color-loss)",
  other_developed: "var(--color-gain)",
};
const GRADES = ["investment_grade", "speculative_grade", "all_rated"];

/** The base rates for the open deal's country (its S&P region), read again when the country changes. */
function useBaseRates(country: string): BaseRates | null | undefined {
  const [found, setFound] = useState<{ country: string; rates: BaseRates | null } | null>(null);
  useEffect(() => {
    let live = true;
    void fetchBaseRates(country || null).then((rates) => {
      if (live) setFound({ country, rates });
    });
    return () => {
      live = false;
    };
  }, [country]);
  return found && found.country === country ? found.rates : undefined;
}

/**
 * Library -> Base rates (PLAN.md 4.5): how often companies of each rating have defaulted, by region
 * and year, and how much lenders recovered, every table cited to its source.
 */
export function BaseRatesStep() {
  return (
    <LibraryGate>
      <BaseRatesScreen />
    </LibraryGate>
  );
}

function BaseRatesScreen() {
  const { inputs } = useDeal();
  const rates = useBaseRates(inputs.country ?? "");
  const t = useTranslations("library");
  const [chosen, setChosen] = useState<CumulativeRegion | null>(null);
  if (rates === undefined) return <Screen rail={null}><LoadingTiles /></Screen>;
  if (!rates?.default || !rates.recovery) {
    return (
      <Screen rail={null}>
        <Notice tone="loss" title={t("unreachableTitle")} role="alert">
          {t("unreachable")}
        </Notice>
      </Screen>
    );
  }
  const available = rates.default.cumulative.map((c) => c.region);
  const region = chosen ?? cumulativeRegionFor(rates.sp_region, available);
  return (
    <Screen rail={<Rail rates={rates} region={region} onRegion={setChosen} available={available} />}>
      <StaleNotice sources={rates.sources} />
      <Tiles>
        <Cumulative rates={rates} region={region} />
        <AnnualByRating rates={rates} />
        <SpeculativeByRegion rates={rates} />
        <Recovery rates={rates} />
      </Tiles>
    </Screen>
  );
}

/** i18n-keys: library.region_* */
function useRegionName() {
  const t = useTranslations("library");
  return (region: string) => (t.has(`region_${region}`) ? t(`region_${region}`) : region);
}

function Rail({ rates, region, onRegion, available }: {
  rates: BaseRates; region: CumulativeRegion; onRegion: (r: CumulativeRegion) => void; available: CumulativeRegion[];
}) {
  const t = useTranslations("library");
  const regionText = useRegionName();
  return (
    <>
      <RailGroup title={t("groupRegion")}>
        <p className="type-body text-[9px]" data-testid="deal-region">
          {rates.country && rates.sp_region
            ? t("dealRegion", { country: regionName(rates.country), region: regionText(rates.sp_region) })
            : t("dealNoCountry")}
        </p>
        <div className="grid gap-px bg-line" role="radiogroup" aria-label={t("cumulativeRegion")}>
          {available.map((r) => (
            <button
              key={r}
              type="button"
              role="radio"
              aria-checked={r === region}
              onClick={() => onRegion(r)}
              className={`px-2.5 py-1.5 text-start ${r === region ? "type-control-value bg-raised shadow-[inset_2px_0_0_var(--color-accent)]" : "type-control bg-panel hover:bg-raised"}`}
            >
              {regionText(r)}
            </button>
          ))}
        </div>
        {rates.sp_region === "other_developed" && <p className="type-body text-[9px]">{t("otherDevelopedNoCumulative")}</p>}
      </RailGroup>
      <RailGroup title={t("groupSources")}>
        <ul className="grid gap-2.5">
          {rates.sources.map((s) => (
            <SourceItem key={s.id} source={s} />
          ))}
        </ul>
      </RailGroup>
    </>
  );
}

/** i18n-keys: library.sample_*, library.note_* */
function SourceItem({ source }: { source: BaseRateSource }) {
  const t = useTranslations("library");
  return (
    <li className="grid gap-0.5" data-source={source.id}>
      <a href={source.url} target="_blank" rel="noreferrer" className="type-control-value text-accent underline">
        {source.publisher}
      </a>
      <span className="type-body text-[9px]">{source.title}</span>
      <span className="font-mono text-[9.5px] text-muted">
        {t(`sample_${source.id}`, {
          count: source.sample.count,
          shown: fmtCount(source.sample.count),
          first: String(source.sample.first_year),
          last: String(source.sample.last_year),
        })}
      </span>
      <span className="font-mono text-[9.5px] text-dim">
        {t("sourceDates", { published: periodLabel(source.published), checked: periodLabel(source.checked_on) })}
      </span>
      <span className="type-body text-[9px]">{t(`note_${source.id}`)}</span>
    </li>
  );
}

function StaleNotice({ sources }: { sources: BaseRateSource[] }) {
  const t = useTranslations("library");
  const stale = sources.filter((s) => s.stale);
  if (!stale.length) return null;
  return (
    <Notice title={t("staleTitle")} role="note">
      {t("stale", { sources: stale.map((s) => s.publisher).join(", ") })}
    </Notice>
  );
}

/** "S&P Global Ratings · Table 25", under every tile. */
function Cite({ rates, table }: { rates: BaseRates; table: string }) {
  const t = useTranslations("library");
  const tb = rates.tables.find((x) => x.id === table);
  const src = tb && rates.sources.find((s) => s.id === tb.source);
  if (!tb || !src) return null;
  return (
    <p className="font-mono text-[9.5px] text-dim" data-cite={table}>
      <a href={src.url} target="_blank" rel="noreferrer" className="underline hover:text-ink">
        {t("cite", { publisher: src.publisher, table: tb.table, title: tb.title })}
      </a>
    </p>
  );
}

/** i18n-keys: library.grade_* */
function Cumulative({ rates, region }: { rates: BaseRates; region: CumulativeRegion }) {
  const t = useTranslations("library");
  const regionText = useRegionName();
  const block = rates.default!.cumulative.find((c) => c.region === region)!;
  const horizons = Array.from({ length: block.horizons }, (_, i) => i + 1);
  const rows = [...rates.default!.ratings, ...GRADES].filter((k) => block.rates[k]);
  const title = t("cumulativeTitle", { region: regionText(region) });
  return (
    <Tile span={12} title={title} unit={t("unitPctIssuers")}>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse font-mono text-[11px]" aria-label={title} data-region={region}>
          <thead>
            <tr>
              <th scope="col" className="type-input-label px-2 py-1 text-start font-normal">{t("colRating")}</th>
              {horizons.map((h) => (
                <th key={h} scope="col" className="border-b border-grid px-2 py-1 text-end font-normal text-muted">
                  {t("yearsAfter", { years: h })}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((k) => (
              <tr key={k} className={GRADES.includes(k) ? "text-bright" : "text-ink"} data-row={k}>
                <th scope="row" className="type-input-label border-b border-grid px-2 py-1 text-start text-[9px] font-normal text-soft">
                  {GRADES.includes(k) ? t(`grade_${k}`) : k}
                </th>
                {block.rates[k].map((v, i) => (
                  <td key={i} className="border-b border-grid px-2 py-1 text-end whitespace-nowrap">
                    {fmtPct(v, 2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Cite rates={rates} table={block.table} />
    </Tile>
  );
}

function AnnualByRating({ rates }: { rates: BaseRates }) {
  const t = useTranslations("library");
  const a = rates.default!.annual_by_rating;
  const title = t("annualTitle");
  return (
    <Tile span={6} title={title} unit={t("unitPctIssuers")}>
      <LineChart
        label={title}
        xLabels={a.years.map(String)}
        lines={RATING_LINES.map((l) => ({ name: l.key, values: a.rates[l.key], color: l.color, width: 1.5 }))}
        yFormat={fmtAxis}
        yMin={0}
      />
      <Cite rates={rates} table="annual_by_rating" />
    </Tile>
  );
}

function SpeculativeByRegion({ rates }: { rates: BaseRates }) {
  const t = useTranslations("library");
  const regionText = useRegionName();
  const s = rates.default!.speculative_by_region;
  const from = firstCompleteYear(s.rates, s.years);
  const title = t("regionTitle");
  return (
    <Tile span={6} title={title} unit={t("unitPctSpeculative")}>
      <LineChart
        label={title}
        xLabels={s.years.slice(from).map(String)}
        lines={s.regions.map((r) => ({
          name: regionText(r),
          values: s.rates[r].slice(from).map((v) => v ?? NaN),
          color: REGION_COLORS[r],
          width: r === rates.sp_region ? 2.5 : 1.25,
        }))}
        yFormat={fmtAxis}
        yMin={0}
      />
      <p className="type-body text-[9px]">{t("regionFrom", { first: String(s.years[from]), firstAll: String(s.years[0]) })}</p>
      <Cite rates={rates} table="speculative_by_region" />
    </Tile>
  );
}

/** i18n-keys: library.seniority_*, library.gcd_* */
function Recovery({ rates }: { rates: BaseRates }) {
  const t = useTranslations("library");
  const r = rates.recovery!;
  const table = (title: string, rows: { key: string; label: string; defaults: number; lgd_pct: number }[], cite: string) => (
    <Tile span={4} title={title} unit={t("unitPctExposure")}>
      <table className="w-full border-collapse font-mono text-[11px]" aria-label={title}>
        <thead>
          <tr className="type-input-label">
            <th scope="col" className="px-2 py-1 text-start font-normal">{t("colGroup")}</th>
            <th scope="col" className="px-2 py-1 text-end font-normal">{t("colBorrowers")}</th>
            <th scope="col" className="px-2 py-1 text-end font-normal">{t("colLgd")}</th>
            <th scope="col" className="px-2 py-1 text-end font-normal">{t("colRecovered")}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} data-row={row.key}>
              <th scope="row" className="type-input-label border-b border-grid px-2 py-1 text-start text-[9px] font-normal text-soft">{row.label}</th>
              <td className="border-b border-grid px-2 py-1 text-end text-muted">{fmtCount(row.defaults)}</td>
              <td className="border-b border-grid px-2 py-1 text-end text-loss">{fmtPct(row.lgd_pct, 0)}</td>
              <td className="border-b border-grid px-2 py-1 text-end text-ink">{fmtPct(100 - row.lgd_pct, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <Cite rates={rates} table={cite} />
    </Tile>
  );
  return (
    <>
      {table(t("seniorityTitle"), r.by_seniority.map((x) => ({ ...x, label: t(`seniority_${x.key}`) })), "lgd_by_seniority")}
      {table(t("gcdRegionTitle"), r.by_region.map((x) => ({ ...x, label: t(`gcd_${x.key}`) })), "lgd_by_region")}
      {table(t("yearTitle"), r.by_year.map((x) => ({ key: String(x.year), label: String(x.year), defaults: x.defaults, lgd_pct: x.lgd_pct })), "lgd_by_year")}
    </>
  );
}
