"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { SecondaryButton } from "@/components/ui/Screen";
import { Tile } from "@/components/ui/Tile";
import { api } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";
import type { DealInputs } from "@/lib/deal/fields";
import { apiInputs } from "@/lib/deal/fields";
import { fmtCount, fmtMultiple, fmtNumber } from "@/lib/format";
import { monthYear } from "@/lib/locale";

import { useDeal } from "./DealProvider";

type Answer = components["schemas"]["MultiplesResponse"];
type Range = components["schemas"]["MultipleRange"];

const RANGES = [
  { key: "entry", field: "entry_mult" },
  { key: "exit", field: "exit_mult" },
] as const;
type RangeName = (typeof RANGES)[number]["key"];

/**
 * Entry and exit multiples by region (PLAN.md 5.4): the deal industry's
 * latest EV/EBITDA among its region's listed companies, the range each
 * multiple has moved within over the years ahead (each shown only where the
 * model card says it beats the region's whole market), "Use suggestion",
 * and the comparables: the industry in other regions, its sector in this
 * one and, while the library is on, reference transactions like it.
 *
 * i18n-keys: multiples.range_*, multiples.hidden_*, multiples.why_*, starting.area_*, library.region_*, library.sector_*,
 * i18n-keys: library.bucket_*, library.event_*
 */
export function Multiples() {
  const { inputs } = useDeal();
  const t = useTranslations("multiples");
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [error, setError] = useState<string>();
  const unavailable = t("unavailable");
  // Only what the ranges read: a change elsewhere in the deal asks nothing
  const { country, industry, hold, currency, unit, ebitda, entry_mult } = inputs;

  useEffect(() => {
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      api
        .POST("/api/ml/multiples", {
          body: { inputs: apiInputs(inputs) },
          signal: ctrl.signal,
        })
        .then(({ data, error: err }) => {
          if (data) {
            setAnswer(data);
            setError(undefined);
          } else setError(String((err as { detail?: unknown })?.detail ?? unavailable));
        })
        .catch(() => {});
    }, 400);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the other inputs don't move the ranges
  }, [country, industry, hold, currency, unit, ebitda, entry_mult, unavailable]);

  return (
    <Tile span={12} title={t("tile")} unit={t("unit")}>
      {error && <p className="font-mono text-[10.5px] text-loss">{error}</p>}
      {!answer && !error && <p className="type-body">{t("reading")}</p>}
      {answer && <Body answer={answer} />}
    </Tile>
  );
}

function Body({ answer }: { answer: Answer }) {
  const t = useTranslations("multiples");
  const area = useTranslations("starting");
  const industry = answer.industry_name ?? answer.industry;

  if (answer.status === "not_enough_data") {
    const thin = [...new Set(answer.skipped.map((s) => area(`area_${s.area}`)))];
    return (
      <div className="grid gap-2" data-multiples-status="not_enough_data">
        <p className="type-body">
          <span className="type-alert text-[9px]">{t("notEnoughData")}</span>{" "}
          {answer.reason === "no_country" ? t("noCountry") : t("noPeers", { min: answer.min_firms, industry, areas: thin.join(", ") })}
        </p>
        {answer.regions.length > 0 && <Regions answer={answer} />}
      </div>
    );
  }

  return (
    <div className="grid gap-3" data-multiples-status="ok">
      <p className="font-mono text-[10.5px] text-ink" data-provenance="multiples">
        {t("latest", {
          multiple: fmtMultiple(answer.latest, 1),
          year: String(answer.latest_year),
          firms: answer.firms ?? 0,
          group: answer.group ? area(`area_${answer.group}`) : "",
          industry,
        })}
      </p>
      <RangesTable answer={answer} />
      <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-4">
        <Regions answer={answer} />
        <Sector answer={answer} />
      </div>
      {answer.deals.enabled && <Deals answer={answer} />}
      <p className="type-body text-[9px]">
        {t("sourceNote", {
          publisher: answer.source.publisher,
          date: answer.published ? monthYear(answer.published.slice(0, 7)) : "",
        })}{" "}
        <a href={answer.source.url} target="_blank" rel="noopener noreferrer" className="text-accent underline">
          {answer.source.title}
        </a>
      </p>
    </div>
  );
}

function RangesTable({ answer }: { answer: Answer }) {
  const t = useTranslations("multiples");
  const lib = useTranslations("library");
  const { inputs, setFields } = useDeal();
  const suggestion: Partial<DealInputs> = {};
  for (const { key, field } of RANGES) {
    const r = answer[key];
    if (r?.shown && r.range) suggestion[field] = r.range.median;
  }
  const differs = RANGES.some(({ field }) => suggestion[field] != null && suggestion[field] !== inputs[field]);
  return (
    <div className="grid gap-1.5">
      <table className="w-full font-mono text-[10.5px]" aria-label={t("tableLabel")}>
        <thead>
          <tr className="text-dim">
            <th className="text-start font-normal">{t("colRange")}</th>
            <th className="text-end font-normal">{t("colLow")}</th>
            <th className="text-end font-normal">{t("colSuggestion")}</th>
            <th className="text-end font-normal">{t("colHigh")}</th>
            <th className="text-end font-normal">{t("colDeal")}</th>
          </tr>
        </thead>
        <tbody>
          {RANGES.map(({ key, field }) => (
            <RangeRow key={key} name={key} r={answer[key]} deal={inputs[field]} />
          ))}
        </tbody>
      </table>
      <div className="flex flex-wrap items-center gap-3">
        {Object.keys(suggestion).length > 0 && (
          <SecondaryButton onClick={() => setFields(suggestion)} disabled={!differs}>
            {t("useSuggestion")}
          </SecondaryButton>
        )}
        <p className="font-mono text-[10px] text-muted">{t("rangeNote", { hold: answer.hold })}</p>
      </div>
      {RANGES.map(({ key }) => {
        const r = answer[key];
        if (!r) return null;
        const c = r.card;
        // The region whose cases the card tested: the peer group's
        const region = c.region ? lib(`region_${c.region}`) : "";
        return (
          <p key={key} className="font-mono text-[10px] text-muted" data-card={key}>
            {r.shown || !r.hidden
              ? t("tested", { range: t(`range_${key}`), cases: c.cases, region, model: fmtNumber(c.model, 1), baseline: fmtNumber(c.baseline, 1) })
              : t(`why_${r.hidden}`, { range: t(`range_${key}`), region, min: answer.min_pairs })}
          </p>
        );
      })}
    </div>
  );
}

function RangeRow({ name, r, deal }: { name: RangeName; r: Range | null | undefined; deal: number }) {
  const t = useTranslations("multiples");
  const b = r?.shown ? r.range : null;
  return (
    <tr data-range={name}>
      <td className="text-ink">
        {t(`range_${name}`)}
        {r && <span className="ms-1 text-dim">{String(r.year)}</span>}
      </td>
      {b ? (
        <>
          <td className="text-end text-muted">{fmtMultiple(b.low, 1)}</td>
          <td className="text-end text-accent">{fmtMultiple(b.median, 1)}</td>
          <td className="text-end text-muted">{fmtMultiple(b.high, 1)}</td>
        </>
      ) : (
        <td colSpan={3} className="text-end text-attention">
          {t(`hidden_${r?.hidden ?? "not_enough_data"}`)}
        </td>
      )}
      <td className="text-end text-ink">{fmtMultiple(deal, 1)}</td>
    </tr>
  );
}

function Regions({ answer }: { answer: Answer }) {
  const t = useTranslations("multiples");
  const area = useTranslations("starting");
  return (
    <div className="grid content-start gap-1">
      <p className="font-mono text-[10px] text-dim">{t("regionsTitle")}</p>
      <ul className="grid gap-0.5" aria-label={t("regionsTitle")}>
        {answer.regions.map((r) => (
          <li key={r.group} className={`flex justify-between gap-2 font-mono text-[10.5px] ${r.used ? "text-accent" : r.enough ? "text-ink" : "text-dim"}`}>
            <span className="truncate">{area(`area_${r.group}`)}</span>
            <span>
              {fmtMultiple(r.multiple, 1)} · {t("firms", { count: r.firms ?? 0 })}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Sector({ answer }: { answer: Answer }) {
  const t = useTranslations("multiples");
  const area = useTranslations("starting");
  if (!answer.sector.length) return <p className="type-body text-[9px]">{t("sectorNone")}</p>;
  return (
    <div className="grid content-start gap-1">
      <p className="font-mono text-[10px] text-dim">{t("sectorTitle", { group: answer.group ? area(`area_${answer.group}`) : "" })}</p>
      <ul className="grid gap-0.5" aria-label={t("sectorLabel")}>
        {answer.sector.map((s) => (
          <li key={s.industry} className={`flex justify-between gap-2 font-mono text-[10.5px] ${s.this ? "text-accent" : "text-ink"}`}>
            <span className="truncate">{s.name}</span>
            <span>
              {fmtMultiple(s.multiple, 1)} · {fmtCount(s.firms)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Deals({ answer }: { answer: Answer }) {
  const t = useTranslations("multiples");
  const lib = useTranslations("library");
  const d = answer.deals;
  if (!d.sector) return <p className="type-body text-[9px]">{t("dealsNoSector")}</p>;
  const region = d.region ? lib(`region_${d.region}`) : "";
  return (
    <div className="grid gap-1" data-multiples-deals>
      <p className="font-mono text-[10px] text-dim">{t("dealsTitle", { count: d.deals.length, region, sector: lib(`sector_${d.sector}`) })}</p>
      {d.deals.length === 0 && <p className="type-body text-[9px]">{t("dealsNone")}</p>}
      <ul className="grid gap-1" aria-label={t("dealsLabel")}>
        {d.deals.map((x) => (
          <li key={x.key} className="font-mono text-[10.5px] text-ink">
            {x.target} · {String(x.year)} · {t("dealEntry", { entry: fmtMultiple(x.entry_multiple, 1) })}
            {x.size && <> · {lib(`bucket_${x.size}`)}</>}
            <span className={`ms-1 ${x.outcome === "distress" ? "text-loss" : x.outcome === "success" ? "text-gain" : "text-muted"}`}>
              {lib(`event_${x.event}`)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
