"use client";

import { useTranslations } from "next-intl";
import { useEffect, useMemo, useState } from "react";

import { useProfile } from "@/components/auth/ProfileProvider";
import { Notice, PrimaryButton, SecondaryButton } from "@/components/ui/Screen";
import { SELECT_CLASS } from "@/components/ui/MoneySelects";
import { Tile } from "@/components/ui/Tile";
import {
  fetchStarting, figureKind, type Starting, type StartingFigure, type StartingField, startingPatch, useIndustries,
} from "@/lib/benchmarks";
import type { DealInputs } from "@/lib/deal/fields";
import { fmtCount, fmtMultiple, fmtNumber, fmtPct } from "@/lib/format";
import { countryOptions, periodLabel, regionName } from "@/lib/locale";

import { useDeal } from "./DealProvider";
import { DealField, MoneyFields, useDealLabel } from "./DealScreen";

const ALL = "all";

/**
 * The deal's starting point (PLAN.md 4.3): its country, industry, currency and size. "Use sourced
 * figures" fills every input published data covers -- margins, multiples, capex, working capital,
 * leverage, growth, tax and the interest rate -- and records the country and industry the deal was
 * sourced for. The choices are a draft until then, so the labels always describe the figures.
 */
export function StartingPointFields() {
  const { inputs, setFields } = useDeal();
  const { profile } = useProfile();
  const t = useTranslations("starting");
  const industries = useIndustries();
  const countries = useMemo(() => countryOptions(), []);
  const [country, setCountry] = useState(inputs.country || profile?.country || "");
  const [industry, setIndustry] = useState(inputs.industry || ALL);
  const [state, setState] = useState<{ status: "idle" | "loading" | "done" | "error"; found?: number; missing?: number }>({ status: "idle" });
  // Another deal opened: its own country and industry become the draft
  const [shownFor, setShownFor] = useState({ country: inputs.country, industry: inputs.industry });
  if (shownFor.country !== inputs.country || shownFor.industry !== inputs.industry) {
    setShownFor({ country: inputs.country, industry: inputs.industry });
    if (inputs.country) setCountry(inputs.country);
    if (inputs.industry) setIndustry(inputs.industry);
  }

  const apply = async () => {
    setState({ status: "loading" });
    const s = await fetchStarting(country, industry, inputs.currency);
    if (!s) {
      setState({ status: "error" });
      return;
    }
    setFields(startingPatch(s));
    setState({ status: "done", found: Object.keys(s.inputs).length, missing: s.missing.length });
  };

  return (
    <>
      <label className="grid grid-cols-[1fr_128px] items-center gap-1.5 py-px">
        <span className="type-input-label">{t("country")}</span>
        <select aria-label={t("countryOf")} value={country} onChange={(e) => setCountry(e.target.value)} className={SELECT_CLASS}>
          <option value="">{t("choose")}</option>
          {countries.map((c) => (
            <option key={c.code} value={c.code}>
              {c.name}
            </option>
          ))}
        </select>
      </label>
      <label className="grid grid-cols-[1fr_128px] items-center gap-1.5 py-px">
        <span className="type-input-label">{t("industry")}</span>
        <select aria-label={t("industryOf")} value={industry} onChange={(e) => setIndustry(e.target.value)} className={SELECT_CLASS}>
          {(industries?.industries.length ? industries.industries : [{ id: ALL, name: "", firms: 0 }]).map((i) => (
            <option key={i.id} value={i.id}>
              {i.id === ALL ? t("allIndustries") : i.name}
            </option>
          ))}
        </select>
      </label>
      <MoneyFields />
      <DealField name="ebitda" />
      <p className="type-body py-1 text-[10px]">{t("sizeNote")}</p>
      <div className="flex flex-wrap items-center gap-2 pt-1">
        <PrimaryButton onClick={() => void apply()} disabled={!country || state.status === "loading"}>
          {state.status === "loading" ? t("applying") : t("apply")}
        </PrimaryButton>
      </div>
      {state.status === "done" && (
        <p role="status" className="font-mono text-[10px] text-dim">
          {t("applied", { found: state.found ?? 0, missing: state.missing ?? 0 })}
        </p>
      )}
      {state.status === "error" && (
        <p role="alert" className="font-mono text-[10px] text-loss">
          {t("applyFailed")}
        </p>
      )}
    </>
  );
}

/** i18n-keys: starting.area_*, starting.source_*, starting.reason_*, starting.note_*, starting.kind_* */
function useFigureText() {
  const t = useTranslations("starting");
  const areaName = (area: string) => (t.has(`area_${area}`) ? t(`area_${area}`) : /^[A-Z]{2}$/.test(area) ? regionName(area) : area);
  const value = (field: StartingField, v: number) => {
    const kind = figureKind(field);
    if (kind === "multiple") return fmtMultiple(v, 2);
    if (kind === "days") return t("days", { days: fmtNumber(v, 1) });
    return fmtPct(v, 2);
  };
  const sample = (f: StartingFigure) =>
    f.sample == null || !f.sample_kind ? "" : t(`kind_${f.sample_kind}`, { count: f.sample, shown: fmtCount(f.sample) });
  const skipped = (f: { skipped: StartingFigure["skipped"] }, min: number) =>
    f.skipped.map((s) => t(`reason_${s.reason}`, { area: areaName(s.area), sample: fmtCount(s.sample ?? 0), min: String(min) })).join(" ");
  return { t, areaName, value, sample, skipped };
}

/**
 * Every starting figure for the deal's country and industry, beside what the deal has now, with
 * its source, sample and date (PLAN.md 4.3). A figure the user changed shows both; "Use" puts the
 * sourced one back. Before a country is chosen the deal's figures are illustrative, and this says so.
 */
export function StartingFiguresTile() {
  const { inputs, setFields } = useDeal();
  const { t, areaName, value, sample, skipped } = useFigureText();
  const label = useDealLabel();
  const [starting, setStarting] = useState<Starting | null>(null);
  const { country, industry, currency } = inputs;

  useEffect(() => {
    if (!country) return;
    let live = true;
    void fetchStarting(country, industry || ALL, currency).then((s) => {
      if (live) setStarting(s);
    });
    return () => {
      live = false;
    };
  }, [country, industry, currency]);

  if (!country) {
    return (
      <Notice tone="attention" title={t("illustrativeTitle")} className="col-span-12">
        {t("illustrativeBody")}
      </Notice>
    );
  }
  // An answer for another country, industry or currency is never shown, nor applied by "Use all"
  if (!starting || starting.country !== country || starting.industry !== (industry || ALL) || starting.currency !== currency) return null;
  const differs = starting.figures.filter((f) => inputs[f.field as keyof DealInputs] !== f.value);
  const title = t("tileTitle", { country: regionName(country), industry: industry && industry !== ALL ? industryName(starting, industry) : t("allIndustries") });

  return (
    <Tile
      span={12}
      title={title}
      action={differs.length > 0 && (
        <SecondaryButton onClick={() => setFields(startingPatch(starting))}>{t("useAll", { count: differs.length })}</SecondaryButton>
      )}
    >
      <div className="overflow-x-auto">
        <table className="w-full border-collapse font-mono text-[11px]" aria-label={title}>
          <thead>
            <tr className="type-input-label text-start">
              <th scope="col" className="px-2 py-1 text-start font-normal">{t("colFigure")}</th>
              <th scope="col" className="px-2 py-1 text-end font-normal">{t("colSourced")}</th>
              <th scope="col" className="px-2 py-1 text-end font-normal">{t("colDeal")}</th>
              <th scope="col" className="px-2 py-1 text-start font-normal">{t("colSource")}</th>
              <th scope="col" className="sr-only">{t("colUse")}</th>
            </tr>
          </thead>
          <tbody>
            {starting.figures.map((f) => {
              const now = inputs[f.field as keyof DealInputs] as number;
              const same = now === f.value;
              return (
                <tr key={f.field} className="border-t border-line align-top" data-field={f.field}>
                  <th scope="row" className="type-input-label px-2 py-1 text-start font-normal">{label(f.field as keyof DealInputs)}</th>
                  <td className="px-2 py-1 text-end text-ink" data-sourced={f.field}>{value(f.field, f.value)}</td>
                  <td className={`px-2 py-1 text-end ${same ? "text-dim" : "text-attention"}`}>{value(f.field, now)}</td>
                  <td className="px-2 py-1 text-start text-muted">
                    <SourceLine f={f} area={areaName(f.area)} sample={sample(f)} />
                    {f.skipped.length > 0 && <span className="block text-attention">{skipped(f, starting.min_firms)}</span>}
                  </td>
                  <td className="px-2 py-1 text-end">
                    {!same && (
                      <button
                        type="button"
                        onClick={() => setFields({ [f.field]: f.value } as Partial<DealInputs>)}
                        className="type-action-secondary px-2 text-accent"
                        aria-label={t("useOne", { figure: label(f.field as keyof DealInputs) })}
                      >
                        {t("use")}
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
            {starting.missing.map((m) => (
              <tr key={m.field} className="border-t border-line" data-field={m.field}>
                <th scope="row" className="type-input-label px-2 py-1 text-start font-normal">{label(m.field as keyof DealInputs)}</th>
                <td colSpan={4} className="px-2 py-1 text-start text-attention">{t("missing")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="grid gap-0.5 font-mono text-[10px] text-dim">
        {starting.notes.map((n) => (
          <li key={n}>{t(`note_${n}`)}</li>
        ))}
        <li>
          {t("attribution", { publisher: starting.source.publisher })}{" "}
          <a href={starting.source.url} target="_blank" rel="noopener noreferrer" className="text-accent underline">
            {starting.source.title}
          </a>
        </li>
      </ul>
    </Tile>
  );
}

function industryName(s: Starting, id: string): string {
  return s.industry_name ?? id;
}

function SourceLine({ f, area, sample }: { f: StartingFigure; area: string; sample: string }) {
  const t = useTranslations("starting");
  const date = f.as_of ? periodLabel(f.as_of) : "";
  const text = [t(`source_${f.source}`), area, sample, date].filter(Boolean).join(" · ");
  return f.url ? (
    <a href={f.url} target="_blank" rel="noopener noreferrer" className="text-accent underline">
      {text}
    </a>
  ) : (
    <span>{text}</span>
  );
}
