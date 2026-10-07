"use client";

import Link from "next/link";
import { useTranslations } from "next-intl";
import { Fragment, type ReactNode, useEffect, useState } from "react";

import { EmptyState, LoadingTiles, Notice, RailGroup, Screen } from "@/components/ui/Screen";
import { Tile, Tiles } from "@/components/ui/Tile";
import { type Schemas } from "@/lib/api/client";
import { fmtCount, fmtMoney, fmtMultiple, fmtPct } from "@/lib/format";
import { fetchReferences, REFERENCE_FIGURES, type ReferenceDeal, type ReferenceDeals, type ReferenceFigureName } from "@/lib/library";
import { periodLabel, regionName } from "@/lib/locale";
import { moneyLabel } from "@/lib/money";
import { stepHref } from "@/lib/nav";

import { LibraryGate } from "./LibraryGate";

type Figure = Schemas["ReferenceFigure"] & { basis?: string; year?: number };
type Part = Schemas["ReferencePart"];
type Source = Schemas["ReferenceSource"];

/**
 * Library -> Reference deals (PLAN.md 4.5b): the sourced reference transactions two administrators
 * approved, every figure with the filing it was read from, where in it and its words.
 */
export function ReferencesStep() {
  return (
    <LibraryGate>
      <ReferencesScreen />
    </LibraryGate>
  );
}

function ReferencesScreen() {
  const t = useTranslations("library");
  const [answer, setAnswer] = useState<ReferenceDeals | null | undefined>(undefined);
  useEffect(() => {
    let live = true;
    void fetchReferences().then((a) => live && setAnswer(a));
    return () => {
      live = false;
    };
  }, []);

  const rail = (
    <RailGroup title={t("groupReferences")}>
      <p className="type-body text-[9px]">{t("referencesRail")}</p>
      {answer && answer.awaiting_review > 0 && (
        <p className="font-mono text-[10.5px] text-attention" data-testid="awaiting-review">
          {t("awaitingReview", { count: answer.awaiting_review, shown: fmtCount(answer.awaiting_review) })}
        </p>
      )}
      <Link href={stepHref("library", "review")} className="type-control text-accent underline">
        {t("openReview")}
      </Link>
    </RailGroup>
  );
  if (answer === undefined) return <Screen rail={rail}><LoadingTiles /></Screen>;
  if (!answer) {
    return (
      <Screen rail={rail}>
        <EmptyState title={t("unreachableTitle")}>{t("unreachable")}</EmptyState>
      </Screen>
    );
  }
  if (!answer.deals.length) {
    return (
      <Screen rail={rail}>
        <EmptyState title={t("noReferencesTitle")}>{t("noReferences")}</EmptyState>
      </Screen>
    );
  }
  return (
    <Screen rail={rail}>
      <Notice tone="info" title={t("referencesNoticeTitle")} role="note">
        {t("referencesNotice", { count: answer.deals.length, shown: fmtCount(answer.deals.length) })}
      </Notice>
      <Tiles>
        {answer.deals.map((d) => (
          <ReferenceDealTile key={d.key} deal={d} />
        ))}
      </Tiles>
    </Screen>
  );
}

/** i18n-keys: library.sector_*, library.kind_*, library.event_*, library.outcome_* */
export function ReferenceDealTile({ deal, aside }: { deal: ReferenceDeal; aside?: ReactNode }) {
  const t = useTranslations("library");
  const d = deal.derived;
  const tone = deal.outcome.kind === "success" ? "text-gain" : deal.outcome.kind === "distress" ? "text-loss" : "text-dim";
  return (
    <Tile
      span={12}
      title={deal.target}
      aside={
        aside ?? (
          <span className="flex gap-1.5">
            <span className="chip text-gain">{t("sourced")}</span>
            <span className={`chip ${tone}`}>{t(`outcome_${deal.outcome.kind}`)}</span>
          </span>
        )
      }
    >
      <div className="grid gap-2" data-reference={deal.key}>
        <p className="font-mono text-[10.5px] text-muted">
          {t("referenceWho", {
            sponsors: deal.sponsors.join(", "),
            closed: periodLabel(deal.closed.date),
            country: regionName(deal.country),
            sector: t(`sector_${deal.sector}`),
            kind: t(`kind_${deal.kind}`),
          })}
        </p>
        <dl className="grid grid-cols-5 gap-2 font-mono text-[11px]" data-derived={deal.key}>
          <Derived label={t("derivedMultiple")} value={fmtMultiple(d.entry_multiple, 1)} />
          <Derived label={t("derivedLeverage")} value={fmtMultiple(d.leverage, 1)} />
          <Derived label={t("derivedTxFee")} value={fmtPct(d.tx_fee_pct, 2)} />
          <Derived label={t("derivedFinFee")} value={fmtPct(d.fin_fee_pct, 2)} />
          <Derived label={t("derivedAmort")} value={fmtPct(d.def_senior_amort, 1)} />
        </dl>
        <FigureTable deal={deal} />
        <p className="type-body text-[9.5px]" data-outcome={deal.key}>
          <span className={tone}>{t(`event_${deal.outcome.event}`, { year: String(deal.outcome.year) })}</span>{" "}
          <Cite deal={deal} cited={deal.outcome} />
        </p>
        {deal.outcome.note && <p className="type-body text-[9px] text-dim">{deal.outcome.note}</p>}
      </div>
    </Tile>
  );
}

function Derived({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="type-input-label">{label}</dt>
      <dd className="text-ink">{value}</dd>
    </div>
  );
}

/** i18n-keys: library.figure_*, library.basis_* */
function FigureTable({ deal }: { deal: ReferenceDeal }) {
  const t = useTranslations("library");
  const money = moneyLabel({ currency: deal.currency, unit: "millions" });
  const usd = moneyLabel({ currency: "USD", unit: "millions" });
  const caption = t("figuresOf", { target: deal.target });
  const shown = REFERENCE_FIGURES.filter((n) => deal.figures[n]);
  const fmt = (name: ReferenceFigureName, v: number) => (name === "senior_amort_pct" ? fmtPct(v, 2) : fmtMoney(v));
  const unit = (name: ReferenceFigureName) => (name === "senior_amort_pct" ? t("perYear") : name === "transaction_value_usd" ? usd : money);
  return (
    <table className="w-full border-collapse font-mono text-[11px]" aria-label={caption}>
      <thead>
        <tr className="type-input-label">
          <th scope="col" className="px-1 py-1 text-start font-normal">{t("colFigure")}</th>
          <th scope="col" className="px-1 py-1 text-end font-normal">{t("colValue")}</th>
          <th scope="col" className="px-1 py-1 text-start font-normal">{t("colSource")}</th>
        </tr>
      </thead>
      <tbody>
        {shown.map((name) => {
          const fig = deal.figures[name] as Figure;
          return (
            <Fragment key={name}>
              <tr data-figure={name} className="align-top">
                <th scope="row" className="border-b border-grid px-1 py-1 text-start text-[10px] font-normal text-soft">
                  {t(`figure_${name}`)}
                  <span className="block text-[9px] text-dim">
                    {unit(name)}
                    {fig.basis ? ` · ${t(`basis_${fig.basis}`, { year: String(fig.year ?? "") })}` : ""}
                  </span>
                </th>
                <td className="border-b border-grid px-1 py-1 text-end text-ink">{fmt(name, fig.value)}</td>
                <td className="border-b border-grid px-1 py-1 text-[10px]">
                  {fig.parts?.length ? <span className="text-dim">{t("sumOfParts", { count: fig.parts.length })}</span> : <Cite deal={deal} cited={fig} />}
                  {fig.note && <span className="type-body block text-[9px] text-dim">{fig.note}</span>}
                </td>
              </tr>
              {fig.parts?.map((part: Part, i: number) => (
                <tr key={`${name}.${i}`} data-part={`${name}.${i}`} className="align-top">
                  <th scope="row" className="border-b border-grid py-0.5 ps-4 pe-1 text-start text-[9.5px] font-normal text-muted">
                    {part.label}
                  </th>
                  <td className="border-b border-grid px-1 py-0.5 text-end text-soft">{fmt(name, part.value)}</td>
                  <td className="border-b border-grid px-1 py-0.5 text-[10px]">
                    <Cite deal={deal} cited={part} />
                  </td>
                </tr>
              ))}
            </Fragment>
          );
        })}
      </tbody>
    </table>
  );
}

/** A figure's filing, as a link, with where in it and its words. */
function Cite({ deal, cited }: { deal: ReferenceDeal; cited: { source?: string | null; where?: string | null; quote?: string | null } }) {
  const t = useTranslations("library");
  const source: Source | undefined = cited.source ? deal.sources[cited.source] : undefined;
  if (!source) return <span className="text-loss">{t("noSource")}</span>;
  return (
    <span className="text-soft">
      <a href={source.url} target="_blank" rel="noreferrer" className="text-accent underline" data-source={cited.source ?? ""}>
        {t("filing", { filer: source.filer, form: source.form, filed: periodLabel(source.filed) })}
      </a>
      {cited.where ? ` · ${cited.where}` : ""}
      {cited.quote ? <span className="block text-[9px] text-dim">{t("quoted", { quote: cited.quote })}</span> : null}
    </span>
  );
}
