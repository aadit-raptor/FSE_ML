"use client";

import { useTranslations } from "next-intl";

import { useMoney } from "@/components/ui/MoneyScope";
import { Tile } from "@/components/ui/Tile";
import { type components } from "@/lib/api/schema";
import { dealFiscal } from "@/lib/deal/fields";
import { fmtCount, fmtMoney, fmtMultiple, fmtPct } from "@/lib/format";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";

import { useDeal } from "./DealProvider";

type Warning = components["schemas"]["RiskWarning"];
type Source = components["schemas"]["RiskSource"];

/**
 * Risk warnings computed from the deal and published data (PLAN.md 2.8).
 *
 * The API sends figures, never sentences: each one is computed from the
 * deal's model run or read from the source the warning names, and the words
 * here come from `warnings.<id>`, which a test keeps free of numbers.
 */
export function DealWarnings() {
  const { inputs, run } = useDeal();
  const t = useTranslations("warnings");
  const warnings = run.result?.risk_warnings;
  if (!warnings) return null;
  return (
    <Tile span={12} title={t("tile")} unit={t("unit")}>
      {warnings.length ? (
        <ul className="grid gap-3" aria-label={t("tile")}>
          {warnings.map((w) => (
            <WarningRow key={w.id} warning={w} fiscal={dealFiscal(inputs)} />
          ))}
        </ul>
      ) : (
        <p className="type-body" data-warnings="none">
          {t("none")}
        </p>
      )}
    </Tile>
  );
}

function WarningRow({ warning, fiscal }: { warning: Warning; fiscal: ReturnType<typeof dealFiscal> }) {
  const t = useTranslations("warnings");
  const { label: mu } = useMoney();
  const fiscalLabels = useFiscalLabels();
  const f = warning.figures;
  const year = (n: number | null | undefined) => (n ? fiscalLabels.deal(n, fiscal)[n - 1] : "");
  const money = (v: number | null | undefined) => `${fmtMoney(v)} ${mu}`;
  // Every argument a message may ask for, formatted the account's way. Years
  // and counts go in as text where grouping would be wrong ("FY2,025").
  const args = {
    leverage: fmtMultiple(f.leverage, 1),
    threshold: fmtMultiple(f.threshold, 1),
    coverage: fmtMultiple(f.coverage, 2),
    band_low: fmtMultiple(f.band_low, 2),
    band_high: fmtMultiple(f.band_high, 2),
    default_pct: fmtPct(f.default_pct, 2),
    years: f.years ?? 0,
    count: f.count ?? 0,
    year: year(f.year),
    worst_year: year(f.worst_year),
    unfunded: money(f.unfunded),
    unfunded_total: money(f.unfunded_total),
    repayment_due: money(f.repayment_due),
    rating: warning.labels?.rating ?? "",
    study_row: warning.labels?.study_row ?? "",
  };
  // The weakest coverage band has no lower bound
  // i18n-keys: warnings.leverage_above_guidance, warnings.implied_rating, warnings.implied_rating_lowest
  // i18n-keys: warnings.interest_exceeds_ebitda, warnings.unfunded_repayment
  const key = warning.id === "implied_rating" && f.band_low == null ? "implied_rating_lowest" : warning.id;
  return (
    <li className="grid gap-1 shadow-[inset_2px_0_0_var(--color-attention)] ps-2" data-warning={warning.id}>
      <p className="type-body text-[10px] text-ink">{t(key, args)}</p>
      <p className="type-body text-[9px] text-muted" data-warning-sources={warning.id}>
        <span className="type-alert text-[9px]">{t("sources")}</span> {warning.sources.map((s) => <SourceLine key={s.id} source={s} />)}
      </p>
    </li>
  );
}

function SourceLine({ source }: { source: Source }) {
  const t = useTranslations("warnings");
  if (source.id === "deal_model") return <span className="me-2">{t("dealModel")}</span>;
  const sample = source.sample
    ? t("sample", {
        count: fmtCount(source.sample.count),
        first: String(source.sample.first_year),
        last: String(source.sample.last_year),
      })
    : null;
  return (
    <span className="me-2" data-source={source.id}>
      {source.url ? (
        <a href={source.url} target="_blank" rel="noreferrer" className="text-accent underline-offset-2 hover:underline">
          {source.publisher}, {source.title}
        </a>
      ) : (
        <>
          {source.publisher}, {source.title}
        </>
      )}
      {" "}({source.published}; {source.detail}
      {sample ? `; ${sample}` : ""})
    </span>
  );
}
