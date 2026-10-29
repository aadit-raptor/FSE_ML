"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { quote } from "@/components/ui/Explain";
import { SecondaryButton } from "@/components/ui/Screen";
import { Kpi, Tile } from "@/components/ui/Tile";
import { fmtMultiple, fmtNumber, fmtRate } from "@/lib/format";
import { unitNameTerm } from "@/lib/glossary";
import { formatDateTime, regionName } from "@/lib/locale";
import {
  isoOfExcelDay, riskFree, valuationValue, withValue, type ModelCompany, type ModelRun, type RiskFree,
} from "@/lib/sectorModel";

/**
 * The valuation (PLAN.md R6), read from the model's own rows: a discounted cash flow and the analyst's target
 * multiple, each a value per share set against the price typed with its date. Every figure is a cell of the
 * workbook too, worked out by the same formulas; the assumptions are the yellow cells in last year's column of the
 * grid below. Nothing here is coloured or worded as a view: the tool states none and picks no rating.
 *
 * A covered company is offered its country's 10-year government yield (economic data, PLAN.md 4.2) as the
 * discount rate's starting point; the analyst adds the premium they judge right.
 *
 * i18n-keys: models.valuationMultiple_*
 * i18n-keys: models.riskFree_*
 * i18n-keys: explain.value_multiple_*
 */
export function ValuationTiles({
  run, company, companyId, onChange,
}: {
  run: ModelRun;
  company: ModelCompany;
  companyId: string | null;
  onChange: (company: ModelCompany) => void;
}) {
  const t = useTranslations("models");
  const x = useTranslations("explain");
  const [offer, setOffer] = useState<{ id: string; answer: RiskFree | null } | null>(null);

  useEffect(() => {
    let cancelled = false;
    if (!companyId) return;
    void riskFree(companyId).then((found) => {
      if (!cancelled) setOffer({ id: companyId, answer: found });
    });
    return () => {
      cancelled = true;
    };
  }, [companyId]);

  if (!run.rows.some((r) => r.key === "v_ps")) return null;          // an API from before R6
  const v = (row: string) => valuationValue(run, row);
  const price = v("v_price");
  const day = isoOfExcelDay(v("v_date"));
  const dated = day ? formatDateTime(new Date(`${day}T00:00:00Z`), { dateStyle: "medium", timeZone: "UTC" }) : "";
  const year = run.choices_made.value_year;
  const yearLabel = run.periods.find((p) => p.name === year)?.label ?? year ?? "";
  const multiple = run.choices_made.value_multiple ?? "ev_ebitda";
  const against = (gap: number | null) =>
    gap !== null ? t("valuationAgainst", { gap: fmtRate(gap, 1) }) : price === null ? t("valuationNoPrice") : t("valuationNoValue");
  const shown = companyId && offer?.id === companyId ? offer.answer : null;     // never another company's answer
  const rate = shown?.status === "ok" && typeof shown.value === "number" ? shown.value : null;
  // Every figure once, as the KPIs show it, for the KPIs and the explanation alike (PLAN.md 6.4)
  const figures = {
    dcf: fmtNumber(v("v_ps"), 2),
    rate: fmtRate(v("v_rate"), 1),
    growth: fmtRate(v("v_g"), 1),
    mps: fmtNumber(v("v_mps"), 2),
    multiple: fmtMultiple(v("v_mult"), 1),
    year: yearLabel,
    price: fmtNumber(price, 2),
  };
  const gapDcf = v("v_gap_dcf");
  const gapMult = v("v_gap_mult");
  const explained = [
    ...(v("v_ps") !== null ? [x.rich("value_dcf", { ...figures, f: quote })] : []),
    ...(v("v_mps") !== null ? [x.rich(`value_multiple_${multiple === "pe" ? "pe" : "ev_ebitda"}`, { ...figures, f: quote })] : []),
    price !== null && gapDcf !== null && gapMult !== null
      ? x.rich("value_price", { price: figures.price, gapDcf: fmtRate(gapDcf, 1), gapMult: fmtRate(gapMult, 1), f: quote })
      : x.rich("value_noPrice"),
    x.rich("value_none"),
  ];

  return (
    <>
      <Kpi span={4} lead term="valueDcf" title={t("valuationDcf")} value={figures.dcf}
        sub={`${t("valuationDcfSub", { rate: figures.rate, growth: figures.growth })} · ${against(gapDcf)}`} />
      <Kpi span={4} lead term="valueMultiple" title={t("valuationMultiple")} value={figures.mps}
        sub={`${t(`valuationMultiple_${multiple}`, { multiple: figures.multiple, year: yearLabel })} · ${against(gapMult)}`} />
      <Kpi span={4} term="priceTyped" title={t("valuationPrice")} value={figures.price}
        sub={price === null ? t("valuationTypePrice") : dated ? t("valuationPriceOn", { date: dated }) : t("valuationNoDate")} />
      <Tile span={12} term="valuation" title={t("valuationTitle")} unit={company.unit_name} unitTerm={unitNameTerm(company.unit_name)} explain={explained}>
        <p className="type-body text-[10.5px]">{t("valuationHow")}</p>
        {shown && rate !== null ? (
          <div className="flex flex-wrap items-center gap-3" data-risk-free={shown.country}>
            <p className="type-body text-[10.5px]">
              {t("riskFree_ok", {
                country: regionName(shown.country ?? ""), rate: fmtRate(rate, 2), period: shown.period ?? "",
                source: shown.source ?? "",
              })}{" "}
              {shown.url && <a className="text-accent underline" href={shown.url} target="_blank" rel="noreferrer">{t("riskFreeSource")}</a>}
            </p>
            <SecondaryButton onClick={() => onChange(withValue(company, "v_rate", "ly", rate))}>
              {t("riskFreeUse")}
            </SecondaryButton>
          </div>
        ) : (
          <p className="type-body text-[10.5px] text-muted">
            {companyId ? t(`riskFree_${shown?.status ?? "no_data"}`) : t("riskFreeCovered")}
          </p>
        )}
      </Tile>
    </>
  );
}
