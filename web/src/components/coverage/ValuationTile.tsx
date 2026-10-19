"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { SecondaryButton } from "@/components/ui/Screen";
import { Kpi, Tile } from "@/components/ui/Tile";
import { fmtMultiple, fmtNumber, fmtRate } from "@/lib/format";
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

  return (
    <>
      <Kpi span={4} lead title={t("valuationDcf")} value={fmtNumber(v("v_ps"), 2)}
        sub={`${t("valuationDcfSub", { rate: fmtRate(v("v_rate"), 1), growth: fmtRate(v("v_g"), 1) })} · ${against(v("v_gap_dcf"))}`} />
      <Kpi span={4} lead title={t("valuationMultiple")} value={fmtNumber(v("v_mps"), 2)}
        sub={`${t(`valuationMultiple_${multiple}`, { multiple: fmtMultiple(v("v_mult"), 1), year: yearLabel })} · ${against(v("v_gap_mult"))}`} />
      <Kpi span={4} title={t("valuationPrice")} value={fmtNumber(price, 2)}
        sub={price === null ? t("valuationTypePrice") : dated ? t("valuationPriceOn", { date: dated }) : t("valuationNoDate")} />
      <Tile span={12} title={t("valuationTitle")} unit={company.unit_name}>
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
