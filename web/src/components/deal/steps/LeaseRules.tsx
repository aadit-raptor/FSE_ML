"use client";

import { useTranslations } from "next-intl";

import { SELECT_CLASS } from "@/components/ui/MoneySelects";
import type { DealInputs } from "@/lib/deal/fields";

import { useDeal } from "../DealProvider";
import { DealField } from "../DealScreen";

type Standard = DealInputs["accounting_standard"];
type LeaseView = DealInputs["lease_view"];

/** i18n-keys: deal.standard_, deal.standard_ifrs, deal.standard_us_gaap */
const STANDARDS: Standard[] = ["", "ifrs", "us_gaap"];
/** i18n-keys: deal.leaseView_, deal.leaseView_pre_ifrs16, deal.leaseView_post_ifrs16 */
const VIEWS: LeaseView[] = ["", "pre_ifrs16", "post_ifrs16"];

/**
 * The deal's accounting standard and leases (PLAN.md 2.6, core/accounting.py):
 * which standard the EBITDA follows (before lease costs under IFRS 16, after
 * them under US GAAP), what the leases cost a year and owe at close, and the
 * view the deal is priced on. With no lease cost and no liability nothing
 * moves, whatever the standard says.
 */
export function LeaseRules() {
  const { inputs, setFields } = useDeal();
  const t = useTranslations("deal");
  const fields = useTranslations("fields");
  const own = inputs.accounting_standard === "ifrs" ? t("leaseView_post_ifrs16") : t("leaseView_pre_ifrs16");
  return (
    <>
      <label className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
        <span className="type-input-label">{fields("accounting_standard")}</span>
        <select
          value={inputs.accounting_standard}
          onChange={(e) => setFields({ accounting_standard: e.target.value as Standard })}
          className={SELECT_CLASS}
        >
          {STANDARDS.map((s) => (
            <option key={s} value={s}>
              {t(`standard_${s}`)}
            </option>
          ))}
        </select>
      </label>
      <DealField name="lease_cost" />
      <DealField name="lease_liability" />
      <label className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
        <span className="type-input-label">{fields("lease_view")}</span>
        <select value={inputs.lease_view} onChange={(e) => setFields({ lease_view: e.target.value as LeaseView })} className={SELECT_CLASS}>
          {VIEWS.map((v) => (
            <option key={v} value={v}>
              {v ? t(`leaseView_${v}`) : t("leaseView_", { own })}
            </option>
          ))}
        </select>
      </label>
      <p className="type-body pt-1.5 text-[9px]" role="note">
        {t("leaseNote")}
      </p>
    </>
  );
}
