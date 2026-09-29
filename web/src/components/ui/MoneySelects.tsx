"use client";

import { useTranslations } from "next-intl";
import { useMemo } from "react";

import { currencyCodes, currencyName, type Money, MONEY_UNITS, type MoneyUnit } from "@/lib/money";

export const SELECT_CLASS =
  "min-w-0 border border-line bg-field px-1 py-0.5 font-mono text-[11px] text-ink outline-none focus:border-accent";

/**
 * Currency (any ISO 4217 code the browser knows) and the unit money is
 * entered and shown in. `of` names what they belong to, for screen readers:
 * "Deal currency", "Company money unit".
 */
export function MoneySelects({ money, onChange, of }: { money: Money; onChange: (money: Money) => void; of: string }) {
  const t = useTranslations("money");
  const codes = useMemo(() => {
    const all = currencyCodes();
    return all.includes(money.currency) ? all : [money.currency, ...all];
  }, [money.currency]);
  return (
    <>
      <label className="grid grid-cols-[1fr_128px] items-center gap-1.5 py-px">
        <span className="type-input-label">{t("currency")}</span>
        <select aria-label={t("currencyOf", { of })} value={money.currency} onChange={(e) => onChange({ ...money, currency: e.target.value })} className={SELECT_CLASS}>
          {codes.map((c) => (
            <option key={c} value={c}>
              {currencyName(c)}
            </option>
          ))}
        </select>
      </label>
      <label className="grid grid-cols-[1fr_128px] items-center gap-1.5 py-px">
        <span className="type-input-label">{t("amountsIn")}</span>
        <select
          aria-label={t("unitOf", { of })}
          value={money.unit}
          onChange={(e) => onChange({ ...money, unit: e.target.value as MoneyUnit })}
          className={SELECT_CLASS}
        >
          {MONEY_UNITS.map((u) => (
            <option key={u.value} value={u.value}>
              {t(u.labelKey)}
            </option>
          ))}
        </select>
      </label>
    </>
  );
}
