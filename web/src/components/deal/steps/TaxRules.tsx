"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { SELECT_CLASS } from "@/components/ui/MoneySelects";
import { Switch } from "@/components/ui/Screen";
import { api, type Schemas } from "@/lib/api/client";
import { type DealInputs, TAX_RULE_DEFAULTS } from "@/lib/deal/fields";
import { regionName } from "@/lib/locale";
import { unitFactor } from "@/lib/money";

import { useDeal } from "../DealProvider";
import { DealField } from "../DealScreen";

type Preset = Schemas["TaxPresetOut"];
type InterestLimit = DealInputs["tax_interest_limit"];

/** i18n-keys: deal.taxLimit_none, deal.taxLimit_ebitda_share, deal.taxLimit_fixed */
const INTEREST_LIMITS: InterestLimit[] = ["none", "ebitda_share", "fixed"];

// The presets never change while the app runs: ask once
let presetsOnce: Promise<Preset[]> | null = null;
function loadPresets(): Promise<Preset[]> {
  presetsOnce ??= api
    .GET("/api/deal/tax-presets")
    .then(({ data }) => data?.presets ?? [])
    .catch(() => {
      presetsOnce = null;
      return [];
    });
  return presetsOnce;
}

function usePresets(): Preset[] {
  const [presets, setPresets] = useState<Preset[]>([]);
  useEffect(() => {
    let live = true;
    void loadPresets().then((p) => live && setPresets(p));
    return () => {
      live = false;
    };
  }, []);
  return presets;
}

/**
 * The deal's tax (PLAN.md 2.5): its rate, a country preset to start from, and
 * the rules a buyout meets -- an interest limit, losses carried forward, a
 * minimum tax. A preset fills every field, all of which stay editable; its
 * amounts come across only in its own currency (core/tax.py apply_preset,
 * mirrored here).
 */
export function TaxRules() {
  const { inputs, setFields } = useDeal();
  const t = useTranslations("deal");
  const fields = useTranslations("fields");
  const presets = usePresets();
  const preset = presets.find((p) => p.code === inputs.tax_preset);
  const limit = inputs.tax_interest_limit;
  const carry = inputs.tax_loss_carryforward;

  const apply = (code: string) => {
    const p = presets.find((x) => x.code === code);
    if (!p) {
      setFields({ ...TAX_RULE_DEFAULTS });
      return;
    }
    const k = inputs.currency === p.currency ? unitFactor("millions", inputs.unit) : 0;
    setFields({
      tax: p.rate,
      tax_preset: p.code as DealInputs["tax_preset"],
      tax_interest_limit: p.interest_limit as InterestLimit,
      tax_interest_limit_pct: p.interest_limit_pct,
      tax_interest_limit_amount: p.interest_limit_amount * k,
      tax_loss_carryforward: p.loss_carryforward,
      tax_loss_limit_pct: p.loss_limit_pct,
      tax_loss_limit_amount: p.loss_limit_amount * k,
      tax_minimum_pct: p.minimum_pct,
    });
  };
  const amountsSkipped =
    preset !== undefined && preset.currency !== inputs.currency && (preset.interest_limit_amount > 0 || preset.loss_limit_amount > 0);

  return (
    <>
      <label className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
        <span className="type-input-label">{fields("tax_preset")}</span>
        <select value={inputs.tax_preset} onChange={(e) => apply(e.target.value)} className={SELECT_CLASS}>
          <option value="">{t("taxPresetNone")}</option>
          {presets.map((p) => (
            <option key={p.code} value={p.code}>
              {regionName(p.code)}
            </option>
          ))}
        </select>
      </label>
      <DealField name="tax" />
      <label className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
        <span className="type-input-label">{fields("tax_interest_limit")}</span>
        <select value={limit} onChange={(e) => setFields({ tax_interest_limit: e.target.value as InterestLimit })} className={SELECT_CLASS}>
          {INTEREST_LIMITS.map((l) => (
            <option key={l} value={l}>
              {t(`taxLimit_${l}`)}
            </option>
          ))}
        </select>
      </label>
      <DealField name="tax_interest_limit_pct" disabled={limit !== "ebitda_share"} />
      <DealField
        name="tax_interest_limit_amount"
        disabled={limit === "none"}
        label={limit === "fixed" ? fields("tax_interest_limit_cap") : fields("tax_interest_limit_amount")}
      />
      <div className="py-1">
        <Switch checked={carry} onChange={(v) => setFields({ tax_loss_carryforward: v })} label={fields("tax_loss_carryforward")} />
      </div>
      <DealField name="tax_loss_limit_amount" disabled={!carry} />
      <DealField name="tax_loss_limit_pct" disabled={!carry} />
      <DealField name="tax_minimum_pct" />
      <div className="grid gap-1 pt-1.5" role="note" data-testid="tax-preset-note">
        <p className="type-alert text-[9px]">{t("taxAdviser")}</p>
        {preset && (
          <p className="type-body text-[9px]">
            {t("taxPresetSource", { country: regionName(preset.code), date: preset.as_of, source: preset.source })}
            {preset.note && ` ${preset.note}`}
          </p>
        )}
        {amountsSkipped && <p className="type-body text-[9px] text-attention">{t("taxAmountsNotApplied", { currency: preset.currency })}</p>}
      </div>
    </>
  );
}
