"use client";

import { useTranslations } from "next-intl";
import Link from "next/link";
import { useCallback, type ReactNode } from "react";

import { FiscalSelects } from "@/components/ui/FiscalSelects";
import { MoneySelects } from "@/components/ui/MoneySelects";
import { NumberField } from "@/components/ui/NumberField";
import { Notice, PrimaryButton, RailGroup, Screen, SecondaryButton, Switch } from "@/components/ui/Screen";
import { multiplesFromPct, pctFromMultiples } from "@/lib/deal/capital";
import { FIELDS, type DealInputs, type FieldSpec, type NumericDealKey } from "@/lib/deal/fields";
import { fmtInput } from "@/lib/format";
import { monthName } from "@/lib/locale";

import { type DealSaveState, useDeal } from "./DealProvider";

export { LoadingTiles } from "@/components/ui/Screen";
export { RailGroup };

/** Deal rail (with auto-update) and results, with the changes / error bar. */
export function DealScreen({ rail, children }: { rail: ReactNode; children: ReactNode }) {
  return (
    <Screen
      rail={
        <>
          <RailHeader />
          {rail}
        </>
      }
      bar={<ChangesBar />}
    >
      {children}
    </Screen>
  );
}

function RailHeader() {
  const { autoUpdate, setAutoUpdate, run, current, saveState } = useDeal();
  const t = useTranslations("deal");
  const state =
    run.status === "running"
      ? t("stateUpdating")
      : run.status === "error"
        ? t("stateError")
        : run.status === "ok"
          ? t("stateUpToDate", { ms: Math.round(run.ms ?? 0) })
          : t("stateLoading");
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1.5 border-b border-line px-3.5 py-2">
      <Switch checked={autoUpdate} onChange={setAutoUpdate} label={t("autoUpdate")} />
      <span
        role="status"
        className={`font-mono text-[10px] whitespace-nowrap ${run.status === "error" ? "text-loss" : run.status === "running" ? "text-attention" : "text-dim"}`}
      >
        {state}
      </span>
      <Link
        href="/deal/saved"
        title={t("openDealTitle")}
        className="flex w-full min-w-0 items-center justify-between gap-2 hover:text-ink"
        data-deal-save={saveState}
      >
        <span className="type-input-label min-w-0 truncate">{current?.name ?? t("unsavedDeal")}</span>
        <span className={`font-mono text-[10px] whitespace-nowrap ${SAVE_TONE[saveState]}`} aria-live="polite">
          {t(SAVE_KEY[saveState])}
        </span>
      </Link>
    </div>
  );
}

/** i18n-keys: deal.saveNotSaved, deal.saveSaving, deal.saveSaved, deal.saveFailed */
const SAVE_KEY: Record<DealSaveState, string> = { unsaved: "saveNotSaved", saving: "saveSaving", saved: "saveSaved", error: "saveFailed" };
const SAVE_TONE: Record<DealSaveState, string> = { unsaved: "text-attention", saving: "text-dim", saved: "text-dim", error: "text-loss" };

/** i18n-keys: fields.* */
export function DealField({ name, disabled }: { name: NumericDealKey; disabled?: boolean }) {
  const { inputs, setField, pending, autoUpdate } = useDeal();
  const fields = useTranslations("fields");
  return (
    <NumberField
      spec={FIELDS[name]}
      label={fields(name)}
      value={inputs[name]}
      onCommit={(v) => setField(name, v)}
      disabled={disabled}
      changed={!autoUpdate && pending.includes(name)}
    />
  );
}

const MULTIPLE_SPEC: FieldSpec = { unit: "x", step: 0.1, decimals: 2, min: 0 };

/** Senior or mezz debt as a multiple of EBITDA; writes back debt % and senior %. */
export function DebtMultipleField({ tranche }: { tranche: "senior" | "mezz" }) {
  const { inputs, setFields, pending, autoUpdate } = useDeal();
  const fields = useTranslations("fields");
  const { seniorX, mezzX } = multiplesFromPct(inputs.entry_mult, inputs.debt_pct, inputs.senior_pct);
  const value = tranche === "senior" ? seniorX : mezzX;
  return (
    <NumberField
      spec={MULTIPLE_SPEC}
      label={tranche === "senior" ? fields("seniorDebt") : fields("mezzanineDebt")}
      value={Number(value.toFixed(4))}
      onCommit={(v) => {
        const next = tranche === "senior" ? pctFromMultiples(inputs.entry_mult, v, mezzX) : pctFromMultiples(inputs.entry_mult, seniorX, v);
        setFields({ debt_pct: Number(next.debtPct.toFixed(6)), senior_pct: Number(next.seniorPct.toFixed(6)) });
      }}
      changed={!autoUpdate && (pending.includes("debt_pct") || pending.includes("senior_pct"))}
    />
  );
}

/**
 * The deal's currency and the unit its money is entered and shown in. A new
 * unit keeps the deal's size (DealProvider.setMoney).
 */
export function MoneyFields() {
  const { money, setMoney } = useDeal();
  const t = useTranslations("deal");
  return <MoneySelects money={money} onChange={setMoney} of={t("of")} />;
}

/** The month the deal's fiscal year ends and its first projected year: labels only (PLAN.md 2.3a). */
export function FiscalFields() {
  const { inputs, setFields } = useDeal();
  const t = useTranslations("deal");
  const fiscal = useTranslations("fiscal");
  return (
    <FiscalSelects
      fiscal={{ endMonth: inputs.fiscal_year_end_month, year: inputs.first_fiscal_year }}
      onChange={(f) => setFields({ fiscal_year_end_month: f.endMonth, first_fiscal_year: f.year })}
      of={t("of")}
      yearLabel={fiscal("firstFiscalYear")}
    />
  );
}

export function WspToggle() {
  const { inputs, setField } = useDeal();
  const fields = useTranslations("fields");
  return (
    <div className="py-1">
      <Switch checked={inputs.wsp_mode} onChange={(v) => setField("wsp_mode", v)} label={fields("wsp_mode")} />
    </div>
  );
}

/**
 * Where a deal that lists its facilities edits its debt. The percentage fields
 * would do nothing for it (the model reads the list instead), so the screens
 * that show them point here rather than keep a control that changes nothing.
 */
export function TranchesOnDebtStep() {
  const { inputs } = useDeal();
  const t = useTranslations("deal");
  return (
    <div className="grid gap-1.5 py-1">
      <p className="type-body text-[9px]">{t("tranchesOnDebtStep", { count: inputs.tranches.length })}</p>
      <Link href="/deal/debt" className="type-action-secondary justify-self-start px-2.5 py-1.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)]">
        {t("editTranches")}
      </Link>
    </div>
  );
}

/** A changed input, named and valued the way the rail shows it. */
export function useDealChange(): (key: keyof DealInputs, value: DealInputs[keyof DealInputs]) => string {
  const fields = useTranslations("fields");
  const t = useTranslations("deal");
  return useCallback(
    (key, v) => {
      if (Array.isArray(v)) return t("facilityCount", { count: v.length });
      if (key === "fiscal_year_end_month" && typeof v === "number") return monthName(v);
      if (key === "first_fiscal_year") return v === null ? fields("none") : String(v);
      if (typeof v === "boolean") return v ? fields("on") : fields("off");
      if (typeof v !== "number") return String(v);
      const spec = FIELDS[key as NumericDealKey];
      return spec ? `${fmtInput(v, spec.decimals)}${spec.unit === "%" ? "%" : spec.unit === "x" ? "x" : ""}` : String(v);
    },
    [fields, t],
  );
}

/** The rail's name for a deal input. i18n-keys: fields.* */
export function useDealLabel(): (key: keyof DealInputs) => string {
  const fields = useTranslations("fields");
  return useCallback((key) => (fields.has(key) ? fields(key) : key), [fields]);
}

/** Error from the last run, or (with auto-update off) the edits waiting to run. */
function ChangesBar() {
  const { run, pending, settingsChanged, autoUpdate, runNow, discard, inputs } = useDeal();
  const t = useTranslations("deal");
  const label = useDealLabel();
  const describe = useDealChange();

  if (run.status === "error") {
    return (
      <Notice tone="loss" title={t("modelDidntRun")} role="alert" actions={<SecondaryButton onClick={runNow}>{t("tryAgain")}</SecondaryButton>}>
        {run.error}
      </Notice>
    );
  }
  if (autoUpdate || (!pending.length && !settingsChanged) || !run.ranFor) return null;

  const ranFor = run.ranFor;
  return (
    <Notice
      title={pending.length ? t("inputsChanged", { count: pending.length }) : t("settingsChanged")}
      actions={
        <>
          <SecondaryButton onClick={discard}>{t("discard")}</SecondaryButton>
          <PrimaryButton onClick={runNow}>{t("runModel")}</PrimaryButton>
        </>
      }
    >
      <span className="font-mono text-[11px]">
        {pending.map((k) => (
          <span key={k} className="me-3">
            {label(k)} {describe(k, ranFor[k])} → <b className="font-medium text-attention">{describe(k, inputs[k])}</b>
          </span>
        ))}
        {settingsChanged && <span>{t("settingsDifferFromRun")}</span>}
      </span>
    </Notice>
  );
}
