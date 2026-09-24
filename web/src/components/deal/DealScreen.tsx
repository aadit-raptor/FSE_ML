"use client";

import Link from "next/link";
import type { ReactNode } from "react";

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
  const state =
    run.status === "running" ? "Updating" : run.status === "error" ? "Error" : run.status === "ok" ? `Up to date · ${Math.round(run.ms ?? 0)} ms` : "Loading";
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1.5 border-b border-line px-3.5 py-2">
      <Switch checked={autoUpdate} onChange={setAutoUpdate} label="Auto-update" />
      <span
        role="status"
        className={`font-mono text-[10px] whitespace-nowrap ${run.status === "error" ? "text-loss" : run.status === "running" ? "text-attention" : "text-dim"}`}
      >
        {state}
      </span>
      <Link
        href="/deal/saved"
        title="Saved deals and version history"
        className="flex w-full min-w-0 items-center justify-between gap-2 hover:text-ink"
        data-deal-save={saveState}
      >
        <span className="type-input-label min-w-0 truncate">{current?.name ?? "Unsaved deal"}</span>
        <span className={`font-mono text-[10px] whitespace-nowrap ${SAVE_TONE[saveState]}`} aria-live="polite">
          {SAVE_TEXT[saveState]}
        </span>
      </Link>
    </div>
  );
}

const SAVE_TEXT: Record<DealSaveState, string> = { unsaved: "Not saved", saving: "Saving", saved: "Saved", error: "Save failed" };
const SAVE_TONE: Record<DealSaveState, string> = { unsaved: "text-attention", saving: "text-dim", saved: "text-dim", error: "text-loss" };

export function DealField({ name, disabled }: { name: NumericDealKey; disabled?: boolean }) {
  const { inputs, setField, pending, autoUpdate } = useDeal();
  return (
    <NumberField
      spec={FIELDS[name]}
      value={inputs[name]}
      onCommit={(v) => setField(name, v)}
      disabled={disabled}
      changed={!autoUpdate && pending.includes(name)}
    />
  );
}

const MULTIPLE_SPEC: FieldSpec = { label: "", unit: "x", step: 0.1, decimals: 2, min: 0 };

/** Senior or mezz debt as a multiple of EBITDA; writes back debt % and senior %. */
export function DebtMultipleField({ tranche }: { tranche: "senior" | "mezz" }) {
  const { inputs, setFields, pending, autoUpdate } = useDeal();
  const { seniorX, mezzX } = multiplesFromPct(inputs.entry_mult, inputs.debt_pct, inputs.senior_pct);
  const value = tranche === "senior" ? seniorX : mezzX;
  return (
    <NumberField
      spec={MULTIPLE_SPEC}
      label={tranche === "senior" ? "Senior debt" : "Mezzanine debt"}
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
  return <MoneySelects money={money} onChange={setMoney} of="Deal" />;
}

/** The month the deal's fiscal year ends and its first projected year: labels only (PLAN.md 2.3a). */
export function FiscalFields() {
  const { inputs, setFields } = useDeal();
  return (
    <FiscalSelects
      fiscal={{ endMonth: inputs.fiscal_year_end_month, year: inputs.first_fiscal_year }}
      onChange={(f) => setFields({ fiscal_year_end_month: f.endMonth, first_fiscal_year: f.year })}
      of="Deal"
      yearLabel="First fiscal year"
    />
  );
}

export function WspToggle() {
  const { inputs, setField } = useDeal();
  return (
    <div className="py-1">
      <Switch checked={inputs.wsp_mode} onChange={(v) => setField("wsp_mode", v)} label="Working capital from days" />
    </div>
  );
}

export function describeDealValue(key: keyof DealInputs, v: DealInputs[keyof DealInputs]): string {
  if (key === "fiscal_year_end_month" && typeof v === "number") return monthName(v);
  if (key === "first_fiscal_year") return v === null ? "none" : String(v);
  if (typeof v === "boolean") return v ? "on" : "off";
  if (typeof v !== "number") return String(v);
  const spec = FIELDS[key as NumericDealKey];
  return spec ? `${fmtInput(v, spec.decimals)}${spec.unit === "%" ? "%" : spec.unit === "x" ? "x" : ""}` : String(v);
}

export function dealLabel(key: keyof DealInputs): string {
  if (key === "wsp_mode") return "Working capital from days";
  if (key === "currency") return "Currency";
  if (key === "unit") return "Amounts in";
  if (key === "fiscal_year_end_month") return "Fiscal year end";
  if (key === "first_fiscal_year") return "First fiscal year";
  return FIELDS[key as NumericDealKey]?.label ?? key;
}

/** Error from the last run, or (with auto-update off) the edits waiting to run. */
function ChangesBar() {
  const { run, pending, settingsChanged, autoUpdate, runNow, discard, inputs } = useDeal();

  if (run.status === "error") {
    return (
      <Notice tone="loss" title="Model didn't run" role="alert" actions={<SecondaryButton onClick={runNow}>Try again</SecondaryButton>}>
        {run.error}
      </Notice>
    );
  }
  if (autoUpdate || (!pending.length && !settingsChanged) || !run.ranFor) return null;

  const ranFor = run.ranFor;
  return (
    <Notice
      title={pending.length ? `${pending.length} input${pending.length > 1 ? "s" : ""} changed` : "Settings changed"}
      actions={
        <>
          <SecondaryButton onClick={discard}>Discard</SecondaryButton>
          <PrimaryButton onClick={runNow}>Run model</PrimaryButton>
        </>
      }
    >
      <span className="font-mono text-[11px]">
        {pending.map((k) => (
          <span key={k} className="mr-3">
            {dealLabel(k)} {describeDealValue(k, ranFor[k])} → <b className="font-medium text-attention">{describeDealValue(k, inputs[k])}</b>
          </span>
        ))}
        {settingsChanged && <span>Settings differ from the last run</span>}
      </span>
    </Notice>
  );
}
