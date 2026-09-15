"use client";

import type { ReactNode } from "react";

import { NumberField } from "@/components/ui/NumberField";
import { multiplesFromPct, pctFromMultiples } from "@/lib/deal/capital";
import { FIELDS, type DealInputs, type FieldSpec, type NumericDealKey } from "@/lib/deal/fields";
import { fmtInput } from "@/lib/format";

import { useDeal } from "./DealProvider";

/** Input rail on the left, results on the right. */
export function DealScreen({ rail, children }: { rail: ReactNode; children: ReactNode }) {
  return (
    <div className="grid h-full min-h-0 grid-cols-[270px_1fr]">
      <aside className="flex min-h-0 flex-col overflow-y-auto border-r border-line bg-panel" aria-label="Deal inputs">
        <RailHeader />
        {rail}
      </aside>
      <div className="min-h-0 overflow-y-auto">
        <ChangesBar />
        {children}
      </div>
    </div>
  );
}

function RailHeader() {
  const { autoUpdate, setAutoUpdate, run } = useDeal();
  const state =
    run.status === "running" ? "Updating" : run.status === "error" ? "Error" : run.status === "ok" ? `Up to date · ${Math.round(run.ms ?? 0)} ms` : "Loading";
  return (
    <div className="flex items-center justify-between gap-2 border-b border-line px-3.5 py-2">
      <button
        type="button"
        role="switch"
        aria-checked={autoUpdate}
        onClick={() => setAutoUpdate(!autoUpdate)}
        className="type-control flex items-center gap-2 hover:text-ink"
      >
        <span className={`relative inline-block h-3.5 w-[26px] ${autoUpdate ? "bg-accent" : "bg-line-strong"}`} aria-hidden>
          <span className={`absolute top-0.5 size-2.5 bg-bg transition-[left] duration-150 ${autoUpdate ? "left-[14px]" : "left-0.5"}`} />
        </span>
        Auto-update
      </button>
      <span
        role="status"
        className={`font-mono text-[10px] ${run.status === "error" ? "text-loss" : run.status === "running" ? "text-attention" : "text-dim"}`}
      >
        {state}
      </span>
    </div>
  );
}

export function RailGroup({ title, children }: { title: string; children: ReactNode }) {
  return (
    <fieldset className="grid gap-0.5 border-b border-line px-3.5 pt-2.5 pb-3">
      <legend className="type-input-group float-left mb-1.5 w-full">{title}</legend>
      {children}
    </fieldset>
  );
}

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

export function WspToggle() {
  const { inputs, setField } = useDeal();
  return (
    <div className="grid grid-cols-[1fr_auto] items-center gap-2 py-1">
      <span className="type-input-label" id="wsp-label">
        Working capital from days
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={inputs.wsp_mode}
        aria-labelledby="wsp-label"
        onClick={() => setField("wsp_mode", !inputs.wsp_mode)}
        className={`relative h-3.5 w-[26px] ${inputs.wsp_mode ? "bg-accent" : "bg-line-strong"}`}
      >
        <span className={`absolute top-0.5 size-2.5 bg-bg transition-[left] duration-150 ${inputs.wsp_mode ? "left-[14px]" : "left-0.5"}`} />
      </button>
    </div>
  );
}

function describeValue(key: keyof DealInputs, v: DealInputs[keyof DealInputs]): string {
  if (typeof v === "boolean") return v ? "on" : "off";
  const spec = FIELDS[key as NumericDealKey];
  return spec ? `${fmtInput(v, spec.decimals)}${spec.unit === "%" ? "%" : spec.unit === "x" ? "x" : ""}` : String(v);
}

/** Error from the last run, or (with auto-update off) the edits waiting to run. */
function ChangesBar() {
  const { run, pending, autoUpdate, runNow, discard, inputs } = useDeal();

  if (run.status === "error") {
    return (
      <div className="flex items-center gap-3.5 bg-[#1d1413] px-3 py-2 shadow-[inset_3px_0_0_var(--color-loss)]" role="alert">
        <span className="type-alert text-loss">Model didn&apos;t run</span>
        <span className="font-mono text-[11px] text-muted">{run.error}</span>
        <span className="flex-1" />
        <button type="button" onClick={runNow} className="type-action-secondary px-2.5 py-1.5 text-muted shadow-[inset_0_0_0_1px_#2a343a] hover:text-ink">
          Try again
        </button>
      </div>
    );
  }
  if (autoUpdate || !pending.length || !run.ranFor) return null;

  const ranFor = run.ranFor;
  return (
    <div className="flex items-center gap-3.5 bg-[#1b1710] px-3 py-2 shadow-[inset_3px_0_0_var(--color-attention)]" role="status">
      <span className="type-alert">
        {pending.length} input{pending.length > 1 ? "s" : ""} changed
      </span>
      <span className="truncate font-mono text-[11px] text-muted">
        {pending.map((k) => (
          <span key={k} className="mr-3">
            {k === "wsp_mode" ? "Working capital from days" : (FIELDS[k as NumericDealKey]?.label ?? k)} {describeValue(k, ranFor[k])} →{" "}
            <b className="font-medium text-attention">{describeValue(k, inputs[k])}</b>
          </span>
        ))}
      </span>
      <span className="flex-1" />
      <button type="button" onClick={discard} className="type-action-secondary px-2.5 py-1.5 text-muted shadow-[inset_0_0_0_1px_#2a343a] hover:text-ink">
        Discard
      </button>
      <button type="button" onClick={runNow} className="type-action bg-accent px-3 py-2 text-bg">
        Run model
      </button>
    </div>
  );
}

/** Placeholder grid while the first result loads. */
export function LoadingTiles() {
  return (
    <div className="grid grid-cols-12 content-start gap-px bg-line" aria-busy="true" aria-label="Loading results">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="col-span-2 grid h-[86px] content-start gap-2 bg-canvas px-3 py-2.5">
          <div className="h-2.5 w-16 bg-raised" />
          <div className="h-5 w-24 bg-raised" />
        </div>
      ))}
      <div className="col-span-6 h-[270px] bg-canvas" />
      <div className="col-span-6 h-[270px] bg-canvas" />
    </div>
  );
}

/** The model's open finding 1, shown wherever minimum cash affects the numbers. */
export function MinCashNotice() {
  const { inputs, run } = useDeal();
  const residual = run.result?.equity_bridge.residual;
  if (!(inputs.mincash > 0)) return null;
  return (
    <div className="col-span-12 flex flex-wrap items-center gap-x-3.5 gap-y-1 bg-[#1b1710] px-3 py-2 shadow-[inset_3px_0_0_var(--color-attention)]" role="note">
      <span className="type-alert">Returns overstated</span>
      <span className="type-body">
        The model doesn&apos;t fund the {inputs.mincash.toFixed(1)} $M minimum cash with sponsor equity, so IRR and MOIC read high
        {typeof residual === "number" ? ` (bridge residual ${residual.toFixed(1)} $M)` : ""}. Open finding 1, not fixed yet.
      </span>
    </div>
  );
}
