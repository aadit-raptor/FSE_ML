"use client";

import { useTranslations } from "next-intl";
import { type ReactNode, useEffect, useId, useRef, useState } from "react";

import { useSettings, num } from "@/components/settings/SettingsProvider";
import { SELECT_CLASS } from "@/components/ui/MoneySelects";
import { NumberField } from "@/components/ui/NumberField";
import { SecondaryButton, Switch } from "@/components/ui/Screen";
import {
  equivalentTranches,
  fullTranche,
  type FullTranche,
  newTranche,
  REFERENCE_RATES,
  type ReferenceRate,
  TRANCHE_KINDS,
  type TrancheKind,
  withKind,
} from "@/lib/deal/capital";
import { dealFiscal } from "@/lib/deal/fields";
import type { FieldSpec } from "@/lib/fields";
import { fmtMoney } from "@/lib/format";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";
import { MONEY } from "@/lib/money";

import { useDeal } from "../DealProvider";

// Bounds mirror TrancheIn in api/schemas.py
const pct = (max: number, min = 0, step = 0.25): FieldSpec => ({ unit: "%", step, decimals: 2, min, max });
const SPECS = {
  amount: { unit: MONEY, step: 5, decimals: 1, min: 0, max: 1e15 } satisfies FieldSpec,
  drawn_pct: pct(100, 0, 5),
  fixed_rate: pct(50, -5),
  reference: pct(50, -5),
  margin: pct(50),
  floor: pct(50, -5),
  maturity_years: { unit: "yr", step: 1, decimals: 0, min: 1, max: 30, integer: true } satisfies FieldSpec,
  amort_pct: pct(100, 0, 0.5),
  upfront_fee_pct: pct(10),
  commitment_fee_pct: pct(5, 0, 0.05),
  sweep_share: pct(100, 0, 5),
  pik_share: pct(100, 0, 5),
};

/**
 * The deal's debt when it is sized by percentages: the four fields, plus the
 * way to write them out as facilities. The first click must not move the
 * deal's own result (lib/deal/capital.ts `equivalentTranches`).
 */
export function UseTranchesButton() {
  const { inputs, setFields } = useDeal();
  const { effective } = useSettings();
  const t = useTranslations("deal");
  return (
    <div className="grid gap-1.5 pt-2">
      <SecondaryButton onClick={() => setFields({ tranches: equivalentTranches(inputs, num(effective, "def_senior_amort", 5)) })}>
        {t("useExplicitTranches")}
      </SecondaryButton>
      <p className="type-body text-[9px]">{t("explicitTranchesNote")}</p>
    </div>
  );
}

/**
 * The deal's facilities, one row each, in sweep order: the first in the list
 * takes the cash sweep first. A row opens into every field the facility has;
 * whatever its kind, all of them stay editable (core/debt.py PRESETS).
 *
 * i18n-keys: trancheKinds.*
 */
export function TrancheList() {
  const { inputs, setFields } = useDeal();
  const t = useTranslations("deal");
  const kinds = useTranslations("trancheKinds");
  const [open, setOpen] = useState<number | null>(null);
  const [adding, setAdding] = useState<TrancheKind>("senior_notes");
  const listId = useId();
  const tranches = inputs.tranches.map(fullTranche);
  // Rows are keyed by position, so a move or a removal remounts them: put the
  // keyboard back on the row that moved (or the one that took its place)
  const headers = useRef<(HTMLButtonElement | null)[]>([]);
  const focusNext = useRef<number | null>(null);
  useEffect(() => {
    const i = focusNext.current;
    if (i === null) return;
    focusNext.current = null;
    headers.current[Math.min(i, tranches.length - 1)]?.focus();
  });

  const write = (next: FullTranche[]) => setFields({ tranches: next });
  const update = (i: number, patch: Partial<FullTranche>) => write(tranches.map((x, j) => (j === i ? { ...x, ...patch } : x)));
  const remove = (i: number) => {
    write(tranches.filter((_, j) => j !== i));
    setOpen(null);
    focusNext.current = i;
  };
  // The list is the sweep order, so a move clears any explicit priority
  const move = (i: number, by: -1 | 1) => {
    const next = tranches.map((x) => ({ ...x, sweep_priority: 0 }));
    [next[i], next[i + by]] = [next[i + by], next[i]];
    write(next);
    setOpen(i + by);
    focusNext.current = i + by;
  };

  return (
    // minmax(0, 1fr) everywhere: a long facility or kind name must truncate, not widen the rail
    <div className="grid min-w-0 grid-cols-[minmax(0,1fr)]">
      <p className="type-body pb-1.5 text-[9px]">{t("sweepOrderNote")}</p>
      <ol className="grid grid-cols-[minmax(0,1fr)] border-t border-line">
        {tranches.map((tr, i) => (
          <li key={i} className="border-b border-line">
            <button
              ref={(el) => {
                headers.current[i] = el;
              }}
              type="button"
              aria-expanded={open === i}
              aria-controls={open === i ? `${listId}-${i}` : undefined}
              onClick={() => setOpen(open === i ? null : i)}
              className="flex w-full items-baseline justify-between gap-2 py-1.5 text-start hover:text-bright"
            >
              <span className="type-input-label min-w-0 truncate">
                {t("trancheRow", { name: tr.name, kind: kinds(tr.kind) })}
              </span>
              <span className="font-mono text-[11px] text-ink">{fmtMoney(tr.amount)}</span>
            </button>
            {open === i && (
              <TrancheFields
                id={`${listId}-${i}`}
                tranche={tr}
                onChange={(patch) => update(i, patch)}
                onKind={(kind) => write(tranches.map((x, j) => (j === i ? withKind(x, kind) : x)))}
                actions={
                  <div className="flex flex-wrap gap-1.5 pt-1.5 pb-2">
                    <SecondaryButton onClick={() => move(i, -1)} disabled={i === 0}>
                      {t("moveUp")}
                    </SecondaryButton>
                    <SecondaryButton onClick={() => move(i, 1)} disabled={i === tranches.length - 1}>
                      {t("moveDown")}
                    </SecondaryButton>
                    <SecondaryButton onClick={() => remove(i)}>{t("removeTranche")}</SecondaryButton>
                  </div>
                }
              />
            )}
          </li>
        ))}
      </ol>
      <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-1.5 pt-2">
        <select aria-label={t("addTrancheKind")} value={adding} onChange={(e) => setAdding(e.target.value as TrancheKind)} className={SELECT_CLASS}>
          {TRANCHE_KINDS.map((k) => (
            <option key={k} value={k}>
              {kinds(k)}
            </option>
          ))}
        </select>
        <SecondaryButton
          onClick={() => {
            write([...tranches, newTranche(adding, kinds(adding))]);
            setOpen(tranches.length);
          }}
        >
          {t("addTranche")}
        </SecondaryButton>
      </div>
    </div>
  );
}

/** i18n-keys: fields.tranche_*, referenceRates.*, trancheKinds.* */
function TrancheFields({
  id,
  tranche: tr,
  onChange,
  onKind,
  actions,
}: {
  id: string;
  tranche: FullTranche;
  onChange: (patch: Partial<FullTranche>) => void;
  onKind: (kind: TrancheKind) => void;
  actions: ReactNode;
}) {
  const fields = useTranslations("fields");
  const kinds = useTranslations("trancheKinds");
  const rates = useTranslations("referenceRates");
  const nameId = useId();
  // The API needs a name: an emptied field waits for the next letter, and
  // leaving it empty puts the last name back
  const [nameDraft, setNameDraft] = useState<string | null>(null);
  const field = (key: keyof typeof SPECS & keyof FullTranche, disabled?: boolean) => (
    <NumberField
      spec={SPECS[key]}
      label={fields(`tranche_${key}`)}
      value={tr[key] as number}
      onCommit={(v) => onChange({ [key]: v })}
      disabled={disabled}
    />
  );

  return (
    <div id={id} className="grid grid-cols-[minmax(0,1fr)] gap-0.5 ps-2">
      <div className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
        <label htmlFor={nameId} className="type-input-label">
          {fields("tranche_name")}
        </label>
        <input
          id={nameId}
          value={nameDraft ?? tr.name}
          maxLength={60}
          autoComplete="off"
          onChange={(e) => {
            setNameDraft(e.target.value);
            if (e.target.value.trim()) onChange({ name: e.target.value });
          }}
          onBlur={() => setNameDraft(null)}
          className="min-w-0 border border-line bg-field px-1.5 py-0.5 font-mono text-[11px] text-ink outline-none focus:border-accent"
        />
      </div>
      <label className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
        <span className="type-input-label">{fields("tranche_kind")}</span>
        <select value={tr.kind} onChange={(e) => onKind(e.target.value as TrancheKind)} className={SELECT_CLASS}>
          {TRANCHE_KINDS.map((k) => (
            <option key={k} value={k}>
              {kinds(k)}
            </option>
          ))}
        </select>
      </label>
      {field("amount")}
      {field("drawn_pct")}
      <div className="py-1">
        <Switch checked={tr.floating} onChange={(v) => onChange({ floating: v })} label={fields("tranche_floating")} />
      </div>
      {tr.floating ? (
        <>
          <label className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
            <span className="type-input-label">{fields("tranche_reference_rate")}</span>
            <select value={tr.reference_rate} onChange={(e) => onChange({ reference_rate: e.target.value as ReferenceRate })} className={SELECT_CLASS}>
              {REFERENCE_RATES.map((r) => (
                <option key={r} value={r}>
                  {rates(r)}
                </option>
              ))}
            </select>
          </label>
          <RatePath tranche={tr} onChange={onChange} />
          {field("floor")}
          {field("margin")}
        </>
      ) : (
        field("fixed_rate")
      )}
      {field("maturity_years")}
      {field("amort_pct")}
      {field("upfront_fee_pct")}
      {field("commitment_fee_pct")}
      <div className="py-1">
        <Switch checked={tr.sweep} onChange={(v) => onChange({ sweep: v })} label={fields("tranche_sweep")} />
      </div>
      {field("sweep_share", !tr.sweep)}
      {field("pik_share")}
      <div className="py-1">
        <Switch checked={tr.allow_redraw} onChange={(v) => onChange({ allow_redraw: v })} label={fields("tranche_allow_redraw")} />
      </div>
      {actions}
    </div>
  );
}

/**
 * The reference rate year by year, one field per year of the hold. A path
 * shorter than the run repeats its last year (core/debt.py), so the grid's
 * longer holds, and a hold made longer later, keep the last rate.
 */
function RatePath({ tranche: tr, onChange }: { tranche: FullTranche; onChange: (patch: Partial<FullTranche>) => void }) {
  const { inputs } = useDeal();
  const t = useTranslations("deal");
  const fields = useTranslations("fields");
  const fiscalLabels = useFiscalLabels();
  const years = fiscalLabels.deal(inputs.hold, dealFiscal(inputs));
  const path = tr.reference_path.length ? tr.reference_path : [tr.reference_level];
  const at = (i: number) => path[Math.min(i, path.length - 1)];
  return (
    <div className="grid gap-0.5">
      {years.map((year, i) => (
        <NumberField
          key={year}
          spec={SPECS.reference}
          label={fields("tranche_reference_year", { year })}
          value={at(i)}
          onCommit={(v) => {
            const next = years.map((_, j) => (j === i ? v : at(j)));
            onChange({ reference_path: next, reference_level: next[0] });
          }}
        />
      ))}
      <p className="type-body pb-1 text-[9px]">{t("ratePathNote")}</p>
    </div>
  );
}
