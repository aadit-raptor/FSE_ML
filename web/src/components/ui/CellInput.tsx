"use client";

import { useState } from "react";

import { fmtInput } from "@/lib/format";
import { parseNumber } from "@/lib/locale";

import { TipText, useTip } from "./Tip";

/**
 * Compact numeric cell for editable grids. Commits valid numbers; invalid text is marked and not applied.
 * With `onClear`, an emptied cell is a figure not known rather than an error. `tip` is the key of its
 * tip under `tips` (PLAN.md 7.1): read as the cell's description and shown on hover.
 */
export function CellInput({
  value,
  onCommit,
  onClear,
  label,
  tip,
  decimals = 1,
}: {
  value: number;
  onCommit: (v: number) => void;
  onClear?: () => void;
  label: string;
  tip: string;
  decimals?: number;
}) {
  const handle = useTip(tip);
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? (Number.isFinite(value) ? fmtInput(value, decimals) : "");
  const blank = draft !== null && draft.trim() === "" && onClear !== undefined;
  const invalid = draft !== null && !blank && !Number.isFinite(parseNumber(draft));
  return (
    <>
    <input
      aria-label={label}
      aria-describedby={handle.id}
      title={handle.text ?? undefined}
      aria-invalid={invalid}
      inputMode="decimal"
      autoComplete="off"
      value={shown}
      onChange={(e) => {
        setDraft(e.target.value);
        if (onClear && e.target.value.trim() === "") return onClear();
        const n = parseNumber(e.target.value);
        if (Number.isFinite(n)) onCommit(n);
      }}
      onBlur={() => !invalid && setDraft(null)}
      className={`w-full min-w-[64px] border bg-field px-1.5 py-0.5 text-end font-mono text-[11px] outline-none focus:border-accent ${invalid ? "border-loss text-loss" : "border-line text-ink"}`}
    />
    <TipText tip={handle} />
    </>
  );
}
