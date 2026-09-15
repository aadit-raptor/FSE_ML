"use client";

import { useState } from "react";

import { fmtInput } from "@/lib/format";

/** Compact numeric cell for editable grids. Commits valid numbers; invalid text is marked and not applied. */
export function CellInput({ value, onCommit, label, decimals = 1 }: { value: number; onCommit: (v: number) => void; label: string; decimals?: number }) {
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? (Number.isFinite(value) ? fmtInput(value, decimals) : "");
  const invalid = draft !== null && (draft.trim() === "" || !Number.isFinite(Number(draft)));
  return (
    <input
      aria-label={label}
      aria-invalid={invalid}
      inputMode="decimal"
      autoComplete="off"
      value={shown}
      onChange={(e) => {
        setDraft(e.target.value);
        const n = Number(e.target.value);
        if (e.target.value.trim() !== "" && Number.isFinite(n)) onCommit(n);
      }}
      onBlur={() => !invalid && setDraft(null)}
      className={`w-full min-w-[64px] border bg-field px-1.5 py-0.5 text-right font-mono text-[11px] outline-none focus:border-accent ${invalid ? "border-loss text-loss" : "border-line text-ink"}`}
    />
  );
}
