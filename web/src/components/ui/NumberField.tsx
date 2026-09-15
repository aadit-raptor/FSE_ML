"use client";

import { useId, useRef, useState } from "react";

import { fmtInput } from "@/lib/format";
import { validate, type FieldSpec } from "@/lib/deal/fields";

type Props = {
  spec: FieldSpec;
  value: number;
  onCommit: (value: number) => void;
  disabled?: boolean;
  /** Edited since the last model run */
  changed?: boolean;
  /** Override the label from the spec */
  label?: string;
};

/**
 * Rail input: label, monospaced value, unit. Commits every valid keystroke
 * (the model reruns on a debounce). Invalid text stays local with a message
 * and puts the model back on the value from before the edit, so results never
 * reflect a half-typed number (typing "20" into Hold must not run hold 2).
 * Arrow up/down step by spec.step.
 */
export function NumberField({ spec, value, onCommit, disabled, changed, label }: Props) {
  const id = useId();
  const [draft, setDraft] = useState<string | null>(null);
  const beforeEdit = useRef(value);
  const shown = draft ?? fmtInput(value, spec.decimals);
  const error = draft === null ? null : validate(spec, draft.trim() === "" ? NaN : Number(draft));

  const commitText = (text: string) => {
    setDraft(text);
    const n = text.trim() === "" ? NaN : Number(text);
    if (!validate(spec, n)) onCommit(n);
    else if (value !== beforeEdit.current) onCommit(beforeEdit.current);
  };

  const nudge = (dir: 1 | -1) => {
    let next = Number((value + dir * spec.step).toFixed(6));
    if (spec.max !== undefined) next = Math.min(next, spec.max);
    if (spec.min !== undefined) next = spec.exclusiveMin ? Math.max(next, spec.min + spec.step) : Math.max(next, spec.min);
    if (!validate(spec, next)) {
      setDraft(null);
      onCommit(next);
    }
  };

  return (
    <div className="grid gap-0.5 py-px">
      <div className="grid grid-cols-[1fr_68px_24px] items-center gap-1.5">
        <label htmlFor={id} className={`type-input-label ${disabled ? "opacity-45" : ""}`}>
          {label ?? spec.label}
        </label>
        <input
          id={id}
          inputMode="decimal"
          autoComplete="off"
          disabled={disabled}
          value={shown}
          aria-invalid={!!error}
          aria-describedby={error ? `${id}-err` : undefined}
          onFocus={() => {
            beforeEdit.current = value;
          }}
          onChange={(e) => commitText(e.target.value)}
          onBlur={() => {
            if (!error) setDraft(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "ArrowUp" || e.key === "ArrowDown") {
              e.preventDefault();
              nudge(e.key === "ArrowUp" ? 1 : -1);
            }
          }}
          className={`border bg-field px-1.5 py-0.5 text-right font-mono text-[11.5px] outline-none focus:border-accent disabled:opacity-45 ${
            error ? "border-loss text-loss" : changed ? "border-attention text-attention" : "border-line text-ink"
          }`}
        />
        <span className={`font-mono text-[10px] text-[#56636a] ${disabled ? "opacity-45" : ""}`}>{spec.unit}</span>
      </div>
      {error && (
        <p id={`${id}-err`} className="text-right font-mono text-[10px] text-loss">
          {error}
        </p>
      )}
    </div>
  );
}
