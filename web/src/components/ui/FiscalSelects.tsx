"use client";

import { useState } from "react";

import { type Fiscal, MONTHS, yearEndNote } from "@/lib/fiscal";
import { monthName } from "@/lib/locale";

const CONTROL =
  "min-w-0 border bg-field px-1 py-0.5 font-mono text-[11px] text-ink outline-none focus:border-accent";

/** A year the API accepts (api/schemas.py DealInputsIn.first_fiscal_year). */
const MIN_YEAR = 1900;
const MAX_YEAR = 2200;

/**
 * The month a fiscal year ends and which fiscal year a column starts at (PLAN.md 2.3a). `of` names what
 * they belong to, for screen readers ("Deal fiscal year end"); `yearLabel` says which column the year is
 * ("First fiscal year" for a deal's projection, "Latest fiscal year" for a company's history). An empty
 * year keeps the plain labels (Y1, LTM).
 */
export function FiscalSelects({
  fiscal,
  onChange,
  of,
  yearLabel,
}: {
  fiscal: Fiscal;
  onChange: (fiscal: Fiscal) => void;
  of: string;
  yearLabel: string;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? (fiscal.year === null ? "" : String(fiscal.year));
  const typed = shown.trim();
  const year = typed === "" ? null : /^\d{4}$/.test(typed) ? Number(typed) : NaN;
  const invalid = year !== null && !(Number.isInteger(year) && year >= MIN_YEAR && year <= MAX_YEAR);
  return (
    <>
      <label className="grid grid-cols-[1fr_128px] items-center gap-1.5 py-px">
        <span className="type-input-label">Year ends in</span>
        <select
          aria-label={`${of} fiscal year end`}
          value={fiscal.endMonth}
          onChange={(e) => onChange({ ...fiscal, endMonth: Number(e.target.value) })}
          className={`${CONTROL} border-line`}
        >
          {MONTHS.map((m) => (
            <option key={m} value={m}>
              {monthName(m)}
            </option>
          ))}
        </select>
      </label>
      <label className="grid grid-cols-[1fr_128px] items-center gap-1.5 py-px">
        <span className="type-input-label">{yearLabel}</span>
        <input
          aria-label={`${of} ${yearLabel.toLowerCase()}`}
          aria-invalid={invalid}
          inputMode="numeric"
          autoComplete="off"
          placeholder="none"
          value={shown}
          onChange={(e) => {
            setDraft(e.target.value);
            const t = e.target.value.trim();
            const y = t === "" ? null : /^\d{4}$/.test(t) ? Number(t) : NaN;
            if (y === null || (y >= MIN_YEAR && y <= MAX_YEAR)) onChange({ ...fiscal, year: y });
          }}
          onBlur={() => !invalid && setDraft(null)}
          className={`${CONTROL} ${invalid ? "border-loss text-loss" : "border-line"}`}
        />
      </label>
      <p className="font-mono text-[9.5px] text-muted">
        {invalid ? `A year from ${MIN_YEAR} to ${MAX_YEAR}, named by when it ends` : yearEndNote(fiscal.endMonth)}
      </p>
    </>
  );
}
