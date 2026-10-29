"use client";

import { useTranslations } from "next-intl";
import { useState } from "react";

import { Term } from "@/components/ui/Term";
import { fmtMultiple, fmtNumber, fmtRate } from "@/lib/format";
import { formatDateTime, formatInput, parseNumber } from "@/lib/locale";
import { excelDay, isoOfExcelDay, type ModelCell, type ModelRow, type ModelRun } from "@/lib/sectorModel";

/**
 * The model's rows down and periods across (PLAN.md R4), as the workbook holds them. Typed figures (blue) and
 * assumptions (yellow) are inputs; everything else is what the model's formula gives, the formula in its
 * tooltip. Shares are typed and shown as percentages and kept as fractions, as the workbook keeps them.
 */
const DECIMALS: Record<string, number> = { money: 2, count: 0, count3: 3, days: 0, rupees: 0, rupees2: 2, check: 2, factor: 4 };

/** What a worked-out (formula) cell shows; the explanation quotes the same text (PLAN.md 6.4). */
export function shown(unit: string | null | undefined, v: number | null | undefined): string {
  if (unit === "share") return fmtRate(v, 1);
  if (unit === "date") {                       // a day number, as Excel holds a date (the price's date, PLAN.md R6)
    const day = isoOfExcelDay(v);
    return day ? formatDateTime(new Date(`${day}T00:00:00Z`), { dateStyle: "medium", timeZone: "UTC" }) : fmtNumber(null);
  }
  if (unit === "times") return fmtMultiple(v, 2);
  return fmtNumber(v, DECIMALS[unit ?? "money"] ?? 2);
}

export function ModelGrid({ run, onEdit }: { run: ModelRun; onEdit: (row: string, period: string, value: number | null) => void }) {
  const t = useTranslations("models");
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse font-mono text-[10.5px]" data-model-grid={run.model}>
        <thead>
          <tr className="border-b border-line">
            <th className="type-input-label sticky start-0 bg-canvas px-2 py-1 text-start">
              <Term id="modelRow">{t("row")}</Term>
            </th>
            <th className="type-input-label px-2 py-1 text-start">
              <Term id="modelUnit">{t("unit")}</Term>
            </th>
            {run.periods.map((p) => (
              <th key={p.name} className="type-input-label px-2 py-1 text-end" data-period={p.name}>
                {p.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {run.rows.map((row, i) =>
            row.key === null || row.key === undefined ? (
              <tr key={`h${i}`} className="bg-raised">
                <td colSpan={run.periods.length + 2} className="type-input-group px-2 py-1">
                  {row.label}
                </td>
              </tr>
            ) : (
              <tr key={row.key} className="border-b border-line/40" data-row={row.key}>
                <td className={`sticky start-0 bg-canvas px-2 py-0.5 whitespace-pre ${row.bold ? "text-bright" : "text-ink"}`}>{row.label}</td>
                <td className="px-2 py-0.5 text-dim">{row.unit_text}</td>
                {run.periods.map((p) => (
                  <td key={p.name} className="px-1 py-0.5 text-end" data-cell={`${row.key}:${p.name}`}>
                    <Cell row={row} cell={row.cells?.[p.name]} onEdit={(v) => onEdit(row.key as string, p.name, v)} />
                  </td>
                ))}
              </tr>
            ),
          )}
        </tbody>
      </table>
    </div>
  );
}

function Cell({ row, cell, onEdit }: { row: ModelRow; cell: ModelCell | undefined; onEdit: (v: number | null) => void }) {
  const t = useTranslations("models");
  if (!cell) return null;
  if (cell.kind === "formula") {
    return (
      <span title={cell.formula ?? undefined} className={row.bold ? "text-bright" : "text-ink"}>
        {shown(row.unit, cell.value)}
      </span>
    );
  }
  const assumption = cell.kind === "assumption";
  const ring = cell.mark ? (cell.mark.level === "red" ? "ring-1 ring-loss" : "ring-1 ring-attention") : "";
  const label = assumption ? t("assumptionCell", { row: row.label.trim() }) : t("figureCell", { row: row.label.trim() });
  if (row.unit === "date") {
    return (
      <input
        type="date"
        key={String(cell.value)}
        aria-label={label}
        title={cell.mark?.why || undefined}
        defaultValue={isoOfExcelDay(cell.value)}
        onBlur={(e) => {
          const next = e.target.value ? excelDay(e.target.value) : null;
          if (next !== (cell.value ?? null)) onEdit(next);
        }}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        }}
        className={`w-32 border border-line px-1 py-0.5 text-end ${assumption ? "bg-[#2a2614]" : "bg-transparent"} ${ring} text-accent`}
      />
    );
  }
  return (
    <TypedInput
      key={String(cell.value)}
      value={cell.value ?? null}
      share={row.unit === "share"}
      label={label}
      title={cell.mark?.why || undefined}
      className={`${assumption ? "bg-[#2a2614]" : "bg-transparent"} ${ring} text-accent`}
      onCommit={onEdit}
    />
  );
}

function TypedInput({
  value, share, label, title, className, onCommit,
}: {
  value: number | null;
  share: boolean;
  label: string;
  title?: string;
  className: string;
  onCommit: (v: number | null) => void;
}) {
  const toText = (v: number | null) => (v === null ? "" : formatInput(share ? v * 100 : v, 0));
  const [draft, setDraft] = useState(toText(value));
  const commit = () => {
    const text = draft.trim();
    if (text === toText(value)) return;          // left as shown: a share read back through /100 would drift
    if (text === "") {
      if (value !== null) onCommit(null);
      return;
    }
    const parsed = parseNumber(text);
    if (!Number.isFinite(parsed)) {
      setDraft(toText(value));
      return;
    }
    const next = share ? parsed / 100 : parsed;
    if (next !== value) onCommit(next);
  };
  return (
    <input
      aria-label={label}
      title={title}
      inputMode="decimal"
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
      }}
      className={`w-24 border border-line px-1 py-0.5 text-end ${className}`}
    />
  );
}
