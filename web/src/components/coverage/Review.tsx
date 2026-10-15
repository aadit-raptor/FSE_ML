"use client";

import { useTranslations } from "next-intl";
import { useMemo, useState } from "react";

import { Notice } from "@/components/ui/Screen";
import { Tile } from "@/components/ui/Tile";
import { downloadFile } from "@/lib/export";
import { fmtCount, fmtNumber } from "@/lib/format";
import { parseNumber } from "@/lib/locale";
import {
  acceptRead, amberCells, cellKey, decimalsOf, workbookName, workbookPath,
  type CellRef, type Correction, type PeriodSummary, type ReadResult, type ReadTable,
} from "@/lib/research";

import { PageView } from "./PageView";

/**
 * The review of a read (PLAN.md R2): the filing's page beside its figures. Every amber figure is answered --
 * accepted as read, or corrected -- and any other figure may be corrected too; **nothing is kept until the
 * analyst accepts the quarter**, and the API refuses a quarter with an amber figure unanswered. Accepting
 * stores the figures from the read the server holds, with these answers, never figures sent from here.
 *
 * i18n-keys: reader.amber_*
 * i18n-keys: reader.statement_*
 * i18n-keys: reader.scope_*
 * i18n-keys: coverage.error_*
 */
type Answer = { kind: "accepted" } | { kind: "corrected"; value: number | null };

const KNOWN_ERRORS = [
  "amber_unreviewed", "bad_review", "quarter_unknown", "unit_unknown", "no_figures", "no_read", "read_not_done",
  "read_expired", "not_found", "too_many_periods", "coverage_full", "no_account", "no_database",
];

export function Review({
  readId, result, company, onAccepted,
}: {
  readId: string;
  result: ReadResult;
  company: { id: string; name: string } | null;
  onAccepted: (periods: PeriodSummary[]) => void;
}) {
  const t = useTranslations("coverage");
  const tr = useTranslations("reader");
  const amber = useMemo(() => amberCells(result), [result]);
  const [answers, setAnswers] = useState<Map<string, Answer>>(new Map());
  const [selected, setSelected] = useState<CellRef | null>(amber[0] ?? null);
  const [page, setPage] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accepted, setAccepted] = useState<PeriodSummary[] | null>(null);

  const unanswered = amber.filter((ref) => !answers.has(cellKey(ref)));
  const pageOf = (ref: CellRef | null) => {
    if (!ref) return null;
    const table = result.tables[ref[0]];
    return table.rows[ref[1]].page ?? table.pages[0] ?? null;
  };
  const shownPage = page ?? pageOf(selected);

  const choose = (ref: CellRef) => {
    setSelected(ref);
    setPage(null);
    const answer = answers.get(cellKey(ref));
    setDraft(answer?.kind === "corrected" && answer.value !== null ? String(answer.value) : "");
  };
  const answer = (ref: CellRef, value: Answer) => {
    setAnswers((old) => new Map(old).set(cellKey(ref), value));
    setError(null);
    // on to the next amber figure still to answer
    const next = amber.find((other) => cellKey(other) !== cellKey(ref) && !answers.has(cellKey(other)));
    if (next) choose(next);
  };

  const accept = async () => {
    if (!company) return;
    setSaving(true);
    setError(null);
    const kept: CellRef[] = [];
    const corrected: Correction[] = [];
    answers.forEach((value, key) => {
      const [table, row, column] = key.split(":").map(Number) as CellRef;
      if (value.kind === "accepted") kept.push([table, row, column]);
      else corrected.push({ table, row, column, value: value.value });
    });
    const done = await acceptRead(company.id, readId, kept, corrected);
    setSaving(false);
    if (done.periods) {
      setAccepted(done.periods);
      onAccepted(done.periods);
    } else setError(done.error ?? "failed");
  };

  const cell = selected ? result.tables[selected[0]].rows[selected[1]].cells[selected[2]] : null;
  const row = selected ? result.tables[selected[0]].rows[selected[1]] : null;
  const column = selected ? result.tables[selected[0]].columns[selected[2]] : null;
  const digits = selected ? decimalsOf(result.tables[selected[0]]) : 0;
  const given = selected ? answers.get(cellKey(selected)) : undefined;
  const parsed = draft.trim() === "" ? null : parseNumber(draft);
  const draftOk = parsed === null || Number.isFinite(parsed);

  return (
    <>
      <Tile span={12} title={t("reviewTitle")}>
        <div className="flex flex-wrap items-center gap-3">
          <p className="type-body text-[10.5px]" data-unanswered={unanswered.length}>
            {unanswered.length > 0 ? t("unanswered", { count: unanswered.length }) : t("allAnswered", { count: amber.length })}
          </p>
          {unanswered.length > 0 && (
            <button type="button" className="type-action border border-line px-2 py-1 text-ink" onClick={() => choose(unanswered[0])}>
              {t("nextAmber")}
            </button>
          )}
          <button
            type="button"
            className="type-action ms-auto bg-accent px-3 py-2 text-bg disabled:opacity-50"
            disabled={!company || unanswered.length > 0 || saving || accepted !== null}
            onClick={() => void accept()}
          >
            {company ? t("acceptInto", { company: company.name }) : t("acceptNoCompany")}
          </button>
        </div>
        {!company && <p className="type-body mt-1 text-[10px]">{t("chooseCompanyFirst")}</p>}
        {error && (
          <Notice tone="loss" title={t("notAccepted")} role="alert">
            <span data-accept-error={error}>{t(`error_${KNOWN_ERRORS.includes(error) ? error : "failed"}`)}</span>
          </Notice>
        )}
        {accepted && company && (
          <Notice tone="info" title={t("acceptedTitle")} role="status">
            <span data-accepted={accepted.length}>{t("acceptedBody", { company: company.name, count: accepted.length })}</span>
            <span className="mt-1 flex flex-wrap gap-2">
              {accepted.map((period) => (
                <button
                  key={period.id}
                  type="button"
                  className="type-action border border-line px-2 py-1 text-ink"
                  onClick={() => void downloadFile(workbookPath(period.id), workbookName(company.name, period))}
                >
                  {t("downloadScope", { scope: tr(`scope_${period.scope}`) })}
                </button>
              ))}
            </span>
          </Notice>
        )}
      </Tile>
      <Tile span={7} title={t("figuresTitle")}>
        <div className="grid gap-4">
          {result.tables.map((table, ti) => (
            <ReviewTable key={ti} index={ti} table={table} selected={selected} answers={answers} onChoose={choose} />
          ))}
        </div>
      </Tile>
      <Tile span={5} title={t("pageTitle")}>
        <div className="sticky top-2 grid gap-3">
          {selected && cell && row && column && (
            <div className="grid gap-1.5 border border-line p-2" data-selected={cellKey(selected)}>
              <p className="font-mono text-[11px] text-ink" dir="auto">
                {row.label}
              </p>
              <p className="font-mono text-[10.5px] text-muted" dir="auto">
                {column.label}
              </p>
              <p className="font-mono text-[12px] text-ink">
                {t("asRead")}: <span data-read-value>{cell.value === null || cell.value === undefined ? t("blank") : fmtNumber(cell.value, digits)}</span>
                {cell.other !== null && cell.other !== undefined && (
                  <span className="text-muted"> · {t("otherRead", { value: fmtNumber(cell.other, digits) })}</span>
                )}
              </p>
              {cell.amber && <p className="type-body text-[10.5px] text-attention">{tr(`amber_${cell.amber}`)}</p>}
              {given && (
                <p className="type-body text-[10.5px] text-gain" data-given={given.kind}>
                  {given.kind === "accepted"
                    ? t("givenAccepted")
                    : t("givenCorrected", { value: given.value === null ? t("blank") : fmtNumber(given.value, digits) })}
                </p>
              )}
              <div className="flex flex-wrap items-end gap-2">
                {cell.amber && (
                  <button type="button" className="type-action border border-line px-2 py-1 text-ink" onClick={() => answer(selected, { kind: "accepted" })}>
                    {t("acceptAsRead")}
                  </button>
                )}
                <label className="grid gap-0.5">
                  <span className="type-input-label">{t("correctLabel")}</span>
                  <input
                    className="w-32 border border-line bg-canvas px-2 py-1 font-mono text-[11px] text-ink"
                    inputMode="decimal"
                    value={draft}
                    onChange={(event) => setDraft(event.target.value)}
                  />
                </label>
                <button
                  type="button"
                  className="type-action border border-line px-2 py-1 text-ink disabled:opacity-40"
                  disabled={!draftOk}
                  onClick={() => answer(selected, { kind: "corrected", value: parsed })}
                >
                  {t("useCorrection")}
                </button>
              </div>
            </div>
          )}
          <PageView fileId={result.file_id} page={shownPage} onPage={setPage} />
        </div>
      </Tile>
    </>
  );
}

function ReviewTable({
  index, table, selected, answers, onChoose,
}: {
  index: number;
  table: ReadTable;
  selected: CellRef | null;
  answers: Map<string, Answer>;
  onChoose: (ref: CellRef) => void;
}) {
  const tr = useTranslations("reader");
  const t = useTranslations("coverage");
  const digits = decimalsOf(table);
  const title = `${tr(`statement_${table.statement}`)} · ${tr(`scope_${table.scope}`)}`;
  return (
    <div className="min-w-0">
      <p className="type-result-title mb-1">{title}</p>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse font-mono text-[11px]" aria-label={title} data-statement={table.statement}>
          <thead>
            <tr className="text-muted">
              <th scope="col" className="border-b border-grid px-2 py-1 text-start font-normal">
                <span className="sr-only">{tr("colLine")}</span>
              </th>
              {table.columns.map((column, c) => (
                <th key={c} scope="col" className="border-b border-grid px-2 py-1 text-end font-normal" dir="auto">
                  {column.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row, r) => (
              <tr key={r} data-line={row.line ?? undefined}>
                <th scope="row" dir="auto" className="max-w-[260px] truncate border-b border-grid px-2 py-1 text-start font-normal text-ink" title={row.label}>
                  {row.label}
                </th>
                {row.cells.map((cell, c) => {
                  const key = cellKey([index, r, c]);
                  const given = answers.get(key);
                  const isSelected = selected !== null && cellKey(selected) === key;
                  const shown = given?.kind === "corrected" ? given.value : cell.value;
                  const tone = given ? "text-gain" : cell.amber ? "bg-[#1b1710] text-attention" : "text-ink";
                  return (
                    <td
                      key={c}
                      className={`border-b border-grid p-0 text-end ${tone} ${isSelected ? "outline outline-1 outline-accent" : ""}`}
                      data-cell={key}
                      data-amber={cell.amber ?? undefined}
                      data-answer={given?.kind}
                    >
                      <button
                        type="button"
                        className="w-full px-2 py-1 text-end"
                        aria-label={t("cellLabel", { row: row.label, column: table.columns[c]?.label ?? "" })}
                        aria-pressed={isSelected}
                        onClick={() => onChoose([index, r, c])}
                      >
                        {shown === null || shown === undefined ? "" : fmtNumber(shown, digits)}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-0.5 font-mono text-[10px] text-muted">{t("tablePages", { pages: table.pages.map((p) => fmtCount(p)).join(", ") })}</p>
    </div>
  );
}
