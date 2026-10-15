"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { PrimaryButton, SecondaryButton } from "@/components/ui/Screen";
import { api } from "@/lib/api/client";
import { uploadFile } from "@/lib/files";
import { fmtNumber } from "@/lib/format";
import { parseNumber } from "@/lib/locale";
import {
  ACCEPT_ERRORS, acceptRead, addCompany, amberCells, cellKey, decimalsOf, readJob, startRead, type CellRef, type Correction,
  type CoverageCompany, READER_ERRORS, type ReaderInfo, type ReadJob,
} from "@/lib/research";

/**
 * Reading a results PDF from the Excel add-in's pane (PLAN.md R3): the same road as Coverage -> Results
 * (upload, read twice and checked, review, accept), in a pane's width. The review lists only the amber
 * figures, each to accept as read or correct; nothing is kept until "Accept quarter and import", which then
 * writes the accepted quarter into the workbook.
 *
 * i18n-keys: reader.amber_*
 * i18n-keys: reader.error_*
 * i18n-keys: coverage.error_*
 */
const POLL_MS = 2000;

type Answer = { do: "accepted" } | { do: "corrected"; value: number | null };

export function ReadInPane({
  companies,
  onCompanies,
  onAccepted,
}: {
  companies: CoverageCompany[];
  onCompanies: (next: CoverageCompany[]) => void;
  onAccepted: (companyId: string, periodIds: string[]) => Promise<void>;
}) {
  const t = useTranslations("excel");
  const reader = useTranslations("reader");
  const coverage = useTranslations("coverage");
  const [info, setInfo] = useState<ReaderInfo | null>(null);
  const [companyId, setCompanyId] = useState("");
  const [newName, setNewName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [quarter, setQuarter] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [stage, setStage] = useState<"idle" | "uploading" | "reading" | "review" | "accepting">("idle");
  const [job, setJob] = useState<ReadJob | null>(null);
  const [answers, setAnswers] = useState<Record<string, Answer>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api.GET("/api/research/reader").then(({ data }) => setInfo(data ?? null));
  }, []);

  const jobId = job?.id;
  useEffect(() => {
    if (stage !== "reading" || !jobId) return;
    const timer = setInterval(async () => {
      const found = await readJob(jobId);
      if (!found || found.status === "queued" || found.status === "running") return;
      setJob(found);
      if (found.status === "succeeded" && found.result) setStage("review");
      else {
        setError(reader(`error_${READER_ERRORS.includes(found.error ?? "") ? found.error : "failed"}`));
        setStage("idle");
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [stage, jobId, reader]);

  const company = companies.find((c) => c.id === companyId) ?? companies[0] ?? null;
  const needsConfirm = info?.confirm_public ?? true;
  const open = !!info && info.status === "on" && info.open;
  const canRead = open && !!company && !!file && (!needsConfirm || confirmed) && stage === "idle";

  const add = async () => {
    setError(null);
    const made = await addCompany(newName.trim());
    if (made.company) {
      onCompanies([made.company, ...companies]);
      setCompanyId(made.company.id);
      setNewName("");
    } else setError(reader(`error_${READER_ERRORS.includes(made.error ?? "") ? made.error : "failed"}`));
  };

  const read = async () => {
    if (!file || !company) return;
    setError(null);
    setAnswers({});
    setStage("uploading");
    const uploaded = await uploadFile(file, () => undefined);
    if (!uploaded.ok || uploaded.file.status !== "clean") {
      setError(reader(uploaded.ok ? "error_not_clean" : "error_upload_failed"));
      setStage("idle");
      return;
    }
    const started = await startRead(uploaded.file.id, confirmed, quarter || null, company.id);
    if (!started.job) {
      setError(reader(`error_${READER_ERRORS.includes(started.error) ? started.error : "failed"}`));
      setStage("idle");
      return;
    }
    setJob(started.job);
    setStage("reading");
  };

  const result = stage === "review" || stage === "accepting" ? job?.result ?? null : null;
  const amber = result ? amberCells(result) : [];
  const left = amber.filter((ref) => !answers[cellKey(ref)]).length;

  const accept = async () => {
    if (!job || !company) return;
    setStage("accepting");
    setError(null);
    const accepted: CellRef[] = [];
    const corrected: Correction[] = [];
    for (const ref of amber) {
      const answer = answers[cellKey(ref)];
      if (answer?.do === "accepted") accepted.push(ref);
      else if (answer?.do === "corrected") corrected.push({ table: ref[0], row: ref[1], column: ref[2], value: answer.value });
    }
    const done = await acceptRead(company.id, job.id, accepted, corrected);
    if (!done.periods) {
      setError(coverage(`error_${ACCEPT_ERRORS.includes(done.error ?? "") ? done.error : "failed"}`));
      setStage("review");
      return;
    }
    setStage("idle");
    setJob(null);
    setFile(null);
    await onAccepted(company.id, done.periods.map((p) => p.id));
  };

  return (
    <section className="grid gap-2.5 border-t border-line pt-4">
      <h2 className="type-input-group">{t("readTitle")}</h2>
      {info && !open && <p className="type-body">{reader("error_reader_not_measured")}</p>}
      <label className="grid gap-1">
        <span className="type-input-label">{t("company")}</span>
        <select
          value={company?.id ?? ""}
          onChange={(e) => setCompanyId(e.target.value)}
          className="border border-line bg-field px-2 py-1.5 text-[12px] text-ink"
          aria-label={t("company")}
        >
          {companies.map((c) => (
            <option key={c.id} value={c.id}>
              {c.name}
            </option>
          ))}
        </select>
      </label>
      <div className="flex gap-2">
        <input
          aria-label={t("newCompany")}
          placeholder={t("newCompany")}
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          className="min-w-0 flex-1 border border-line bg-field px-2 py-1.5 text-[12px] text-ink"
        />
        <SecondaryButton disabled={!newName.trim()} onClick={() => void add()}>
          {t("addCompany")}
        </SecondaryButton>
      </div>
      <label className="grid gap-1">
        <span className="type-input-label">{t("pdf")}</span>
        <input
          type="file"
          accept=".pdf,application/pdf"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          className="text-[11px] text-ink"
        />
      </label>
      <label className="grid gap-1">
        <span className="type-input-label">{t("quarterEnd")}</span>
        <input
          type="date"
          value={quarter}
          onChange={(e) => setQuarter(e.target.value)}
          className="border border-line bg-field px-2 py-1.5 font-mono text-[12px] text-ink"
        />
      </label>
      {needsConfirm && (
        <label className="flex items-start gap-2">
          <input type="checkbox" checked={confirmed} onChange={(e) => setConfirmed(e.target.checked)} />
          <span className="type-body">{t("confirmPublic")}</span>
        </label>
      )}
      <div>
        <PrimaryButton disabled={!canRead} onClick={() => void read()}>
          {t("read")}
        </PrimaryButton>
      </div>
      {stage === "uploading" && <p className="type-step" role="status">{t("uploading")}</p>}
      {stage === "reading" && <p className="type-step" role="status">{t("reading")}</p>}
      {result && (
        <div className="grid gap-2" data-unanswered={left}>
          <h3 className="type-input-group">{t("reviewTitle")}</h3>
          <p className="type-body">{t("amberLeft", { count: left })}</p>
          {amber.map((ref) => {
            const [ti, ri, ci] = ref;
            const table = result.tables[ti];
            const row = table.rows[ri];
            const cell = row.cells[ci];
            const digits = decimalsOf(table);
            const where = row.page
              ? t("figureAt", { label: row.label, column: table.columns[ci]?.label ?? "", page: String(row.page) })
              : t("figureAtNoPage", { label: row.label, column: table.columns[ci]?.label ?? "" });
            return (
              <AmberFigure
                key={cellKey(ref)}
                where={where}
                read={fmtNumber(cell.value, digits)}
                other={cell.other === null || cell.other === undefined ? null : fmtNumber(cell.other, digits)}
                reason={reader(`amber_${cell.amber}`)}
                answer={answers[cellKey(ref)] ?? null}
                digits={digits}
                cellId={cellKey(ref)}
                onAnswer={(answer) => setAnswers((old) => ({ ...old, [cellKey(ref)]: answer }))}
              />
            );
          })}
          <div>
            <PrimaryButton disabled={left > 0 || stage === "accepting"} onClick={() => void accept()}>
              {t("acceptAndImport")}
            </PrimaryButton>
          </div>
        </div>
      )}
      {error && <p className="type-alert" role="alert">{error}</p>}
    </section>
  );
}

function AmberFigure({
  where, read, other, reason, answer, digits, cellId, onAnswer,
}: {
  where: string;
  read: string;
  other: string | null;
  reason: string;
  answer: Answer | null;
  digits: number;
  cellId: string;
  onAnswer: (answer: Answer) => void;
}) {
  const t = useTranslations("excel");
  const [text, setText] = useState("");
  const value = text.trim() === "" ? null : parseNumber(text);
  return (
    <div className="grid gap-1.5 border border-line p-2" data-amber={cellId} data-answer={answer?.do ?? ""}>
      <span className="type-body">{where}</span>
      <span className="type-body text-attention">{reason}</span>
      <span className="font-mono text-[12px] text-ink">
        {t("readValue")} {read}
        {other !== null && ` · ${t("otherValue")} ${other}`}
      </span>
      {answer && (
        <span className="type-body text-accent">
          {answer.do === "accepted"
            ? t("answerAccepted")
            : t("answerCorrected", { value: answer.value === null ? "-" : fmtNumber(answer.value, digits) })}
        </span>
      )}
      <div className="flex flex-wrap gap-2">
        <SecondaryButton onClick={() => onAnswer({ do: "accepted" })}>{t("acceptAsRead")}</SecondaryButton>
        <input
          aria-label={t("correctFigure")}
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="w-28 border border-line bg-field px-2 py-1 font-mono text-[12px] text-ink"
        />
        <SecondaryButton
          disabled={text.trim() !== "" && (value === null || Number.isNaN(value))}
          onClick={() => onAnswer({ do: "corrected", value })}
        >
          {t("useCorrection")}
        </SecondaryButton>
      </div>
    </div>
  );
}
