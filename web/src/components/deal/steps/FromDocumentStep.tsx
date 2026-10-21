"use client";

import { useTranslations } from "next-intl";
import { useCallback, useEffect, useRef, useState } from "react";

import { Notice, SecondaryButton } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { api } from "@/lib/api/client";
import type { DealInputs } from "@/lib/deal/fields";
import {
  dealFromAccepted, documentJob, DOCUMENT_ERRORS, readerInfo, splitDeal, startDocumentRead, type DocumentDeal,
  type DocumentFigure, type DocumentJob, type DocumentLabel, type DocumentReaderInfo, type DocumentResult,
} from "@/lib/documents";
import { megabytes, refusedLocally, uploadFile, type FileList, type UploadProblem } from "@/lib/files";
import { fmtCount, fmtNumber, fmtPct } from "@/lib/format";
import { formatDateTime } from "@/lib/locale";

import { DealScreen, RailGroup } from "../DealScreen";
import { useDeal } from "../DealProvider";

/**
 * Deal -> From a document (PLAN.md 6.2): read an annual report or an information pack, any language, twice by two
 * AI models; review every value beside its page and quote; accept each one; then put what was accepted into the
 * deal. The deal inputs come from the API, worked out from the read's own result and only the accepted values.
 *
 * i18n-keys: documents.error_*
 * i18n-keys: files.problem_*
 * i18n-keys: documents.amber_*
 * i18n-keys: documents.figure_*
 * i18n-keys: documents.label_*
 * i18n-keys: documents.kind_*
 * i18n-keys: documents.unit_*
 * i18n-keys: documents.standard_*
 * i18n-keys: documents.scope_*
 * i18n-keys: documents.input_*
 * i18n-keys: documents.note_*
 */
const POLL_MS = 2000;

export function FromDocumentStep() {
  const t = useTranslations("documents");
  const [info, setInfo] = useState<DocumentReaderInfo | null>(null);
  const [files, setFiles] = useState<FileList | null>(null);
  const [chosen, setChosen] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [job, setJob] = useState<DocumentJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadProblem, setUploadProblem] = useState<UploadProblem | null>(null);
  const tf = useTranslations("files");

  const loadFiles = useCallback(async () => {
    const { data } = await api.GET("/api/files");
    if (data) setFiles(data);
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([readerInfo(), api.GET("/api/files")]).then(([found, list]) => {
      if (cancelled) return;
      if (found) setInfo(found);
      if (list.data) setFiles(list.data);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const running = job !== null && (job.status === "queued" || job.status === "running");
  const jobId = job?.id;
  useEffect(() => {
    if (!running || !jobId) return;
    const timer = setInterval(async () => {
      const found = await documentJob(jobId);
      if (!found) return;
      setJob(found);
      if (found.status === "failed") setError(found.error ?? "failed");
      if (found.status !== "queued" && found.status !== "running") void readerInfo().then((i) => i && setInfo(i));
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [running, jobId]);

  const pdfs = (files?.files ?? []).filter((f) => f.kind === "pdf" && f.status === "clean");
  const selected = chosen || pdfs[0]?.id || "";
  const needsConfirm = info?.confirm_public ?? true;
  const canRead = !!info && info.status === "on" && info.open && !!selected && (!needsConfirm || confirmed) && !running;

  const read = async () => {
    setError(null);
    const started = await startDocumentRead(selected, confirmed);
    if (started.job) setJob(started.job);
    else setError(started.error ?? "failed");
  };

  const upload = async (file: File) => {
    if (!files) return;
    setError(null);
    setUploadProblem(null);
    const early = refusedLocally(file, files.usage, ["pdf"]);
    if (early) {
      // the upload's own reason (too large, past the quota ...), in the Documents step's words
      if (early === "bad_type") setError("not_pdf");
      else setUploadProblem(early);
      return;
    }
    setUploading(true);
    const result = await uploadFile(file, () => undefined);
    setUploading(false);
    await loadFiles();
    setConfirmed(false);
    if (result.ok && result.file.status === "clean") setChosen(result.file.id);
    else if (result.ok) setError("not_clean");
    else setUploadProblem(result.problem);
  };

  const result = job?.status === "succeeded" ? job.result ?? null : null;
  const stage = (job?.progress ?? 0) < 0.6 ? t("stageReadA") : (job?.progress ?? 0) < 0.95 ? t("stageReadB") : t("stageChecking");

  return (
    <DealScreen
      rail={
        <Rail
          info={info}
          pdfs={pdfs}
          selected={selected}
          onChoose={(id) => {
            // a confirmation is for one document: choosing another asks again
            setChosen(id);
            setConfirmed(false);
          }}
          confirmed={confirmed}
          onConfirm={setConfirmed}
          needsConfirm={needsConfirm}
          canRead={canRead}
          running={running}
          uploading={uploading}
          ready={files !== null}
          onRead={() => void read()}
          onUpload={(file) => void upload(file)}
        />
      }
    >
      {info && info.status === "off" && (
        <Notice tone="info" title={t("offTitle")} role="note">
          {t("offBody")}
        </Notice>
      )}
      {info && info.status === "on" && !info.open && (
        <Notice tone="attention" title={t("closedTitle")} role="note">
          {t("closedBody")}
        </Notice>
      )}
      {info && info.status === "on" && info.open && info.measured.meets_bar === null && (
        <Notice tone="attention" title={t("unmeasuredTitle")} role="note">
          {t("unmeasuredBody", { cases: info.measured.cases })}
        </Notice>
      )}
      {info && info.status === "on" && info.open && info.measured.meets_bar === false && (
        <Notice tone="attention" title={t("belowBarTitle")} role="note">
          {t("belowBarBody", { cases: info.measured.cases })}
        </Notice>
      )}
      {uploadProblem && files && (
        <Notice tone="loss" title={t("notUploaded")} role="alert">
          <span data-upload-problem={uploadProblem}>
            {tf(`problem_${uploadProblem}`, {
              limit: tf("megabytes", { value: fmtNumber(megabytes(files.usage.max_file_bytes).value, 0) }),
              quota: tf("megabytes", { value: fmtNumber(megabytes(files.usage.quota_bytes).value, 0) }),
              maxFiles: files.usage.quota_files,
            })}
          </span>
        </Notice>
      )}
      {running && (
        <Notice tone="info" title={t("reading")} role="status">
          <span data-read-stage>{stage}</span>
        </Notice>
      )}
      {error && (
        <Notice tone="loss" title={t("notRead")} role="alert">
          <span data-read-error={error}>{t(`error_${DOCUMENT_ERRORS.includes(error) ? error : "failed"}`)}</span>
        </Notice>
      )}
      {result && job ? (
        <ReviewDocument key={job.id} jobId={job.id} result={result} />
      ) : (
        !running && (
          <Tiles>
            <Tile span={12} title={t("howTitle")}>
              <p className="type-body text-[10.5px]">{t("howBody")}</p>
            </Tile>
          </Tiles>
        )
      )}
    </DealScreen>
  );
}

function Rail(props: {
  info: DocumentReaderInfo | null;
  pdfs: { id: string; name: string }[];
  selected: string;
  onChoose: (id: string) => void;
  confirmed: boolean;
  onConfirm: (value: boolean) => void;
  needsConfirm: boolean;
  canRead: boolean;
  running: boolean;
  uploading: boolean;
  /** The file list has loaded: an upload before it would have nowhere to go */
  ready: boolean;
  onRead: () => void;
  onUpload: (file: File) => void;
}) {
  const t = useTranslations("documents");
  const input = useRef<HTMLInputElement>(null);
  const { info } = props;
  return (
    <>
      <RailGroup title={t("fileGroup")}>
        <label className="grid gap-1">
          <span className="type-input-label">{t("fileLabel")}</span>
          <select
            className="border border-line bg-canvas px-2 py-1 font-mono text-[11px] text-ink"
            value={props.selected}
            onChange={(event) => props.onChoose(event.target.value)}
            disabled={props.pdfs.length === 0}
          >
            {props.pdfs.length === 0 && <option value="">{t("noFiles")}</option>}
            {props.pdfs.map((file) => (
              <option key={file.id} value={file.id}>
                {file.name}
              </option>
            ))}
          </select>
        </label>
        <input
          ref={input}
          type="file"
          className="sr-only"
          aria-hidden
          tabIndex={-1}
          accept=".pdf"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (file) props.onUpload(file);
          }}
        />
        <button
          type="button"
          disabled={!props.ready || props.uploading || props.running}
          onClick={() => input.current?.click()}
          className="type-action mt-1.5 border border-line px-3 py-1.5 text-ink disabled:opacity-50"
        >
          {props.uploading ? t("uploading") : t("uploadPdf")}
        </button>
      </RailGroup>
      {props.needsConfirm && (
        <RailGroup title={t("publicGroup")}>
          <label className="flex items-start gap-2">
            <input type="checkbox" className="mt-0.5" checked={props.confirmed} onChange={(event) => props.onConfirm(event.target.checked)} />
            <span className="type-input-label">{t("publicConfirm")}</span>
          </label>
          <p className="type-body text-[10px]" data-public-warning>
            {t("publicWarning")}
          </p>
        </RailGroup>
      )}
      <RailGroup title={t("readGroup")}>
        <button type="button" disabled={!props.canRead} onClick={props.onRead} className="type-action bg-accent px-3 py-2 text-bg disabled:opacity-50">
          {props.running ? t("reading") : t("read")}
        </button>
        {info && (
          <p className="mt-1.5 font-mono text-[10.5px] text-muted" data-pages-left>
            {t("pagesLeft", { left: info.pages_left, total: info.pages_per_day })}
          </p>
        )}
      </RailGroup>
    </>
  );
}

const decimals = (value: number) => (Number.isInteger(value) ? 0 : Math.min(4, String(value).split(".")[1]?.length ?? 0));
const day = (iso: string) => formatDateTime(new Date(`${iso}T00:00:00Z`), { dateStyle: "medium", timeZone: "UTC" });

function ReviewDocument({ jobId, result }: { jobId: string; result: DocumentResult }) {
  const t = useTranslations("documents");
  const { setMoney, setFields } = useDeal();
  const [accepted, setAccepted] = useState<Set<string>>(new Set());
  const [applied, setApplied] = useState<DocumentDeal | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const toggle = (id: string, on: boolean) =>
    setAccepted((old) => {
      const next = new Set(old);
      if (on) next.add(id);
      else next.delete(id);
      return next;
    });

  const labelValue = (label: DocumentLabel | { key: string; value: string | null | undefined }) => {
    const value = label.value ?? "";
    if (!value) return "";
    if (label.key === "unit") return t(`unit_${value}`);
    if (label.key === "accounting_standard") return t(`standard_${value}`);
    if (label.key === "scope") return t(`scope_${value}`);
    if (label.key === "fiscal_year_end") return day(value);
    return value;
  };

  const use = async () => {
    setBusy(true);
    setError(null);
    const answer = await dealFromAccepted(jobId, [...accepted]);
    setBusy(false);
    if (!answer.deal) {
      setError(answer.error ?? "failed");
      return;
    }
    const { money, fields } = splitDeal(answer.deal.inputs);
    // the deal's other money is converted to the document's unit first, so nothing else changes size
    if (money) setMoney(money);
    if (Object.keys(fields).length) setFields(fields as Partial<DealInputs>);
    setApplied(answer.deal);
  };

  const unit = result.labels.find((l) => l.key === "unit");
  const currency = result.labels.find((l) => l.key === "currency");
  return (
    <Tiles>
      <Kpi title={t("kpiValues")} value={fmtCount(result.summary.figures + result.summary.labels)} sub={t("kpiFigures", { count: result.summary.figures })} lead />
      <Kpi title={t("kpiAmber")} value={fmtCount(result.summary.amber)} sub={t("kpiAmberSub")} tone={result.summary.amber > 0 ? "attention" : "gain"} />
      <Kpi title={t("kpiAccepted")} value={fmtCount(accepted.size)} sub={t("kpiAcceptedSub")} />
      <Tile span={3} title={t("documentTitle")}>
        <p className="font-mono text-[11px] text-ink" dir="auto" data-company>
          {result.company}
        </p>
        <p className="font-mono text-[10.5px] text-muted">
          {[t(`kind_${result.document_kind}`), result.language.toUpperCase(), t("pagesCount", { count: result.document.pages })].join(" · ")}
        </p>
        <p className="font-mono text-[10.5px] text-muted">{t("readBy", { a: result.reads.A.model, b: result.reads.B?.model ?? "-" })}</p>
      </Tile>
      <Tile span={12} title={t("labelsTitle")} unit={t("labelsUnit")}>
        <ValueTable
          rows={result.labels.map((label) => ({
            id: label.id,
            what: t(`label_${label.key}`),
            period: "",
            value: labelValue(label),
            other: label.other ? labelValue({ key: label.key, value: label.other }) : null,
            page: label.page ?? null,
            quote: label.quote,
            amber: label.amber ?? null,
          }))}
          accepted={accepted}
          onToggle={toggle}
        />
      </Tile>
      <Tile span={12} title={t("figuresTitle")} unit={unit && currency ? t("figuresUnit", { unit: labelValue(unit), currency: currency.value }) : t("figuresUnitUnknown")}>
        <ValueTable
          rows={result.figures.map((figure: DocumentFigure) => ({
            id: figure.id,
            what: t(`figure_${figure.key}`),
            period: day(figure.period_end),
            value: fmtNumber(figure.value, decimals(figure.value)),
            other: figure.other === null || figure.other === undefined ? null : fmtNumber(figure.other, decimals(figure.other)),
            page: figure.page ?? null,
            quote: figure.quote,
            amber: figure.amber ?? null,
          }))}
          accepted={accepted}
          onToggle={toggle}
        />
      </Tile>
      <Tile span={12} title={t("dealTitle")}>
        <p className="type-body text-[10.5px]">{t("dealBody")}</p>
        <div className="flex flex-wrap gap-2">
          <SecondaryButton disabled={accepted.size === 0 || busy} onClick={() => void use()}>
            {t("useInDeal", { count: accepted.size })}
          </SecondaryButton>
        </div>
        {error && (
          <p className="type-body text-loss" role="alert">
            {t(`error_${DOCUMENT_ERRORS.includes(error) ? error : "failed"}`)}
          </p>
        )}
        {applied && <Applied deal={applied} />}
      </Tile>
    </Tiles>
  );
}

type Row = { id: string; what: string; period: string; value: string; other: string | null; page: number | null; quote: string; amber: string | null };

function ValueTable({ rows, accepted, onToggle }: { rows: Row[]; accepted: Set<string>; onToggle: (id: string, on: boolean) => void }) {
  const t = useTranslations("documents");
  if (rows.length === 0) return <p className="type-body" role="note">{t("noValues")}</p>;
  return (
    <table className="w-full table-fixed border-collapse font-mono text-[11px]">
      <thead>
        <tr className="text-muted">
          {[t("colAccept"), t("colWhat"), t("colPeriod"), t("colValue"), t("colPage"), t("colQuote"), t("colCheck")].map((label, column) => (
            <th
              key={label}
              scope="col"
              className={`border-b border-grid px-2 py-1 font-normal ${column === 3 ? "text-end" : "text-start"} ${column === 0 ? "w-16" : column === 5 ? "w-[38%]" : ""}`}
            >
              {label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.id} data-item={row.id} data-amber={row.amber ?? ""} className="text-ink">
            <td className="border-b border-grid px-2 py-1">
              <input
                type="checkbox"
                aria-label={t("acceptOne", { what: row.what, period: row.period })}
                checked={accepted.has(row.id)}
                onChange={(event) => onToggle(row.id, event.target.checked)}
              />
            </td>
            <th scope="row" className="truncate border-b border-grid px-2 py-1 text-start font-normal">
              {row.what}
            </th>
            <td className="border-b border-grid px-2 py-1 text-muted">{row.period}</td>
            <td className="border-b border-grid px-2 py-1 text-end" data-value>
              {row.value}
            </td>
            <td className="border-b border-grid px-2 py-1 text-muted">{row.page === null ? "" : String(row.page)}</td>
            <td className="truncate border-b border-grid px-2 py-1 text-muted" dir="auto" title={row.quote} data-quote>
              {row.quote}
            </td>
            <td className={`border-b border-grid px-2 py-1 ${row.amber ? "text-attention" : "text-gain"}`}>
              {row.amber ? t(`amber_${row.amber}`, { other: row.other ?? "" }) : t("checked")}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function Applied({ deal }: { deal: DocumentDeal }) {
  const t = useTranslations("documents");
  const set = Object.entries(deal.inputs).filter(([, value]) => value !== null && value !== undefined);
  const shown = (name: string, value: unknown) => {
    if (typeof value === "number" && ["gross_margin", "opex", "da", "capex", "tax", "growth"].includes(name)) return fmtPct(value, 1);
    if (typeof value === "number" && name === "ebitda") return fmtNumber(value, 2);
    if (name === "accounting_standard" && typeof value === "string") return t(`standard_${value}`);
    if (name === "unit" && typeof value === "string") return t(`unit_${value}`);
    return String(value);
  };
  return (
    <div className="grid gap-1" data-applied>
      <p className="type-input-label">{set.length ? t("appliedTitle", { count: set.length }) : t("appliedNone")}</p>
      <ul className="grid gap-0.5 font-mono text-[10.5px] text-ink">
        {set.map(([name, value]) => (
          <li key={name} data-input={name}>
            {t(`input_${name}`)}: {shown(name, value)}
          </li>
        ))}
      </ul>
      {deal.left_out.map((left) => (
        <p key={left.input} className="font-mono text-[10px] text-attention" data-left-out={left.input}>
          {t("leftOut", { input: t(`input_${left.input}`), count: left.needs.length })}
        </p>
      ))}
      {deal.notes.map((note) => (
        <p key={note} className="font-mono text-[10px] text-attention">
          {t(`note_${note}`)}
        </p>
      ))}
    </div>
  );
}
