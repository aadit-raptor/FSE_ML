"use client";

import { useTranslations } from "next-intl";
import { useCallback, useEffect, useRef, useState } from "react";

import { Notice, RailGroup, Screen } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { api } from "@/lib/api/client";
import { refusedLocally, uploadFile, type FileList } from "@/lib/files";
import { fmtCount, fmtRate } from "@/lib/format";
import { formatDateTime } from "@/lib/locale";
import {
  addCompany, listCoverage, readJob, startRead, type CoverageCompany, type ReaderInfo, type ReadJob, type ReadResult,
  READER_ERRORS,
} from "@/lib/research";

import { Review } from "./Review";

/**
 * Coverage -> Results (PLAN.md R1, R2): read a results filing twice and check it, for a covered company, then
 * review it with the page beside the figures and accept the quarter. Nothing is kept until it is accepted.
 *
 * i18n-keys: reader.amber_*
 * i18n-keys: reader.problem_*
 * i18n-keys: reader.error_*
 * i18n-keys: reader.statement_*
 * i18n-keys: reader.scope_*
 * i18n-keys: reader.unit_*
 */
const POLL_MS = 2000;

export function ResultsStep() {
  const t = useTranslations("reader");
  const [info, setInfo] = useState<ReaderInfo | null>(null);
  const [files, setFiles] = useState<FileList | null>(null);
  const [chosen, setChosen] = useState("");
  const [quarter, setQuarter] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [job, setJob] = useState<ReadJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [companies, setCompanies] = useState<CoverageCompany[] | null>(null);
  const [companyId, setCompanyId] = useState("");

  const loadFiles = useCallback(async () => {
    const { data } = await api.GET("/api/files");
    if (data) setFiles(data);
    return data ?? null;
  }, []);
  const loadInfo = useCallback(async () => {
    const { data } = await api.GET("/api/research/reader");
    if (data) setInfo(data);
  }, []);

  useEffect(() => {
    let cancelled = false;
    Promise.all([api.GET("/api/research/reader"), api.GET("/api/files"), listCoverage()]).then(([reader, list, coverage]) => {
      if (cancelled) return;
      if (reader.data) setInfo(reader.data);
      if (list.data) setFiles(list.data);
      if (coverage) setCompanies(coverage.companies);
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
      const found = await readJob(jobId);
      if (!found) return;
      setJob(found);
      if (found.status === "failed") setError(found.error ?? "failed");
      if (found.status !== "queued" && found.status !== "running") void loadInfo();
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [running, jobId, loadInfo]);

  const pdfs = (files?.files ?? []).filter((f) => f.kind === "pdf" && f.status === "clean");
  const selected = chosen || pdfs[0]?.id || "";
  const needsConfirm = info?.confirm_public ?? true;
  const company = (companies ?? []).find((c) => c.id === companyId) ?? companies?.[0] ?? null;
  const canRead = !!info && info.status === "on" && info.open && !!selected && !!company && (!needsConfirm || confirmed) && !running;

  const read = async () => {
    setError(null);
    const started = await startRead(selected, confirmed, quarter || null, company?.id ?? null);
    if (started.job) setJob(started.job);
    else setError(started.error);
  };

  const add = async (name: string) => {
    setError(null);
    const made = await addCompany(name);
    if (made.company) {
      const found = made.company;
      setCompanies((old) => [found, ...(old ?? [])]);
      setCompanyId(found.id);
    } else setError(made.error ?? "failed");
  };

  const upload = async (file: File) => {
    if (!files) return;
    setError(null);
    const early = refusedLocally(file, files.usage, ["pdf"]);
    if (early) {
      setError(early === "bad_type" ? "not_pdf" : "upload_failed");
      return;
    }
    setUploading(true);
    const result = await uploadFile(file, () => undefined);
    setUploading(false);
    await loadFiles();
    setConfirmed(false);
    if (result.ok && result.file.status === "clean") setChosen(result.file.id);
    else setError(result.ok ? "not_clean" : "upload_failed");
  };

  const result = job?.status === "succeeded" ? job.result ?? null : null;
  const stage = (job?.progress ?? 0) < 0.55 ? t("stageReadA") : (job?.progress ?? 0) < 0.95 ? t("stageReadB") : t("stageChecking");
  const errorText = error ? t(`error_${READER_ERRORS.includes(error) ? error : "failed"}`) : null;

  return (
    <Screen
      rail={
        <ReadRail
          info={info}
          companies={companies ?? []}
          company={company?.id ?? ""}
          onCompany={setCompanyId}
          onAdd={(name) => void add(name)}
          pdfs={pdfs}
          selected={selected}
          onChoose={(id) => {
            // a confirmation is for one file: choosing another asks again
            setChosen(id);
            setConfirmed(false);
          }}
          quarter={quarter}
          onQuarter={setQuarter}
          confirmed={confirmed}
          onConfirm={setConfirmed}
          needsConfirm={needsConfirm}
          canRead={canRead}
          running={running}
          uploading={uploading}
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
          {t("belowBarBody", { cases: info.measured.cases, share: fmtRate(info.measured.amber_share, 2) })}
        </Notice>
      )}
      {running && (
        <Notice tone="info" title={t("reading")} role="status">
          <span data-read-stage>{stage}</span>
        </Notice>
      )}
      {errorText && (
        <Notice tone="loss" title={t("notRead")} role="alert">
          <span data-read-error={error ?? ""}>{errorText}</span>
        </Notice>
      )}
      {result && job ? (
        <Tiles>
          <ReadView result={result} />
          <Review key={job.id} readId={job.id} result={result} company={company} onAccepted={() => void listCoverage().then((c) => c && setCompanies(c.companies))} />
        </Tiles>
      ) : !running && (
        <Tiles>
          <Tile span={12} title={t("howTitle")}>
            <p className="type-body text-[10.5px]">{t("howBody")}</p>
          </Tile>
        </Tiles>
      )}
    </Screen>
  );
}

function ReadRail(props: {
  info: ReaderInfo | null;
  companies: CoverageCompany[];
  company: string;
  onCompany: (id: string) => void;
  onAdd: (name: string) => void;
  pdfs: { id: string; name: string }[];
  selected: string;
  onChoose: (id: string) => void;
  quarter: string;
  onQuarter: (value: string) => void;
  confirmed: boolean;
  onConfirm: (value: boolean) => void;
  needsConfirm: boolean;
  canRead: boolean;
  running: boolean;
  uploading: boolean;
  onRead: () => void;
  onUpload: (file: File) => void;
}) {
  const t = useTranslations("reader");
  const tc = useTranslations("coverage");
  const input = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const { info } = props;
  return (
    <>
      <RailGroup title={tc("companyGroup")}>
        <label className="grid gap-1">
          <span className="type-input-label">{tc("companyLabel")}</span>
          <select
            className="border border-line bg-canvas px-2 py-1 font-mono text-[11px] text-ink"
            value={props.company}
            onChange={(event) => props.onCompany(event.target.value)}
            disabled={props.companies.length === 0}
          >
            {props.companies.length === 0 && <option value="">{tc("noCompanies")}</option>}
            {props.companies.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
        <form
          className="mt-1.5 grid grid-cols-[minmax(0,1fr)_auto] items-end gap-1.5"
          onSubmit={(event) => {
            event.preventDefault();
            if (name.trim()) props.onAdd(name.trim());
            setName("");
          }}
        >
          <label className="grid min-w-0 gap-1">
            <span className="type-input-label">{tc("newCompany")}</span>
            <input
              className="min-w-0 border border-line bg-canvas px-2 py-1 font-mono text-[11px] text-ink"
              value={name}
              maxLength={120}
              onChange={(event) => setName(event.target.value)}
            />
          </label>
          <button type="submit" disabled={!name.trim()} className="type-action border border-line px-2 py-1 text-ink disabled:opacity-50">
            {tc("addCompany")}
          </button>
        </form>
      </RailGroup>
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
          disabled={props.uploading || props.running}
          onClick={() => input.current?.click()}
          className="type-action mt-1.5 border border-line px-3 py-1.5 text-ink disabled:opacity-50"
        >
          {props.uploading ? t("uploading") : t("uploadPdf")}
        </button>
        <label className="mt-2 grid gap-1">
          <span className="type-input-label">{t("quarterLabel")}</span>
          <input
            type="date"
            className="border border-line bg-canvas px-2 py-1 font-mono text-[11px] text-ink"
            value={props.quarter}
            onChange={(event) => props.onQuarter(event.target.value)}
          />
        </label>
        <p className="type-body text-[10px]">{t("quarterHelp")}</p>
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

function ReadView({ result }: { result: ReadResult }) {
  const t = useTranslations("reader");
  const share = result.summary.amber_share;
  return (
    <>
      <Kpi title={t("kpiFigures")} value={fmtCount(result.summary.figures)} sub={t("kpiTables", { count: result.summary.tables })} lead />
      <Kpi
        title={t("kpiAmber")}
        value={fmtCount(result.summary.amber)}
        sub={share === null || share === undefined ? undefined : t("kpiAmberShare", { share: fmtRate(share, 2) })}
        tone={result.summary.amber > 0 ? "attention" : "gain"}
      />
      <Kpi title={t("kpiInSums")} value={fmtCount(result.summary.in_sums)} sub={t("kpiInSumsSub")} />
      <Kpi
        title={t("kpiPeriods")}
        value={fmtCount(result.summary.period_problems)}
        sub={result.quarter_end ? t("quarterEnded", { day: formatDateTime(new Date(`${result.quarter_end}T00:00:00Z`), { dateStyle: "medium", timeZone: "UTC" }) }) : t("quarterUnknown")}
        tone={result.summary.period_problems > 0 ? "attention" : undefined}
      />
      <Tile span={4} title={t("filingTitle")}>
        <p className="font-mono text-[11px] text-ink" dir="auto" data-company>
          {result.company}
        </p>
        <p className="font-mono text-[10.5px] text-muted">
          {result.unit ? t(`unit_${result.unit}`) : t("unitUnknown")}
          {result.currency ? ` · ${result.currency}` : ""}
          {" · "}
          {t("readBy", { a: result.reads.A.model, b: result.reads.B?.model ?? "-" })}
        </p>
      </Tile>
      {result.problems.length > 0 && (
        <Tile span={12} title={t("problemsTitle")}>
          <ul className="grid gap-0.5">
            {result.problems.map((problem, i) => (
              <li key={i} className="type-body text-attention" data-problem={problem.code}>
                {problem.statement ? `${t(`statement_${problem.statement}`)}: ` : ""}
                {t(`problem_${problem.code}`, { column: problem.column === null || problem.column === undefined ? "" : String(problem.column + 1), detail: problem.detail })}
              </li>
            ))}
          </ul>
        </Tile>
      )}
    </>
  );
}
