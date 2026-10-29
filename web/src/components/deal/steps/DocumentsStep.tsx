"use client";

import { useTranslations } from "next-intl";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { EmptyGuide, Notice, SecondaryButton } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { api } from "@/lib/api/client";
import { acceptedExtensions, megabytes, refusalCode, refusedLocally, uploadFile, type FileList, type StoredFile, type UploadProblem } from "@/lib/files";
import { fmtCount, fmtNumber } from "@/lib/format";
import { formatDateTime } from "@/lib/locale";

import { DealScreen, RailGroup } from "../DealScreen";
import { TipText, useTip } from "@/components/ui/Tip";

/**
 * i18n-keys: files.problem_*
 * i18n-keys: files.reason_*
 * i18n-keys: files.status_*
 * i18n-keys: files.kind_*
 */
type Progress = { name: string; stage: "sending" | "checking" };

/** How often the list is read again while a file is still being checked. */
const RECHECK_MS = 2500;

const day = (iso: string) => formatDateTime(new Date(iso), { dateStyle: "medium" });

/** Upload, list, download and delete documents (PLAN.md 6.1). */
export function DocumentsStep() {
  const t = useTranslations("files");
  const [list, setList] = useState<FileList | null>(null);
  const [problems, setProblems] = useState<{ name: string; problem: UploadProblem }[]>([]);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [loadFailed, setLoadFailed] = useState(false);

  const load = useCallback(async () => {
    const { data } = await api.GET("/api/files");
    setLoadFailed(!data);
    if (data) setList(data);
    return data ?? null;
  }, []);

  useEffect(() => {
    let cancelled = false;
    api.GET("/api/files").then(({ data }) => {
      if (cancelled) return;
      setLoadFailed(!data);
      if (data) setList(data);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  // A check can outlast the upload that started it (a simulation ahead of it,
  // a store that could not be reached): while one is unfinished the list is
  // read again, which is also what queues a lost check again on the server
  const checking = !progress && !!list?.files.some((f) => f.status === "scanning");
  useEffect(() => {
    if (!checking) return;
    const timer = setInterval(() => void load(), RECHECK_MS);
    return () => clearInterval(timer);
  }, [checking, load]);

  const size = (bytes: number) => {
    // A small file in kilobytes: "0.0 MB" says nothing
    if (bytes > 0 && bytes < 1024 * 1024) return t("kilobytes", { value: fmtNumber(Math.max(1, Math.round(bytes / 1024)), 0) });
    const { value, digits } = megabytes(bytes);
    return t("megabytes", { value: fmtNumber(value, digits) });
  };

  const upload = async (chosen: File[]) => {
    const failed: { name: string; problem: UploadProblem }[] = [];
    let current = list;
    for (const file of chosen) {
      if (!current) break;
      const early = refusedLocally(file, current.usage, current.kinds);
      if (early) {
        failed.push({ name: file.name, problem: early });
        continue;
      }
      setProgress({ name: file.name, stage: "sending" });
      const result = await uploadFile(file, (stage) => setProgress({ name: file.name, stage }));
      if (!result.ok) failed.push({ name: file.name, problem: result.problem });
      current = (await load()) ?? current;
    }
    setProgress(null);
    setProblems(failed);
  };

  const usage = list?.usage;
  return (
    <DealScreen rail={<UploadRail list={list} busy={progress !== null} onChoose={(files) => void upload(files)} size={size} />}>
      {loadFailed && (
        <Notice tone="loss" title={t("notLoaded")} role="alert">
          {t("notLoadedBody")}
        </Notice>
      )}
      {list && !list.enabled && (
        <Notice tone="info" title={t("notSetUp")} role="note">
          {t("notSetUpBody")}
        </Notice>
      )}
      {usage?.near_limit && (
        <Notice tone={usage.full ? "loss" : "attention"} title={usage.full ? t("quotaFull") : t("quotaNear")} role="status">
          <span data-quota-message>
            {t("quotaBody", {
              used: size(usage.bytes),
              quota: size(usage.quota_bytes),
              files: usage.files,
              maxFiles: usage.quota_files,
            })}
          </span>
        </Notice>
      )}
      {progress && (
        <Notice tone="info" title={progress.stage === "sending" ? t("sending") : t("checking")} role="status">
          <span dir="auto">{progress.name}</span>
        </Notice>
      )}
      {problems.length > 0 && (
        <Notice tone="loss" title={t("notUploaded")} role="alert" actions={<SecondaryButton onClick={() => setProblems([])}>{t("dismiss")}</SecondaryButton>}>
          <span className="grid gap-0.5">
            {problems.map((p, i) => (
              <span key={i} data-upload-problem={p.problem}>
                <span dir="auto">{p.name}</span>
                {": "}
                {t(`problem_${p.problem}`, { limit: usage ? size(usage.max_file_bytes) : "", quota: usage ? size(usage.quota_bytes) : "", maxFiles: usage?.quota_files ?? 0 })}
              </span>
            ))}
          </span>
        </Notice>
      )}
      <Tiles>
        <Kpi title={t("kpiFiles")} value={usage ? fmtCount(usage.files) : t("none")} sub={usage ? t("ofFiles", { count: usage.quota_files }) : undefined} lead />
        <Kpi
          title={t("kpiSpace")}
          value={usage ? size(usage.bytes) : t("none")}
          sub={usage ? t("ofSpace", { quota: size(usage.quota_bytes) }) : undefined}
          tone={usage?.full ? "loss" : usage?.near_limit ? "attention" : undefined}
        />
        <Kpi title={t("kpiLargest")} value={usage ? size(usage.max_file_bytes) : t("none")} sub={t("perFile")} />
        <Tile span={6} title={t("howTitle")}>
          <p className="type-body text-[10.5px]">{t("howBody", { days: usage?.retention_days ?? 90 })}</p>
        </Tile>
        <Documents list={list} onChange={() => void load()} size={size} />
      </Tiles>
    </DealScreen>
  );
}

function UploadRail({ list, busy, onChoose, size }: { list: FileList | null; busy: boolean; onChoose: (files: File[]) => void; size: (bytes: number) => string }) {
  const t = useTranslations("files");
  const input = useRef<HTMLInputElement>(null);
  const usage = list?.usage;
  const off = !list?.enabled || busy || !!usage?.full;
  const share = usage ? Math.min(1, usage.bytes / usage.quota_bytes) : 0;
  const uploadTip = useTip("uploadFiles");
  return (
    <>
      <RailGroup title={t("uploadGroup")}>
        <input aria-describedby={uploadTip.id}
          ref={input}
          type="file"
          multiple
          className="sr-only"
          // The button below is the control; this is only the browser's file dialog
          aria-hidden
          tabIndex={-1}
          accept={list ? acceptedExtensions(list.kinds) : undefined}
          disabled={off}
          onChange={(event) => {
            const files = Array.from(event.target.files ?? []);
            event.target.value = ""; // the same file can be chosen again
            if (files.length) onChoose(files);
          }}
        />
        <TipText tip={uploadTip} />
        <button type="button" disabled={off} aria-describedby={uploadTip.id} onClick={() => input.current?.click()} className="type-action bg-accent px-3 py-2 text-bg disabled:opacity-50">
          {busy ? t("uploading") : t("chooseFiles")}
        </button>
        <p className="type-body mt-2 text-[10px]">{t("accepted", { limit: usage ? size(usage.max_file_bytes) : "" })}</p>
      </RailGroup>
      <RailGroup title={t("spaceGroup")}>
        <div className="h-1.5 bg-line-strong" role="img" aria-label={t("spaceUsed", { share: fmtNumber(share * 100, 0) })}>
          <div className={`h-full ${usage?.full ? "bg-loss" : usage?.near_limit ? "bg-attention" : "bg-accent"}`} style={{ inlineSize: `${share * 100}%` }} />
        </div>
        <p className="mt-1.5 font-mono text-[10.5px] text-muted" data-usage>
          {usage ? t("spaceLine", { used: size(usage.bytes), quota: size(usage.quota_bytes) }) : t("none")}
        </p>
        <p className="font-mono text-[10.5px] text-muted">{usage ? t("filesLine", { files: usage.files, maxFiles: usage.quota_files }) : ""}</p>
      </RailGroup>
    </>
  );
}

function Documents({ list, onChange, size }: { list: FileList | null; onChange: () => void; size: (bytes: number) => string }) {
  const t = useTranslations("files");
  const guide = useTranslations("guides");
  const [confirming, setConfirming] = useState<string | null>(null);
  const [failed, setFailed] = useState<string | null>(null);

  const download = async (file: StoredFile) => {
    const { data, error } = await api.POST("/api/files/{file_id}/link", { params: { path: { file_id: file.id } } });
    if (!data) {
      setFailed(t(`problem_${refusalCode(error) ?? "failed"}`, { limit: "", quota: "", maxFiles: 0 }));
      return;
    }
    setFailed(null);
    // The link answers with "save this file", so the page stays where it is
    window.location.assign(data.url);
  };

  const remove = async (file: StoredFile) => {
    const { response } = await api.DELETE("/api/files/{file_id}", { params: { path: { file_id: file.id } } });
    setConfirming(null);
    setFailed(response.ok || response.status === 404 ? null : t("deleteFailed"));
    onChange();
  };

  const files = list?.files ?? [];
  return (
    <Tile span={12} title={t("listTitle")} unit={t("listUnit")}>
      {failed && (
        <p className="type-body text-loss" role="alert">
          {failed}
        </p>
      )}
      {list === null ? (
        <p className="type-body" role="note">
          {t("loading")}
        </p>
      ) : files.length === 0 && !list.enabled ? (
        <p className="type-body" role="note">
          {t("notSetUpBody")}
        </p>
      ) : files.length === 0 ? (
        <EmptyGuide
          id="documents"
          title={guide("documentsTitle")}
          steps={[guide("documentsStep1"), guide("documentsStep2"), guide("documentsStep3")]}
          actions={
            <Link href="/deal/from-document" className="type-action-secondary px-2.5 py-1.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)]">
              {guide("documentsRead")}
            </Link>
          }
        >
          {t("empty")}
        </EmptyGuide>
      ) : (
        <table className="w-full border-collapse font-mono text-[11px]" aria-label={t("listTitle")}>
          <thead>
            <tr className="text-start text-muted">
              {[t("colName"), t("colKind"), t("colSize"), t("colStatus"), t("colUploaded"), t("colDeleted")].map((label, column) => (
                <th key={label} scope="col" className={`border-b border-grid px-2 py-1 font-normal ${column === 2 ? "text-end" : "text-start"}`}>
                  {label}
                </th>
              ))}
              <th scope="col" className="border-b border-grid px-2 py-1 text-end font-normal">
                <span className="sr-only">{t("colActions")}</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {files.map((file) => (
              <tr key={file.id} data-file={file.name} data-status={file.status} className="text-ink">
                <th scope="row" dir="auto" className="max-w-[320px] truncate border-b border-grid px-2 py-1.5 text-start font-normal" title={file.name}>
                  {file.name}
                </th>
                <td className="border-b border-grid px-2 py-1.5 text-muted">{t(`kind_${file.kind}`)}</td>
                <td className="border-b border-grid px-2 py-1.5 text-end">{size(file.size)}</td>
                <td className={`border-b border-grid px-2 py-1.5 ${file.status === "refused" ? "text-loss" : file.status === "clean" ? "text-gain" : "text-attention"}`}>
                  {t(`status_${file.status}`)}
                  {file.reason && <span className="text-muted"> {t(`reason_${file.reason}`)}</span>}
                </td>
                <td className="border-b border-grid px-2 py-1.5 text-muted">{day(file.created_at)}</td>
                <td className="border-b border-grid px-2 py-1.5 text-muted">{day(file.expires_at)}</td>
                <td className="border-b border-grid px-2 py-1">
                  <span className="flex justify-end gap-1.5">
                    {confirming === file.id ? (
                      <>
                        <span className="type-body self-center text-loss">{t("deleteForGood")}</span>
                        <SecondaryButton onClick={() => void remove(file)}>{t("confirmDelete")}</SecondaryButton>
                        <SecondaryButton onClick={() => setConfirming(null)}>{t("cancel")}</SecondaryButton>
                      </>
                    ) : (
                      <>
                        {file.status === "clean" && <SecondaryButton onClick={() => void download(file)}>{t("download")}</SecondaryButton>}
                        <SecondaryButton onClick={() => setConfirming(file.id)}>{t("delete")}</SecondaryButton>
                      </>
                    )}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Tile>
  );
}
