"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { DraftBlock } from "@/components/ui/DraftBlock";
import { Notice, PrimaryButton, RailGroup, Screen, SecondaryButton } from "@/components/ui/Screen";
import { Tile, Tiles } from "@/components/ui/Tile";
import { parseNumber } from "@/lib/locale";
import {
  downloadNote, fieldsFrom, getDetails, getReport, plain, putDetails, putReport, refusedKey, writeSentences,
  type Report, type ReportDetails, type ReportFields,
} from "@/lib/report";
import { listCoverage, type CoverageCompany } from "@/lib/research";
import { listVersions, type ModelVersion } from "@/lib/savedModel";

/**
 * Coverage -> Report (PLAN.md R8): a results note drafted from a covered company's saved sector model. The API
 * fills every number from the model and checks each against its cell; the linking sentences are the templates'
 * until the AI is asked to write them (it is sent no company and no figure, and a sentence that breaks a rule goes
 * back to its template). The rating, the target and the argument are the analyst's own fields, empty until typed:
 * the tool picks none. The firm name and disclosures belong to the account and go on every note.
 *
 * i18n-keys: report.refused_*
 * i18n-keys: report.source_*
 * i18n-keys: report.section_*
 */
const field = "border border-line bg-canvas px-2 py-1 text-[11px] text-ink";

export function ReportStep() {
  const t = useTranslations("report");
  const [companies, setCompanies] = useState<CoverageCompany[] | null>(null);
  const [companyId, setCompanyId] = useState("");
  const [report, setReport] = useState<Report | null>(null);
  const [refused, setRefused] = useState<string | null>(null);
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [details, setDetails] = useState<ReportDetails | null>(null);

  useEffect(() => {
    let cancelled = false;
    void Promise.all([listCoverage(), getDetails()]).then(([list, found]) => {
      if (cancelled) return;
      const all = list?.companies ?? [];
      setCompanies(all);
      setDetails(found ?? { firm: "", disclosures: "" });
      if (all.length) setCompanyId((now) => now || all[0].id);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const [drawn, setDrawn] = useState(0);         // bumped to draw the note again (the account's details changed)

  useEffect(() => {
    let cancelled = false;
    if (!companyId) return;
    void Promise.all([getReport(companyId), listVersions(companyId)]).then(([answer, kept]) => {
      if (cancelled) return;
      setReport(answer.report ?? null);
      setRefused(answer.refused ?? null);
      setVersions(kept);
    });
    return () => {
      cancelled = true;
    };
  }, [companyId, drawn]);

  const take = (answer: { report?: Report; refused?: string }) => {
    if (answer.report) setReport(answer.report);
    setRefused(answer.refused ?? null);
  };

  return (
    <Screen
      rail={
        <>
          <RailGroup title={t("companyGroup")}>
            {companies === null ? (
              <p className="type-step">{t("loading")}</p>
            ) : companies.length === 0 ? (
              <p className="type-body text-[10.5px]">{t("noCompanies")}</p>
            ) : (
              <label className="grid gap-1">
                <span className="type-input-label">{t("company")}</span>
                <select className={field} value={companyId} onChange={(e) => setCompanyId(e.target.value)}>
                  {companies.map((c) => (
                    <option key={c.id} value={c.id}>{c.name}</option>
                  ))}
                </select>
              </label>
            )}
          </RailGroup>
          {report && (
            <ViewGroup key={`${report.company_id}:${JSON.stringify(report.fields)}`} report={report} versions={versions}
              onSave={async (fields) => take(await putReport(report.company_id, fields))} />
          )}
          {report && <SentencesGroup report={report} onWrite={async () => take(await writeSentences(report.company_id))} />}
          {details && (
            <DetailsGroup key={JSON.stringify(details)} details={details} onSave={async (next) => {
              const saved = await putDetails(next);
              if (!saved) return false;
              setDetails(saved);
              setDrawn((n) => n + 1);
              return true;
            }} />
          )}
          {report && (
            <RailGroup title={t("downloadGroup")}>
              <p className="type-body mb-1.5 text-[10.5px]">{t("downloadHint")}</p>
              <div className="flex flex-wrap gap-1.5">
                <PrimaryButton onClick={() => void downloadNote(report.company_id, report.company_name, "word")}>
                  {t("downloadWord")}
                </PrimaryButton>
                <SecondaryButton onClick={() => void downloadNote(report.company_id, report.company_name, "pdf")}>
                  {t("downloadPdf")}
                </SecondaryButton>
              </div>
            </RailGroup>
          )}
        </>
      }
    >
      <Notice tone="info" title={t("toolTitle")} role="note">{t("tool")}</Notice>
      {refused && (
        <Notice tone={refused === "no_model" ? "attention" : "loss"} title={t("refusedTitle")} role="alert">
          {refused === "limit" ? t("limit") : t(refusedKey(refused))}
        </Notice>
      )}
      <Tiles>
        {!report && !refused && (
          <Tile span={12} title={t("emptyTitle")}>
            <p className="type-body text-[10.5px]">{t("emptyBody")}</p>
          </Tile>
        )}
        {report && (
          <Tile span={12} title={t("previewTitle", { name: report.company_name })}>
            <article className="grid max-w-[760px] gap-2" aria-label={t("previewLabel")}>
              {report.blocks.map((block, i) => (
                <DraftBlock key={i} block={block} chartLabel={t("chartLabel", { title: plain(block.title) })} />
              ))}
            </article>
          </Tile>
        )}
      </Tiles>
    </Screen>
  );
}

function ViewGroup({ report, versions, onSave }: {
  report: Report; versions: ModelVersion[]; onSave: (fields: ReportFields) => Promise<void>;
}) {
  const t = useTranslations("report");
  const start = fieldsFrom(report);
  const [rating, setRating] = useState(start.rating ?? "");
  const [target, setTarget] = useState(start.target == null ? "" : String(start.target));
  const [date, setDate] = useState(start.note_date ?? "");
  const [argument, setArgument] = useState(start.argument ?? "");
  const [compare, setCompare] = useState(start.compare_version == null ? "" : String(start.compare_version));
  const [busy, setBusy] = useState(false);
  const parsed = target.trim() ? parseNumber(target) : null;
  const badTarget = parsed !== null && !(Number.isFinite(parsed) && parsed >= 0);

  const save = async () => {
    setBusy(true);
    try {
      await onSave({
        rating, target: badTarget ? null : parsed, note_date: date || null, argument,
        compare_version: compare ? Number(compare) : null,
      });
    } finally {
      setBusy(false);
    }
  };

  return (
    <RailGroup title={t("viewGroup")}>
      <p className="type-body mb-1.5 text-[10.5px]">{t("viewHint")}</p>
      <label className="grid gap-1">
        <span className="type-input-label">{t("ratingLabel")}</span>
        <input className={field} value={rating} maxLength={40} onChange={(e) => setRating(e.target.value)} />
      </label>
      <label className="grid gap-1">
        <span className="type-input-label">{t("targetLabel")}</span>
        <input className={field} value={target} inputMode="decimal" onChange={(e) => setTarget(e.target.value)}
          aria-invalid={badTarget} />
      </label>
      {badTarget && <p className="type-body text-[10px] text-loss">{t("targetInvalid")}</p>}
      <label className="grid gap-1">
        <span className="type-input-label">{t("dateLabel")}</span>
        <input className={field} type="date" value={date} onChange={(e) => setDate(e.target.value)} />
      </label>
      <label className="grid gap-1">
        <span className="type-input-label">{t("compareLabel")}</span>
        <select className={field} value={compare} onChange={(e) => setCompare(e.target.value)}>
          <option value="">{t("compareNone")}</option>
          {versions.map((v) => (
            <option key={v.number} value={String(v.number)}>
              {v.label ? t("versionNamed", { number: String(v.number), label: v.label }) : t("version", { number: String(v.number) })}
            </option>
          ))}
        </select>
      </label>
      <label className="grid gap-1">
        <span className="type-input-label">{t("argumentLabel")}</span>
        <textarea className={`${field} min-h-[110px]`} value={argument} maxLength={20000}
          onChange={(e) => setArgument(e.target.value)} />
      </label>
      <div className="mt-1.5">
        <PrimaryButton onClick={() => void save()} disabled={busy || badTarget}>{t("saveView")}</PrimaryButton>
      </div>
    </RailGroup>
  );
}

function SentencesGroup({ report, onWrite }: { report: Report; onWrite: () => Promise<void> }) {
  const t = useTranslations("report");
  const [busy, setBusy] = useState(false);
  return (
    <RailGroup title={t("sentencesGroup")}>
      <p className="type-body mb-1.5 text-[10.5px]">{t("sentencesHint")}</p>
      <ul className="mb-1.5 grid gap-0.5">
        {report.sentences.map((s) => (
          <li key={s.section} className="type-body flex justify-between gap-2 text-[10px]" data-section={s.section}>
            <span>{t(`section_${s.section}`)}</span>
            <span className={s.source === "ai" ? "text-accent" : "text-muted"}>{t(`source_${s.source}`)}</span>
          </li>
        ))}
      </ul>
      <SecondaryButton disabled={busy} onClick={() => {
        setBusy(true);
        void onWrite().finally(() => setBusy(false));
      }}>
        {busy ? t("writing") : t("write")}
      </SecondaryButton>
    </RailGroup>
  );
}

function DetailsGroup({ details, onSave }: { details: ReportDetails; onSave: (d: ReportDetails) => Promise<boolean> }) {
  const t = useTranslations("report");
  const [firm, setFirm] = useState(details.firm);
  const [disclosures, setDisclosures] = useState(details.disclosures);
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  return (
    <RailGroup title={t("detailsGroup")}>
      <p className="type-body mb-1.5 text-[10.5px]">{t("detailsHint")}</p>
      <label className="grid gap-1">
        <span className="type-input-label">{t("firmLabel")}</span>
        <input className={field} value={firm} maxLength={200} onChange={(e) => setFirm(e.target.value)} />
      </label>
      <label className="grid gap-1">
        <span className="type-input-label">{t("disclosuresLabel")}</span>
        <textarea className={`${field} min-h-[90px]`} value={disclosures} maxLength={8000}
          onChange={(e) => setDisclosures(e.target.value)} />
      </label>
      <div className="mt-1.5">
        <SecondaryButton disabled={busy} onClick={() => {
          setBusy(true);
          void onSave({ firm, disclosures }).then((ok) => setFailed(!ok)).finally(() => setBusy(false));
        }}>
          {t("saveDetails")}
        </SecondaryButton>
      </div>
      {failed && <p className="type-body text-[10px] text-loss" role="alert">{t("detailsFailed")}</p>}
    </RailGroup>
  );
}
