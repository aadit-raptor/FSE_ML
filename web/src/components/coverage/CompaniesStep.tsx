"use client";

import Link from "next/link";
import { useTranslations } from "next-intl";
import { useCallback, useEffect, useState } from "react";

import { EmptyGuide, Notice, RailGroup, Screen } from "@/components/ui/Screen";
import { Tile, Tiles } from "@/components/ui/Tile";
import { downloadFile } from "@/lib/export";
import { fmtCount, fmtNumber } from "@/lib/format";
import { formatDateTime } from "@/lib/locale";
import {
  addCompany, listCoverage, removeCompany, removePeriod, storedPeriod, workbookName, workbookPath,
  type CoverageCompany, type CoverageList, type PeriodSummary, type StoredPeriod,
} from "@/lib/research";

import { PageView } from "./PageView";
import { TipLabel, TipText, useTip } from "@/components/ui/Tip";

/**
 * Coverage -> Companies (PLAN.md R2): the companies an analyst follows and the quarters they accepted, each
 * downloadable as the checked workbook and viewable with the page beside its figures while the filing is
 * kept (6.1's 90 days; the figures stay after it goes).
 *
 * i18n-keys: reader.statement_*
 * i18n-keys: reader.scope_*
 * i18n-keys: reader.amber_*
 * i18n-keys: reader.unit_*
 */
const day = (iso: string) => formatDateTime(new Date(`${iso.slice(0, 10)}T00:00:00Z`), { dateStyle: "medium", timeZone: "UTC" });

export function CompaniesStep() {
  const t = useTranslations("coverage");
  const guide = useTranslations("guides");
  const nameTip = useTip("coverageNewCompany");
  const [list, setList] = useState<CoverageList | null>(null);
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<StoredPeriod | null>(null);

  const load = useCallback(async () => {
    const found = await listCoverage();
    if (found) setList(found);
  }, []);
  useEffect(() => {
    let cancelled = false;
    void listCoverage().then((found) => {
      if (!cancelled && found) setList(found);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  const add = async () => {
    setError(null);
    const made = await addCompany(name.trim());
    if (made.company) {
      setName("");
      await load();
    } else setError(made.error ?? "failed");
  };

  return (
    <Screen
      rail={
        <RailGroup title={t("companyGroup")}>
          <form
            className="grid gap-1.5"
            onSubmit={(event) => {
              event.preventDefault();
              if (name.trim()) void add();
            }}
          >
            <label className="grid gap-1">
              <span className="type-input-label"><TipLabel tip={nameTip}>{t("newCompany")}</TipLabel></span>
              <input aria-describedby={nameTip.id}
                className="border border-line bg-canvas px-2 py-1 font-mono text-[11px] text-ink"
                value={name}
                maxLength={120}
                onChange={(event) => setName(event.target.value)}
              />
            </label>
            <TipText tip={nameTip} />
            <button type="submit" disabled={!name.trim()} className="type-action bg-accent px-3 py-2 text-bg disabled:opacity-50">
              {t("addCompany")}
            </button>
          </form>
          {list && (
            <p className="mt-1.5 font-mono text-[10.5px] text-muted">
              {t("limits", { companies: fmtCount(list.limits.companies), periods: fmtCount(list.limits.periods) })}
            </p>
          )}
        </RailGroup>
      }
    >
      {error && (
        <Notice tone="loss" title={t("notAdded")} role="alert">
          {t(`error_${["too_many_companies", "bad_name", "no_account"].includes(error) ? error : "failed"}`)}
        </Notice>
      )}
      <Tiles>
        {list && list.companies.length === 0 && (
          <Tile span={12} title={t("emptyTitle")}>
            <EmptyGuide
              id="coverage"
              title={guide("coverageTitle")}
              steps={[guide("coverageStep1"), guide("coverageStep2"), guide("coverageStep3"), guide("coverageStep4")]}
            >
              {t("emptyBody")}
            </EmptyGuide>
          </Tile>
        )}
        {list?.companies.map((company) => (
          <CompanyTile
            key={company.id}
            company={company}
            onOpen={async (period) => setOpen(await storedPeriod(period.id))}
            onChanged={async () => {
              setOpen(null);
              await load();
            }}
          />
        ))}
        {open && <PeriodView key={open.id} period={open} onClose={() => setOpen(null)} />}
      </Tiles>
    </Screen>
  );
}

function CompanyTile({
  company, onOpen, onChanged,
}: {
  company: CoverageCompany;
  onOpen: (period: PeriodSummary) => void;
  onChanged: () => void;
}) {
  const t = useTranslations("coverage");
  const tr = useTranslations("reader");
  return (
    <Tile span={12} title={company.name} unit={t("periodCount", { count: company.periods.length })}>
      <div className="grid gap-2" data-company-tile={company.name}>
        {company.periods.length === 0 ? (
          <p className="type-body text-[10.5px]">
            {t("noPeriods")} <Link href="/coverage/results" className="text-accent underline">{t("readResults")}</Link>
          </p>
        ) : (
          <table className="w-full border-collapse font-mono text-[11px]" aria-label={t("periodsLabel", { company: company.name })}>
            <thead>
              <tr className="text-muted">
                <th scope="col" className="border-b border-grid px-2 py-1 text-start font-normal">{t("colPeriod")}</th>
                <th scope="col" className="border-b border-grid px-2 py-1 text-start font-normal">{t("colScope")}</th>
                <th scope="col" className="border-b border-grid px-2 py-1 text-start font-normal">{t("colUnit")}</th>
                <th scope="col" className="border-b border-grid px-2 py-1 text-start font-normal">{t("colAccepted")}</th>
                <th scope="col" className="border-b border-grid px-2 py-1 text-end font-normal">
                  <span className="sr-only">{t("colActions")}</span>
                </th>
              </tr>
            </thead>
            <tbody>
              {company.periods.map((period) => (
                <tr key={period.id} data-period={`${period.period_end}|${period.scope}`}>
                  <td className="border-b border-grid px-2 py-1 text-ink">
                    {t("periodLabel", { day: day(period.period_end), months: String(period.months) })}
                  </td>
                  <td className="border-b border-grid px-2 py-1 text-ink">{tr(`scope_${period.scope}`)}</td>
                  <td className="border-b border-grid px-2 py-1 text-ink">
                    {tr(`unit_${period.unit}`)}
                    {period.currency ? ` · ${period.currency}` : ""}
                  </td>
                  <td className="border-b border-grid px-2 py-1 text-muted">{day(period.accepted_at)}</td>
                  <td className="border-b border-grid px-2 py-1 text-end">
                    <span className="inline-flex gap-1.5">
                      <button type="button" className="type-action border border-line px-2 py-0.5 text-ink" onClick={() => onOpen(period)}>
                        {t("show")}
                      </button>
                      <button
                        type="button"
                        className="type-action border border-line px-2 py-0.5 text-ink"
                        onClick={() => void downloadFile(workbookPath(period.id), workbookName(company.name, period))}
                      >
                        {t("download")}
                      </button>
                      <button
                        type="button"
                        className="type-action border border-line px-2 py-0.5 text-loss"
                        onClick={async () => {
                          if (await removePeriod(period.id)) onChanged();
                        }}
                      >
                        {t("removePeriod")}
                      </button>
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <div>
          <button
            type="button"
            className="type-action border border-line px-2 py-0.5 text-loss"
            onClick={async () => {
              if (await removeCompany(company.id)) onChanged();
            }}
          >
            {t("removeCompany")}
          </button>
        </div>
      </div>
    </Tile>
  );
}

/** A stored period, read-only: every figure with its mark, and the page it came from while the file is kept. */
function PeriodView({ period, onClose }: { period: StoredPeriod; onClose: () => void }) {
  const t = useTranslations("coverage");
  const tr = useTranslations("reader");
  const [page, setPage] = useState<number | null>(period.figures.tables[0]?.pages[0] ?? null);
  return (
    <>
      <Tile span={7} title={t("storedTitle", { company: period.company_name, day: day(period.period_end) })}>
        <div className="grid gap-4" data-stored-period={period.id}>
          <p className="type-body text-[10.5px]">
            {t("storedCounts", { figures: period.counts.figures, corrected: period.counts.corrected, accepted: period.counts.accepted })}
          </p>
          {period.figures.tables.map((table, ti) => {
            const digits = table.rows.some((row) => row.values.some((v) => v !== null && v !== undefined && !Number.isInteger(v))) ? 2 : 0;
            const title = `${tr(`statement_${table.statement}`)} · ${tr(`scope_${period.scope}`)}`;
            return (
              <div key={ti} className="min-w-0">
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
                            <button type="button" className="text-start" onClick={() => setPage(row.page ?? table.pages[0] ?? null)}>
                              {row.label}
                            </button>
                          </th>
                          {row.values.map((value, c) => {
                            const mark = row.marks?.[c] ?? null;
                            const note = mark
                              ? mark.do === "corrected"
                                ? t("markCorrected", { read: mark.read === null || mark.read === undefined ? t("blank") : fmtNumber(mark.read, digits) })
                                : t("markAccepted", { why: mark.amber ? tr(`amber_${mark.amber}`) : "" })
                              : undefined;
                            return (
                              <td
                                key={c}
                                className={`border-b border-grid px-2 py-1 text-end ${mark ? "text-gain" : "text-ink"}`}
                                title={note}
                                data-mark={mark?.do}
                              >
                                {value === null || value === undefined ? "" : fmtNumber(value, digits)}
                              </td>
                            );
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}
          <div>
            <button type="button" className="type-action border border-line px-2 py-1 text-ink" onClick={onClose}>
              {t("close")}
            </button>
          </div>
        </div>
      </Tile>
      <Tile span={5} title={t("pageTitle")}>
        <div className="sticky top-2">
          <PageView fileId={period.file_id} page={page} onPage={setPage} />
        </div>
      </Tile>
    </>
  );
}
