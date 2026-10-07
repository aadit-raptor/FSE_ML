"use client";

import { useTranslations } from "next-intl";
import Link from "next/link";
import { useEffect, useState } from "react";

import { EmptyState, LoadingTiles, Notice, RailGroup, Screen, Switch } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { api, type Schemas } from "@/lib/api/client";
import { fmtCount, fmtNumber, fmtPct } from "@/lib/format";
import { formatDateTime } from "@/lib/locale";
import { stepHref } from "@/lib/nav";

type Report = Schemas["ValidationReportOut"];
type Check = Schemas["ValidationCheck"];
type Cell = Schemas["ValidationCell"];
type Bucket = Schemas["ValidationBucket"];
type RangeStats = Schemas["RangeStats"];
type ProbabilityStats = Schemas["ProbabilityStats"];
type SampleId = "out_of_time" | "in_sample";

const METHODOLOGY = "https://github.com/aadit-raptor/FSE_ML/blob/main/docs/methodology.md#model-validation";
const SAMPLES: SampleId[] = ["out_of_time", "in_sample"];

const isRange = (s: Cell["stats"]): s is RangeStats => !!s && "levels" in s;

/**
 * Backtest -> Validation (PLAN.md 4.6): how often the model's ranges and default risk came true,
 * overall and by region, sector, size and era. The nightly validation-report task writes the report;
 * this reads the newest. The headline is out of time (outcomes newer than the prediction's data);
 * in-sample cases are shown apart, on request.
 */
export function ValidationStep() {
  const t = useTranslations("modelValidation");
  const [report, setReport] = useState<Report | null | undefined>(undefined);
  const [failed, setFailed] = useState(false);
  const [sample, setSample] = useState<SampleId>("out_of_time");
  useEffect(() => {
    let live = true;
    api
      .GET("/api/validation/report")
      .then(({ data }) => {
        if (!live) return;
        if (data) setReport(data.report ?? null);
        else setFailed(true);
      })
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, []);
  /** i18n-keys: modelValidation.sample_*, modelValidation.sampleNote_* */
  const rail = (
    <>
      <RailGroup title={t("groupSample")}>
        <div className="grid gap-px bg-line" role="radiogroup" aria-label={t("groupSample")}>
          {SAMPLES.map((s) => (
            <button
              key={s}
              type="button"
              role="radio"
              aria-checked={sample === s}
              onClick={() => setSample(s)}
              className={`grid gap-0.5 px-2.5 py-1.5 text-start ${sample === s ? "bg-raised shadow-[inset_2px_0_0_var(--color-accent)]" : "bg-panel hover:bg-raised"}`}
            >
              <span className={sample === s ? "type-control-value" : "type-control"}>{t(`sample_${s}`)}</span>
            </button>
          ))}
        </div>
        <p className="type-body text-[9px]">{t(`sampleNote_${sample}`)}</p>
      </RailGroup>
      <RailGroup title={t("groupAbout")}>
        <p className="type-body text-[9px]">{t("about")}</p>
        <p className="type-body text-[9px]">{t("anonymity")}</p>
        <a href={METHODOLOGY} target="_blank" rel="noreferrer" className="type-body text-[9px] text-accent underline">
          {t("methodology")}
        </a>
        <Link href={stepHref("backtest", "actuals")} className="type-body text-[9px] text-accent underline">
          {t("contribute")}
        </Link>
      </RailGroup>
    </>
  );
  return (
    <Screen rail={rail}>
      {failed ? (
        <Notice tone="loss" title={t("unreachableTitle")} role="alert">
          {t("unreachable")}
        </Notice>
      ) : report === undefined ? (
        <LoadingTiles />
      ) : report === null ? (
        <EmptyState title={t("noReportTitle")}>{t("noReport")}</EmptyState>
      ) : (
        <ReportTiles report={report} sample={sample} />
      )}
    </Screen>
  );
}

function ReportTiles({ report, sample }: { report: Report; sample: SampleId }) {
  const t = useTranslations("modelValidation");
  return (
    <Tiles>
      <Kpi title={t("kpiGenerated")} value={formatDateTime(new Date(report.generated_at))} sub={t("engineVersion", { version: report.engine_version })} />
      <Kpi title={t("kpiLibrary")} value={fmtCount(report.cases.library)} sub={report.library_included ? t("libraryOn") : t("libraryOff")} />
      <Kpi
        title={t("kpiContributed")}
        value={report.cases.contributed_deals == null ? t("fewerThan", { count: report.rules.min_contributed }) : fmtCount(report.cases.contributed_deals)}
        sub={t("contributedSub")}
      />
      <Kpi title={t("kpiFitUntil")} value={String(report.fit_until.default ?? "")} sub={t("fitUntilSub")} />
      {report.checks.map((check) => (
        <CheckTile key={check.id} check={check} sample={sample} report={report} />
      ))}
    </Tiles>
  );
}

/** i18n-keys: modelValidation.check_*, modelValidation.claim_* */
function CheckTile({ check, sample, report }: { check: Check; sample: SampleId; report: Report }) {
  const t = useTranslations("modelValidation");
  const data = check.samples[sample];
  return (
    <Tile span={12} title={t(`check_${check.id}`)} aside={<span className="chip text-muted">{t(`sample_${sample}`)}</span>}>
      <div className="grid gap-2" data-check={check.id} data-sample={sample}>
        <p className="type-body text-[9.5px]">{t(`claim_${check.id}`)}</p>
        <Overall cell={data.overall} minCases={report.rules.min_cases} minContributed={report.rules.min_contributed} />
        <div className="grid grid-cols-2 gap-3">
          {report.dimensions.map((d) => (
            <SplitTable key={d} check={check.id} dimension={d} buckets={data.splits[d] ?? []} />
          ))}
        </div>
      </div>
    </Tile>
  );
}

function Overall({ cell, minCases, minContributed }: { cell: Cell; minCases: number; minContributed: number }) {
  const t = useTranslations("modelValidation");
  if (cell.suppressed) return <p className="font-mono text-[10.5px] text-attention" data-overall="suppressed">{t("suppressedOverall", { count: minContributed })}</p>;
  if (!cell.enough || !cell.stats) {
    return (
      <p className="font-mono text-[10.5px] text-attention" data-overall="not-enough">
        {t("notEnoughOverall", { n: cell.n ?? 0, count: minCases })}
      </p>
    );
  }
  const s = cell.stats;
  return (
    <p className="font-mono text-[11px] text-ink" data-overall="stats">
      {isRange(s)
        ? t("overallRange", {
            n: cell.n ?? 0,
            levels: s.levels.map((l) => t("levelInside", { claimed: fmtPct(l.claimed_pct, 0), inside: fmtPct(l.inside_pct) })).join(", "),
            bias: fmtNumber(s.bias_pp, 1, true),
            verdict: s.consistent ? t("consistent") : t("inconsistent"),
          })
        : t("overallProbability", {
            n: cell.n ?? 0,
            predicted: fmtPct((s as ProbabilityStats).predicted_pct),
            observed: fmtPct((s as ProbabilityStats).observed_pct),
            bias: fmtNumber((s as ProbabilityStats).bias_pp, 1, true),
            verdict: (s as ProbabilityStats).consistent ? t("consistent") : t("inconsistent"),
          })}
    </p>
  );
}

/** i18n-keys: library.region_*, library.sector_*, library.bucket_*, library.dimension_* */
function useBucketName() {
  const lib = useTranslations("library");
  const t = useTranslations("modelValidation");
  return (dimension: string, bucket: string) => {
    if (bucket === "unknown") return t("unknown");
    const key = dimension === "region" ? `region_${bucket}` : dimension === "sector" ? `sector_${bucket}` : `bucket_${bucket}`;
    return lib.has(key) ? lib(key) : bucket;
  };
}

function SplitTable({ check, dimension, buckets }: { check: Check["id"]; dimension: string; buckets: Bucket[] }) {
  const t = useTranslations("modelValidation");
  const lib = useTranslations("library");
  const bucketName = useBucketName();
  const range = check === "irr_range";
  const heads = range ? [t("colInside50"), t("colInside80"), t("colInside90"), t("colBias")] : [t("colPredicted"), t("colObserved"), t("colBias"), t("colBrier")];
  const title = lib(`dimension_${dimension}`);
  return (
    <table className="w-full self-start border-collapse font-mono text-[11px]" aria-label={t("splitTable", { check: t(`check_${check}`), dimension: title })} data-dimension={dimension}>
      <thead>
        <tr className="type-input-label">
          <th scope="col" className="px-1 py-1 text-start font-normal">{title}</th>
          <th scope="col" className="px-1 py-1 text-end font-normal">{t("colCases")}</th>
          {heads.map((h) => (
            <th key={h} scope="col" className="px-1 py-1 text-end font-normal">
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {buckets.map((b) => (
          <tr key={b.bucket} data-bucket={`${dimension}:${b.bucket}`}>
            <th scope="row" className="border-b border-grid px-1 py-0.5 text-start text-[10px] font-normal text-soft">
              {bucketName(dimension, b.bucket)}
            </th>
            <td className={`border-b border-grid px-1 py-0.5 text-end ${b.n ? "text-ink" : "text-dim"}`}>{b.suppressed ? t("hidden") : fmtCount(b.n)}</td>
            <BucketFigures bucket={b} range={range} />
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function BucketFigures({ bucket, range }: { bucket: Bucket; range: boolean }) {
  const t = useTranslations("modelValidation");
  const cls = "border-b border-grid px-1 py-0.5 text-end";
  const s = bucket.stats;
  if (!s) {
    return (
      <td colSpan={4} className={`${cls} text-dim`}>
        {bucket.suppressed ? t("hiddenNote") : t("notEnough")}
      </td>
    );
  }
  const tone = (ok: boolean | null | undefined) => (ok === false ? "text-attention" : "text-ink");
  if (range && isRange(s)) {
    return (
      <>
        {s.levels.map((l) => (
          <td key={l.claimed_pct} className={`${cls} ${tone(l.consistent)}`}>
            {fmtPct(l.inside_pct, 0)}
          </td>
        ))}
        <td className={`${cls} text-ink`}>{fmtNumber(s.bias_pp, 1, true)}</td>
      </>
    );
  }
  const p = s as ProbabilityStats;
  return (
    <>
      <td className={`${cls} text-ink`}>{fmtPct(p.predicted_pct)}</td>
      <td className={`${cls} ${tone(p.consistent)}`}>{fmtPct(p.observed_pct)}</td>
      <td className={`${cls} text-ink`}>{fmtNumber(p.bias_pp, 1, true)}</td>
      <td className={`${cls} text-ink`}>{fmtNumber(p.brier, 3)}</td>
    </>
  );
}

/**
 * The owner's choice, on Backtest -> Plan and actuals, to let a saved deal's plan-vs-actual result
 * count, anonymised, in the validation report (PLAN.md 4.6). Off until switched on. Mounted with the
 * deal's id as its key, so another deal starts from a fresh read.
 */
export function ValidationConsent({ dealId }: { dealId: string }) {
  const t = useTranslations("modelValidation");
  const [optIn, setOptIn] = useState<boolean | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    let live = true;
    api
      .GET("/api/deals/{deal_id}/validation", { params: { path: { deal_id: dealId } } })
      .then(({ data }) => live && (data ? setOptIn(data.opt_in) : setFailed(true)))
      .catch(() => live && setFailed(true));
    return () => {
      live = false;
    };
  }, [dealId]);
  const change = async (next: boolean) => {
    setFailed(false);
    setOptIn(next);
    try {
      const { data } = await api.PUT("/api/deals/{deal_id}/validation", { params: { path: { deal_id: dealId } }, body: { opt_in: next } });
      if (data) setOptIn(data.opt_in);
      else {
        setOptIn(!next);
        setFailed(true);
      }
    } catch {
      setOptIn(!next);
      setFailed(true);
    }
  };
  return (
    <RailGroup title={t("groupConsent")}>
      {optIn !== null && <Switch checked={optIn} onChange={(v) => void change(v)} label={t("consentLabel")} />}
      <p className="type-body text-[9px]">{t("consentNote")}</p>
      {failed && (
        <p className="type-body text-[9px] text-loss" role="alert">
          {t("consentFailed")}
        </p>
      )}
      <Link href={stepHref("backtest", "validation")} className="type-body text-[9px] text-accent underline">
        {t("toReport")}
      </Link>
    </RailGroup>
  );
}
