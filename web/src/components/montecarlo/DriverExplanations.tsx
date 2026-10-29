"use client";

import { useTranslations } from "next-intl";

import { DivergingBars } from "@/components/charts/Bars";
import { quote } from "@/components/ui/Explain";
import { Term } from "@/components/ui/Term";
import { Tile } from "@/components/ui/Tile";
import type { components } from "@/lib/api/schema";
import { fmtCount, fmtNumber, fmtPct, fmtRate, isNum } from "@/lib/format";
import { driverTerm } from "@/lib/glossary";
import { useEngineLabel } from "@/lib/i18n/useEngineText";

export type Explanations = components["schemas"]["DriverExplanations"];
type Case = components["schemas"]["ExplainedCase"];

/** A contribution in percentage points of IRR, always signed. */
const points = (v: number) => fmtNumber(v * 100, 2, true);

/**
 * Why the worst and the best simulated paths are where they are (PLAN.md 5.6): one tile a tail,
 * from the IRR at the mean assumptions, through each driver's contribution, to the tail's mean IRR.
 * The API's figures add up exactly; nothing is recomputed here.
 */
export function DriverExplanationTiles({ explanations }: { explanations: Explanations | null | undefined }) {
  // A run too slow to explain within the timeout answers null (api/limits.py), and an API from
  // before 5.6 (a deploy rolling out, a rollback) answers without the field at all
  if (!explanations?.cases) return null;
  return (
    <>
      {explanations.cases.map((c) => (
        <CaseTile key={c.case} explained={c} base={explanations.base_irr} share={explanations.share} />
      ))}
    </>
  );
}

/** i18n-keys: explain.tail_downside, explain.tail_upside */
function CaseTile({ explained, base, share }: { explained: Case; base: number | null | undefined; share: number }) {
  const t = useTranslations("explanations");
  const driverName = useEngineLabel("driver");
  const title = explained.case === "downside" ? t("titleDownside", { share: fmtPct(share * 100, 0) }) : t("titleUpside", { share: fmtPct(share * 100, 0) });
  // A contribution the API could not give (null) is left out, never drawn as a zero
  const rows = explained.contributions
    .flatMap((c) => (isNum(c.irr) ? [{ label: driverName(c.driver), value: c.irr, term: driverTerm(c.driver) }] : []))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const sampled = explained.tail_paths > explained.paths;
  const x = useTranslations("explain");
  // The figures as this tile shows them (PLAN.md 6.4): the two IRRs and the largest bar
  const shown = { share: fmtPct(share * 100, 0), base: fmtRate(base, 2), total: fmtRate(explained.irr, 2) };
  const largest = rows[0];
  const lines = [
    x.rich(explained.case === "downside" ? "tail_downside" : "tail_upside", { ...shown, f: quote }),
    ...(largest ? [x.rich("tail_largest", { driver: largest.label, points: points(largest.value), f: quote })] : []),
    x.rich("tail_adds"),
  ];

  return (
    <Tile span={6} term={explained.case === "downside" ? "tailDownside" : "tailUpside"} title={title} unit={t("unit")} explain={lines} data={{ "data-explained-case": explained.case }}>
      <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1">
        <dt className="type-input-label text-[9px] text-soft">
          <Term id="irrAtMean">{t("base")}</Term>
        </dt>
        <dd className="type-figure text-end text-[12px] text-ink" data-explained="base">
          {shown.base}
        </dd>
      </dl>
      <DivergingBars label={t("chart", { title })} format={points} rows={rows} />
      <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1 border-t border-line pt-1.5">
        <dt className="type-input-label text-[9px] text-soft">
          <Term id="tailMeanIrr">{t("total")}</Term>
        </dt>
        <dd className="type-figure text-end text-[12px] text-accent" data-explained="total">
          {shown.total}
        </dd>
      </dl>
      <p className="type-body text-[9px]">
        {t("note")}
        {sampled && ` ${t("sampled", { paths: fmtCount(explained.paths), tail: fmtCount(explained.tail_paths) })}`}
      </p>
    </Tile>
  );
}
