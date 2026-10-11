"use client";

import { useTranslations } from "next-intl";

import { DivergingBars } from "@/components/charts/Bars";
import { Tile } from "@/components/ui/Tile";
import type { components } from "@/lib/api/schema";
import { fmtCount, fmtNumber, fmtPct, fmtRate, isNum } from "@/lib/format";
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

function CaseTile({ explained, base, share }: { explained: Case; base: number | null | undefined; share: number }) {
  const t = useTranslations("explanations");
  const driverName = useEngineLabel("driver");
  const title = explained.case === "downside" ? t("titleDownside", { share: fmtPct(share * 100, 0) }) : t("titleUpside", { share: fmtPct(share * 100, 0) });
  // A contribution the API could not give (null) is left out, never drawn as a zero
  const rows = explained.contributions
    .flatMap((c) => (isNum(c.irr) ? [{ label: driverName(c.driver), value: c.irr }] : []))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const sampled = explained.tail_paths > explained.paths;

  return (
    <Tile span={6} title={title} unit={t("unit")}>
      <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1">
        <dt className="type-input-label text-[9px] text-soft">{t("base")}</dt>
        <dd className="type-figure text-end text-[12px] text-ink" data-explained="base">
          {fmtRate(base, 2)}
        </dd>
      </dl>
      <DivergingBars label={t("chart", { title })} format={points} rows={rows} />
      <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-3 gap-y-1 border-t border-line pt-1.5">
        <dt className="type-input-label text-[9px] text-soft">{t("total")}</dt>
        <dd className="type-figure text-end text-[12px] text-accent" data-explained="total">
          {fmtRate(explained.irr, 2)}
        </dd>
      </dl>
      <p className="type-body text-[9px]">
        {t("note")}
        {sampled && ` ${t("sampled", { paths: fmtCount(explained.paths), tail: fmtCount(explained.tail_paths) })}`}
      </p>
    </Tile>
  );
}
