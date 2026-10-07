"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { LoadingTiles, Notice, RailGroup, Screen } from "@/components/ui/Screen";
import { Tile, Tiles } from "@/components/ui/Tile";
import { fmtCount } from "@/lib/format";
import { type Coverage, fetchCoverage } from "@/lib/library";

import { LibraryGate, LibrarySwitch } from "./LibraryGate";
import { useLibrary } from "./LibraryProvider";

const DIMENSIONS = ["region", "size", "sector", "era", "outcome"] as const;

/**
 * Library -> Coverage (PLAN.md 4.5): how many deals the library holds in each region, size, sector,
 * era and outcome, empty buckets included, and what the base rates span. Administrators switch the
 * library here; the switch stays reachable while it is off.
 */
export function CoverageStep() {
  const { state } = useLibrary();
  const t = useTranslations("library");
  const rail = (
    <RailGroup title={t("groupAdmin")}>
      {state?.can_switch ? <LibrarySwitch /> : <p className="type-body text-[9px]">{state?.locked_off ? t("offLocked") : t("adminOnly")}</p>}
    </RailGroup>
  );
  return (
    <Screen rail={rail}>
      <LibraryGate switchElsewhere>
        <CoverageResults />
      </LibraryGate>
    </Screen>
  );
}

function CoverageResults() {
  const t = useTranslations("library");
  const [coverage, setCoverage] = useState<Coverage | null | undefined>(undefined);
  useEffect(() => {
    let live = true;
    void fetchCoverage().then((c) => live && setCoverage(c));
    return () => {
      live = false;
    };
  }, []);
  if (coverage === undefined) return <LoadingTiles />;
  if (!coverage?.enabled) {
    return (
      <Notice tone="loss" title={t("unreachableTitle")} role="alert">
        {t("unreachable")}
      </Notice>
    );
  }
  return (
    <Tiles>
      {coverage.collections.map((c) => (
        <Collection key={c.id} collection={c} />
      ))}
      <BaseRateCoverage coverage={coverage} />
    </Tiles>
  );
}

/** i18n-keys: library.collection_*, library.dimension_*, library.bucket_*, library.region_*, library.empty_*, library.sector_* */
function useBucketName() {
  const t = useTranslations("library");
  return (dimension: string, bucket: string) => {
    // The examples' sectors are their own words; the reference transactions' are GICS sectors
    const key = dimension === "region" ? `region_${bucket}` : dimension === "sector" ? `sector_${bucket}` : `bucket_${bucket}`;
    return t.has(key) ? t(key) : bucket;
  };
}

function Collection({ collection }: { collection: Coverage["collections"][number] }) {
  const t = useTranslations("library");
  const bucketName = useBucketName();
  const title = t(`collection_${collection.id}`, { count: collection.count, shown: fmtCount(collection.count) });
  return (
    <Tile
      span={12}
      title={title}
      aside={<span className={`chip ${collection.sourced ? "text-gain" : "text-attention"}`}>{collection.sourced ? t("sourced") : t("unsourced")}</span>}
    >
      {collection.count === 0 && <p className="type-body text-[9.5px]">{t(`empty_${collection.id}`)}</p>}
      {!!collection.awaiting_review && (
        <p className="font-mono text-[10.5px] text-attention" data-testid={`awaiting-${collection.id}`}>
          {t("awaitingReview", { count: collection.awaiting_review, shown: fmtCount(collection.awaiting_review) })}
        </p>
      )}
      <div className="grid grid-cols-5 gap-3" data-collection={collection.id}>
        {DIMENSIONS.map((d) => (
          <table key={d} className="w-full self-start border-collapse font-mono text-[11px]" aria-label={t("dimensionTable", { collection: title, dimension: t(`dimension_${d}`) })}>
            <thead>
              <tr>
                <th scope="col" colSpan={2} className="type-input-label px-1 py-1 text-start font-normal">
                  {t(`dimension_${d}`)}
                </th>
              </tr>
            </thead>
            <tbody>
              {collection.dimensions[d]?.length ? (
                collection.dimensions[d]!.map((b) => (
                  <tr key={b.bucket} data-bucket={`${d}:${b.bucket}`}>
                    <th scope="row" className="border-b border-grid px-1 py-0.5 text-start text-[10px] font-normal text-soft">
                      {bucketName(d, b.bucket)}
                    </th>
                    <td className={`border-b border-grid px-1 py-0.5 text-end ${b.count ? "text-ink" : "text-dim"}`}>{fmtCount(b.count)}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={2} className="px-1 py-0.5 text-dim">
                    {t("none")}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        ))}
      </div>
    </Tile>
  );
}

/** i18n-keys: library.table_* */
function BaseRateCoverage({ coverage }: { coverage: Coverage }) {
  const t = useTranslations("library");
  const title = t("baseRateCoverageTitle");
  return (
    <Tile span={12} title={title}>
      <table className="w-full border-collapse font-mono text-[11px]" aria-label={title}>
        <thead>
          <tr className="type-input-label">
            <th scope="col" className="px-2 py-1 text-start font-normal">{t("colTable")}</th>
            <th scope="col" className="px-2 py-1 text-end font-normal">{t("colRegions")}</th>
            <th scope="col" className="px-2 py-1 text-end font-normal">{t("colBands")}</th>
            <th scope="col" className="px-2 py-1 text-end font-normal">{t("colYears")}</th>
            <th scope="col" className="px-2 py-1 text-end font-normal">{t("colObservations")}</th>
          </tr>
        </thead>
        <tbody>
          {coverage.base_rates.map((r) => (
            <tr key={r.table} data-table={r.table}>
              <th scope="row" className="type-input-label border-b border-grid px-2 py-1 text-start text-[9px] font-normal text-soft">
                {t(`table_${r.table}`)}
              </th>
              <td className="border-b border-grid px-2 py-1 text-end text-ink">{fmtCount(r.regions)}</td>
              <td className="border-b border-grid px-2 py-1 text-end text-ink">{fmtCount(r.bands)}</td>
              <td className="border-b border-grid px-2 py-1 text-end text-ink">{t("yearSpan", { first: String(r.first_year), last: String(r.last_year) })}</td>
              <td className="border-b border-grid px-2 py-1 text-end text-ink">{r.observations == null ? t("none") : fmtCount(r.observations)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </Tile>
  );
}
