"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { useSettings } from "@/components/settings/SettingsProvider";
import { SecondaryButton } from "@/components/ui/Screen";
import { Tile } from "@/components/ui/Tile";
import { api } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";
import { apiInputs, type DealInputs } from "@/lib/deal/fields";
import { fmtCount, fmtNumber, fmtPct } from "@/lib/format";
import { regionName } from "@/lib/locale";

import { useMonteCarlo } from "./MonteCarloProvider";

export type Growth = components["schemas"]["GrowthResponse"];
type GrowthSettings = components["schemas"]["GrowthSettings"];

// The normal quantile of 0.9: a draw's central 80% is its mean +- Z80 spreads (ml/growth_calibrator.py)
const Z80 = 1.2816;

/** Answers by what they read, so the rail and the tile ask once between them. */
const pending = new Map<string, Promise<Growth | null>>();

function fetchGrowth(inputs: DealInputs): Promise<Growth | null> {
  const key = `${inputs.country}|${inputs.industry}|${inputs.hold}`;
  let found = pending.get(key);
  if (!found) {
    found = api
      .POST("/api/ml/growth", { body: { inputs: apiInputs(inputs) } })
      .then(({ data }) => data ?? null)
      .catch(() => null)
      .then((answer) => {
        if (!answer) pending.delete(key);
        return answer;
      });
    pending.set(key, found);
  }
  return found;
}

/**
 * The revenue growth range for the open deal's sector and region over its hold (PLAN.md 5.5), or
 * null before the deal has a country, while loading, or when the answer is for another deal.
 */
export function useGrowth(): Growth | null {
  const { inputs } = useDeal();
  const { country, industry, hold } = inputs;
  const [answer, setAnswer] = useState<Growth | null>(null);
  useEffect(() => {
    if (!country) return;
    let live = true;
    const id = setTimeout(() => {
      void fetchGrowth(inputs).then((a) => {
        if (live) setAnswer(a);
      });
    }, 300);
    return () => {
      live = false;
      clearTimeout(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps -- the range reads only these three
  }, [country, industry, hold]);
  const forThisDeal = !!answer && answer.country === country && answer.industry === industry && answer.hold === hold;
  return country && forThisDeal ? answer : null;
}

/** Whether the calibrated Settings are the ones in use, and the call that applies them. */
export function useCalibrate(settings: GrowthSettings | null | undefined) {
  const { overrides, replace } = useSettings();
  const { sim, clearSimEdits } = useMonteCarlo();
  const inUse = !!settings && sim.growth_mean === settings.mc_growth_mean && sim.growth_std === settings.mc_growth_std;
  const apply = () => {
    if (!settings) return;
    replace({ ...overrides, ...settings });
    // The rail's growth fields show the Settings again, now the calibrated ones
    clearSimEdits(["growth_mean", "growth_std"]);
  };
  return { inUse, apply };
}

/** i18n-keys: growth.why_*, library.region_*, library.sector_* */
function useWhy() {
  const t = useTranslations("growth");
  const lib = useTranslations("library");
  return (g: Growth) => {
    if (g.reason) return t(`why_${g.reason}`);
    const hidden = g.hidden ?? "not_enough_data";
    return t(`why_${hidden}`, {
      region: g.region ? lib(`region_${g.region}`) : "",
      min: String(g.min_companies),
      first: String(g.horizons[0]),
      last: String(g.horizons[g.horizons.length - 1]),
    });
  };
}

/** The rail's button: "Calibrate from sector and region", or why there is nothing to calibrate from. */
export function GrowthCalibrationRail() {
  const growth = useGrowth();
  const t = useTranslations("growth");
  const why = useWhy();
  const { inUse, apply } = useCalibrate(growth?.settings);
  if (!growth) return null;
  if (!growth.shown || !growth.settings) {
    return (
      <p className="type-body pt-1.5 text-[9px]" data-testid="growth-calibration-hidden">
        {why(growth)}
      </p>
    );
  }
  return inUse ? (
    <p className="type-body pt-1.5 text-[9px]" data-testid="growth-calibrated">
      {t("calibrated")}
    </p>
  ) : (
    <div className="pt-1.5">
      <SecondaryButton onClick={apply}>{t("calibrate")}</SecondaryButton>
    </div>
  );
}

/**
 * What the calibrated growth range rests on: the companies like the deal's in its region, how
 * their revenue grew against their economy, the economy's growth the range is centred on, and
 * what the model card found on years it had not seen.
 *
 * i18n-keys: growth.note_*, library.region_*, library.sector_*, starting.area_*
 */
export function GrowthCalibrationTile() {
  const growth = useGrowth();
  if (!growth) return null;
  return <Body g={growth} />;
}

function Body({ g }: { g: Growth }) {
  const t = useTranslations("growth");
  const lib = useTranslations("library");
  const area = useTranslations("starting");
  const why = useWhy();
  const { sim } = useMonteCarlo();
  const { inUse, apply } = useCalibrate(g.settings);
  const sector = g.group_sector ? lib(`sector_${g.group_sector}`) : t("allSectors");
  const region = g.region ? lib(`region_${g.region}`) : "";
  const anchorArea = g.anchor ? (area.has(`area_${g.anchor.area}`) ? area(`area_${g.anchor.area}`) : regionName(g.anchor.area)) : "";
  return (
    <Tile
      span={12}
      title={t("tile", { sector, region })}
      unit={t("unit")}
      action={g.shown && !inUse && <SecondaryButton onClick={apply}>{t("calibrate")}</SecondaryButton>}
    >
      <div className="grid gap-2" data-growth-status={g.shown ? "shown" : (g.hidden ?? g.reason ?? "hidden")}>
        {!g.shown && (
          <p className="type-body">
            <span className="type-alert text-[9px]">{t("notShown")}</span> {why(g)}
          </p>
        )}
        {g.range && g.settings && (
          <table className="w-full border-collapse font-mono text-[11px]" aria-label={t("rangeLabel")}>
            <thead>
              <tr className="type-input-label">
                <th scope="col" className="px-2 py-1 text-start font-normal">{t("colFigure")}</th>
                <th scope="col" className="px-2 py-1 text-end font-normal">{t("colCalibrated")}</th>
                <th scope="col" className="px-2 py-1 text-end font-normal">{t("colInUse")}</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="px-2 py-1">{t("rowRange", { hold: String(g.hold) })}</td>
                <td className="px-2 py-1 text-end text-bright" data-testid="growth-range">
                  {t("range", { low: fmtPct(g.range.low, 1), high: fmtPct(g.range.high, 1) })}
                </td>
                <td className="px-2 py-1 text-end text-muted">
                  {t("range", { low: fmtPct(sim.growth_mean - Z80 * sim.growth_std, 1), high: fmtPct(sim.growth_mean + Z80 * sim.growth_std, 1) })}
                </td>
              </tr>
              <tr>
                <td className="px-2 py-1">{t("rowMean")}</td>
                <td className="px-2 py-1 text-end text-ink" data-testid="growth-mean">{fmtPct(g.settings.mc_growth_mean, 2)}</td>
                <td className="px-2 py-1 text-end text-muted">{fmtPct(sim.growth_mean, 2)}</td>
              </tr>
              <tr>
                <td className="px-2 py-1">{t("rowStd")}</td>
                <td className="px-2 py-1 text-end text-ink" data-testid="growth-std">{fmtPct(g.settings.mc_growth_std, 2)}</td>
                <td className="px-2 py-1 text-end text-muted">{fmtPct(sim.growth_std, 2)}</td>
              </tr>
            </tbody>
          </table>
        )}
        {g.observed && (
          <p className="font-mono text-[10.5px] text-ink" data-provenance="growth">
            {t("observed", {
              companies: g.observed.companies,
              shown: fmtCount(g.observed.companies),
              sector,
              region,
              hold: String(g.hold),
              first: String(g.observed.first),
              last: String(g.observed.last),
              low: fmtNumber(g.observed.low, 1, true),
              median: fmtNumber(g.observed.median, 1, true),
              high: fmtNumber(g.observed.high, 1, true),
            })}
          </p>
        )}
        {g.anchor && (
          <p className="font-mono text-[10.5px] text-muted">
            {t("anchor", { growth: fmtPct(g.anchor.value, 2), area: anchorArea, year: g.anchor.as_of ?? "" })}
          </p>
        )}
        {g.card.verdict !== "not_enough_data" && g.card.model != null && g.card.baseline != null && (
          <p className="font-mono text-[10.5px] text-muted" data-testid="growth-card">
            {t("card", {
              region,
              cases: g.card.cases,
              shown: fmtCount(g.card.cases),
              model: fmtNumber(g.card.model, 1),
              baseline: fmtNumber(g.card.baseline, 1),
            })}
          </p>
        )}
        <ul className="type-body grid gap-0.5 text-[9px]">
          {g.notes.map((n) => (
            <li key={n}>{t(`note_${n}`, { floor: fmtCount(g.floor_usd_m) })}</li>
          ))}
        </ul>
        <p className="type-body text-[9px]">
          {t("sourcesNote")}{" "}
          {Object.entries(g.sources).map(([id, s], i) => (
            <span key={id}>
              {i > 0 && " · "}
              <a href={s.url} target="_blank" rel="noopener noreferrer" className="text-accent underline">
                {s.publisher}
              </a>
            </span>
          ))}
        </p>
      </div>
    </Tile>
  );
}
