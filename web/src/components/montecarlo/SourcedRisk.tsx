"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { useSettings } from "@/components/settings/SettingsProvider";
import { RailGroup, SecondaryButton } from "@/components/ui/Screen";
import { Tile } from "@/components/ui/Tile";
import { fetchRisk, type Risk, type RiskFigure, type RiskScenario, type RiskSetting, riskKind } from "@/lib/benchmarks";
import { fmtCount, fmtMultiple, fmtNumber, fmtPct, isNum } from "@/lib/format";
import { useProvenance } from "@/lib/i18n/useProvenance";
import { regionName } from "@/lib/locale";

import { useCalibrate, useGrowth } from "./GrowthCalibration";
import { type SimKey, useMonteCarlo } from "./MonteCarloProvider";

/** The rail fields the sourced Settings fill: a rail edit of one of them overrides the sourced figure. */
const RAIL_SETTING: Partial<Record<SimKey, RiskSetting>> = {
  growth_mean: "mc_growth_mean",
  growth_std: "mc_growth_std",
  exit_mean: "mc_exit_mean",
  exit_std: "mc_exit_std",
  rate_mean: "mc_rate_mean",
  rate_std: "mc_rate_std",
  gm_mean: "mc_gm_mean",
  gm_std: "mc_gm_std",
};
const RAIL_KEYS = Object.keys(RAIL_SETTING) as SimKey[];

const ALL = "all";
const SCENARIO_ORDER = ["recession", "stagflation", "bull"] as const;

/**
 * The Monte Carlo figures sourced for the open deal's country, industry and currency (PLAN.md 4.4),
 * or null: before the deal has a country, while loading, or when the answer is for another deal.
 */
export function useSourcedRisk(): { risk: Risk | null; country: string } {
  const { inputs } = useDeal();
  const { country, industry, currency } = inputs;
  const ind = industry || ALL;
  const [risk, setRisk] = useState<Risk | null>(null);
  useEffect(() => {
    if (!country) return;
    let live = true;
    void fetchRisk(country, ind, currency).then((r) => {
      if (live) setRisk(r);
    });
    return () => {
      live = false;
    };
  }, [country, ind, currency]);
  const forThisDeal = !!risk && risk.country === country && risk.industry === ind && risk.currency === currency;
  return { risk: country && forThisDeal ? risk : null, country };
}

/**
 * The sourced Settings that differ from the ones in use, and the call that applies them all. Growth
 * calibrated from the deal's sector and region (PLAN.md 5.5) stands in for the economy-wide growth
 * figures while it is in use: neither counts as differing, and "Use sourced figures" keeps it.
 */
function useApply(risk: Risk | null) {
  const { effective, overrides, replace } = useSettings();
  const { sim, clearSimEdits } = useMonteCarlo();
  const growth = useGrowth();
  const { inUse: calibrated } = useCalibrate(growth?.settings);
  const kept = new Set<RiskSetting>(calibrated ? ["mc_growth_mean", "mc_growth_std"] : []);
  // A Setting differs when Settings hold another value, or the rail overrides it
  const railValue = (k: RiskSetting) => {
    const field = RAIL_KEYS.find((f) => RAIL_SETTING[f] === k);
    return field ? sim[field] : effective[k];
  };
  const differing = risk
    ? (Object.keys(risk.settings) as RiskSetting[]).filter(
        (k) => !kept.has(k) && (effective[k] !== risk.settings[k] || railValue(k) !== risk.settings[k]),
      )
    : [];
  const apply = () => {
    if (!risk) return;
    const sourced = Object.fromEntries(Object.entries(risk.settings).filter(([k]) => !kept.has(k as RiskSetting)));
    replace({ ...overrides, ...sourced });
    // The rail's ranges show the Settings again, now the sourced ones; paths, hurdle and seed stay as edited
    clearSimEdits(RAIL_KEYS);
  };
  return { differing, apply, effective, railValue };
}

/** i18n-keys: starting.area_* */
function useAreaName() {
  const s = useTranslations("starting");
  return (area: string) => (s.has(`area_${area}`) ? s(`area_${area}`) : /^[A-Z]{2}$/.test(area) ? regionName(area) : area);
}

/**
 * The rail's word on where the simulation's assumptions come from: illustrative until the deal has
 * a country and its sourced figures are applied, with the button that applies them.
 */
export function SourcedRiskRail() {
  const { risk, country } = useSourcedRisk();
  const { differing, apply } = useApply(risk);
  const t = useTranslations("risk");
  const m = useTranslations("montecarlo");
  const provenance = useProvenance();
  const areaName = useAreaName();
  const sourced = !!risk && Object.keys(risk.settings).length > 0 && differing.length === 0;
  return (
    <RailGroup title={t("group")}>
      {sourced ? (
        <p className="type-body text-[9px]" data-testid="risk-sourced">
          {t("sourcedFor", { country: areaName(risk.country), industry: risk.industry_name ?? t("allIndustries") })}
        </p>
      ) : (
        <div className="grid gap-1" role="note">
          <p className="type-alert text-[9px]">{provenance.illustrative}</p>
          <p className="type-body text-[9px]">{country ? t("illustrativeWithCountry") : m("illustrativeDetail")}</p>
        </div>
      )}
      {risk && differing.length > 0 && (
        <div className="pt-1.5">
          <SecondaryButton onClick={apply}>{t("useAll", { count: differing.length })}</SecondaryButton>
        </div>
      )}
    </RailGroup>
  );
}

/** i18n-keys: risk.label_*, risk.source_*, risk.kind_*, risk.reason_*, risk.note_*, risk.rule_*, risk.scenario_* */
function useFigureText(risk: Risk) {
  const t = useTranslations("risk");
  const areaName = useAreaName();
  const value = (key: RiskSetting, v: unknown) => {
    if (typeof v !== "number" || !isNum(v)) return t("none");
    const kind = riskKind(key);
    if (kind === "multiple") return fmtMultiple(v, 2);
    if (kind === "multiplier") return t("times", { value: fmtNumber(v, 3) });
    if (kind === "points") return t("points", { value: fmtNumber(v, 2, true) });
    if (kind === "correlation") return fmtNumber(v, 2, true);
    return fmtPct(v, 2);
  };
  const period = (p: unknown) => {
    if (typeof p !== "string" || !p.includes("-")) return "";
    const [first, last] = p.split("-");
    return first === last ? first : t("period", { first, last });
  };
  const source = (f: RiskFigure) =>
    [
      t(`source_${f.source.replace(/\+/g, "_")}`),
      areaName(f.area),
      f.sample == null || !f.sample_kind ? "" : t(`kind_${f.sample_kind}`, { count: f.sample, shown: fmtCount(f.sample) }),
      period(f.detail.period),
    ]
      .filter(Boolean)
      .join(" · ");
  const skipped = (s: RiskFigure["skipped"]) =>
    s.map((x) => t(`reason_${x.reason}`, { area: areaName(x.area), sample: fmtCount(x.sample ?? 0), min: String(risk.min_years) })).join(" ");
  const periods = (sc: RiskScenario) =>
    sc.periods.map((p) => (p.start === p.end ? String(p.start) : t("period", { first: String(p.start), last: String(p.end) }))).join(", ");
  return { t, areaName, value, source, skipped, periods, period };
}

/**
 * Every sourced Monte Carlo figure beside the one in use, with its source, sample and years, and
 * the historical years each scenario preset is built from (PLAN.md 4.4).
 */
export function SourcedRiskTile() {
  const { risk } = useSourcedRisk();
  if (!risk) return null;
  return <RiskTable risk={risk} />;
}

function RiskTable({ risk }: { risk: Risk }) {
  const { differing, apply, railValue } = useApply(risk);
  const { t, areaName, value, source, skipped, periods, period } = useFigureText(risk);
  const title = t("tileTitle", { country: areaName(risk.country), industry: risk.industry_name ?? t("allIndustries") });
  const corr = risk.correlation;
  return (
    <Tile
      span={12}
      title={title}
      action={differing.length > 0 && <SecondaryButton onClick={apply}>{t("useAll", { count: differing.length })}</SecondaryButton>}
    >
      <ul className="grid gap-1 font-mono text-[11px]" aria-label={t("scenariosLabel")}>
        {SCENARIO_ORDER.filter((id) => risk.scenarios[id]).map((id) => {
          const sc = risk.scenarios[id]!;
          return (
            <li key={id} data-scenario={id}>
              <span className="type-input-label">{t(`scenario_${id}`)}</span>{" "}
              <span className="text-muted">
                {t(`rule_${sc.rule}`, { area: areaName(sc.area), window: period(sc.window), count: sc.years.length, of: sc.of_years })}
              </span>{" "}
              <span className="text-ink" data-testid={`periods-${id}`}>
                {periods(sc)}
              </span>
            </li>
          );
        })}
      </ul>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse font-mono text-[11px]" aria-label={title}>
          <thead>
            <tr className="type-input-label">
              <th scope="col" className="px-2 py-1 text-start font-normal">{t("colFigure")}</th>
              <th scope="col" className="px-2 py-1 text-end font-normal">{t("colSourced")}</th>
              <th scope="col" className="px-2 py-1 text-end font-normal">{t("colInUse")}</th>
              <th scope="col" className="px-2 py-1 text-start font-normal">{t("colSource")}</th>
            </tr>
          </thead>
          <tbody>
            {risk.figures.map((f) => {
              const now = railValue(f.field);
              return (
                <tr key={f.field} className="border-t border-line align-top" data-setting={f.field}>
                  <th scope="row" className="type-input-label px-2 py-1 text-start font-normal">{t(`label_${f.field}`)}</th>
                  <td className="px-2 py-1 text-end text-ink" data-sourced={f.field}>{value(f.field, f.value)}</td>
                  <td className={`px-2 py-1 text-end ${now === f.value ? "text-dim" : "text-attention"}`}>{value(f.field, now)}</td>
                  <td className="px-2 py-1 text-start text-muted">
                    {f.url ? (
                      <a href={f.url} target="_blank" rel="noopener noreferrer" className="text-accent underline">
                        {source(f)}
                      </a>
                    ) : (
                      source(f)
                    )}
                    {f.skipped.length > 0 && <span className="block text-attention">{skipped(f.skipped)}</span>}
                  </td>
                </tr>
              );
            })}
            {risk.missing.map((m) => (
              <tr key={m.field} className="border-t border-line align-top" data-setting={m.field}>
                <th scope="row" className="type-input-label px-2 py-1 text-start font-normal">{t(`label_${m.field}`)}</th>
                <td colSpan={3} className="px-2 py-1 text-start text-attention">
                  {t("missing")} {skipped(m.skipped)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ul className="grid gap-0.5 font-mono text-[10px] text-dim">
        <li>
          {t("correlationNote", {
            group: areaName(corr.group),
            rows: fmtCount(corr.rows),
            period: period(corr.period),
            shrink: fmtNumber(corr.shrink, 2),
          })}
        </li>
        {risk.notes.map((n) => (
          <li key={n}>{t(`note_${n}`)}</li>
        ))}
      </ul>
    </Tile>
  );
}
