"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { SecondaryButton } from "@/components/ui/Screen";
import { Tile } from "@/components/ui/Tile";
import { api } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";
import { fmtCount, fmtNumber, fmtPct, fmtRate } from "@/lib/format";
import { formatDateTime, regionName } from "@/lib/locale";

import { SCENARIOS, type Scenario, useMonteCarlo } from "./MonteCarloProvider";

export type Regime = components["schemas"]["RegimeResponse"];

/** Answers by country, so a visit asks once. */
const pending = new Map<string, Promise<Regime | null>>();

function fetchRegime(country: string): Promise<Regime | null> {
  let found = pending.get(country);
  if (!found) {
    found = api
      .POST("/api/ml/macro-regime", { body: { country } })
      .then(({ data }) => data ?? null)
      .catch(() => null)
      .then((answer) => {
        if (!answer) pending.delete(country);
        return answer;
      });
    pending.set(country, found);
  }
  return found;
}

/** The economic regime of the open deal's region (PLAN.md 5.7), or null while loading. */
export function useRegime(): Regime | null {
  const { inputs } = useDeal();
  const country = inputs.country ?? "";
  const [answer, setAnswer] = useState<Regime | null>(null);
  useEffect(() => {
    let live = true;
    void fetchRegime(country).then((a) => {
      if (live) setAnswer(a);
    });
    return () => {
      live = false;
    };
  }, [country]);
  return answer && answer.country === country ? answer : null;
}

/** A day the API sends as `YYYY-MM-DD`, in the account's locale. */
const day = (iso: string) => formatDateTime(new Date(`${iso}T00:00:00Z`), { dateStyle: "medium", timeZone: "UTC" });

/**
 * The deal region's economic regime this year, every region's beside it, what it was read from
 * and what its model card found. Shown with the sources, not the results: it never goes stale
 * with the simulation.
 *
 * i18n-keys: regime.region_*, regime.why_*, regime.basis_*, regime.note_*
 * i18n-keys: montecarlo.scenario*
 * i18n-keys: regime.card, regime.cardAutumn
 */
export function EconomicRegimeTile() {
  const regime = useRegime();
  if (!regime) return null;
  return <Body g={regime} />;
}

function Body({ g }: { g: Regime }) {
  const t = useTranslations("regime");
  const mc = useTranslations("montecarlo");
  const { scenario, setScenario } = useMonteCarlo();
  const label = (id: string) => mc(SCENARIOS.find((s) => s.id === id)!.labelKey);
  const region = t(`region_${g.region}`);
  const fit = g.regime;
  const inUse = !!fit && (scenario ?? "base") === fit.regime;
  return (
    <Tile
      span={12}
      title={t("tile", { region })}
      unit={t("unit", { trail: String(g.trail) })}
      action={
        fit &&
        !inUse && (
          // Marks Monte Carlo out of date; the bar above reruns it
          <SecondaryButton onClick={() => setScenario(fit.regime as Scenario)}>{t("usePreset", { label: label(fit.regime) })}</SecondaryButton>
        )
      }
    >
      <div className="grid gap-2" data-regime-status={g.shown ? "shown" : (g.hidden ?? g.status)} data-regime-region={g.region}>
        {!g.own_region && (
          <p className="type-body text-[9px]" data-testid="regime-stand-in">
            {g.country ? t("worldStandsIn", { country: regionName(g.country) }) : t("noCountry")}
          </p>
        )}
        {g.status === "no_data" && <p className="type-body">{t("noData")}</p>}
        {g.status === "ok" && !g.shown && (
          <p className="type-body">
            <span className="type-alert text-[9px]">{t("notShown")}</span>{" "}
            {t(`why_${g.hidden ?? "not_enough_data"}`, { region, days: String(g.max_age_days) })}
          </p>
        )}
        {fit && (
          <>
            <p className="font-mono text-[11.5px] text-ink" data-testid="regime-current">
              <b className="text-bright">{label(fit.regime)}</b> · {t("dated", { year: String(fit.year), basis: t(`basis_${fit.basis}`) })}
              {g.read_on && <span className="text-muted"> · {t("readOn", { date: day(g.read_on) })}</span>}
            </p>
            <p className="font-mono text-[10.5px] text-ink" data-provenance="regime">
              {t("figures", {
                region,
                year: String(fit.year),
                growth: fmtPct(fit.growth, 1),
                growthNorm: fmtPct(fit.growth_norm, 1),
                growthGap: fmtNumber(fit.growth_gap, 1, true),
                inflation: fmtPct(fit.inflation, 1),
                inflationNorm: fmtPct(fit.inflation_norm, 1),
                inflationGap: fmtNumber(fit.inflation_gap, 1, true),
                trail: String(g.trail),
              })}
            </p>
            <p className="font-mono text-[10.5px] text-muted">
              {t("cutoffs", {
                weak: fmtNumber(fit.weak_growth, 1, true),
                strong: fmtNumber(fit.strong_growth, 1, true),
                high: fmtNumber(fit.high_inflation, 1, true),
                years: fit.history_years,
                first: String(fit.first_year),
                share: fmtRate(g.share, 0),
              })}
            </p>
            {fit.recent.length > 0 && (
              <p className="font-mono text-[10.5px] text-muted" data-testid="regime-recent">
                {t("recent")} {fit.recent.map((r) => `${r.year} ${label(r.regime)}`).join(" · ")}
              </p>
            )}
          </>
        )}
        {g.regions.length > 0 && (
          <table className="w-full border-collapse font-mono text-[11px]" aria-label={t("tableLabel")}>
            <thead>
              <tr className="type-input-label">
                <th scope="col" className="px-2 py-1 text-start font-normal">{t("colRegion")}</th>
                <th scope="col" className="px-2 py-1 text-start font-normal">{t("colRegime")}</th>
                <th scope="col" className="px-2 py-1 text-end font-normal">{t("colYear")}</th>
              </tr>
            </thead>
            <tbody>
              {g.regions.map((r) => (
                <tr key={r.region} className={r.region === g.region ? "text-bright" : "text-ink"} data-regime-row={r.region}>
                  <th scope="row" className="px-2 py-1 text-start font-normal">{t(`region_${r.region}`)}</th>
                  <td className="px-2 py-1 text-start">{r.regime ? label(r.regime) : t("rowNotShown")}</td>
                  <td className="px-2 py-1 text-end text-muted">
                    {r.year != null && r.basis ? t("dated", { year: String(r.year), basis: t(`basis_${r.basis}`) }) : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {g.card.verdict !== "not_enough_data" && g.card.model != null && g.card.baseline != null && (
          <p className="font-mono text-[10.5px] text-muted" data-testid="regime-card">
            {t(g.card.set === "autumn" ? "cardAutumn" : "card", {
              region,
              cases: g.card.cases,
              shown: fmtCount(g.card.cases),
              model: fmtRate(g.card.model, 0),
              baseline: fmtRate(g.card.baseline, 0),
            })}
          </p>
        )}
        <ul className="type-body grid gap-0.5 text-[9px]">
          {g.notes.map((n) => (
            <li key={n}>{t(`note_${n}`, { first: String(g.first_year) })}</li>
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
