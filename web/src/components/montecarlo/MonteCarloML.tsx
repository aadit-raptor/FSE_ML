"use client";

import { useTranslations } from "next-intl";
import { useEffect, useMemo, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { debtShareOfEv } from "@/lib/deal/capital";
import { apiInputs } from "@/lib/deal/fields";
import { useSettings } from "@/components/settings/SettingsProvider";
import { EmptyState, Notice, PrimaryButton, SecondaryButton } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { api, type Schemas } from "@/lib/api/client";
import { useCapabilities } from "@/lib/capabilities";
import { fmtCount, fmtNumber, fmtRate } from "@/lib/format";

import { useMonteCarlo } from "./MonteCarloProvider";
import { MonteCarloScreen } from "./MonteCarloScreen";
import { TipLabel, TipText, useTips } from "@/components/ui/Tip";

type Sliders = Schemas["LiveSliders"];
type Estimate = Schemas["LiveResponse"];

/**
 * Where each slider's track usually runs. The simulation takes any value the inputs do (LiveSliders in
 * api/schemas.py), so a deal outside a track widens it rather than being clamped to it. i18n-keys: montecarlo.slider*
 */
const SLIDERS: { key: keyof Sliders; labelKey: string; tipKey: string; min: number; max: number; lowest: number; highest: number; step: number; unit: string }[] = [
  { key: "growth_mean", labelKey: "sliderGrowthMean", tipKey: "live_growth_mean", min: -5, max: 20, lowest: -50, highest: 100, step: 0.1, unit: "%" },
  { key: "exit_mean", labelKey: "sliderExitMean", tipKey: "live_exit_mean", min: 4, max: 20, lowest: 1, highest: 100, step: 0.1, unit: "x" },
  { key: "interest_mean", labelKey: "sliderInterestMean", tipKey: "live_interest_mean", min: 1, max: 15, lowest: 0, highest: 50, step: 0.1, unit: "%" },
  { key: "gross_margin_mean", labelKey: "sliderGrossMarginMean", tipKey: "live_gross_margin_mean", min: 10, max: 80, lowest: 1, highest: 99, step: 0.5, unit: "%" },
  { key: "debt_pct", labelKey: "sliderDebtPct", tipKey: "live_debt_pct", min: 20, max: 90, lowest: 0, highest: 99, step: 1, unit: "%" },
  { key: "exit_std", labelKey: "sliderExitStd", tipKey: "live_exit_std", min: 0.3, max: 5, lowest: 0.1, highest: 50, step: 0.1, unit: "x" },
];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** A slider's track for a deal that starts at `own`: the usual one, stretched to hold it. */
export function track(s: (typeof SLIDERS)[number], own: number): { min: number; max: number } {
  return {
    min: Math.max(s.lowest, Math.min(s.min, Math.floor(own))),
    max: Math.min(s.highest, Math.max(s.max, Math.ceil(own))),
  };
}

export function LiveStep() {
  return (
    <MonteCarloScreen>
      <Live />
    </MonteCarloScreen>
  );
}

function Live() {
  const caps = useCapabilities();
  const { sim, runNow } = useMonteCarlo();
  const { inputs: deal } = useDeal();
  const { overrides } = useSettings();
  const t = useTranslations("montecarlo");
  const e = useTranslations("errors");
  const tips = useTips(SLIDERS.map((s) => s.tipKey));
  /** The deal's own assumptions: where the sliders start, and what "Reset to rail" goes back to. */
  const own = useMemo(
    (): Sliders => ({
      growth_mean: clamp(sim.growth_mean, -50, 100),
      exit_mean: clamp(sim.exit_mean, 1, 100),
      interest_mean: clamp(sim.rate_mean, 0, 50),
      gross_margin_mean: clamp(sim.gm_mean, 1, 99),
      debt_pct: clamp(debtShareOfEv(deal), 0, 99),
      exit_std: clamp(sim.exit_std, 0.1, 50),
    }),
    [sim, deal],
  );
  const [sliders, setSliders] = useState<Sliders>(own);
  // A deal or rail edited elsewhere (or a saved deal arriving after this mounted) restarts the sliders
  // from it: they are moves away from the deal on screen, never a remembered other deal.
  const ownKey = JSON.stringify(own);
  const [basis, setBasis] = useState(ownKey);
  if (basis !== ownKey) {
    setBasis(ownKey);
    setSliders(own);
  }
  const [estimate, setEstimate] = useState<Estimate | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [error, setError] = useState<string>();
  const unavailable = t("liveUnavailableShort");
  const unreachable = e("apiUnreachable");
  /** A listed structure with nothing drawn at close has no debt to scale. */
  const nothingDrawn = deal.tranches.length > 0 && own.debt_pct < 0.01;

  useEffect(() => {
    if (!caps?.live) return;
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      setStatus("loading");
      api
        .POST("/api/montecarlo/live", {
          body: { mc: { ...sim, ebitda: deal.ebitda, entry_mult: deal.entry_mult, hold: deal.hold }, deal: apiInputs(deal), settings: overrides, sliders },
          signal: ctrl.signal,
        })
        .then(({ data, error: err }) => {
          if (data) {
            setEstimate(data);
            setStatus("idle");
          } else {
            const detail = (err as { detail?: unknown })?.detail;
            setError(typeof detail === "string" ? detail : unavailable);
            setStatus("error");
          }
        })
        .catch((err) => {
          if ((err as Error)?.name !== "AbortError") {
            setError(unreachable);
            setStatus("error");
          }
        });
    }, 300);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [caps?.live, sliders, sim, deal, overrides, unavailable, unreachable]);

  if (!caps) return null;
  if (!caps.live) {
    return <EmptyState title={t("liveUnavailable")}>{t("liveUnavailableBody")}</EmptyState>;
  }
  const p = estimate?.prediction;
  const hurdle = sim.hurdle / 100;
  const [low, high] = estimate?.median_interval ?? [null, null];
  return (
    <Tiles>
      {estimate && (
        <Notice title={t("liveSimulationTitle")} role="note" className="col-span-12">
          <span data-provenance="live">{t("liveSimulationBody", { paths: fmtCount(estimate.paths), full: fmtCount(sim.n) })}</span>
        </Notice>
      )}
      <Kpi
        title={t("kpiMedianIrr")}
        value={fmtRate(p?.irr_p50)}
        sub={status === "loading" ? t("updating") : estimate ? t("liveInterval", { low: fmtRate(low), high: fmtRate(high) }) : undefined}
        lead
      />
      <Kpi title={t("kpiMeanIrr")} value={fmtRate(p?.irr_mean)} sub={t("stdSub", { std: fmtRate(p?.irr_std) })} />
      <Kpi title={t("kpiP5")} value={fmtRate(p?.irr_p5)} sub={t("oneInTwentyBelow")} />
      <Kpi title={t("kpiP95")} value={fmtRate(p?.irr_p95)} sub={t("oneInTwentyAbove")} />
      <Kpi title={t("kpiAboveHurdle", { hurdle: fmtRate(hurdle, 0) })} value={fmtRate(p?.p_above_hurdle)} sub={t("shareOfPaths")} />
      <Kpi title={t("kpiWipeout")} value={fmtRate(p?.p_wipeout)} sub={t("shareOfPaths")} tone={(p?.p_wipeout ?? 0) > 0.05 ? "loss" : undefined} />

      <Tile span={5} title={t("tileAssumptions")} aside={<SecondaryButton onClick={() => setSliders(own)}>{t("resetToRail")}</SecondaryButton>}>
        <div className="grid gap-3">
          {SLIDERS.map((s) => {
            const { min, max } = track(s, own[s.key]);
            const fixed = s.key === "debt_pct" && nothingDrawn;
            return (
              <label key={s.key} className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1">
                <span className="type-input-label">
                  <TipLabel tip={tips[s.tipKey]}>{t(s.labelKey)}</TipLabel>
                </span>
                <span className="font-mono text-[11.5px] text-ink">
                  {fmtNumber(sliders[s.key], s.step < 1 ? 1 : 0)} {s.unit}
                </span>
                <input
                  type="range"
                  aria-label={t(s.labelKey)}
                  aria-describedby={tips[s.tipKey].id}
                  min={min}
                  max={max}
                  step={s.step}
                  disabled={fixed}
                  value={sliders[s.key]}
                  onChange={(ev) => setSliders((v) => ({ ...v, [s.key]: Number(ev.target.value) }))}
                  className="col-span-2 accent-[var(--color-accent)]"
                />
                {fixed && <span className="type-body col-span-2 text-[9px]">{t("liveNothingDrawn")}</span>}
                <TipText tip={tips[s.tipKey]} />
              </label>
            );
          })}
        </div>
        {status === "error" && <p className="font-mono text-[10.5px] text-loss">{error}</p>}
      </Tile>
      <Tile span={7} title={t("tileEstimatedRange")} unit={t("estimatedRangeUnit")}>
        {p && <RangeBand p={p} hurdle={hurdle} />}
        <p className="type-body text-[9px]">{t("liveNote")}</p>
        <div>
          <PrimaryButton onClick={runNow}>{t("runNow")}</PrimaryButton>
        </div>
      </Tile>
    </Tiles>
  );
}

function RangeBand({ p, hurdle }: { p: Record<string, number | null>; hurdle: number }) {
  const t = useTranslations("montecarlo");
  const W = 620;
  const H = 110;
  const lo = -0.3;
  const hi = 0.8;
  const x = (v: number) => 20 + ((W - 40) * (clamp(v, lo, hi) - lo)) / (hi - lo);
  const v = (k: string) => p[k] ?? 0;
  const ticks = [-0.2, 0, 0.2, 0.4, 0.6, 0.8];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={t("chartEstimatedRange")} className="block h-auto w-full">
      {ticks.map((tick) => (
        <g key={tick}>
          <line x1={x(tick)} x2={x(tick)} y1={18} y2={80} stroke="var(--color-grid)" />
          <text x={x(tick)} y={98} textAnchor="middle" className="chart-tick">
            {fmtRate(tick, 0)}
          </text>
        </g>
      ))}
      <rect x={x(v("irr_p5"))} y={34} width={Math.max(1, x(v("irr_p95")) - x(v("irr_p5")))} height={30} fill="var(--color-accent)" fillOpacity={0.18} />
      <rect x={x(v("irr_p25"))} y={34} width={Math.max(1, x(v("irr_p75")) - x(v("irr_p25")))} height={30} fill="var(--color-accent)" fillOpacity={0.4} />
      <line x1={x(v("irr_p50"))} x2={x(v("irr_p50"))} y1={28} y2={70} stroke="var(--color-accent)" strokeWidth={3} />
      <line x1={x(hurdle)} x2={x(hurdle)} y1={16} y2={80} stroke="var(--color-attention)" strokeDasharray="3 3" />
      <text x={x(hurdle) + 5} y={14} className="chart-reference" style={{ fill: "var(--color-attention)" }}>
        {t("markerHurdle", { value: fmtRate(hurdle, 0) })}
      </text>
    </svg>
  );
}
