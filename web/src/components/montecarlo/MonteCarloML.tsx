"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { apiInputs } from "@/lib/deal/fields";
import { useSettings } from "@/components/settings/SettingsProvider";
import { useMoney } from "@/components/ui/MoneyScope";
import { EmptyState, Notice, PrimaryButton, SecondaryButton } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { api, type Schemas } from "@/lib/api/client";
import { useCapabilities } from "@/lib/capabilities";
import { fmtMultiple, fmtNumber, fmtRate } from "@/lib/format";

import { SCENARIOS, type Scenario, useMonteCarlo } from "./MonteCarloProvider";
import { MonteCarloScreen } from "./MonteCarloScreen";

type Sliders = Schemas["SurrogateSliders"];
type Prediction = Schemas["SurrogateResponse"];

/** Bounds mirror SurrogateSliders in api/schemas.py (the surrogate's training range). i18n-keys: montecarlo.slider* */
const SLIDERS: { key: keyof Sliders; labelKey: string; min: number; max: number; step: number; unit: string }[] = [
  { key: "growth_mean", labelKey: "sliderGrowthMean", min: -5, max: 20, step: 0.1, unit: "%" },
  { key: "exit_mean", labelKey: "sliderExitMean", min: 4, max: 20, step: 0.1, unit: "x" },
  { key: "interest_mean", labelKey: "sliderInterestMean", min: 1, max: 15, step: 0.1, unit: "%" },
  { key: "gross_margin_mean", labelKey: "sliderGrossMarginMean", min: 10, max: 80, step: 0.5, unit: "%" },
  { key: "debt_pct", labelKey: "sliderDebtPct", min: 20, max: 90, step: 1, unit: "%" },
  { key: "exit_std", labelKey: "sliderExitStd", min: 0.3, max: 5, step: 0.1, unit: "x" },
];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

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
  const { label: mu } = useMoney();
  const { overrides } = useSettings();
  const t = useTranslations("montecarlo");
  const e = useTranslations("errors");
  const units = useTranslations("units");
  /** A training-deal term in its unit (rates arrive as fractions; "money" is the deal's currency and unit). */
  const term = (v: number, unit: string, decimals: number): string => {
    if (unit === "percent") return fmtRate(v, decimals);
    if (unit === "multiple") return fmtMultiple(v, decimals);
    if (unit === "years") return units("years", { value: fmtNumber(v, decimals) });
    if (unit === "money") return `${fmtNumber(v, decimals)} ${mu}`;
    return fmtNumber(v, decimals);
  };
  const initial = (): Sliders => ({
    growth_mean: clamp(sim.growth_mean, -5, 20),
    exit_mean: clamp(sim.exit_mean, 4, 20),
    interest_mean: clamp(sim.rate_mean, 1, 15),
    gross_margin_mean: clamp(sim.gm_mean, 10, 80),
    debt_pct: clamp(deal.debt_pct, 20, 90),
    exit_std: clamp(sim.exit_std, 0.3, 5),
  });
  const [sliders, setSliders] = useState<Sliders>(initial);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "error">("idle");
  const [error, setError] = useState<string>();
  const unavailable = t("liveUnavailableShort");
  const unreachable = e("apiUnreachable");

  useEffect(() => {
    if (!caps?.surrogate) return;
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      setStatus("loading");
      api
        .POST("/api/ml/surrogate", {
          body: { mc: { ...sim, ebitda: deal.ebitda, entry_mult: deal.entry_mult, hold: deal.hold }, deal: apiInputs(deal), settings: overrides, sliders },
          signal: ctrl.signal,
        })
        .then(({ data, error: err }) => {
          if (data) {
            setPred(data);
            setStatus("idle");
          } else {
            setError(String((err as { detail?: unknown })?.detail ?? unavailable));
            setStatus("error");
          }
        })
        .catch((err) => {
          if ((err as Error)?.name !== "AbortError") {
            setError(unreachable);
            setStatus("error");
          }
        });
    }, 120);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [caps?.surrogate, sliders, sim, deal, overrides, unavailable, unreachable]);

  if (!caps) return null;
  if (!caps.surrogate) {
    return (
      <EmptyState title={t("liveUnavailable")}>
        {t.rich("liveUnavailableBody", { code: (chunks) => <code>{chunks}</code> })}
      </EmptyState>
    );
  }
  const p = pred?.prediction;
  const hurdle = sim.hurdle / 100;
  return (
    <Tiles>
      {pred && (
        <Notice title={t("liveTrainedTitle")} role="note" className="col-span-12">
          <span data-provenance="live">{pred.training_deal.map((d) => `${d.term} ${term(d.model_value, d.unit, d.decimals)}`).join(" · ")}</span>
          {t("liveTrainedAfter")}
        </Notice>
      )}
      {pred && pred.term_differences.length > 0 && (
        <Notice title={t("liveDirectionalTitle")} role="note" className="col-span-12">
          {t("liveDirectionalBody", {
            differences: pred.term_differences
              .map((d) =>
                t("liveDifference", {
                  term: d.term,
                  value: fmtNumber(d.value, d.decimals + (d.unit === "percent" ? 2 : 0)),
                  modelValue: fmtNumber(d.model_value, d.decimals + (d.unit === "percent" ? 2 : 0)),
                }),
              )
              .join(", "),
          })}
        </Notice>
      )}
      {pred?.tail_unreliable && (
        <Notice title={t("liveTailTitle")} role="note" className="col-span-12">
          {t("liveTailBody")}
        </Notice>
      )}
      <Kpi title={t("kpiMedianIrr")} value={fmtRate(p?.irr_p50)} sub={status === "loading" ? t("updating") : t("surrogateEstimate")} lead />
      <Kpi title={t("kpiMeanIrr")} value={fmtRate(p?.irr_mean)} sub={t("stdSub", { std: fmtRate(p?.irr_std) })} />
      <Kpi title={t("kpiP5")} value={pred?.tail_unreliable ? fmtRate(null) : fmtRate(p?.irr_p5)} sub={t("oneInTwentyBelow")} />
      <Kpi title={t("kpiP95")} value={fmtRate(p?.irr_p95)} sub={t("oneInTwentyAbove")} />
      <Kpi title={t("kpiAbove20")} value={fmtRate(p?.p_above_20)} sub={t("asTrained")} />
      <Kpi title={t("kpiWipeout")} value={fmtRate(p?.p_wipeout)} sub={t("shareOfPaths")} tone={(p?.p_wipeout ?? 0) > 0.05 ? "loss" : undefined} />

      <Tile span={5} title={t("tileAssumptions")} aside={<SecondaryButton onClick={() => setSliders(initial())}>{t("resetToRail")}</SecondaryButton>}>
        <div className="grid gap-3">
          {SLIDERS.map((s) => (
            <label key={s.key} className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1">
              <span className="type-input-label">{t(s.labelKey)}</span>
              <span className="font-mono text-[11.5px] text-ink">
                {fmtNumber(sliders[s.key], s.step < 1 ? 1 : 0)} {s.unit}
              </span>
              <input
                type="range"
                aria-label={t(s.labelKey)}
                min={s.min}
                max={s.max}
                step={s.step}
                value={sliders[s.key]}
                onChange={(ev) => setSliders((v) => ({ ...v, [s.key]: Number(ev.target.value) }))}
                className="col-span-2 accent-[var(--color-accent)]"
              />
            </label>
          ))}
        </div>
        {status === "error" && <p className="font-mono text-[10.5px] text-loss">{error}</p>}
      </Tile>
      <Tile span={7} title={t("tileEstimatedRange")} unit={t("estimatedRangeUnit")}>
        {p && <RangeBand p={p} hurdle={hurdle} />}
        <p className="type-body text-[9px]">{t("surrogateNote")}</p>
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

type Regime = { regime: Scenario; confidence: number; data_as_of: string; label_probabilities: Record<string, number> };

/** Current macro regime from FRED data, with a shortcut to that scenario preset. */
export function MacroRegime() {
  const caps = useCapabilities();
  const { setScenario } = useMonteCarlo();
  const t = useTranslations("montecarlo");
  const [regime, setRegime] = useState<Regime | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "error">("idle");
  const [error, setError] = useState<string>();
  if (!caps?.macro_regime_installed) return null;
  if (!caps.macro_regime_trained) {
    return (
      <Tile span={12} title={t("tileMacroRegime")} unit={t("macroSource")}>
        <p className="type-body">{t.rich("macroUntrained", { code: (chunks) => <code>{chunks}</code> })}</p>
      </Tile>
    );
  }
  const preset = SCENARIOS.find((s) => s.id === regime?.regime);
  const label = preset ? t(preset.labelKey) : undefined;
  return (
    <Tile span={12} title={t("tileMacroRegime")} unit={t("macroUnit")}>
      <div className="flex flex-wrap items-center gap-4">
        <SecondaryButton
          disabled={state === "loading"}
          onClick={async () => {
            setState("loading");
            const { data, error: err } = await api.POST("/api/ml/macro-regime");
            if (data) {
              setRegime(data as Regime);
              setState("idle");
            } else {
              setError(String((err as unknown as { detail?: unknown } | undefined)?.detail ?? t("macroFailed")));
              setState("error");
            }
          }}
        >
          {state === "loading" ? t("macroClassifying") : t("macroDetect")}
        </SecondaryButton>
        {regime && (
          <>
            <span className="font-mono text-[11.5px] text-ink">
              <b className="text-bright">{label ?? regime.regime}</b> ·{" "}
              {t("macroConfidence", { confidence: fmtRate(regime.confidence, 0), asOf: regime.data_as_of })}
            </span>
            <span className="font-mono text-[10.5px] text-muted">
              {Object.entries(regime.label_probabilities)
                .map(([k, v]) => t("macroProbability", { label: k, share: fmtRate(v, 0) }))
                .join(" · ")}
            </span>
            {label && (
              <SecondaryButton
                // Marks Monte Carlo out of date; the bar above reruns it
                onClick={() => setScenario(regime.regime)}
              >
                {t("macroUsePreset", { label })}
              </SecondaryButton>
            )}
          </>
        )}
        {state === "error" && <span className="font-mono text-[10.5px] text-loss">{error}</span>}
      </div>
    </Tile>
  );
}
