"use client";

import { useEffect, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { useSettings } from "@/components/settings/SettingsProvider";
import { EmptyState, Notice, PrimaryButton, SecondaryButton } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { api, type Schemas } from "@/lib/api/client";
import { useCapabilities } from "@/lib/capabilities";
import { fmtRate } from "@/lib/format";

import { SCENARIOS, type Scenario, useMonteCarlo } from "./MonteCarloProvider";
import { MonteCarloScreen } from "./MonteCarloScreen";

type Sliders = Schemas["SurrogateSliders"];
type Prediction = Schemas["SurrogateResponse"];

/** Bounds mirror SurrogateSliders in api/schemas.py (the surrogate's training range). */
const SLIDERS: { key: keyof Sliders; label: string; min: number; max: number; step: number; unit: string }[] = [
  { key: "growth_mean", label: "Revenue growth mean", min: -5, max: 20, step: 0.1, unit: "%" },
  { key: "exit_mean", label: "Exit multiple mean", min: 4, max: 20, step: 0.1, unit: "x" },
  { key: "interest_mean", label: "Interest rate mean", min: 1, max: 15, step: 0.1, unit: "%" },
  { key: "gross_margin_mean", label: "Gross margin mean", min: 10, max: 80, step: 0.5, unit: "%" },
  { key: "debt_pct", label: "Debt / EV", min: 20, max: 90, step: 1, unit: "%" },
  { key: "exit_std", label: "Exit multiple std dev", min: 0.3, max: 5, step: 0.1, unit: "x" },
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
  const { overrides } = useSettings();
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

  useEffect(() => {
    if (!caps?.surrogate) return;
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      setStatus("loading");
      api
        .POST("/api/ml/surrogate", {
          body: { mc: { ...sim, ebitda: deal.ebitda, entry_mult: deal.entry_mult, hold: deal.hold }, deal, settings: overrides, sliders },
          signal: ctrl.signal,
        })
        .then(({ data, error: err }) => {
          if (data) {
            setPred(data);
            setStatus("idle");
          } else {
            setError(String((err as { detail?: unknown })?.detail ?? "Live mode is unavailable"));
            setStatus("error");
          }
        })
        .catch((e) => {
          if ((e as Error)?.name !== "AbortError") {
            setError("Can't reach the API.");
            setStatus("error");
          }
        });
    }, 120);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [caps?.surrogate, sliders, sim, deal, overrides]);

  if (!caps) return null;
  if (!caps.surrogate) {
    return (
      <EmptyState title="Live mode isn't available on this server">
        It needs the optional ML layer (<code>pip install -r requirements-ml.txt</code>) and the committed surrogate model files. Monte Carlo itself works without it.
      </EmptyState>
    );
  }
  const p = pred?.prediction;
  const hurdle = sim.hurdle / 100;
  return (
    <Tiles>
      {pred && pred.term_differences.length > 0 && (
        <Notice title="Directional only" role="note" className="col-span-12">
          The surrogate learned a fixed deal and this one differs in{" "}
          {pred.term_differences.map((d) => `${d.term} (${d.value.toFixed(d.decimals + (d.unit === "percent" ? 2 : 0))} vs ${d.model_value.toFixed(d.decimals + (d.unit === "percent" ? 2 : 0))})`).join(", ")}.
          Use Run Monte Carlo for exact results.
        </Notice>
      )}
      {pred?.tail_unreliable && (
        <Notice title="Downside tail unreliable" role="note" className="col-span-12">
          With this much wipeout risk the 5th percentile sits near the -100% floor, where the surrogate isn&apos;t reliable. Run the simulation for the downside.
        </Notice>
      )}
      <Kpi title="Median IRR" value={fmtRate(p?.irr_p50)} sub={status === "loading" ? "updating" : "surrogate estimate"} lead />
      <Kpi title="Mean IRR" value={fmtRate(p?.irr_mean)} sub={`std ${fmtRate(p?.irr_std)}`} />
      <Kpi title="P5" value={pred?.tail_unreliable ? "n/a" : fmtRate(p?.irr_p5)} sub="1 in 20 below" />
      <Kpi title="P95" value={fmtRate(p?.irr_p95)} sub="1 in 20 above" />
      <Kpi title="P(IRR > 20%)" value={fmtRate(p?.p_above_20)} sub="as trained" />
      <Kpi title="Wipeout" value={fmtRate(p?.p_wipeout)} sub="share of paths" tone={(p?.p_wipeout ?? 0) > 0.05 ? "loss" : undefined} />

      <Tile span={5} title="Assumptions" aside={<SecondaryButton onClick={() => setSliders(initial())}>Reset to rail</SecondaryButton>}>
        <div className="grid gap-3">
          {SLIDERS.map((s) => (
            <label key={s.key} className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1">
              <span className="type-input-label">{s.label}</span>
              <span className="font-mono text-[11.5px] text-ink">
                {sliders[s.key].toFixed(s.step < 1 ? 1 : 0)} {s.unit}
              </span>
              <input
                type="range"
                aria-label={s.label}
                min={s.min}
                max={s.max}
                step={s.step}
                value={sliders[s.key]}
                onChange={(e) => setSliders((v) => ({ ...v, [s.key]: Number(e.target.value) }))}
                className="col-span-2 accent-[var(--color-accent)]"
              />
            </label>
          ))}
        </div>
        {status === "error" && <p className="font-mono text-[10.5px] text-loss">{error}</p>}
      </Tile>
      <Tile span={7} title="Estimated IRR range" unit="P5-P95, P25-P75, median">
        {p && <RangeBand p={p} hurdle={hurdle} />}
        <p className="type-body text-[9px]">
          A neural network trained on the simulation answers in milliseconds as you drag. The first answer can take a few seconds while the model loads.
        </p>
        <div>
          <PrimaryButton onClick={runNow}>Run Monte Carlo</PrimaryButton>
        </div>
      </Tile>
    </Tiles>
  );
}

function RangeBand({ p, hurdle }: { p: Record<string, number | null>; hurdle: number }) {
  const W = 620;
  const H = 110;
  const lo = -0.3;
  const hi = 0.8;
  const x = (v: number) => 20 + ((W - 40) * (clamp(v, lo, hi) - lo)) / (hi - lo);
  const v = (k: string) => p[k] ?? 0;
  const ticks = [-0.2, 0, 0.2, 0.4, 0.6, 0.8];
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Estimated IRR range" className="block h-auto w-full">
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={18} y2={80} stroke="var(--color-grid)" />
          <text x={x(t)} y={98} textAnchor="middle" className="chart-tick">
            {(t * 100).toFixed(0)}%
          </text>
        </g>
      ))}
      <rect x={x(v("irr_p5"))} y={34} width={Math.max(1, x(v("irr_p95")) - x(v("irr_p5")))} height={30} fill="var(--color-accent)" fillOpacity={0.18} />
      <rect x={x(v("irr_p25"))} y={34} width={Math.max(1, x(v("irr_p75")) - x(v("irr_p25")))} height={30} fill="var(--color-accent)" fillOpacity={0.4} />
      <line x1={x(v("irr_p50"))} x2={x(v("irr_p50"))} y1={28} y2={70} stroke="var(--color-accent)" strokeWidth={3} />
      <line x1={x(hurdle)} x2={x(hurdle)} y1={16} y2={80} stroke="var(--color-attention)" strokeDasharray="3 3" />
      <text x={x(hurdle) + 5} y={14} className="chart-reference" style={{ fill: "var(--color-attention)" }}>
        Hurdle {(hurdle * 100).toFixed(0)}%
      </text>
    </svg>
  );
}

type Regime = { regime: Scenario; confidence: number; data_as_of: string; label_probabilities: Record<string, number> };

/** Current macro regime from FRED data, with a shortcut to that scenario preset. */
export function MacroRegime() {
  const caps = useCapabilities();
  const { setScenario } = useMonteCarlo();
  const [regime, setRegime] = useState<Regime | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "error">("idle");
  const [error, setError] = useState<string>();
  if (!caps?.macro_regime_installed) return null;
  if (!caps.macro_regime_trained) {
    return (
      <Tile span={12} title="Macro regime" unit="FRED">
        <p className="type-body">
          Installed but not trained. Set <code>FRED_API_KEY</code> on the API server and run <code>python -m ml.macro_regime</code> to classify the current regime here.
        </p>
      </Tile>
    );
  }
  const label = SCENARIOS.find((s) => s.id === regime?.regime)?.label;
  return (
    <Tile span={12} title="Macro regime" unit="hidden Markov model on FRED data">
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
              setError(String((err as unknown as { detail?: unknown } | undefined)?.detail ?? "Could not classify the regime"));
              setState("error");
            }
          }}
        >
          {state === "loading" ? "Classifying" : "Detect current regime"}
        </SecondaryButton>
        {regime && (
          <>
            <span className="font-mono text-[11.5px] text-ink">
              <b className="text-bright">{label ?? regime.regime}</b> · {(regime.confidence * 100).toFixed(0)}% confidence · data as of {regime.data_as_of}
            </span>
            <span className="font-mono text-[10.5px] text-muted">
              {Object.entries(regime.label_probabilities)
                .map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`)
                .join(" · ")}
            </span>
            {label && (
              <SecondaryButton
                // Marks Monte Carlo out of date; the bar above reruns it
                onClick={() => setScenario(regime.regime)}
              >
                Use {label} preset
              </SecondaryButton>
            )}
          </>
        )}
        {state === "error" && <span className="font-mono text-[10.5px] text-loss">{error}</span>}
      </div>
    </Tile>
  );
}
