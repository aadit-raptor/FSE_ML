"use client";

import { useEffect, type ReactNode } from "react";

import { DealField } from "@/components/deal/DealScreen";
import { NumberField } from "@/components/ui/NumberField";
import { EmptyState, LoadingTiles, Notice, PrimaryButton, RailGroup, Screen, Switch } from "@/components/ui/Screen";
import type { FieldSpec } from "@/lib/fields";
import { ILLUSTRATIVE } from "@/lib/provenance";

import { SCENARIOS, SIM_FIELDS, type SimKey, useMonteCarlo } from "./MonteCarloProvider";

const SEED_SPEC: FieldSpec = { label: "Seed", unit: "", step: 1, decimals: 0, min: 0, integer: true };

function SimField({ name, label }: { name: SimKey; label?: string }) {
  const { sim, setSim } = useMonteCarlo();
  return <NumberField spec={SIM_FIELDS[name]} label={label} value={sim[name]} onCommit={(v) => setSim(name, v)} />;
}

function Rail() {
  const { scenario, setScenario, seed, setSeed } = useMonteCarlo();
  return (
    <>
      <RailGroup title="From the deal">
        <DealField name="ebitda" />
        <DealField name="entry_mult" />
        <DealField name="hold" />
      </RailGroup>
      <RailGroup title="Simulation">
        <SimField name="n" />
        <SimField name="hurdle" />
        <div className="grid grid-cols-[1fr_auto] items-center gap-2 py-1">
          <Switch checked={seed !== null} onChange={(on) => setSeed(on ? 42 : null)} label="Fixed seed" />
        </div>
        {seed !== null && <NumberField spec={SEED_SPEC} value={seed} onCommit={setSeed} />}
      </RailGroup>
      <RailGroup title="Scenario preset">
        <div className="grid grid-cols-2 gap-px bg-line" role="radiogroup" aria-label="Scenario preset">
          {[{ id: null, label: "None" }, ...SCENARIOS].map((s) => (
            <button
              key={s.label}
              type="button"
              role="radio"
              aria-checked={scenario === s.id}
              onClick={() => setScenario(s.id)}
              className={`type-action-secondary px-2 py-1.5 ${scenario === s.id ? "bg-raised text-bright shadow-[inset_0_-2px_0_var(--color-accent)]" : "bg-panel text-muted hover:text-ink"}`}
            >
              {s.label}
            </button>
          ))}
        </div>
        <p className="type-body pt-1.5 text-[9px]">Applies Settings multipliers to the distributions below. The Scenarios step always shows all four.</p>
      </RailGroup>
      <div className="grid gap-1 border-b border-line px-3.5 py-2.5" role="note">
        <p className="type-alert text-[9px]">{ILLUSTRATIVE}</p>
        <p className="type-body text-[9px]">Applies to the ranges below, the driver correlations and the scenario presets.</p>
      </div>
      <RailGroup title="Revenue growth">
        <SimField name="growth_mean" />
        <SimField name="growth_std" />
      </RailGroup>
      <RailGroup title="Exit multiple">
        <SimField name="exit_mean" />
        <SimField name="exit_std" />
      </RailGroup>
      <RailGroup title="Interest rate">
        <SimField name="rate_mean" />
        <SimField name="rate_std" />
      </RailGroup>
      <RailGroup title="Gross margin">
        <SimField name="gm_mean" />
        <SimField name="gm_std" />
      </RailGroup>
    </>
  );
}

/** Where a background run is (PLAN.md 1.9); the rest of the app stays usable meanwhile. */
function RunProgress() {
  const { run, cancel } = useMonteCarlo();
  const pct = Math.round((run.progress?.fraction ?? 0) * 100);
  return (
    <Notice
      tone="info"
      title="Running simulation"
      actions={
        <button type="button" onClick={cancel} className="type-action-secondary border border-line px-2.5 py-1.5 text-muted hover:text-ink">
          Cancel
        </button>
      }
    >
      <span className="flex items-center gap-3">
        <span
          role="progressbar"
          aria-label="Simulation progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
          aria-valuetext={run.progress?.stage}
          className="relative h-[3px] w-40 flex-none bg-line"
        >
          <span className="absolute inset-y-0 left-0 bg-accent transition-[width] duration-300" style={{ width: `${pct}%` }} />
        </span>
        <span className="font-mono text-[11px] tabular-nums">{pct}%</span>
        <span data-testid="run-stage">{run.progress?.stage}</span>
      </span>
    </Notice>
  );
}

function Bar() {
  const { run, stale, changes, runNow } = useMonteCarlo();
  if (run.status === "error") {
    return (
      <Notice tone="loss" title="Simulation didn't run" role="alert" actions={<PrimaryButton onClick={runNow}>Try again</PrimaryButton>}>
        {run.error}
      </Notice>
    );
  }
  if (run.status === "running") return <RunProgress />;
  if (!stale) return null;
  return (
    <Notice title="Monte Carlo is out of date" actions={<PrimaryButton onClick={runNow}>Run Monte Carlo</PrimaryButton>}>
      <span className="font-mono text-[11px]">{changes.join("   ")}</span>
    </Notice>
  );
}

/** Shared frame for every Monte Carlo step. Runs once on first visit. */
export function MonteCarloScreen({ children }: { children: ReactNode }) {
  const { run, runNow } = useMonteCarlo();

  useEffect(() => {
    if (run.status === "idle") runNow();
    // Only on first visit; later runs are explicit
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Screen rail={<Rail />} bar={<Bar />}>
      {run.result ? (
        <div className={run.status === "running" ? "opacity-60 transition-opacity" : ""}>{children}</div>
      ) : run.status === "error" ? (
        <EmptyState title="No simulation yet" action={<PrimaryButton onClick={runNow}>Run Monte Carlo</PrimaryButton>}>
          Fix the problem above and run again.
        </EmptyState>
      ) : run.status === "cancelled" ? (
        <EmptyState title="Simulation cancelled" action={<PrimaryButton onClick={runNow}>Run Monte Carlo</PrimaryButton>}>
          Nothing ran to the end. Run it when you are ready.
        </EmptyState>
      ) : (
        <LoadingTiles />
      )}
    </Screen>
  );
}

/** Dim tiles whose numbers no longer match the inputs. */
export function useStaleClass(): string {
  const { stale } = useMonteCarlo();
  return stale ? "[&_section>*:not(header)]:opacity-40" : "";
}
