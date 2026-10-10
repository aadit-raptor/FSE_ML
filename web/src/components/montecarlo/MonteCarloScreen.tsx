"use client";

import { useTranslations } from "next-intl";
import { useEffect, type ReactNode } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { DealField } from "@/components/deal/DealScreen";
import { NumberField } from "@/components/ui/NumberField";
import { EmptyState, LoadingTiles, Notice, PrimaryButton, RailGroup, Screen, Switch } from "@/components/ui/Screen";
import { floatingCount } from "@/lib/deal/capital";
import type { FieldSpec } from "@/lib/fields";

import { SCENARIOS, SIM_FIELDS, SIM_LABEL_KEY, type SimKey, useMonteCarlo } from "./MonteCarloProvider";
import { GrowthCalibrationRail } from "./GrowthCalibration";
import { SourcedRiskRail } from "./SourcedRisk";

const SEED_SPEC: FieldSpec = { unit: "", step: 1, decimals: 0, min: 0, integer: true };

function SimField({ name }: { name: SimKey }) {
  const { sim, setSim } = useMonteCarlo();
  const t = useTranslations("montecarlo");
  return <NumberField spec={SIM_FIELDS[name]} label={t(SIM_LABEL_KEY[name])} value={sim[name]} onCommit={(v) => setSim(name, v)} />;
}

/** i18n-keys: montecarlo.presetNone */
function Rail() {
  const { scenario, setScenario, seed, setSeed } = useMonteCarlo();
  const { inputs: deal } = useDeal();
  const t = useTranslations("montecarlo");
  return (
    <>
      <RailGroup title={t("groupFromDeal")}>
        <DealField name="ebitda" />
        <DealField name="entry_mult" />
        <DealField name="hold" />
      </RailGroup>
      <RailGroup title={t("groupSimulation")}>
        <SimField name="n" />
        <SimField name="hurdle" />
        <div className="grid grid-cols-[1fr_auto] items-center gap-2 py-1">
          <Switch checked={seed !== null} onChange={(on) => setSeed(on ? 42 : null)} label={t("fixedSeed")} />
        </div>
        {seed !== null && <NumberField spec={SEED_SPEC} label={t("seed")} value={seed} onCommit={setSeed} />}
      </RailGroup>
      <RailGroup title={t("groupScenarioPreset")}>
        <div className="grid grid-cols-2 gap-px bg-line" role="radiogroup" aria-label={t("groupScenarioPreset")}>
          {[{ id: null, labelKey: "presetNone" }, ...SCENARIOS].map((s) => (
            <button
              key={s.labelKey}
              type="button"
              role="radio"
              aria-checked={scenario === s.id}
              onClick={() => setScenario(s.id)}
              className={`type-action-secondary px-2 py-1.5 ${scenario === s.id ? "bg-raised text-bright shadow-[inset_0_-2px_0_var(--color-accent)]" : "bg-panel text-muted hover:text-ink"}`}
            >
              {t(s.labelKey)}
            </button>
          ))}
        </div>
        <p className="type-body pt-1.5 text-[9px]">{t("presetNote")}</p>
      </RailGroup>
      <SourcedRiskRail />
      <RailGroup title={t("groupGrowth")}>
        <SimField name="growth_mean" />
        <SimField name="growth_std" />
        <GrowthCalibrationRail />
      </RailGroup>
      <RailGroup title={t("groupExitMultiple")}>
        <SimField name="exit_mean" />
        <SimField name="exit_std" />
      </RailGroup>
      <RailGroup title={t("groupInterestRate")}>
        <SimField name="rate_mean" />
        <SimField name="rate_std" />
        {/* A deal listing its facilities: the draw moves only the floating ones (PLAN.md 2.4b) */}
        {deal.tranches.length > 0 && (
          <p className="type-body pt-1.5 text-[9px]" data-testid="rate-tranches-note">
            {t("rateTranchesNote", { count: floatingCount(deal.tranches) })}
          </p>
        )}
      </RailGroup>
      <RailGroup title={t("groupGrossMargin")}>
        <SimField name="gm_mean" />
        <SimField name="gm_std" />
      </RailGroup>
    </>
  );
}

/** Where a background run is (PLAN.md 1.9); the rest of the app stays usable meanwhile. */
function RunProgress() {
  const { run, cancel } = useMonteCarlo();
  const t = useTranslations("montecarlo");
  const pct = Math.round((run.progress?.fraction ?? 0) * 100);
  return (
    <Notice
      tone="info"
      title={t("running")}
      actions={
        <button type="button" onClick={cancel} className="type-action-secondary border border-line px-2.5 py-1.5 text-muted hover:text-ink">
          {t("cancel")}
        </button>
      }
    >
      <span className="flex items-center gap-3">
        <span
          role="progressbar"
          aria-label={t("progress")}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={pct}
          aria-valuetext={run.progress?.stage}
          className="relative h-[3px] w-40 flex-none bg-line"
        >
          <span className="absolute inset-y-0 start-0 bg-accent transition-[width] duration-300" style={{ width: `${pct}%` }} />
        </span>
        <span className="font-mono text-[11px] tabular-nums">{pct}%</span>
        <span data-testid="run-stage">{run.progress?.stage}</span>
      </span>
    </Notice>
  );
}

function Bar() {
  const { run, stale, changes, runNow } = useMonteCarlo();
  const t = useTranslations("montecarlo");
  if (run.status === "error") {
    return (
      <Notice tone="loss" title={t("didntRun")} role="alert" actions={<PrimaryButton onClick={runNow}>{t("tryAgain")}</PrimaryButton>}>
        {run.error}
      </Notice>
    );
  }
  if (run.status === "running") return <RunProgress />;
  if (!stale) return null;
  return (
    <Notice title={t("outOfDate")} actions={<PrimaryButton onClick={runNow}>{t("runNow")}</PrimaryButton>}>
      <span className="font-mono text-[11px]">{changes.join("   ")}</span>
    </Notice>
  );
}

/** Shared frame for every Monte Carlo step. Runs once on first visit. */
export function MonteCarloScreen({ children }: { children: ReactNode }) {
  const { run, runNow } = useMonteCarlo();
  const t = useTranslations("montecarlo");

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
        <EmptyState title={t("noSimulation")} action={<PrimaryButton onClick={runNow}>{t("runNow")}</PrimaryButton>}>
          {t("noSimulationBody")}
        </EmptyState>
      ) : run.status === "cancelled" ? (
        <EmptyState title={t("cancelled")} action={<PrimaryButton onClick={runNow}>{t("runNow")}</PrimaryButton>}>
          {t("cancelledBody")}
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
