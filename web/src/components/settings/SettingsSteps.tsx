"use client";

import { useTranslations } from "next-intl";
import type { ReactNode } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { heat } from "@/components/charts/HeatTable";
import { CellInput } from "@/components/ui/CellInput";
import { NumberField } from "@/components/ui/NumberField";
import { EmptyState, LoadingTiles, Notice, PrimaryButton, RailGroup, Screen, SecondaryButton, Switch } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { useMoney } from "@/components/ui/MoneyScope";
import type { DealInputs } from "@/lib/deal/fields";
import type { FieldSpec } from "@/lib/fields";
import { fmtInput, fmtMoney, fmtNumber, fmtRate } from "@/lib/format";
import { useProvenance } from "@/lib/i18n/useProvenance";
import { MAX_SIMULATION_PATHS } from "@/lib/limits";
import { MONEY } from "@/lib/money";

import { useSettings } from "./SettingsProvider";

/**
 * A setting's key doubles as its key in the `settings` namespace (PLAN.md 2.3b).
 *
 * i18n-keys: settings.def_*, settings.sens_*, settings.mc_*
 * i18n-keys: settings.tx_fee_pct, settings.fin_fee_pct, settings.other_uses
 */
type Def = { key: string; spec: FieldSpec };

const pct = (step = 0.5, decimals = 1, extra: Partial<FieldSpec> = {}): FieldSpec => ({ unit: "%", step, decimals, ...extra });

/** Deal default key -> deal input it seeds. */
const DEAL_DEFAULTS: (Def & { input?: keyof DealInputs })[] = [
  { key: "def_ebitda", input: "ebitda", spec: { unit: MONEY, step: 5, decimals: 1, min: 0, exclusiveMin: true } },
  { key: "def_entry_mult", input: "entry_mult", spec: { unit: "x", step: 0.5, decimals: 1, min: 0, exclusiveMin: true } },
  { key: "def_exit_mult", input: "exit_mult", spec: { unit: "x", step: 0.5, decimals: 1, min: 0, exclusiveMin: true } },
  { key: "def_hold", input: "hold", spec: { unit: "yr", step: 1, decimals: 0, min: 1, max: 15, integer: true } },
  { key: "def_growth", input: "growth", spec: pct() },
  { key: "def_gross_margin", input: "gross_margin", spec: pct(1) },
  { key: "def_opex", input: "opex", spec: pct(1) },
  { key: "def_tax", input: "tax", spec: pct(1) },
  { key: "def_da", input: "da", spec: pct() },
  { key: "def_debt_pct", input: "debt_pct", spec: pct(1, 1, { min: 0, max: 99 }) },
  { key: "def_senior_pct", input: "senior_pct", spec: pct(1, 1, { min: 0, max: 100 }) },
  { key: "def_base_rate", input: "base_rate", spec: pct(0.25, 2, { min: 0 }) },
  { key: "def_mezz_spread", input: "mezz_spread", spec: pct(0.25, 2, { min: 0 }) },
  { key: "def_capex", input: "capex", spec: pct() },
  { key: "def_nwc", input: "nwc", spec: pct(0.25, 2) },
  { key: "def_mincash", input: "mincash", spec: { unit: MONEY, step: 5, decimals: 1, min: 0 } },
  { key: "def_senior_amort", spec: pct() },
];

const SENSITIVITY: Def[] = [
  { key: "sens_em_min", spec: { unit: "x", step: 0.5, decimals: 1, min: 0 } },
  { key: "sens_em_max", spec: { unit: "x", step: 0.5, decimals: 1, min: 0 } },
  { key: "sens_em_steps", spec: { unit: "", step: 1, decimals: 0, min: 2, integer: true } },
  { key: "sens_hp_min", spec: { unit: "yr", step: 1, decimals: 0, min: 1, integer: true } },
  { key: "sens_hp_max", spec: { unit: "yr", step: 1, decimals: 0, min: 1, integer: true } },
];

const FEES: Def[] = [
  { key: "tx_fee_pct", spec: pct(0.1, 2, { min: 0 }) },
  { key: "fin_fee_pct", spec: pct(0.1, 2, { min: 0 }) },
  { key: "other_uses", spec: { unit: MONEY, step: 1, decimals: 1, min: 0 } },
];

const MC: Def[] = [
  { key: "mc_n", spec: { unit: "n", step: 5000, decimals: 0, min: 1000, max: MAX_SIMULATION_PATHS, integer: true } },
  { key: "mc_hurdle", spec: pct(1, 1, { min: 0 }) },
  { key: "mc_growth_mean", spec: pct() },
  { key: "mc_growth_std", spec: pct(0.5, 1, { min: 0.1 }) },
  { key: "mc_exit_mean", spec: { unit: "x", step: 0.5, decimals: 1, min: 1 } },
  { key: "mc_exit_std", spec: { unit: "x", step: 0.25, decimals: 2, min: 0.1 } },
  { key: "mc_rate_mean", spec: pct(0.25, 2, { min: 0 }) },
  { key: "mc_rate_std", spec: pct(0.25, 2, { min: 0.1 }) },
  { key: "mc_gm_mean", spec: pct(1, 1, { min: 1, max: 99 }) },
  { key: "mc_gm_std", spec: pct(0.5, 1, { min: 0.1 }) },
  { key: "mc_n_passes", spec: { unit: "", step: 1, decimals: 0, min: 1, max: 10, integer: true } },
];

/** i18n-keys: settings.driver* */
const DRIVERS = [
  { id: "g", labelKey: "driverGrowth" },
  { id: "em", labelKey: "driverExitMultiple" },
  { id: "ir", labelKey: "driverInterest" },
  { id: "gm", labelKey: "driverGrossMargin" },
  { id: "sh", labelKey: "driverShock" },
];

/** i18n-keys: settings.preset* */
const PRESETS = [
  { id: "bull", labelKey: "presetBull", cols: [["bull_growth_mult", "×"], ["bull_exit_mult", "×"], ["bull_rate_mult", "×"], ["bull_margin_mult", "×"]] },
  {
    id: "rec",
    labelKey: "presetRecession",
    cols: [["rec_growth_adj", "pts"], ["rec_exit_mult", "×"], ["rec_rate_mult", "×"], ["rec_margin_mult", "×"]],
    floor: "rec_growth_floor",
  },
  {
    id: "stag",
    labelKey: "presetStagflation",
    cols: [["stag_growth_adj", "pts"], ["stag_exit_mult", "×"], ["stag_rate_mult", "×"], ["stag_margin_mult", "×"]],
    floor: "stag_growth_floor",
  },
] as const;

function SettingField({ def }: { def: Def }) {
  const { effective, defaults, set, reset, overrides } = useSettings();
  const t = useTranslations("settings");
  const v = effective[def.key];
  const changed = def.key in overrides;
  const label = t(def.key);
  if (typeof v === "boolean") {
    return (
      <div className="flex items-center justify-between py-1">
        <Switch checked={v} onChange={(on) => set(def.key, on)} label={label} />
      </div>
    );
  }
  if (typeof v !== "number") return null;
  return (
    <div className="grid grid-cols-[1fr_auto] items-start gap-2">
      <NumberField spec={def.spec} label={label} value={v} onCommit={(n) => set(def.key, n)} changed={changed} />
      <span className="flex h-[22px] w-[74px] items-center justify-end gap-1.5">
        {changed && defaults && (
          <button
            type="button"
            onClick={() => reset(def.key)}
            className="font-mono text-[10px] text-accent hover:underline"
            title={t("defaultValue", { value: String(defaults[def.key]) })}
          >
            {t("reset")}
          </button>
        )}
      </span>
    </div>
  );
}

function Rail() {
  const { overrides, defaults, reset, correlation, error } = useSettings();
  const t = useTranslations("settings");
  const changed = Object.keys(overrides);
  return (
    <>
      <RailGroup title={t("groupChanged")} aside={<span className="chip text-dim">{changed.length}</span>}>
        {!changed.length && <p className="type-body text-[9px]">{t("allAtDefaults")}</p>}
        {changed.map((k) => (
          <div key={k} className="grid grid-cols-[1fr_auto] items-center gap-2 py-0.5">
            <span className="font-mono text-[10.5px] text-ink">
              {k} <span className="text-muted">{String(defaults?.[k])}</span> → <span className="text-attention">{String(overrides[k])}</span>
            </span>
            <button type="button" onClick={() => reset(k)} className="font-mono text-[10px] text-accent hover:underline">
              {t("reset")}
            </button>
          </div>
        ))}
        {changed.length > 0 && (
          <div className="pt-1.5">
            <SecondaryButton onClick={() => reset()}>{t("resetAll")}</SecondaryButton>
          </div>
        )}
      </RailGroup>
      <RailGroup title={t("groupStatus")}>
        <p className="font-mono text-[10.5px] text-ink">
          {t("correlationsLabel")}{" "}
          {correlation ? (
            <span className={correlation.valid ? "text-gain" : "text-loss"}>{correlation.valid ? t("correlationsValid") : t("correlationsInvalid")}</span>
          ) : (
            <span className="text-dim">{t("correlationsChecking")}</span>
          )}
        </p>
        {error && <p className="font-mono text-[10px] text-loss">{error}</p>}
        <p className="type-body pt-1 text-[9px]">{t("statusNote")}</p>
      </RailGroup>
    </>
  );
}

function SettingsScreen({ children }: { children: ReactNode }) {
  const { defaults, error } = useSettings();
  const t = useTranslations("settings");
  const provenance = useProvenance();
  return (
    <Screen
      rail={<Rail />}
      bar={
        <Notice tone="attention" title={provenance.illustrative} role="note">
          {provenance.illustrativeDetail}
        </Notice>
      }
    >
      {defaults ? children : error ? <EmptyState title={t("unavailable")}>{error}</EmptyState> : <LoadingTiles />}
    </Screen>
  );
}

const Form = ({ defs }: { defs: Def[] }) => (
  <div className="grid gap-0.5">
    {defs.map((d) => (
      <SettingField key={d.key} def={d} />
    ))}
  </div>
);

export function DealDefaultsStep() {
  return (
    <SettingsScreen>
      <DealDefaults />
    </SettingsScreen>
  );
}

function DealDefaults() {
  const { effective } = useSettings();
  const { setFields } = useDeal();
  const t = useTranslations("settings");
  const apply = () => {
    const patch: Partial<DealInputs> = {};
    DEAL_DEFAULTS.forEach((d) => {
      const v = effective[d.key];
      if (d.input && typeof v === "number") (patch as Record<string, number>)[d.input] = v;
    });
    setFields(patch);
  };
  return (
    <Tiles>
      <Notice tone="info" title={t("startingValues")} className="col-span-12" actions={<PrimaryButton onClick={apply}>{t("applyToDeal")}</PrimaryButton>}>
        {t("startingValuesBody")}
      </Notice>
      <Tile span={6} title={t("tileEntryOperations")}>
        <Form defs={DEAL_DEFAULTS.slice(0, 9)} />
      </Tile>
      <Tile span={6} title={t("tileCapitalCashFlow")}>
        <Form defs={DEAL_DEFAULTS.slice(9)} />
      </Tile>
      <Tile span={6} title={t("tileSensitivity")}>
        <Form defs={SENSITIVITY} />
        <p className="type-body text-[9px]">{t("sensitivityNote")}</p>
      </Tile>
    </Tiles>
  );
}

export function FeesStep() {
  return (
    <SettingsScreen>
      <Fees />
    </SettingsScreen>
  );
}

function Fees() {
  const { label: mu } = useMoney();
  const { run } = useDeal();
  const t = useTranslations("settings");
  const r = run.result;
  return (
    <Tiles>
      <Tile span={6} title={t("tileFees")}>
        <Form defs={FEES} />
        <p className="type-body text-[9px]">{t("feesNote")}</p>
      </Tile>
      <Kpi title={t("kpiDealIrr")} value={fmtRate(r?.returns.irr)} sub={t("updatesAsYouEdit")} lead />
      <Kpi title={t("kpiFeesAtEntry")} value={fmtMoney(r ? -(r.equity_bridge.entry_costs ?? 0) : null)} sub={t("currentDealSub", { money: mu })} />
      <Kpi title={t("kpiEquityIn")} value={fmtMoney(r?.returns.entry_equity)} sub={t("currentDealSub", { money: mu })} />
    </Tiles>
  );
}

export function MonteCarloDefaultsStep() {
  return (
    <SettingsScreen>
      <MonteCarloDefaults />
    </SettingsScreen>
  );
}

function MonteCarloDefaults() {
  const t = useTranslations("settings");
  return (
    <Tiles>
      <Tile span={6} title={t("tileSimulationDefaults")}>
        <Form defs={MC} />
        <p className="type-body text-[9px]">{t("simulationDefaultsNote")}</p>
      </Tile>
      <Tile span={6} title={t("tileClipping")}>
        <Form defs={[{ key: "mc_clip_irr", spec: { unit: "", step: 1, decimals: 0 } }]} />
        <p className="type-body text-[9px]">{t("clippingNote")}</p>
      </Tile>
    </Tiles>
  );
}

export function CorrelationsStep() {
  return (
    <SettingsScreen>
      <Correlations />
    </SettingsScreen>
  );
}

function Correlations() {
  const { effective, set, correlation } = useSettings();
  const t = useTranslations("settings");
  const key = (a: string, b: string) => {
    const order = DRIVERS.map((d) => d.id);
    const [x, y] = order.indexOf(a) < order.indexOf(b) ? [a, b] : [b, a];
    return `corr_${x}_${y}`;
  };
  return (
    <Tiles>
      {correlation && !correlation.valid && (
        <Notice tone="loss" title={t("invalidMatrixTitle")} role="alert" className="col-span-12">
          {t("invalidMatrixBody")}
        </Notice>
      )}
      <Tile span={7} title={t("tileDriverCorrelations")} unit={t("editAboveDiagonal")}>
        <table className="w-full border-separate border-spacing-px">
          <thead>
            <tr>
              <th />
              {DRIVERS.map((d) => (
                <th key={d.id} scope="col" className="px-1 py-1 text-end font-mono text-[10.5px] font-normal text-muted">
                  {t(d.labelKey)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {DRIVERS.map((row, i) => (
              <tr key={row.id}>
                <th scope="row" className="type-input-label pe-2 text-start text-[9px] font-normal whitespace-nowrap">
                  {t(row.labelKey)}
                </th>
                {DRIVERS.map((col, j) => {
                  if (i === j)
                    return (
                      <td key={col.id} className="px-2 py-1 text-end font-mono text-[11px] text-dim">
                        {fmtNumber(1, 2)}
                      </td>
                    );
                  const k = key(row.id, col.id);
                  const v = effective[k];
                  if (typeof v !== "number") return <td key={col.id} />;
                  return j > i ? (
                    <td key={col.id} className="px-1 py-0.5">
                      <CellInput
                        label={t("correlationCell", { row: t(row.labelKey), column: t(col.labelKey) })}
                        value={v}
                        decimals={2}
                        onCommit={(n) => set(k, Math.max(-1, Math.min(1, n)))}
                      />
                    </td>
                  ) : (
                    <td key={col.id} className="px-2 py-1 text-end font-mono text-[11px] text-ink" style={{ background: heat(v, 0, 1) }}>
                      {fmtInput(v, 2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </Tile>
      <Tile span={5} title={t("tileMatrixUsed")} unit={t("fromTheApi")}>
        {correlation && (
          <table className="w-full border-separate border-spacing-px font-mono text-[11px]">
            <tbody>
              {correlation.matrix.map((r, i) => (
                <tr key={i}>
                  {r.map((v, j) => (
                    <td key={j} className="px-2 py-1 text-end" style={i === j ? { color: "var(--color-dim)" } : { background: heat(v, 0, 1) }}>
                      {fmtNumber(v, 2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="type-body text-[9px]">{t("matrixOrder")}</p>
      </Tile>
    </Tiles>
  );
}

export function PresetsStep() {
  return (
    <SettingsScreen>
      <Presets />
    </SettingsScreen>
  );
}

function Presets() {
  const { effective, set } = useSettings();
  const t = useTranslations("settings");
  const units = useTranslations("units");
  const headers = [t("presetColGrowth"), t("presetColExit"), t("presetColRate"), t("presetColMargin"), t("presetColFloor")];
  // "x" multiplies; "pts" adds percentage points, and is a word
  const unitLabel = (unit: string) => (unit === "pts" ? units("pts") : unit);
  return (
    <Tiles>
      <Tile span={12} title={t("tilePresets")} unit={t("presetsUnit")}>
        <table className="w-full border-separate border-spacing-x-1 border-spacing-y-0.5">
          <thead>
            <tr>
              <th />
              {headers.map((h) => (
                <th key={h} scope="col" className="px-1 py-1 text-end font-mono text-[10.5px] font-normal text-muted">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {PRESETS.map((p) => (
              <tr key={p.id}>
                <th scope="row" className="type-input-label pe-2 text-start text-[9.5px] font-normal">
                  {t(p.labelKey)}
                </th>
                {p.cols.map(([k, unit], i) => (
                  <td key={k}>
                    <div className="grid grid-cols-[1fr_24px] items-center gap-1">
                      <CellInput
                        label={t("presetCell", { preset: t(p.labelKey), column: headers[i] })}
                        value={Number(effective[k])}
                        decimals={2}
                        onCommit={(n) => set(k, n)}
                      />
                      <span className="font-mono text-[10px] text-[#56636a]">{unitLabel(unit)}</span>
                    </div>
                  </td>
                ))}
                <td>
                  {"floor" in p ? (
                    <div className="grid grid-cols-[1fr_24px] items-center gap-1">
                      <CellInput
                        label={t("presetFloorCell", { preset: t(p.labelKey) })}
                        value={Number(effective[p.floor])}
                        decimals={1}
                        onCommit={(n) => set(p.floor, n)}
                      />
                      <span className="font-mono text-[10px] text-[#56636a]">%</span>
                    </div>
                  ) : (
                    <span className="block pe-7 text-end font-mono text-[10px] text-dim">{t("presetNoFloor")}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="type-body text-[9px]">{t("presetsNote")}</p>
      </Tile>
    </Tiles>
  );
}
