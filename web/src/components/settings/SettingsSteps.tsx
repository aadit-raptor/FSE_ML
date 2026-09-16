"use client";

import type { ReactNode } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { heat } from "@/components/charts/HeatTable";
import { CellInput } from "@/components/ui/CellInput";
import { NumberField } from "@/components/ui/NumberField";
import { EmptyState, LoadingTiles, Notice, PrimaryButton, RailGroup, Screen, SecondaryButton, Switch } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import type { DealInputs } from "@/lib/deal/fields";
import type { FieldSpec } from "@/lib/fields";
import { fmtInput, fmtMoney, fmtRate } from "@/lib/format";
import { MAX_SIMULATION_PATHS } from "@/lib/limits";
import { ILLUSTRATIVE, ILLUSTRATIVE_DETAIL } from "@/lib/provenance";

import { useSettings } from "./SettingsProvider";

type Def = { key: string; label: string; spec: Omit<FieldSpec, "label"> };

const pct = (step = 0.5, decimals = 1, extra: Partial<FieldSpec> = {}) => ({ unit: "%", step, decimals, ...extra });

/** Deal default key -> deal input it seeds. */
const DEAL_DEFAULTS: (Def & { input?: keyof DealInputs })[] = [
  { key: "def_ebitda", input: "ebitda", label: "EBITDA", spec: { unit: "$M", step: 5, decimals: 1, min: 0, exclusiveMin: true } },
  { key: "def_entry_mult", input: "entry_mult", label: "Entry multiple", spec: { unit: "x", step: 0.5, decimals: 1, min: 0, exclusiveMin: true } },
  { key: "def_exit_mult", input: "exit_mult", label: "Exit multiple", spec: { unit: "x", step: 0.5, decimals: 1, min: 0, exclusiveMin: true } },
  { key: "def_hold", input: "hold", label: "Hold", spec: { unit: "yr", step: 1, decimals: 0, min: 1, max: 15, integer: true } },
  { key: "def_growth", input: "growth", label: "Revenue growth", spec: pct() },
  { key: "def_gross_margin", input: "gross_margin", label: "Gross margin", spec: pct(1) },
  { key: "def_opex", input: "opex", label: "Opex", spec: pct(1) },
  { key: "def_tax", input: "tax", label: "Tax rate", spec: pct(1) },
  { key: "def_da", input: "da", label: "D&A", spec: pct() },
  { key: "def_debt_pct", input: "debt_pct", label: "Debt / EV", spec: pct(1, 1, { min: 0, max: 99 }) },
  { key: "def_senior_pct", input: "senior_pct", label: "Senior share", spec: pct(1, 1, { min: 0, max: 100 }) },
  { key: "def_base_rate", input: "base_rate", label: "Senior rate", spec: pct(0.25, 2, { min: 0 }) },
  { key: "def_mezz_spread", input: "mezz_spread", label: "Mezz spread", spec: pct(0.25, 2, { min: 0 }) },
  { key: "def_capex", input: "capex", label: "Capex", spec: pct() },
  { key: "def_nwc", input: "nwc", label: "NWC change", spec: pct(0.25, 2) },
  { key: "def_mincash", input: "mincash", label: "Minimum cash", spec: { unit: "$M", step: 5, decimals: 1, min: 0 } },
  { key: "def_senior_amort", label: "Senior amortisation", spec: pct() },
];

const SENSITIVITY: Def[] = [
  { key: "sens_em_min", label: "Exit multiple from", spec: { unit: "x", step: 0.5, decimals: 1, min: 0 } },
  { key: "sens_em_max", label: "Exit multiple to", spec: { unit: "x", step: 0.5, decimals: 1, min: 0 } },
  { key: "sens_em_steps", label: "Exit multiple steps", spec: { unit: "", step: 1, decimals: 0, min: 2, integer: true } },
  { key: "sens_hp_min", label: "Hold from", spec: { unit: "yr", step: 1, decimals: 0, min: 1, integer: true } },
  { key: "sens_hp_max", label: "Hold to", spec: { unit: "yr", step: 1, decimals: 0, min: 1, integer: true } },
];

const FEES: Def[] = [
  { key: "tx_fee_pct", label: "Transaction fees, % of EV", spec: pct(0.1, 2, { min: 0 }) },
  { key: "fin_fee_pct", label: "Financing fees, % of debt", spec: pct(0.1, 2, { min: 0 }) },
  { key: "other_uses", label: "Other uses", spec: { unit: "$M", step: 1, decimals: 1, min: 0 } },
];

const MC: Def[] = [
  { key: "mc_n", label: "Paths", spec: { unit: "n", step: 5000, decimals: 0, min: 1000, max: MAX_SIMULATION_PATHS, integer: true } },
  { key: "mc_hurdle", label: "Hurdle IRR", spec: pct(1, 1, { min: 0 }) },
  { key: "mc_growth_mean", label: "Growth mean", spec: pct() },
  { key: "mc_growth_std", label: "Growth std dev", spec: pct(0.5, 1, { min: 0.1 }) },
  { key: "mc_exit_mean", label: "Exit multiple mean", spec: { unit: "x", step: 0.5, decimals: 1, min: 1 } },
  { key: "mc_exit_std", label: "Exit multiple std dev", spec: { unit: "x", step: 0.25, decimals: 2, min: 0.1 } },
  { key: "mc_rate_mean", label: "Rate mean", spec: pct(0.25, 2, { min: 0 }) },
  { key: "mc_rate_std", label: "Rate std dev", spec: pct(0.25, 2, { min: 0.1 }) },
  { key: "mc_gm_mean", label: "Gross margin mean", spec: pct(1, 1, { min: 1, max: 99 }) },
  { key: "mc_gm_std", label: "Gross margin std dev", spec: pct(0.5, 1, { min: 0.1 }) },
  { key: "mc_n_passes", label: "Interest passes", spec: { unit: "", step: 1, decimals: 0, min: 1, max: 10, integer: true } },
];

const DRIVERS = [
  { id: "g", label: "Growth" },
  { id: "em", label: "Exit multiple" },
  { id: "ir", label: "Interest" },
  { id: "gm", label: "Gross margin" },
  { id: "sh", label: "EBITDA shock" },
];

const PRESETS = [
  { id: "bull", label: "Bull", cols: [["bull_growth_mult", "×"], ["bull_exit_mult", "×"], ["bull_rate_mult", "×"], ["bull_margin_mult", "×"]] },
  { id: "rec", label: "Recession", cols: [["rec_growth_adj", "pts"], ["rec_exit_mult", "×"], ["rec_rate_mult", "×"], ["rec_margin_mult", "×"]], floor: "rec_growth_floor" },
  { id: "stag", label: "Stagflation", cols: [["stag_growth_adj", "pts"], ["stag_exit_mult", "×"], ["stag_rate_mult", "×"], ["stag_margin_mult", "×"]], floor: "stag_growth_floor" },
] as const;

function SettingField({ def }: { def: Def }) {
  const { effective, defaults, set, reset, overrides } = useSettings();
  const v = effective[def.key];
  const changed = def.key in overrides;
  if (typeof v === "boolean") {
    return (
      <div className="flex items-center justify-between py-1">
        <Switch checked={v} onChange={(on) => set(def.key, on)} label={def.label} />
      </div>
    );
  }
  if (typeof v !== "number") return null;
  return (
    <div className="grid grid-cols-[1fr_auto] items-start gap-2">
      <NumberField spec={{ ...def.spec, label: def.label }} value={v} onCommit={(n) => set(def.key, n)} changed={changed} />
      <span className="flex h-[22px] w-[74px] items-center justify-end gap-1.5">
        {changed && defaults && (
          <button type="button" onClick={() => reset(def.key)} className="font-mono text-[10px] text-accent hover:underline" title={`Default ${String(defaults[def.key])}`}>
            reset
          </button>
        )}
      </span>
    </div>
  );
}

function Rail() {
  const { overrides, defaults, reset, correlation, error } = useSettings();
  const changed = Object.keys(overrides);
  return (
    <>
      <RailGroup title="Changed settings" aside={<span className="chip text-dim">{changed.length}</span>}>
        {!changed.length && <p className="type-body text-[9px]">All settings are at their defaults.</p>}
        {changed.map((k) => (
          <div key={k} className="grid grid-cols-[1fr_auto] items-center gap-2 py-0.5">
            <span className="font-mono text-[10.5px] text-ink">
              {k} <span className="text-muted">{String(defaults?.[k])}</span> → <span className="text-attention">{String(overrides[k])}</span>
            </span>
            <button type="button" onClick={() => reset(k)} className="font-mono text-[10px] text-accent hover:underline">
              reset
            </button>
          </div>
        ))}
        {changed.length > 0 && (
          <div className="pt-1.5">
            <SecondaryButton onClick={() => reset()}>Reset all</SecondaryButton>
          </div>
        )}
      </RailGroup>
      <RailGroup title="Status">
        <p className="font-mono text-[10.5px] text-ink">
          Correlations{" "}
          {correlation ? (
            <span className={correlation.valid ? "text-gain" : "text-loss"}>{correlation.valid ? "valid" : "invalid"}</span>
          ) : (
            <span className="text-dim">checking</span>
          )}
        </p>
        {error && <p className="font-mono text-[10px] text-loss">{error}</p>}
        <p className="type-body pt-1 text-[9px]">Saved in this browser. Changes apply to the next deal, simulation and backtest run.</p>
      </RailGroup>
    </>
  );
}

function SettingsScreen({ children }: { children: ReactNode }) {
  const { defaults, error } = useSettings();
  return (
    <Screen
      rail={<Rail />}
      bar={
        <Notice tone="attention" title={ILLUSTRATIVE} role="note">
          {ILLUSTRATIVE_DETAIL}
        </Notice>
      }
    >
      {defaults ? children : error ? <EmptyState title="Settings unavailable">{error}</EmptyState> : <LoadingTiles />}
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
      <Notice
        tone="info"
        title="Starting values"
        className="col-span-12"
        actions={<PrimaryButton onClick={apply}>Apply to current deal</PrimaryButton>}
      >
        These seed a new deal. Applying them replaces the current deal&apos;s inputs, which then rerun.
      </Notice>
      <Tile span={6} title="Entry, exit and operations">
        <Form defs={DEAL_DEFAULTS.slice(0, 9)} />
      </Tile>
      <Tile span={6} title="Capital structure and cash flow">
        <Form defs={DEAL_DEFAULTS.slice(9)} />
      </Tile>
      <Tile span={6} title="Sensitivity grid">
        <Form defs={SENSITIVITY} />
        <p className="type-body text-[9px]">Rows and columns of the IRR sensitivity on the Returns screen.</p>
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
  const { run } = useDeal();
  const r = run.result;
  return (
    <Tiles>
      <Tile span={6} title="Fees and other uses">
        <Form defs={FEES} />
        <p className="type-body text-[9px]">Funded by sponsor equity at close, so higher fees lower IRR. Used by the deal, Monte Carlo and backtests.</p>
      </Tile>
      <Kpi title="Current deal IRR" value={fmtRate(r?.returns.irr)} sub="updates as you edit" lead />
      <Kpi title="Fees at entry" value={fmtMoney(r ? -(r.equity_bridge.entry_costs ?? 0) : null)} sub="$M, current deal" />
      <Kpi title="Equity in" value={fmtMoney(r?.returns.entry_equity)} sub="$M, current deal" />
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
  return (
    <Tiles>
      <Tile span={6} title="Simulation defaults">
        <Form defs={MC} />
        <p className="type-body text-[9px]">
          The Monte Carlo screen starts from these. Any value you edit there takes precedence. Interest passes is read by the simulation engine.
        </p>
      </Tile>
      <Tile span={6} title="Clipping">
        <Form defs={[{ key: "mc_clip_irr", label: "Clip IRR to -100%...500%", spec: { unit: "", step: 1, decimals: 0 } }]} />
        <p className="type-body text-[9px]">Limits extreme simulated IRRs so a few paths don&apos;t distort the mean. Applies to Monte Carlo and backtests.</p>
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
  const key = (a: string, b: string) => {
    const order = DRIVERS.map((d) => d.id);
    const [x, y] = order.indexOf(a) < order.indexOf(b) ? [a, b] : [b, a];
    return `corr_${x}_${y}`;
  };
  return (
    <Tiles>
      {correlation && !correlation.valid && (
        <Notice tone="loss" title="Not a valid correlation matrix" role="alert" className="col-span-12">
          These correlations can&apos;t all hold at once (the matrix isn&apos;t positive semi-definite). Monte Carlo will refuse to run until they&apos;re changed.
        </Notice>
      )}
      <Tile span={7} title="Driver correlations" unit="edit above the diagonal">
        <table className="w-full border-separate border-spacing-px">
          <thead>
            <tr>
              <th />
              {DRIVERS.map((d) => (
                <th key={d.id} scope="col" className="px-1 py-1 text-right font-mono text-[10.5px] font-normal text-muted">
                  {d.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {DRIVERS.map((row, i) => (
              <tr key={row.id}>
                <th scope="row" className="type-input-label pr-2 text-left text-[9px] font-normal whitespace-nowrap">
                  {row.label}
                </th>
                {DRIVERS.map((col, j) => {
                  if (i === j)
                    return (
                      <td key={col.id} className="px-2 py-1 text-right font-mono text-[11px] text-dim">
                        1.00
                      </td>
                    );
                  const k = key(row.id, col.id);
                  const v = effective[k];
                  if (typeof v !== "number") return <td key={col.id} />;
                  return j > i ? (
                    <td key={col.id} className="px-1 py-0.5">
                      <CellInput label={`Correlation ${row.label} and ${col.label}`} value={v} decimals={2} onCommit={(n) => set(k, Math.max(-1, Math.min(1, n)))} />
                    </td>
                  ) : (
                    <td key={col.id} className="px-2 py-1 text-right font-mono text-[11px] text-ink" style={{ background: heat(v, 0, 1) }}>
                      {fmtInput(v, 2)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </Tile>
      <Tile span={5} title="Matrix used" unit="from the API">
        {correlation && (
          <table className="w-full border-separate border-spacing-px font-mono text-[11px]">
            <tbody>
              {correlation.matrix.map((r, i) => (
                <tr key={i}>
                  {r.map((v, j) => (
                    <td key={j} className="px-2 py-1 text-right" style={i === j ? { color: "var(--color-dim)" } : { background: heat(v, 0, 1) }}>
                      {v.toFixed(2)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
        <p className="type-body text-[9px]">Order: growth, exit multiple, interest, gross margin, EBITDA shock.</p>
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
  const headers = ["Growth", "Exit multiple", "Interest rate", "Gross margin", "Growth floor"];
  return (
    <Tiles>
      <Tile span={12} title="Scenario presets" unit="applied to the Monte Carlo base case">
        <table className="w-full border-separate border-spacing-x-1 border-spacing-y-0.5">
          <thead>
            <tr>
              <th />
              {headers.map((h) => (
                <th key={h} scope="col" className="px-1 py-1 text-right font-mono text-[10.5px] font-normal text-muted">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {PRESETS.map((p) => (
              <tr key={p.id}>
                <th scope="row" className="type-input-label pr-2 text-left text-[9.5px] font-normal">
                  {p.label}
                </th>
                {p.cols.map(([k, unit]) => (
                  <td key={k}>
                    <div className="grid grid-cols-[1fr_24px] items-center gap-1">
                      <CellInput label={`${p.label} ${k}`} value={Number(effective[k])} decimals={2} onCommit={(n) => set(k, n)} />
                      <span className="font-mono text-[10px] text-[#56636a]">{unit}</span>
                    </div>
                  </td>
                ))}
                <td>
                  {"floor" in p ? (
                    <div className="grid grid-cols-[1fr_24px] items-center gap-1">
                      <CellInput label={`${p.label} growth floor`} value={Number(effective[p.floor])} decimals={1} onCommit={(n) => set(p.floor, n)} />
                      <span className="font-mono text-[10px] text-[#56636a]">%</span>
                    </div>
                  ) : (
                    <span className="block pr-7 text-right font-mono text-[10px] text-dim">none</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="type-body text-[9px]">× multiplies the base case; pts adds percentage points to growth, which then can&apos;t fall below the floor.</p>
      </Tile>
    </Tiles>
  );
}
