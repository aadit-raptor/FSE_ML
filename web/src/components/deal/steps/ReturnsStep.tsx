"use client";

import { DataTable } from "@/components/charts/DataTable";
import { SensitivityTable } from "@/components/charts/HeatTable";
import { StackedBars } from "@/components/charts/StackedBars";
import { Waterfall } from "@/components/charts/Waterfall";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { fmtMoney, fmtMultiple } from "@/lib/format";

import { useDeal } from "../DealProvider";
import { DealField, DealScreen, LoadingTiles, RailGroup } from "../DealScreen";
import { baseCell, debtSeries, hurdleSub, totals, yearLabels } from "./shared";

export function ReturnsStep() {
  const { run } = useDeal();
  return (
    <DealScreen
      rail={
        <>
          <RailGroup title="Entry and exit">
            <DealField name="ebitda" />
            <DealField name="entry_mult" />
            <DealField name="exit_mult" />
            <DealField name="hold" />
          </RailGroup>
          <RailGroup title="Operations">
            <DealField name="growth" />
            <DealField name="gross_margin" />
            <DealField name="opex" />
            <DealField name="tax" />
          </RailGroup>
          <RailGroup title="Capital structure">
            <DealField name="debt_pct" />
            <DealField name="senior_pct" />
            <DealField name="base_rate" />
            <DealField name="mezz_spread" />
          </RailGroup>
        </>
      }
    >
      {run.result ? <ReturnsResults /> : <LoadingTiles />}
    </DealScreen>
  );
}

function ReturnsResults() {
  const { inputs, run, hurdle } = useDeal();
  const res = run.result!;
  const r = res.returns;
  const om = res.operating_model;
  const base = baseCell(res, inputs.exit_mult, inputs.hold);
  const begin = totals(res, "total_beginning_debt");
  const years = yearLabels(om.revenue?.length ?? 0);

  return (
    <Tiles>
      <Kpi title="IRR" value={r.irr == null ? "n/a" : `${(r.irr * 100).toFixed(1)}%`} lead {...hurdleSub(r.irr, hurdle)} />
      <Kpi title="MOIC" value={fmtMultiple(r.moic)} sub={`${r.holding_period ?? inputs.hold} yr hold`} />
      <Kpi title="Equity in" value={fmtMoney(r.entry_equity)} sub="$M at close" />
      <Kpi title="Equity out" value={fmtMoney(r.net_exit_equity)} sub="$M at exit" />
      <Kpi title="Exit EV" value={fmtMoney(r.exit_ev)} sub={`${fmtMultiple(r.exit_multiple, 1)} on ${fmtMoney(r.exit_ebitda)} EBITDA`} />
      <Kpi title="Net debt at exit" value={fmtMoney(r.net_debt_at_exit)} sub={`from ${fmtMoney(begin[0])} $M`} />

      <Tile span={6} title="Equity bridge" unit="$M">
        <Waterfall
          label={`Equity bridge from ${fmtMoney(r.entry_equity)} to ${fmtMoney(r.net_exit_equity)} $M`}
          // `key` is the API's short axis label (Entry, Fees, EBITDA growth, Multiple, Deleverage, Exit)
          steps={res.bridge_steps.map((s) => ({ label: s.key, value: s.value ?? 0, isTotal: s.is_total }))}
        />
      </Tile>
      <Tile span={6} title="IRR sensitivity" unit="exit multiple / hold">
        <SensitivityTable
          exitMultiples={res.exit_sensitivity.exit_multiples}
          holds={res.exit_sensitivity.holding_periods}
          table={res.exit_sensitivity.table}
          hurdle={hurdle}
          baseRow={base.row}
          baseCol={base.col}
        />
        <p className="type-body text-[9px]">Every cell is a full model run for that exit multiple and hold. Ranges are set in Settings, Deal defaults.</p>
      </Tile>
      <Tile span={6} title="Debt paydown" unit="by tranche, $M">
        <StackedBars {...debtSeries(res)} />
      </Tile>
      <Tile span={6} title="Operating summary" unit="$M">
        <DataTable
          caption="Operating summary by year"
          columns={years}
          rows={[
            { label: "Revenue", values: om.revenue ?? [] },
            { label: "EBITDA", values: om.ebitda ?? [] },
            { label: "EBITDA margin", values: om.ebitda_margin ?? [], kind: "rate" },
            { label: "Interest", values: om.interest_expense ?? [], kind: "outflow" },
            { label: "Levered FCF", values: res.cash_flow.levered_fcf ?? [], total: true },
          ]}
        />
      </Tile>
    </Tiles>
  );
}
