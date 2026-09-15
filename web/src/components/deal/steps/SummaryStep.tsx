"use client";

import Link from "next/link";

import { DataTable } from "@/components/charts/DataTable";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { multiplesFromPct } from "@/lib/deal/capital";
import { FIELDS, type NumericDealKey } from "@/lib/deal/fields";
import { fmtDelta, fmtInput, fmtMoney, fmtMultiple, fmtPct } from "@/lib/format";

import { useDeal } from "../DealProvider";
import { DealScreen, LoadingTiles, RailGroup } from "../DealScreen";
import { hurdleSub, totals, yearLabels } from "./shared";

const ASSUMPTIONS: { title: string; keys: NumericDealKey[] }[] = [
  { title: "Entry and exit", keys: ["ebitda", "entry_mult", "exit_mult", "hold"] },
  { title: "Operations", keys: ["growth", "gross_margin", "opex", "tax", "da"] },
  { title: "Capital structure", keys: ["debt_pct", "senior_pct", "base_rate", "mezz_spread"] },
  { title: "Cash flow", keys: ["capex", "nwc", "mincash"] },
];

export function SummaryStep() {
  const { inputs, run } = useDeal();
  const { seniorX, mezzX } = multiplesFromPct(inputs.entry_mult, inputs.debt_pct, inputs.senior_pct);
  return (
    <DealScreen
      rail={
        <>
          {ASSUMPTIONS.map((g) => (
            <RailGroup key={g.title} title={g.title}>
              <dl className="grid grid-cols-[1fr_auto] gap-x-2 gap-y-1">
                {g.keys.map((k) => (
                  <div key={k} className="contents">
                    <dt className="type-input-label">{FIELDS[k].label}</dt>
                    <dd className="text-right font-mono text-[11.5px] text-ink">
                      {fmtInput(inputs[k], FIELDS[k].decimals)} <span className="text-[10px] text-[#56636a]">{FIELDS[k].unit}</span>
                    </dd>
                  </div>
                ))}
              </dl>
            </RailGroup>
          ))}
          <div className="grid gap-2 px-3.5 py-3">
            <p className="type-body">
              Debt {fmtMultiple(seniorX, 1)} senior and {fmtMultiple(mezzX, 1)} mezz. Working capital {inputs.wsp_mode ? "from days" : "as % of revenue"}.
            </p>
            <Link href="/deal/inputs" className="type-action-secondary justify-self-start px-2.5 py-1.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)]">
              Edit inputs
            </Link>
          </div>
        </>
      }
    >
      {run.result ? <SummaryResults /> : <LoadingTiles />}
    </DealScreen>
  );
}

function SummaryResults() {
  const { run, hurdle } = useDeal();
  const res = run.result!;
  const om = res.operating_model;
  const cf = res.cash_flow;
  const r = res.returns;
  const years = yearLabels(om.revenue?.length ?? 0);
  const br = res.equity_bridge;

  return (
    <Tiles>
      <Kpi title="IRR" value={r.irr == null ? "n/a" : `${(r.irr * 100).toFixed(1)}%`} lead {...hurdleSub(r.irr, hurdle)} />
      <Kpi title="MOIC" value={fmtMultiple(r.moic)} sub={`${r.holding_period} yr hold`} />
      <Kpi title="Equity in" value={fmtMoney(r.entry_equity)} sub="$M" />
      <Kpi title="Equity out" value={fmtMoney(r.net_exit_equity)} sub="$M" />
      <Kpi title="Total gain" value={fmtMoney(br.total_gain)} sub="$M" />
      <Kpi title="Bridge residual" value={fmtMoney(br.residual)} sub="$M, should be 0" tone={Math.abs(br.residual ?? 0) > 0.05 ? "attention" : undefined} />

      <Tile span={12} title="Income statement" unit="$M">
        <DataTable
          caption="Income statement by year"
          columns={years}
          rows={[
            { label: "Revenue", values: om.revenue ?? [] },
            { label: "Gross profit", values: om.gross_profit ?? [] },
            { label: "Opex", values: om.opex ?? [], kind: "outflow" },
            { label: "EBITDA", values: om.ebitda ?? [], total: true },
            { label: "D&A", values: om.da ?? [], kind: "outflow" },
            { label: "EBIT", values: om.ebit ?? [] },
            { label: "Interest expense", values: om.interest_expense ?? [], kind: "outflow" },
            { label: "Pre-tax income", values: om.ebt ?? [] },
            { label: "Taxes", values: om.taxes ?? [], kind: "outflow" },
            { label: "Net income", values: om.net_income ?? [], total: true },
            { label: "EBITDA margin", values: om.ebitda_margin ?? [], kind: "rate" },
          ]}
        />
      </Tile>
      <Tile span={6} title="Cash flow" unit="$M">
        <DataTable
          caption="Cash flow by year"
          columns={years}
          rows={[
            { label: "Net income", values: cf.net_income ?? [] },
            { label: "D&A", values: cf.da ?? [] },
            { label: "Capex", values: cf.capex ?? [], kind: "outflow" },
            { label: "Change in NWC", values: cf.delta_nwc ?? [], kind: "outflow" },
            { label: "Levered FCF", values: cf.levered_fcf ?? [], total: true },
          ]}
        />
      </Tile>
      <Tile span={6} title="Debt" unit="all tranches, $M">
        <DataTable
          caption="Debt schedule totals by year"
          columns={years}
          rows={[
            { label: "Opening", values: totals(res, "total_beginning_debt") },
            { label: "Mandatory", values: totals(res, "total_mandatory_repayment"), kind: "outflow" },
            { label: "Cash sweep", values: totals(res, "total_cash_sweep"), kind: "outflow" },
            { label: "Closing", values: totals(res, "total_ending_debt"), total: true },
            { label: "Interest", values: totals(res, "total_interest_expense"), kind: "outflow" },
          ]}
        />
      </Tile>
      <Tile span={12} title="Equity bridge" unit="$M and share of gain">
        <table className="w-full border-collapse font-mono text-[11.5px]">
          <caption className="sr-only">Equity bridge</caption>
          <tbody>
            {res.bridge_steps.map((s) => (
              <tr key={s.key} className={s.is_total ? "text-bright" : "text-ink"}>
                <th scope="row" className="type-input-label border-b border-grid py-1.5 text-left font-normal text-soft">
                  {s.label}
                </th>
                <td className={`border-b border-grid py-1.5 text-right ${!s.is_total && (s.value ?? 0) < 0 ? "text-loss" : ""} ${s.is_total ? "font-semibold" : ""}`}>
                  {s.is_total ? fmtMoney(s.value) : fmtDelta(s.value)}
                </td>
                <td className="w-24 border-b border-grid py-1.5 text-right text-muted">{s.pct_of_gain == null ? "" : fmtPct(s.pct_of_gain)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Tile>
    </Tiles>
  );
}
