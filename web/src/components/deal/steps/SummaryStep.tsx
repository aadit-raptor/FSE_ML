"use client";

import Link from "next/link";
import { useState } from "react";

import { DataTable, type Row } from "@/components/charts/DataTable";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { NumberField } from "@/components/ui/NumberField";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { useMoney } from "@/components/ui/MoneyScope";
import { multiplesFromPct } from "@/lib/deal/capital";
import { FIELDS, type NumericDealKey } from "@/lib/deal/fields";
import { downloadWorkbook, sheet, tableSheet } from "@/lib/export";
import type { FieldSpec } from "@/lib/fields";
import { fmtDelta, fmtInput, fmtMoney, fmtMultiple, fmtPct } from "@/lib/format";
import { fieldUnit } from "@/lib/money";

import { useDeal } from "../DealProvider";
import { DealScreen, LoadingTiles, RailGroup } from "../DealScreen";
import { hurdleSub, totals, yearLabels } from "./shared";

const SBC_SPEC: FieldSpec = { label: "Stock-based comp", unit: "%", step: 0.5, decimals: 1, min: 0, max: 50 };

const ASSUMPTIONS: { title: string; keys: NumericDealKey[] }[] = [
  { title: "Entry and exit", keys: ["ebitda", "entry_mult", "exit_mult", "hold"] },
  { title: "Operations", keys: ["growth", "gross_margin", "opex", "tax", "da"] },
  { title: "Capital structure", keys: ["debt_pct", "senior_pct", "base_rate", "mezz_spread"] },
  { title: "Cash flow", keys: ["capex", "nwc", "mincash"] },
];

export function SummaryStep() {
  const { inputs, run, money } = useDeal();
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
                      {fmtInput(inputs[k], FIELDS[k].decimals)} <span className="text-[10px] text-[#56636a]">{fieldUnit(FIELDS[k].unit, money)}</span>
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
  const { label: mu, money } = useMoney();
  const { run, hurdle, inputs } = useDeal();
  const [sbcPct, setSbcPct] = useState(2);
  const res = run.result!;
  const om = res.operating_model;
  const cf = res.cash_flow;
  const r = res.returns;
  const years = yearLabels(om.revenue?.length ?? 0);
  const br = res.equity_bridge;
  const revenue = (om.revenue ?? []).map((v) => v ?? 0);
  const cogs = (om.cogs ?? []).map((v) => Math.abs(v ?? 0));

  const incomeRows: Row[] = [
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
  ];
  const cashRows: Row[] = [
    { label: "Net income", values: cf.net_income ?? [] },
    { label: "D&A", values: cf.da ?? [] },
    { label: "Capex", values: cf.capex ?? [], kind: "outflow" },
    { label: "Change in NWC", values: cf.delta_nwc ?? [], kind: "outflow" },
    { label: "Levered FCF", values: cf.levered_fcf ?? [], total: true },
  ];
  const debtRows: Row[] = [
    { label: "Opening", values: totals(res, "total_beginning_debt") },
    { label: "Mandatory", values: totals(res, "total_mandatory_repayment"), kind: "outflow" },
    { label: "Cash sweep", values: totals(res, "total_cash_sweep"), kind: "outflow" },
    { label: "Closing", values: totals(res, "total_ending_debt"), total: true },
    { label: "Interest", values: totals(res, "total_interest_expense"), kind: "outflow" },
  ];

  // PP&E roll-forward. The deal model has no balance sheet, so the opening
  // balance is an estimate (3x year-one capex), as on the Streamlit summary.
  const capex = (cf.capex ?? []).map((v) => Math.abs(v ?? 0));
  const da = (om.da ?? []).map((v) => Math.abs(v ?? 0));
  const ppeBegin: number[] = [];
  const ppeEnd: number[] = [];
  let ppe = (capex[0] ?? 0) * 3;
  capex.forEach((c, i) => {
    ppeBegin.push(ppe);
    ppe = ppe + c - (da[i] ?? 0);
    ppeEnd.push(ppe);
  });
  const ppeRows: Row[] = [
    { label: "Opening PP&E (est.)", values: ppeBegin },
    { label: "Capex", values: capex },
    { label: "Depreciation", values: da, kind: "outflow" },
    { label: "Closing PP&E", values: ppeEnd, total: true },
  ];

  const ar = revenue.map((v) => (v * inputs.ar_days) / 365);
  const inv = cogs.map((v) => (v * inputs.inv_days) / 365);
  const ap = cogs.map((v) => (v * inputs.ap_days) / 365);
  const wcRows: Row[] = [
    { label: "Receivables", values: ar },
    { label: "Inventory", values: inv },
    { label: "Payables", values: ap, kind: "outflow" },
    { label: "Net working capital", values: ar.map((v, i) => v + (inv[i] ?? 0) - (ap[i] ?? 0)), total: true },
  ];

  const sbc = revenue.map((v) => (v * sbcPct) / 100);
  const adj = (om.ebitda ?? []).map((v, i) => (v ?? 0) + (sbc[i] ?? 0));
  const adjRows: Row[] = [
    { label: "EBITDA", values: om.ebitda ?? [] },
    { label: "Stock-based comp", values: sbc },
    { label: "Adjusted EBITDA", values: adj, total: true },
    { label: "Adjusted margin", values: adj.map((v, i) => (revenue[i] ? v / revenue[i] : NaN)), kind: "rate" },
  ];

  const bridgeSheet = sheet(
    "Equity bridge",
    ["Component", `Value (${mu})`, "% of gain"],
    res.bridge_steps.map((s) => [s.label, s.value ?? null, s.pct_of_gain ?? null]),
  );
  const allSheets = () => [
    tableSheet("P&L", years, incomeRows),
    tableSheet("Cash flow", years, cashRows),
    tableSheet("Debt schedule", years, debtRows),
    ...Object.entries(res.tranches).map(([name, rows]) =>
      sheet(
        name,
        ["Year", "Opening", "Mandatory", "Cash sweep", "Closing", "Interest", "Rate (%)"],
        rows.map((t) => [
          t.year,
          t.beginning_balance ?? null,
          t.mandatory_repayment ?? null,
          t.cash_sweep ?? null,
          t.ending_balance ?? null,
          t.interest_expense ?? null,
          t.interest_rate == null ? null : t.interest_rate * 100,
        ]),
      ),
    ),
    bridgeSheet,
    tableSheet("PP&E", years, ppeRows),
    ...(inputs.wsp_mode ? [tableSheet("Working capital", years, wcRows)] : []),
    tableSheet("Adjusted EBITDA", years, adjRows),
  ];

  return (
    <Tiles>
      <Kpi title="IRR" value={r.irr == null ? "n/a" : `${(r.irr * 100).toFixed(1)}%`} lead {...hurdleSub(r.irr, hurdle)} />
      <Kpi title="MOIC" value={fmtMultiple(r.moic)} sub={`${r.holding_period} yr hold`} />
      <Kpi title="Equity in" value={fmtMoney(r.entry_equity)} sub={mu} />
      <Kpi title="Equity out" value={fmtMoney(r.net_exit_equity)} sub={mu} />
      <Kpi title="Total gain" value={fmtMoney(br.total_gain)} sub={mu} />
      <Kpi title="Bridge residual" value={fmtMoney(br.residual)} sub={`${mu}, should be 0`} tone={Math.abs(br.residual ?? 0) > 0.05 ? "attention" : undefined} />

      <div className="col-span-12 flex items-center justify-between gap-4 bg-canvas px-3 py-2">
        <p className="type-body">Every table on this page, plus each tranche&apos;s schedule, in one workbook.</p>
        <DownloadButton label="All tables" onDownload={() => downloadWorkbook("lbo_summary.xlsx", allSheets(), money)} />
      </div>

      <Tile span={12} title="Income statement" unit={mu} action={<DownloadButton onDownload={() => downloadWorkbook("pl.xlsx", [tableSheet("P&L", years, incomeRows)], money)} />}>
        <DataTable caption="Income statement by year" columns={years} rows={incomeRows} />
      </Tile>
      <Tile span={6} title="Cash flow" unit={mu} action={<DownloadButton onDownload={() => downloadWorkbook("cashflow.xlsx", [tableSheet("Cash flow", years, cashRows)], money)} />}>
        <DataTable caption="Cash flow by year" columns={years} rows={cashRows} />
      </Tile>
      <Tile span={6} title="Debt" unit={`all tranches, ${mu}`} action={<DownloadButton onDownload={() => downloadWorkbook("debt_schedule.xlsx", [tableSheet("Debt schedule", years, debtRows)], money)} />}>
        <DataTable caption="Debt schedule totals by year" columns={years} rows={debtRows} />
      </Tile>
      <Tile span={6} title="PP&E roll-forward" unit={mu} action={<DownloadButton onDownload={() => downloadWorkbook("ppe_schedule.xlsx", [tableSheet("PP&E", years, ppeRows)], money)} />}>
        <DataTable caption="PP&E roll-forward by year" columns={years} rows={ppeRows} />
        <p className="type-body text-[9px]">The deal model has no balance sheet: opening PP&amp;E is estimated at three times year-one capex.</p>
      </Tile>
      <Tile
        span={6}
        title="Working capital"
        unit={`days method, ${mu}`}
        action={inputs.wsp_mode ? <DownloadButton onDownload={() => downloadWorkbook("wc_schedule.xlsx", [tableSheet("Working capital", years, wcRows)], money)} /> : undefined}
      >
        {inputs.wsp_mode ? (
          <>
            <DataTable caption="Working capital from days by year" columns={years} rows={wcRows} />
            <p className="type-body text-[9px]">
              Receivables {inputs.ar_days}d of revenue; inventory {inputs.inv_days}d and payables {inputs.ap_days}d of cost of sales.
            </p>
          </>
        ) : (
          <p className="type-body">
            Working capital is a flat share of revenue. Turn on{" "}
            <Link href="/deal/debt" className="text-accent underline">
              working capital from days
            </Link>{" "}
            to see the schedule.
          </p>
        )}
      </Tile>
      <Tile span={6} title="Adjusted EBITDA" unit={mu} action={<DownloadButton onDownload={() => downloadWorkbook("adj_ebitda.xlsx", [tableSheet("Adjusted EBITDA", years, adjRows)], money)} />}>
        <div className="max-w-[260px]">
          <NumberField spec={SBC_SPEC} value={sbcPct} onCommit={setSbcPct} />
        </div>
        <DataTable caption="Adjusted EBITDA by year" columns={years} rows={adjRows} />
        <p className="type-body text-[9px]">Stock-based comp is added back for presentation only; it doesn&apos;t change the model&apos;s returns.</p>
      </Tile>
      <Tile span={6} title="Equity bridge" unit={`${mu} and share of gain`} action={<DownloadButton onDownload={() => downloadWorkbook("equity_bridge.xlsx", [bridgeSheet], money)} />}>
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
