"use client";

import { DataTable } from "@/components/charts/DataTable";
import { StackedBars } from "@/components/charts/StackedBars";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { useMoney } from "@/components/ui/MoneyScope";
import { downloadWorkbook, sheet } from "@/lib/export";
import { fmtMoney, fmtPct } from "@/lib/format";

import { useDeal } from "../DealProvider";
import { DealField, DealScreen, LoadingTiles, RailGroup, WspToggle } from "../DealScreen";
import { debtSeries, totals, yearLabels } from "./shared";

export function DebtStep() {
  const { inputs, run } = useDeal();
  const res = run.result;

  return (
    <DealScreen
      rail={
        <>
          <RailGroup title="Capital structure">
            <DealField name="debt_pct" />
            <DealField name="senior_pct" />
            <DealField name="base_rate" />
            <DealField name="mezz_spread" />
          </RailGroup>
          <RailGroup title="Cash flow, % of revenue">
            <DealField name="capex" />
            <DealField name="da" />
            <DealField name="nwc" disabled={inputs.wsp_mode} />
            <DealField name="mincash" />
          </RailGroup>
          <RailGroup title="Working capital">
            <WspToggle />
            <DealField name="ar_days" disabled={!inputs.wsp_mode} />
            <DealField name="inv_days" disabled={!inputs.wsp_mode} />
            <DealField name="ap_days" disabled={!inputs.wsp_mode} />
          </RailGroup>
        </>
      }
    >
      {!res ? (
        <LoadingTiles />
      ) : (
        <DebtResults />
      )}
    </DealScreen>
  );
}

function DebtResults() {
  const { label: mu, money } = useMoney();
  const { inputs, run } = useDeal();
  const res = run.result!;
  const begin = totals(res, "total_beginning_debt");
  const end = totals(res, "total_ending_debt");
  const interest = totals(res, "total_interest_expense");
  const cf = res.cash_flow;
  const years = yearLabels(end.length);
  const atClose = begin[0];
  const lastEnd = end.at(-1);
  const repaidPct = atClose > 0 && lastEnd !== undefined ? ((atClose - lastEnd) / atClose) * 100 : NaN;
  const cumFcf = cf.cumulative_fcf?.at(-1);
  // How far the income statement's interest is from the final debt schedule;
  // the interest loop iterates until this is under a cent
  const plInterest = res.operating_model.interest_expense ?? [];
  const interestGap = Math.max(0, ...interest.map((v, i) => Math.abs(v - (plInterest[i] ?? v))));

  return (
    <Tiles>
      <Kpi title="Debt at close" value={fmtMoney(atClose)} sub={`${fmtPct(inputs.debt_pct)} of EV, ${mu}`} lead />
      <Kpi title="Debt at exit" value={fmtMoney(lastEnd)} sub={`${fmtPct(repaidPct)} repaid`} />
      <Kpi title="Net debt at exit" value={fmtMoney(res.returns.net_debt_at_exit)} sub={`after cash, ${mu}`} />
      <Kpi title="Interest, year 1" value={fmtMoney(interest[0])} sub={mu} />
      <Kpi title="Cumulative FCF" value={fmtMoney(cumFcf)} sub={`levered, ${end.length} yr`} />
      <Kpi
        title="Interest loop"
        value={res.interest_converged ? "Converged" : "Not converged"}
        sub={`P&L vs schedule gap ${fmtMoney(interestGap)} ${mu}`}
        tone={res.interest_converged ? "gain" : "attention"}
      />

      <Tile span={6} title="Debt balance" unit={`by tranche, ${mu}`}>
        <StackedBars {...debtSeries(res)} />
      </Tile>
      <Tile span={6} title="Cash flow" unit={mu}>
        <DataTable
          caption="Levered free cash flow by year"
          columns={years}
          rows={[
            { label: "Net income", values: cf.net_income ?? [] },
            { label: "D&A", values: cf.da ?? [] },
            { label: "Capex", values: cf.capex ?? [], kind: "outflow" },
            { label: "Change in NWC", values: cf.delta_nwc ?? [], kind: "outflow" },
            { label: "Levered FCF", values: cf.levered_fcf ?? [], total: true },
            { label: "Cumulative FCF", values: cf.cumulative_fcf ?? [] },
          ]}
        />
      </Tile>

      {Object.entries(res.tranches).map(([name, rows]) => (
        <Tile
          key={name}
          span={6}
          title={name}
          unit={`${fmtPct((rows[0]?.interest_rate ?? NaN) * 100, 2)} rate, ${mu}`}
          action={
            <DownloadButton
              onDownload={() =>
                downloadWorkbook("debt_schedule.xlsx", [
                  sheet(
                    name,
                    ["Year", "Opening", "Mandatory", "Cash sweep", "Closing", "Interest"],
                    rows.map((t) => [t.year, t.beginning_balance ?? null, t.mandatory_repayment ?? null, t.cash_sweep ?? null, t.ending_balance ?? null, t.interest_expense ?? null]),
                  ),
                ], money)
              }
            />
          }
        >
          <DataTable
            caption={`${name} schedule`}
            columns={rows.map((r) => `Y${r.year}`)}
            rows={[
              { label: "Opening", values: rows.map((r) => r.beginning_balance) },
              { label: "Mandatory", values: rows.map((r) => r.mandatory_repayment), kind: "outflow" },
              { label: "Cash sweep", values: rows.map((r) => r.cash_sweep), kind: "outflow" },
              { label: "Closing", values: rows.map((r) => r.ending_balance), total: true },
              { label: "Interest", values: rows.map((r) => r.interest_expense), kind: "outflow" },
            ]}
          />
        </Tile>
      ))}
    </Tiles>
  );
}
