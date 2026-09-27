"use client";

import { useTranslations } from "next-intl";

import { DataTable } from "@/components/charts/DataTable";
import { StackedBars } from "@/components/charts/StackedBars";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { useMoney } from "@/components/ui/MoneyScope";
import { dealFiscal } from "@/lib/deal/fields";
import { downloadWorkbook, sheet } from "@/lib/export";
import { fmtMoney, fmtPct } from "@/lib/format";
import { useEngineLabel } from "@/lib/i18n/useEngineText";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";

import { useDeal } from "../DealProvider";
import { DealField, DealScreen, LoadingTiles, RailGroup, WspToggle } from "../DealScreen";
import { debtSeries, debtYears, totals } from "./shared";

export function DebtStep() {
  const { inputs, run } = useDeal();
  const t = useTranslations("deal");
  const res = run.result;

  return (
    <DealScreen
      rail={
        <>
          <RailGroup title={t("groupCapital")}>
            <DealField name="debt_pct" />
            <DealField name="senior_pct" />
            <DealField name="base_rate" />
            <DealField name="mezz_spread" />
          </RailGroup>
          <RailGroup title={t("groupCashFlow")}>
            <DealField name="capex" />
            <DealField name="da" />
            <DealField name="nwc" disabled={inputs.wsp_mode} />
            <DealField name="mincash" />
          </RailGroup>
          <RailGroup title={t("groupWorkingCapital")}>
            <WspToggle />
            <DealField name="ar_days" disabled={!inputs.wsp_mode} />
            <DealField name="inv_days" disabled={!inputs.wsp_mode} />
            <DealField name="ap_days" disabled={!inputs.wsp_mode} />
          </RailGroup>
        </>
      }
    >
      {!res ? <LoadingTiles /> : <DebtResults />}
    </DealScreen>
  );
}

function DebtResults() {
  const { label: mu, money } = useMoney();
  const { inputs, run } = useDeal();
  const t = useTranslations("deal");
  const x = useTranslations("export");
  const fiscalLabels = useFiscalLabels();
  const trancheName = useEngineLabel("tranche");
  const res = run.result!;
  const begin = totals(res, "total_beginning_debt");
  const end = totals(res, "total_ending_debt");
  const interest = totals(res, "total_interest_expense");
  const cf = res.cash_flow;
  const fiscal = dealFiscal(inputs);
  const years = fiscalLabels.deal(end.length, fiscal);
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
      <Kpi title={t("kpiDebtAtClose")} value={fmtMoney(atClose)} sub={t("debtAtCloseSub", { share: fmtPct(inputs.debt_pct), money: mu })} lead />
      <Kpi title={t("kpiDebtAtExit")} value={fmtMoney(lastEnd)} sub={t("repaidSub", { share: fmtPct(repaidPct) })} />
      <Kpi title={t("kpiNetDebtAtExit")} value={fmtMoney(res.returns.net_debt_at_exit)} sub={t("afterCash", { money: mu })} />
      <Kpi title={t("kpiInterestYear1")} value={fmtMoney(interest[0])} sub={mu} />
      <Kpi title={t("kpiCumulativeFcf")} value={fmtMoney(cumFcf)} sub={t("leveredYears", { years: end.length })} />
      <Kpi
        title={t("kpiInterestLoop")}
        value={res.interest_converged ? t("converged") : t("notConverged")}
        sub={t("interestGapSub", { gap: fmtMoney(interestGap), money: mu })}
        tone={res.interest_converged ? "gain" : "attention"}
      />

      <Tile span={6} title={t("tileDebtBalance")} unit={t("byTranche", { money: mu })}>
        <StackedBars {...debtSeries(res, fiscalLabels.deal(debtYears(res), fiscal), { close: t("close"), chart: t("debtBalanceByTranche"), trancheName })} />
      </Tile>
      <Tile span={6} title={t("tileCashFlow")} unit={mu}>
        <DataTable
          caption={t("leveredFcfByYear")}
          columns={years}
          rows={[
            { label: t("rowNetIncome"), values: cf.net_income ?? [] },
            { label: t("rowDa"), values: cf.da ?? [] },
            { label: t("rowCapex"), values: cf.capex ?? [], kind: "outflow" },
            { label: t("rowChangeInNwc"), values: cf.delta_nwc ?? [], kind: "outflow" },
            { label: t("rowLeveredFcf"), values: cf.levered_fcf ?? [], total: true },
            { label: t("rowCumulativeFcf"), values: cf.cumulative_fcf ?? [] },
          ]}
        />
      </Tile>

      {Object.entries(res.tranches).map(([name, rows]) => (
        <Tile
          key={name}
          span={6}
          title={trancheName(name)}
          unit={t("rateAndMoney", { rate: fmtPct((rows[0]?.interest_rate ?? NaN) * 100, 2), money: mu })}
          action={
            <DownloadButton
              onDownload={() =>
                downloadWorkbook(
                  x("fileDebtSchedule"),
                  [
                    sheet(
                      trancheName(name),
                      [x("colYear"), x("colOpening"), x("colMandatory"), x("colCashSweep"), x("colClosing"), x("colInterest")],
                      rows.map((t2) => [
                        years[t2.year - 1] ?? t2.year,
                        t2.beginning_balance ?? null,
                        t2.mandatory_repayment ?? null,
                        t2.cash_sweep ?? null,
                        t2.ending_balance ?? null,
                        t2.interest_expense ?? null,
                      ]),
                      { columns: ["text", "money", "money", "money", "money", "money"] },
                    ),
                  ],
                  money,
                )
              }
            />
          }
        >
          <DataTable
            caption={t("trancheSchedule", { tranche: trancheName(name) })}
            columns={rows.map((r) => years[r.year - 1] ?? String(r.year))}
            rows={[
              { label: t("rowOpening"), values: rows.map((r) => r.beginning_balance) },
              { label: t("rowMandatory"), values: rows.map((r) => r.mandatory_repayment), kind: "outflow" },
              { label: t("rowCashSweep"), values: rows.map((r) => r.cash_sweep), kind: "outflow" },
              { label: t("rowClosing"), values: rows.map((r) => r.ending_balance), total: true },
              { label: t("rowInterest"), values: rows.map((r) => r.interest_expense), kind: "outflow" },
            ]}
          />
        </Tile>
      ))}
    </Tiles>
  );
}
