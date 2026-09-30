"use client";

import { useTranslations } from "next-intl";
import Link from "next/link";
import { useState } from "react";

import { DataTable, type Row } from "@/components/charts/DataTable";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { NumberField } from "@/components/ui/NumberField";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { useMoney } from "@/components/ui/MoneyScope";
import { multiplesFromPct } from "@/lib/deal/capital";
import { dealFiscal, FIELDS, type NumericDealKey } from "@/lib/deal/fields";
import { downloadWorkbook, fraction, sheet, tableSheet } from "@/lib/export";
import type { FieldSpec } from "@/lib/fields";
import { fmtDelta, fmtInput, fmtMoney, fmtMultiple, fmtPct, fmtRate } from "@/lib/format";
import { useEngineLabel } from "@/lib/i18n/useEngineText";
import { useUnitLabel } from "@/lib/i18n/useFieldText";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";
import { MONEY } from "@/lib/money";

import { useDeal } from "../DealProvider";
import { DealScreen, LoadingTiles, RailGroup } from "../DealScreen";
import { totals, useHurdleSub } from "./shared";

const SBC_SPEC: FieldSpec = { unit: "%", step: 0.5, decimals: 1, min: 0, max: 50 };

/** i18n-keys: deal.group* */
const ASSUMPTIONS: { titleKey: string; keys: NumericDealKey[] }[] = [
  { titleKey: "groupEntryExit", keys: ["ebitda", "entry_mult", "exit_mult", "hold"] },
  { titleKey: "groupOperations", keys: ["growth", "gross_margin", "opex", "tax", "da"] },
  { titleKey: "groupCapital", keys: ["debt_pct", "senior_pct", "base_rate", "mezz_spread"] },
  { titleKey: "groupCashFlowPlain", keys: ["capex", "nwc", "mincash"] },
];

export function SummaryStep() {
  const { inputs, run, money } = useDeal();
  const t = useTranslations("deal");
  const fields = useTranslations("fields");
  const unitLabel = useUnitLabel();
  const { seniorX, mezzX } = multiplesFromPct(inputs.entry_mult, inputs.debt_pct, inputs.senior_pct);
  const listed = inputs.tranches.length > 0;
  const trancheName = useEngineLabel("tranche");
  return (
    <DealScreen
      rail={
        <>
          {ASSUMPTIONS.map((g) => (
            <RailGroup key={g.titleKey} title={t(g.titleKey)}>
              <dl className="grid grid-cols-[1fr_auto] gap-x-2 gap-y-1">
                {listed && g.titleKey === "groupCapital"
                  ? inputs.tranches.map((tr, i) => (
                      <div key={i} className="contents">
                        <dt className="type-input-label truncate">{trancheName(tr.name)}</dt>
                        <dd className="text-end font-mono text-[11.5px] text-ink">
                          {fmtMoney(tr.amount)} <span className="text-[10px] text-[#56636a]">{unitLabel(MONEY, money)}</span>
                        </dd>
                      </div>
                    ))
                  : g.keys.map((k) => (
                  <div key={k} className="contents">
                    <dt className="type-input-label">{fields(k)}</dt>
                    <dd className="text-end font-mono text-[11.5px] text-ink">
                      {fmtInput(inputs[k], FIELDS[k].decimals)} <span className="text-[10px] text-[#56636a]">{unitLabel(FIELDS[k].unit, money)}</span>
                    </dd>
                  </div>
                ))}
              </dl>
            </RailGroup>
          ))}
          <div className="grid gap-2 px-3.5 py-3">
            <p className="type-body">
              {listed
                ? t("summaryListedNote", {
                    count: inputs.tranches.length,
                    method: inputs.wsp_mode ? t("methodFromDays") : t("methodShareOfRevenue"),
                  })
                : t("summaryDebtNote", {
                    senior: fmtMultiple(seniorX, 1),
                    mezz: fmtMultiple(mezzX, 1),
                    method: inputs.wsp_mode ? t("methodFromDays") : t("methodShareOfRevenue"),
                  })}
            </p>
            <Link href="/deal/inputs" className="type-action-secondary justify-self-start px-2.5 py-1.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)]">
              {t("editInputs")}
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
  const t = useTranslations("deal");
  const x = useTranslations("export");
  const units = useTranslations("units");
  const hurdleSub = useHurdleSub();
  const fiscalLabels = useFiscalLabels();
  const trancheName = useEngineLabel("tranche");
  const bridgeRow = useEngineLabel("bridgeRow");
  const [sbcPct, setSbcPct] = useState(2);
  const res = run.result!;
  const om = res.operating_model;
  const cf = res.cash_flow;
  const r = res.returns;
  const years = fiscalLabels.deal(om.revenue?.length ?? 0, dealFiscal(inputs));
  const br = res.equity_bridge;
  const revenue = (om.revenue ?? []).map((v) => v ?? 0);
  const cogs = (om.cogs ?? []).map((v) => Math.abs(v ?? 0));

  const incomeRows: Row[] = [
    { label: t("rowRevenue"), values: om.revenue ?? [] },
    { label: t("rowGrossProfit"), values: om.gross_profit ?? [] },
    { label: t("rowOpex"), values: om.opex ?? [], kind: "outflow" },
    { label: t("rowEbitda"), values: om.ebitda ?? [], total: true },
    { label: t("rowDa"), values: om.da ?? [], kind: "outflow" },
    { label: t("rowEbit"), values: om.ebit ?? [] },
    { label: t("rowInterestExpense"), values: om.interest_expense ?? [], kind: "outflow" },
    { label: t("rowPretaxIncome"), values: om.ebt ?? [] },
    { label: t("rowTaxes"), values: om.taxes ?? [], kind: "outflow" },
    { label: t("rowNetIncome"), values: om.net_income ?? [], total: true },
    { label: t("rowEbitdaMargin"), values: om.ebitda_margin ?? [], kind: "rate" },
  ];
  // The deal's tax rules at work (PLAN.md 2.5); a deal with none has no tax
  // block. The minimum-tax row shows only when it ever raises the tax.
  const tx = res.tax;
  const taxRows: Row[] | null = tx
    ? [
        { label: t("rowTaxableIncome"), values: tx.taxable_income ?? [] },
        { label: t("rowInterestDeducted"), values: tx.interest_deductible ?? [] },
        { label: t("rowInterestCarried"), values: tx.interest_carried ?? [] },
        { label: t("rowLossesUsed"), values: tx.losses_used ?? [] },
        { label: t("rowLossesCarried"), values: tx.losses_carried ?? [] },
        ...((tx.minimum_tax_topup ?? []).some((v) => (v ?? 0) > 0)
          ? [{ label: t("rowMinimumTaxTopup"), values: tx.minimum_tax_topup ?? [] }]
          : []),
        { label: t("rowTaxes"), values: tx.taxes ?? [], kind: "outflow", total: true },
      ]
    : null;
  const cashRows: Row[] = [
    { label: t("rowNetIncome"), values: cf.net_income ?? [] },
    { label: t("rowDa"), values: cf.da ?? [] },
    { label: t("rowCapex"), values: cf.capex ?? [], kind: "outflow" },
    { label: t("rowChangeInNwc"), values: cf.delta_nwc ?? [], kind: "outflow" },
    { label: t("rowLeveredFcf"), values: cf.levered_fcf ?? [], total: true },
  ];
  const debtRows: Row[] = [
    { label: t("rowOpening"), values: totals(res, "total_beginning_debt") },
    { label: t("rowMandatory"), values: totals(res, "total_mandatory_repayment"), kind: "outflow" },
    { label: t("rowCashSweep"), values: totals(res, "total_cash_sweep"), kind: "outflow" },
    { label: t("rowClosing"), values: totals(res, "total_ending_debt"), total: true },
    { label: t("rowInterest"), values: totals(res, "total_interest_expense"), kind: "outflow" },
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
    { label: t("rowOpeningPpeEstimate"), values: ppeBegin },
    { label: t("rowCapex"), values: capex },
    { label: t("rowDepreciation"), values: da, kind: "outflow" },
    { label: t("rowClosingPpe"), values: ppeEnd, total: true },
  ];

  const ar = revenue.map((v) => (v * inputs.ar_days) / 365);
  const inv = cogs.map((v) => (v * inputs.inv_days) / 365);
  const ap = cogs.map((v) => (v * inputs.ap_days) / 365);
  const wcRows: Row[] = [
    { label: t("rowReceivables"), values: ar },
    { label: t("rowInventory"), values: inv },
    { label: t("rowPayables"), values: ap, kind: "outflow" },
    { label: t("rowNetWorkingCapital"), values: ar.map((v, i) => v + (inv[i] ?? 0) - (ap[i] ?? 0)), total: true },
  ];

  const sbc = revenue.map((v) => (v * sbcPct) / 100);
  const adj = (om.ebitda ?? []).map((v, i) => (v ?? 0) + (sbc[i] ?? 0));
  const adjRows: Row[] = [
    { label: t("rowEbitda"), values: om.ebitda ?? [] },
    { label: t("rowStockBasedComp"), values: sbc },
    { label: t("rowAdjustedEbitda"), values: adj, total: true },
    { label: t("rowAdjustedMargin"), values: adj.map((v, i) => (revenue[i] ? v / revenue[i] : NaN)), kind: "rate" },
  ];

  const bridgeSheet = sheet(
    x("sheetEquityBridge"),
    [x("colComponent"), x("colValue", { money: mu }), x("colShareOfGain")],
    res.bridge_steps.map((s) => [bridgeRow(s.label), s.value ?? null, fraction(s.pct_of_gain, 100)]),
    { columns: ["text", "money", "percent"] },
  );
  const allSheets = () => [
    tableSheet(x("sheetPl"), years, incomeRows),
    ...(taxRows ? [tableSheet(x("sheetTax"), years, taxRows)] : []),
    tableSheet(x("sheetCashFlow"), years, cashRows),
    tableSheet(x("sheetDebtSchedule"), years, debtRows),
    ...Object.entries(res.tranches).map(([name, rows]) =>
      sheet(
        trancheName(name),
        [x("colYear"), x("colOpening"), x("colMandatory"), x("colCashSweep"), x("colClosing"), x("colInterest"), x("colRate")],
        rows.map((tr) => [
          years[tr.year - 1] ?? tr.year,
          tr.beginning_balance ?? null,
          tr.mandatory_repayment ?? null,
          tr.cash_sweep ?? null,
          tr.ending_balance ?? null,
          tr.interest_expense ?? null,
          fraction(tr.interest_rate),
        ]),
        { columns: ["text", "money", "money", "money", "money", "money", "percent"] },
      ),
    ),
    bridgeSheet,
    tableSheet(x("sheetPpe"), years, ppeRows),
    ...(inputs.wsp_mode ? [tableSheet(x("sheetWorkingCapital"), years, wcRows)] : []),
    tableSheet(x("sheetAdjustedEbitda"), years, adjRows),
  ];

  return (
    <Tiles>
      <Kpi title={t("kpiIrr")} value={fmtRate(r.irr)} lead {...hurdleSub(r.irr, hurdle)} />
      <Kpi title={t("kpiMoic")} value={fmtMultiple(r.moic)} sub={units("holdYears", { years: r.holding_period ?? inputs.hold })} />
      <Kpi title={t("kpiEquityIn")} value={fmtMoney(r.entry_equity)} sub={mu} />
      <Kpi title={t("kpiEquityOut")} value={fmtMoney(r.net_exit_equity)} sub={mu} />
      <Kpi title={t("kpiTotalGain")} value={fmtMoney(br.total_gain)} sub={mu} />
      <Kpi
        title={t("kpiBridgeResidual")}
        value={fmtMoney(br.residual)}
        sub={t("shouldBeZero", { money: mu })}
        tone={Math.abs(br.residual ?? 0) > 0.05 ? "attention" : undefined}
      />

      <div className="col-span-12 flex items-center justify-between gap-4 bg-canvas px-3 py-2">
        <p className="type-body">{t("allTablesNote")}</p>
        <DownloadButton label={t("allTables")} onDownload={() => downloadWorkbook(x("fileSummary"), allSheets(), money)} />
      </div>

      <Tile
        span={12}
        title={t("tileIncomeStatement")}
        unit={mu}
        action={<DownloadButton onDownload={() => downloadWorkbook(x("filePl"), [tableSheet(x("sheetPl"), years, incomeRows)], money)} />}
      >
        <DataTable caption={t("incomeStatementByYear")} columns={years} rows={incomeRows} />
      </Tile>
      {taxRows && (
        <Tile
          span={12}
          title={t("tileTax")}
          unit={mu}
          action={<DownloadButton onDownload={() => downloadWorkbook(x("fileTax"), [tableSheet(x("sheetTax"), years, taxRows)], money)} />}
        >
          <DataTable caption={t("taxByYear")} columns={years} rows={taxRows} />
        </Tile>
      )}
      <Tile
        span={6}
        title={t("tileCashFlow")}
        unit={mu}
        action={<DownloadButton onDownload={() => downloadWorkbook(x("fileCashFlow"), [tableSheet(x("sheetCashFlow"), years, cashRows)], money)} />}
      >
        <DataTable caption={t("cashFlowByYear")} columns={years} rows={cashRows} />
      </Tile>
      <Tile
        span={6}
        title={t("tileDebt")}
        unit={t("allTranches", { money: mu })}
        action={
          <DownloadButton onDownload={() => downloadWorkbook(x("fileDebtSchedule"), [tableSheet(x("sheetDebtSchedule"), years, debtRows)], money)} />
        }
      >
        <DataTable caption={t("debtTotalsByYear")} columns={years} rows={debtRows} />
      </Tile>
      <Tile
        span={6}
        title={t("tilePpe")}
        unit={mu}
        action={<DownloadButton onDownload={() => downloadWorkbook(x("filePpe"), [tableSheet(x("sheetPpe"), years, ppeRows)], money)} />}
      >
        <DataTable caption={t("ppeByYear")} columns={years} rows={ppeRows} />
        <p className="type-body text-[9px]">{t("ppeEstimateNote")}</p>
      </Tile>
      <Tile
        span={6}
        title={t("tileWorkingCapital")}
        unit={t("daysMethod", { money: mu })}
        action={
          inputs.wsp_mode ? (
            <DownloadButton onDownload={() => downloadWorkbook(x("fileWorkingCapital"), [tableSheet(x("sheetWorkingCapital"), years, wcRows)], money)} />
          ) : undefined
        }
      >
        {inputs.wsp_mode ? (
          <>
            <DataTable caption={t("workingCapitalByYear")} columns={years} rows={wcRows} />
            <p className="type-body text-[9px]">
              {t("workingCapitalDaysNote", {
                ar: units("days", { value: inputs.ar_days }),
                inv: units("days", { value: inputs.inv_days }),
                ap: units("days", { value: inputs.ap_days }),
              })}
            </p>
          </>
        ) : (
          <p className="type-body">
            {t.rich("workingCapitalOffNote", {
              link: (chunks) => (
                <Link href="/deal/debt" className="text-accent underline">
                  {chunks}
                </Link>
              ),
            })}
          </p>
        )}
      </Tile>
      <Tile
        span={6}
        title={t("tileAdjustedEbitda")}
        unit={mu}
        action={<DownloadButton onDownload={() => downloadWorkbook(x("fileAdjEbitda"), [tableSheet(x("sheetAdjustedEbitda"), years, adjRows)], money)} />}
      >
        <div className="max-w-[260px]">
          <NumberField spec={SBC_SPEC} label={t("rowStockBasedComp")} value={sbcPct} onCommit={setSbcPct} />
        </div>
        <DataTable caption={t("adjustedEbitdaByYear")} columns={years} rows={adjRows} />
        <p className="type-body text-[9px]">{t("sbcNote")}</p>
      </Tile>
      <Tile
        span={6}
        title={t("tileEquityBridge")}
        unit={t("bridgeUnit", { money: mu })}
        action={<DownloadButton onDownload={() => downloadWorkbook(x("fileEquityBridge"), [bridgeSheet], money)} />}
      >
        <table className="w-full border-collapse font-mono text-[11.5px]">
          <caption className="sr-only">{t("tileEquityBridge")}</caption>
          <tbody>
            {res.bridge_steps.map((s) => (
              <tr key={s.key} className={s.is_total ? "text-bright" : "text-ink"}>
                <th scope="row" className="type-input-label border-b border-grid py-1.5 text-start font-normal text-soft">
                  {bridgeRow(s.label)}
                </th>
                <td className={`border-b border-grid py-1.5 text-end ${!s.is_total && (s.value ?? 0) < 0 ? "text-loss" : ""} ${s.is_total ? "font-semibold" : ""}`}>
                  {s.is_total ? fmtMoney(s.value) : fmtDelta(s.value)}
                </td>
                <td className="w-24 border-b border-grid py-1.5 text-end text-muted">{s.pct_of_gain == null ? "" : fmtPct(s.pct_of_gain)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Tile>
    </Tiles>
  );
}
