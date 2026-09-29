"use client";

import { useTranslations } from "next-intl";

import { DataTable } from "@/components/charts/DataTable";
import { SensitivityTable } from "@/components/charts/HeatTable";
import { StackedBars } from "@/components/charts/StackedBars";
import { Waterfall } from "@/components/charts/Waterfall";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { useMoney } from "@/components/ui/MoneyScope";
import { dealFiscal } from "@/lib/deal/fields";
import { downloadWorkbook, fraction, sheet } from "@/lib/export";
import { fmtMoney, fmtMultiple, fmtRate } from "@/lib/format";
import { useEngineLabel } from "@/lib/i18n/useEngineText";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";

import { useDeal } from "../DealProvider";
import { DealField, DealScreen, LoadingTiles, RailGroup } from "../DealScreen";
import { baseCell, debtSeries, debtYears, totals, useHurdleSub } from "./shared";

export function ReturnsStep() {
  const { run } = useDeal();
  const t = useTranslations("deal");
  return (
    <DealScreen
      rail={
        <>
          <RailGroup title={t("groupEntryExit")}>
            <DealField name="ebitda" />
            <DealField name="entry_mult" />
            <DealField name="exit_mult" />
            <DealField name="hold" />
          </RailGroup>
          <RailGroup title={t("groupOperations")}>
            <DealField name="growth" />
            <DealField name="gross_margin" />
            <DealField name="opex" />
            <DealField name="tax" />
          </RailGroup>
          <RailGroup title={t("groupCapital")}>
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
  const { label: mu, money } = useMoney();
  const { inputs, run, hurdle } = useDeal();
  const t = useTranslations("deal");
  const x = useTranslations("export");
  const units = useTranslations("units");
  const hurdleSub = useHurdleSub();
  const fiscalLabels = useFiscalLabels();
  const trancheName = useEngineLabel("tranche");
  const bridgeAxis = useEngineLabel("bridgeAxis");
  const res = run.result!;
  const r = res.returns;
  const om = res.operating_model;
  const base = baseCell(res, inputs.exit_mult, inputs.hold);
  const begin = totals(res, "total_beginning_debt");
  const fiscal = dealFiscal(inputs);
  const years = fiscalLabels.deal(om.revenue?.length ?? 0, fiscal);

  return (
    <Tiles>
      <Kpi title={t("kpiIrr")} value={fmtRate(r.irr)} lead {...hurdleSub(r.irr, hurdle)} />
      <Kpi title={t("kpiMoic")} value={fmtMultiple(r.moic)} sub={units("holdYears", { years: r.holding_period ?? inputs.hold })} />
      <Kpi title={t("kpiEquityIn")} value={fmtMoney(r.entry_equity)} sub={t("atClose", { money: mu })} />
      <Kpi title={t("kpiEquityOut")} value={fmtMoney(r.net_exit_equity)} sub={t("atExit", { money: mu })} />
      <Kpi
        title={t("kpiExitEv")}
        value={fmtMoney(r.exit_ev)}
        sub={t("exitEvSub", { multiple: fmtMultiple(r.exit_multiple, 1), ebitda: fmtMoney(r.exit_ebitda) })}
      />
      <Kpi title={t("kpiNetDebtAtExit")} value={fmtMoney(r.net_debt_at_exit)} sub={t("fromDebt", { debt: fmtMoney(begin[0]), money: mu })} />

      <Tile span={6} title={t("tileEquityBridge")} unit={mu}>
        <Waterfall
          label={t("equityBridgeFrom", { from: fmtMoney(r.entry_equity), to: fmtMoney(r.net_exit_equity), money: mu })}
          // `key` is the API's short axis label (Entry, Fees, EBITDA growth, Multiple, Deleverage, Exit)
          steps={res.bridge_steps.map((s) => ({ label: bridgeAxis(s.key), value: s.value ?? 0, isTotal: s.is_total }))}
        />
      </Tile>
      <Tile
        span={6}
        title={t("tileIrrSensitivity")}
        unit={t("sensitivityUnit")}
        action={
          <DownloadButton
            onDownload={() =>
              downloadWorkbook(
                x("fileIrrSensitivity"),
                [
                  sheet(
                    x("sheetIrrSensitivity"),
                    [x("colExitMultiple"), ...res.exit_sensitivity.holding_periods.map((h) => units("holdColumn", { years: h }))],
                    res.exit_sensitivity.table.map((row, i) => [res.exit_sensitivity.exit_multiples[i], ...row.map((v) => fraction(v))]),
                    { columns: ["multiple", ...res.exit_sensitivity.holding_periods.map(() => "percent" as const)] },
                  ),
                ],
                money,
              )
            }
          />
        }
      >
        <SensitivityTable
          exitMultiples={res.exit_sensitivity.exit_multiples}
          holds={res.exit_sensitivity.holding_periods}
          table={res.exit_sensitivity.table}
          hurdle={hurdle}
          baseRow={base.row}
          baseCol={base.col}
        />
        <p className="type-body text-[9px]">{t("sensitivityNote")}</p>
      </Tile>
      <Tile span={6} title={t("tileDebtPaydown")} unit={t("byTranche", { money: mu })}>
        <StackedBars {...debtSeries(res, fiscalLabels.deal(debtYears(res), fiscal), { close: t("close"), chart: t("debtBalanceByTranche"), trancheName })} />
      </Tile>
      <Tile span={6} title={t("tileOperatingSummary")} unit={mu}>
        <DataTable
          caption={t("operatingSummaryByYear")}
          columns={years}
          rows={[
            { label: t("rowRevenue"), values: om.revenue ?? [] },
            { label: t("rowEbitda"), values: om.ebitda ?? [] },
            { label: t("rowEbitdaMargin"), values: om.ebitda_margin ?? [], kind: "rate" },
            { label: t("rowInterest"), values: om.interest_expense ?? [], kind: "outflow" },
            { label: t("rowLeveredFcf"), values: res.cash_flow.levered_fcf ?? [], total: true },
          ]}
        />
      </Tile>
    </Tiles>
  );
}
