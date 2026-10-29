"use client";

import { useTranslations } from "next-intl";

import { DataTable } from "@/components/charts/DataTable";
import { SensitivityTable } from "@/components/charts/HeatTable";
import { StackedBars } from "@/components/charts/StackedBars";
import { Waterfall } from "@/components/charts/Waterfall";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { quote } from "@/components/ui/Explain";
import { Term } from "@/components/ui/Term";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { useMoney } from "@/components/ui/MoneyScope";
import { dealFiscal } from "@/lib/deal/fields";
import type { Schemas } from "@/lib/api/client";
import { downloadWorkbook, fraction, sheet } from "@/lib/export";
import { fmtMoney, fmtMultiple, fmtRate, isNum } from "@/lib/format";
import { bridgeTerm, statementTerm, unitTerm } from "@/lib/glossary";
import { useEngineLabel } from "@/lib/i18n/useEngineText";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";
import { useStandardLabel } from "@/lib/i18n/useStandardLabel";

import { useDeal } from "../DealProvider";
import { Multiples } from "../Multiples";
import { DealField, DealScreen, LoadingTiles, RailGroup, TranchesOnDebtStep } from "../DealScreen";
import { baseCell, debtSeries, debtYears, hurdleFigures, totals, useHurdleSub } from "./shared";

export function ReturnsStep() {
  const { inputs, run } = useDeal();
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
            {inputs.tranches.length ? (
              <TranchesOnDebtStep />
            ) : (
              <>
                <DealField name="debt_pct" />
                <DealField name="senior_pct" />
                <DealField name="base_rate" />
                <DealField name="mezz_spread" />
              </>
            )}
          </RailGroup>
        </>
      }
    >
      {run.result ? <ReturnsResults /> : <LoadingTiles />}
    </DealScreen>
  );
}

/** i18n-keys: explain.returns_clears, explain.returns_misses */
function ReturnsResults() {
  const { label: mu, money } = useMoney();
  const { inputs, run, hurdle, current } = useDeal();
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
  const std = useStandardLabel(inputs.accounting_standard);
  const leases = res.leases;
  const x2 = useTranslations("explain");
  // Every figure once, as the tiles show it: the KPIs and the explanation read the same strings (PLAN.md 6.4)
  const hold = r.holding_period ?? inputs.hold;
  const shown = {
    irr: fmtRate(r.irr),
    moic: fmtMultiple(r.moic),
    equityIn: fmtMoney(r.entry_equity),
    equityOut: fmtMoney(r.net_exit_equity),
    exitEv: fmtMoney(r.exit_ev),
    multiple: fmtMultiple(r.exit_multiple, 1),
    ebitda: fmtMoney(r.exit_ebitda),
    netDebt: fmtMoney(r.net_debt_at_exit),
    debtAtClose: fmtMoney(begin[0]),
    hold: String(hold),
  };
  const vsHurdle = hurdleFigures(r.irr, hurdle);
  // A sentence is said only when every figure it quotes exists: never "an IRR of n/a"
  const returnsExplained = [
    ...([r.irr, r.moic, r.entry_equity, r.net_exit_equity].every(isNum) ? [x2.rich("returns_money", { ...shown, money: mu, f: quote })] : []),
    ...(vsHurdle ? [x2.rich(vsHurdle.clears ? "returns_clears" : "returns_misses", { diff: vsHurdle.diff, hurdle: vsHurdle.hurdle, f: quote })] : []),
    ...([r.exit_ev, r.exit_ebitda, r.exit_multiple, r.net_debt_at_exit].every(isNum) ? [x2.rich("returns_exit", { ...shown, f: quote })] : []),
    x2.rich("returns_bridge"),
  ];
  // The grid as its cells and headings show it: the outlined cell is the deal's own exit multiple and hold
  const s = res.exit_sensitivity;
  const cells = s.table.flat().filter(isNum);
  const baseIrr = base.row >= 0 && base.col >= 0 ? s.table[base.row]?.[base.col] : null;
  const sensitivityExplained = [
    isNum(baseIrr)
      ? x2.rich("sensitivity_base", {
          multiple: fmtMultiple(s.exit_multiples[base.row], 1),
          hold: units("holdColumn", { years: s.holding_periods[base.col] }),
          irr: fmtRate(baseIrr),
          f: quote,
        })
      : x2.rich("sensitivity_noBase"),
    ...(cells.length
      ? [x2.rich("sensitivity_range", { low: fmtRate(Math.min(...cells)), high: fmtRate(Math.max(...cells)), hurdle: fmtRate(hurdle, 0), f: quote })]
      : []),
  ];

  return (
    <Tiles>
      <Kpi term="irr" title={t("kpiIrr")} value={shown.irr} lead {...hurdleSub(r.irr, hurdle)} />
      <Kpi term="moic" title={t("kpiMoic")} value={shown.moic} sub={units("holdYears", { years: hold })} />
      <Kpi term="equityIn" title={t("kpiEquityIn")} value={shown.equityIn} sub={t("atClose", { money: mu })} />
      <Kpi term="equityOut" title={t("kpiEquityOut")} value={shown.equityOut} sub={t("atExit", { money: mu })} />
      <Kpi term="exitEv" title={t("kpiExitEv")} value={shown.exitEv} sub={t("exitEvSub", { multiple: shown.multiple, ebitda: shown.ebitda })} />
      <Kpi term="netDebtAtExit" title={t("kpiNetDebtAtExit")} value={shown.netDebt} sub={t("fromDebt", { debt: shown.debtAtClose, money: mu })} />

      <Tile span={6} term="equityBridge" title={t("tileEquityBridge")} unit={mu} unitTerm={unitTerm(money.unit)} explain={returnsExplained}>
        <Waterfall
          label={t("equityBridgeFrom", { from: shown.equityIn, to: shown.equityOut, money: mu })}
          // `key` is the API's short axis label (Entry, Fees, EBITDA growth, Multiple, Deleverage, Exit)
          steps={res.bridge_steps.map((s) => ({ label: bridgeAxis(s.key), value: s.value ?? 0, isTotal: s.is_total, term: bridgeTerm(s.key) }))}
        />
      </Tile>
      <Tile
        span={6}
        term="irrSensitivity"
        title={t("tileIrrSensitivity")}
        unit={t("sensitivityUnit")}
        explain={sensitivityExplained}
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
                run.result?.model,
                current?.id ?? null,
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
      <Tile span={6} term="debtPaydown" title={t("tileDebtPaydown")} unit={t("byTranche", { money: mu })}>
        <StackedBars {...debtSeries(res, fiscalLabels.deal(debtYears(res), fiscal), { close: t("close"), chart: t("debtBalanceByTranche"), trancheName })} />
      </Tile>
      <Tile span={6} term="operatingSummary" title={t("tileOperatingSummary")} unit={mu} unitTerm={unitTerm(money.unit)}>
        <DataTable
          caption={t("operatingSummaryByYear")}
          columns={years}
          rows={[
            { label: t("rowRevenue"), values: om.revenue ?? [], term: "revenue" },
            // With leases the model grows EBITDA after lease costs (PLAN.md 2.6), whatever the standard
            leases
              ? { label: t("rowEbitdaAfterLeases"), values: om.ebitda ?? [], term: "ebitdaAfterLeases" }
              : { label: t("rowEbitda"), values: om.ebitda ?? [], term: "ebitda" },
            { label: t("rowEbitdaMargin"), values: om.ebitda_margin ?? [], kind: "rate", term: "ebitdaMargin" },
            {
              label: std("rowInterest", t("rowInterest")),
              values: om.interest_expense ?? [],
              kind: "outflow",
              term: statementTerm("interest", inputs.accounting_standard),
            },
            { label: t("rowLeveredFcf"), values: res.cash_flow.levered_fcf ?? [], total: true, term: "leveredFcf" },
          ]}
        />
      </Tile>
      {leases && <LeasesTile leases={leases} />}
      <Multiples />
    </Tiles>
  );
}

/** What the deal's leases did to its value (PLAN.md 2.6): the answer's `leases` block. */
function LeasesTile({ leases }: { leases: NonNullable<Schemas["DealRunResponse"]["leases"]> }) {
  const { label: mu, money } = useMoney();
  const t = useTranslations("deal");
  const rows: [string, number, string][] = [
    [t("rowLeaseCost"), leases.lease_cost, "leaseCost"],
    [t("rowOperatingEbitda"), leases.operating_ebitda, "ebitdaAfterLeases"],
    [t("rowValuationEbitda"), leases.valuation_ebitda, "valuationEbitda"],
    [t("rowEntryEv"), leases.entry_ev, "entryEv"],
    [t("rowLeaseLiability"), leases.lease_liability, "leaseLiability"],
    [t("rowNetDebtAtEntry"), leases.net_debt_at_entry, "netDebtAtEntry"],
  ];
  return (
    <Tile span={6} term="leases" title={t("tileLeases")} unit={mu} unitTerm={unitTerm(money.unit)} aside={<span className="chip">{t("leasesSub", { view: t(`leaseView_${leases.view}`) })}</span>}>
      <table className="w-full border-collapse font-mono text-[11.5px]" data-testid="leases">
        <tbody>
          {rows.map(([label, v, term]) => (
            <tr key={label}>
              <th scope="row" className="type-input-label border-b border-grid py-1.5 text-start font-normal text-soft">
                <Term id={term}>{label}</Term>
              </th>
              <td className="border-b border-grid py-1.5 text-end text-ink">{fmtMoney(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="type-body pt-1.5 text-[9px]">{leases.counted_as_debt ? t("leasesAsDebt") : t("leasesNotDebt")}</p>
    </Tile>
  );
}
