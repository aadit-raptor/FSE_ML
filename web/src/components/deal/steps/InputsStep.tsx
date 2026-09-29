"use client";

import { useTranslations } from "next-intl";
import Link from "next/link";
import { useEffect, useState } from "react";

import { useSettings } from "@/components/settings/SettingsProvider";
import { useMoney } from "@/components/ui/MoneyScope";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { api, type Schemas } from "@/lib/api/client";
import { debtShareOfEv, drawnDebt, floatingCount, multiplesFromPct } from "@/lib/deal/capital";
import type { Tranche } from "@/lib/deal/fields";
import { fmtCount, fmtMoney, fmtMultiple, fmtPct, fmtRate } from "@/lib/format";
import { useEngineLabel } from "@/lib/i18n/useEngineText";

import { useDeal } from "../DealProvider";
import { DealRisk } from "../DealRisk";
import { DealField, DealScreen, DebtMultipleField, FiscalFields, LoadingTiles, MoneyFields, RailGroup, TranchesOnDebtStep } from "../DealScreen";
import { useHurdleSub } from "./shared";

type SourcesUses = Schemas["SourcesUsesResponse"];

function useSourcesAndUses(ebitda: number, entryMult: number, seniorX: number, mezzX: number, mincash: number, tranches: Tranche[]) {
  const { overrides } = useSettings();
  const { money } = useMoney();
  const [su, setSu] = useState<SourcesUses | null>(null);
  useEffect(() => {
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      api
        .POST("/api/deal/sources-and-uses", {
          // A deal that lists its facilities is sourced by them (core/deal.py sources_and_uses_for)
          body: { ebitda, entry_mult: entryMult, senior_x: seniorX, mezz_x: mezzX, mincash, settings: overrides, money, ...(tranches.length ? { tranches } : {}) },
          signal: ctrl.signal,
        })
        .then(({ data }) => data && setSu(data))
        .catch(() => {});
    }, 300);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [ebitda, entryMult, seniorX, mezzX, mincash, overrides, money, tranches]);
  return su;
}

export function InputsStep() {
  const { inputs, run, hurdle } = useDeal();
  const t = useTranslations("deal");
  const hurdleSub = useHurdleSub();
  const { seniorX, mezzX } = multiplesFromPct(inputs.entry_mult, inputs.debt_pct, inputs.senior_pct);
  const su = useSourcesAndUses(inputs.ebitda, inputs.entry_mult, Number(seniorX.toFixed(6)), Number(mezzX.toFixed(6)), inputs.mincash, inputs.tranches);
  const listed = inputs.tranches.length > 0;
  const trancheName = useEngineLabel("tranche");
  const r = run.result?.returns;
  const ev = inputs.ebitda * inputs.entry_mult;
  const { label: mu } = useMoney();
  const units = useTranslations("units");

  return (
    <DealScreen
      rail={
        <>
          <RailGroup title={t("groupMoney")}>
            <MoneyFields />
          </RailGroup>
          <RailGroup title={t("groupFiscal")}>
            <FiscalFields />
          </RailGroup>
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
          <RailGroup title={t("groupFinancing")}>
            {inputs.tranches.length ? (
              <TranchesOnDebtStep />
            ) : (
              <>
                <DebtMultipleField tranche="senior" />
                <DebtMultipleField tranche="mezz" />
                <DealField name="base_rate" />
                <DealField name="mezz_spread" />
              </>
            )}
          </RailGroup>
        </>
      }
    >
      {!r ? (
        <LoadingTiles />
      ) : (
        <Tiles>
          <Kpi title={t("kpiIrr")} value={fmtRate(r.irr)} lead {...hurdleSub(r.irr, hurdle)} />
          <Kpi title={t("kpiMoic")} value={fmtMultiple(r.moic)} sub={units("holdYears", { years: inputs.hold })} />
          <Kpi title={t("kpiEnterpriseValue")} value={fmtMoney(ev)} sub={t("evSub", { multiple: fmtMultiple(inputs.entry_mult, 1), money: mu })} />
          <Kpi
            title={t("kpiTotalDebt")}
            value={fmtMoney(listed ? drawnDebt(inputs.tranches) : (ev * inputs.debt_pct) / 100)}
            sub={t("shareOfEv", { share: fmtPct(debtShareOfEv(inputs)) })}
          />
          <Kpi title={t("kpiSponsorEquity")} value={fmtMoney(su?.sponsor_equity)} sub={t("inclFees", { money: mu })} />
          {listed ? (
            <Kpi
              title={t("kpiFacilities")}
              value={fmtCount(inputs.tranches.length)}
              sub={t("floatingSub", { count: floatingCount(inputs.tranches) })}
            />
          ) : (
            <Kpi
              title={t("kpiSeniorShare")}
              value={fmtPct(inputs.senior_pct)}
              sub={t("seniorMezzSub", { senior: fmtMultiple(seniorX, 1), mezz: fmtMultiple(mezzX, 1) })}
            />
          )}

          <Tile span={6} title={t("tileSources")} unit={mu}>
            <SuTable
              rows={[
                ...(listed
                  ? (su?.tranches ?? []).map((tr): [string, number] => [trancheName(tr.name), tr.amount])
                  : ([
                      [t("rowSeniorTermLoan"), su?.senior_debt],
                      [t("rowMezzanine"), su?.mezz_debt],
                    ] as [string, number | null | undefined][])),
                [t("rowSponsorEquity"), su?.sponsor_equity],
              ]}
              total={[t("rowTotalSources"), su?.total_sources]}
            />
          </Tile>
          <Tile
            span={6}
            title={t("tileUses")}
            unit={mu}
            aside={su && <span className={`chip ${su.balanced ? "text-gain" : "text-loss"}`}>{su.balanced ? t("balanced") : t("outOfBalance")}</span>}
          >
            <SuTable
              rows={[
                [t("rowPurchasePrice"), su?.equity_purchase_price],
                [t("rowTransactionFees"), su?.transaction_fees],
                [t("rowFinancingFees"), su?.financing_fees],
                ...(listed ? [[t("rowTrancheFees"), su?.tranche_fees] as [string, number | undefined]] : []),
                [t("rowOtherUses"), su?.other_uses],
                [t("rowCashToBalanceSheet"), su?.cash_to_balance_sheet],
              ]}
              total={[t("rowTotalUses"), su?.total_uses]}
            />
          </Tile>
          <DealRisk />
          <div className="col-span-12 flex items-center justify-between gap-4 bg-canvas px-3 py-3">
            <p className="type-body">{listed ? t("debtListedNote") : t("debtSizedNote")}</p>
            <Link href="/deal/debt" className="type-action-secondary px-2.5 py-1.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)]">
              {t("nextDebt")}
            </Link>
          </div>
        </Tiles>
      )}
    </DealScreen>
  );
}

function SuTable({ rows, total }: { rows: [string, number | null | undefined][]; total: [string, number | null | undefined] }) {
  return (
    <table className="w-full border-collapse font-mono text-[11.5px]">
      <tbody>
        {rows.map(([label, v]) => (
          <tr key={label}>
            <th scope="row" className="type-input-label border-b border-grid py-1.5 text-start font-normal text-soft">
              {label}
            </th>
            <td className="border-b border-grid py-1.5 text-end text-ink">{fmtMoney(v)}</td>
          </tr>
        ))}
        <tr>
          <th scope="row" className="type-input-label py-1.5 text-start font-normal text-bright">
            {total[0]}
          </th>
          <td className="py-1.5 text-end font-semibold text-bright" data-total={total[0]}>
            {fmtMoney(total[1])}
          </td>
        </tr>
      </tbody>
    </table>
  );
}
