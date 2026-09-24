"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { useSettings } from "@/components/settings/SettingsProvider";
import { useMoney } from "@/components/ui/MoneyScope";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { api, type Schemas } from "@/lib/api/client";
import { multiplesFromPct } from "@/lib/deal/capital";
import { fmtMoney, fmtMultiple, fmtPct, fmtRate } from "@/lib/format";

import { useDeal } from "../DealProvider";
import { DealRisk } from "../DealRisk";
import { DealField, DealScreen, DebtMultipleField, FiscalFields, LoadingTiles, MoneyFields, RailGroup } from "../DealScreen";
import { hurdleSub } from "./shared";

type SourcesUses = Schemas["SourcesUsesResponse"];

function useSourcesAndUses(ebitda: number, entryMult: number, seniorX: number, mezzX: number, mincash: number) {
  const { overrides } = useSettings();
  const { money } = useMoney();
  const [su, setSu] = useState<SourcesUses | null>(null);
  useEffect(() => {
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      api
        .POST("/api/deal/sources-and-uses", {
          body: { ebitda, entry_mult: entryMult, senior_x: seniorX, mezz_x: mezzX, mincash, settings: overrides, money },
          signal: ctrl.signal,
        })
        .then(({ data }) => data && setSu(data))
        .catch(() => {});
    }, 300);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [ebitda, entryMult, seniorX, mezzX, mincash, overrides, money]);
  return su;
}

export function InputsStep() {
  const { inputs, run, hurdle } = useDeal();
  const { seniorX, mezzX } = multiplesFromPct(inputs.entry_mult, inputs.debt_pct, inputs.senior_pct);
  const su = useSourcesAndUses(inputs.ebitda, inputs.entry_mult, Number(seniorX.toFixed(6)), Number(mezzX.toFixed(6)), inputs.mincash);
  const r = run.result?.returns;
  const ev = inputs.ebitda * inputs.entry_mult;
  const { label: mu } = useMoney();

  return (
    <DealScreen
      rail={
        <>
          <RailGroup title="Money">
            <MoneyFields />
          </RailGroup>
          <RailGroup title="Fiscal years">
            <FiscalFields />
          </RailGroup>
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
          <RailGroup title="Financing, x EBITDA">
            <DebtMultipleField tranche="senior" />
            <DebtMultipleField tranche="mezz" />
            <DealField name="base_rate" />
            <DealField name="mezz_spread" />
          </RailGroup>
        </>
      }
    >
      {!r ? (
        <LoadingTiles />
      ) : (
        <Tiles>
          <Kpi title="IRR" value={fmtRate(r.irr)} lead {...hurdleSub(r.irr, hurdle)} />
          <Kpi title="MOIC" value={fmtMultiple(r.moic)} sub={`${inputs.hold} yr hold`} />
          <Kpi title="Enterprise value" value={fmtMoney(ev)} sub={`${fmtMultiple(inputs.entry_mult, 1)} EBITDA, ${mu}`} />
          <Kpi title="Total debt" value={fmtMoney((ev * inputs.debt_pct) / 100)} sub={`${fmtPct(inputs.debt_pct)} of EV`} />
          <Kpi title="Sponsor equity" value={fmtMoney(su?.sponsor_equity)} sub={`${mu} incl. fees`} />
          <Kpi title="Senior share" value={fmtPct(inputs.senior_pct)} sub={`${fmtMultiple(seniorX, 1)} senior, ${fmtMultiple(mezzX, 1)} mezz`} />

          <Tile span={6} title="Sources" unit={mu}>
            <SuTable
              rows={[
                ["Senior term loan", su?.senior_debt],
                ["Mezzanine", su?.mezz_debt],
                ["Sponsor equity", su?.sponsor_equity],
              ]}
              total={["Total sources", su?.total_sources]}
            />
          </Tile>
          <Tile span={6} title="Uses" unit={mu} aside={su && <span className={`chip ${su.balanced ? "text-gain" : "text-loss"}`}>{su.balanced ? "balanced" : "out of balance"}</span>}>
            <SuTable
              rows={[
                ["Purchase price", su?.equity_purchase_price],
                ["Transaction fees", su?.transaction_fees],
                ["Financing fees", su?.financing_fees],
                ["Other uses", su?.other_uses],
                ["Cash to balance sheet", su?.cash_to_balance_sheet],
              ]}
              total={["Total uses", su?.total_uses]}
            />
          </Tile>
          <DealRisk />
          <div className="col-span-12 flex items-center justify-between gap-4 bg-canvas px-3 py-3">
            <p className="type-body">Debt is sized as a multiple of EBITDA here and as a share of EV on the next step. Both edit the same deal.</p>
            <Link href="/deal/debt" className="type-action-secondary px-2.5 py-1.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)]">
              Next: debt &amp; cash flow
            </Link>
          </div>
        </Tiles>
      )}
    </DealScreen>
  );
}

function SuTable({ rows, total }: { rows: [string, number | undefined][]; total: [string, number | undefined] }) {
  return (
    <table className="w-full border-collapse font-mono text-[11.5px]">
      <tbody>
        {rows.map(([label, v]) => (
          <tr key={label}>
            <th scope="row" className="type-input-label border-b border-grid py-1.5 text-left font-normal text-soft">
              {label}
            </th>
            <td className="border-b border-grid py-1.5 text-right text-ink">{fmtMoney(v)}</td>
          </tr>
        ))}
        <tr>
          <th scope="row" className="type-input-label py-1.5 text-left font-normal text-bright">
            {total[0]}
          </th>
          <td className="py-1.5 text-right font-semibold text-bright" data-total={total[0]}>
            {fmtMoney(total[1])}
          </td>
        </tr>
      </tbody>
    </table>
  );
}
