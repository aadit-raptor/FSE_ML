"use client";

import { useTranslations } from "next-intl";

import { DataTable, type Row } from "@/components/charts/DataTable";
import { StackedBars } from "@/components/charts/StackedBars";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { useMoney } from "@/components/ui/MoneyScope";
import { debtShareOfEv } from "@/lib/deal/capital";
import { dealFiscal, type DealRun } from "@/lib/deal/fields";
import { downloadWorkbook, sheet } from "@/lib/export";
import { fmtMoney, fmtMultiple, fmtPct, fmtRate, isNum } from "@/lib/format";
import { useEngineLabel } from "@/lib/i18n/useEngineText";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";
import { useStandardLabel } from "@/lib/i18n/useStandardLabel";

import { useDeal } from "../DealProvider";
import { BusinessRiskSelect, DealDistressTile } from "../Distress";
import { DealField, DealScreen, LoadingTiles, RailGroup, WspToggle } from "../DealScreen";
import { debtSeries, debtYears, totals } from "./shared";
import { TrancheList, UseTranchesButton } from "./TrancheList";

type ScheduleRow = NonNullable<DealRun["tranches"][string]>[number];
type DetailKey = "cash_interest" | "pik_interest" | "commitment_fee" | "redrawn";

/**
 * The schedule rows a facility needs beyond the five every loan has: what
 * accrued in kind, the commitment fee, what was drawn. Each appears only when
 * it is not zero in some year (or when `when` says so), the way the equity
 * bridge hides a fee step a deal does not have.
 */
function detailRows(
  rows: ScheduleRow[],
  wanted: [DetailKey, string, Row["kind"], ((r: ScheduleRow) => boolean)?][],
): Row[] {
  return wanted
    .filter(([key, , , when]) => rows.some((r) => (when ? when(r) : (r[key] ?? 0) !== 0)))
    .map(([key, label, kind]) => ({ label, values: rows.map((r) => r[key]), kind }));
}

/** One row per facility: what it is, how big, how priced, when it is due, what it sweeps and accrues. */
function CapitalStructureTable({ rows }: { rows: NonNullable<DealRun["capital_structure"]> }) {
  const t = useTranslations("deal");
  const kinds = useTranslations("trancheKinds");
  const rates = useTranslations("referenceRates");
  const trancheName = useEngineLabel("tranche");
  const head = [t("colFacility"), t("colAmount"), t("colTimesEbitda"), t("colRate"), t("colMaturity"), t("colSweep"), t("colPik")];
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse font-mono text-[11px]">
        <caption className="sr-only">{t("tileCapitalStructure")}</caption>
        <thead>
          <tr>
            {head.map((h, i) => (
              <th key={h} scope="col" className={`border-b border-grid px-2 py-1 font-normal text-muted ${i === 0 ? "text-start" : "text-end"}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} className="text-ink" data-facility={r.name}>
              <th scope="row" className="border-b border-grid px-2 py-1 text-start font-normal">
                <span className="type-input-label block text-soft">{trancheName(r.name)}</span>
                {/* i18n-keys: trancheKinds.*, referenceRates.* */}
                <span className="block text-[9.5px] text-dim">
                  {!r.floating
                    ? t("kindFixed", { kind: kinds(r.kind) })
                    : r.reference_rate && r.reference_rate !== "custom"
                      ? t("kindFloating", { kind: kinds(r.kind), reference: rates(r.reference_rate) })
                      : t("kindFloatingOwnRate", { kind: kinds(r.kind) })}
                </span>
              </th>
              <td className="border-b border-grid px-2 py-1 text-end">
                {fmtMoney(r.amount)}
                {isNum(r.commitment) && <span className="block text-[9.5px] text-dim">{t("ofCommitment", { commitment: fmtMoney(r.commitment) })}</span>}
              </td>
              <td className="border-b border-grid px-2 py-1 text-end">{fmtMultiple(r.x_ebitda, 1)}</td>
              <td className="border-b border-grid px-2 py-1 text-end">{fmtRate(r.rate, 2)}</td>
              <td className="border-b border-grid px-2 py-1 text-end">{t("years", { years: r.maturity_years })}</td>
              <td className="border-b border-grid px-2 py-1 text-end">{fmtPct(r.sweep_share, 0)}</td>
              <td className="border-b border-grid px-2 py-1 text-end">{fmtPct(r.pik_share, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function DebtStep() {
  const { inputs, run } = useDeal();
  const t = useTranslations("deal");
  const res = run.result;

  return (
    <DealScreen
      rail={
        <>
          <RailGroup title={t("groupCapital")}>
            {inputs.tranches.length ? (
              <TrancheList />
            ) : (
              <>
                <DealField name="debt_pct" />
                <DealField name="senior_pct" />
                <DealField name="base_rate" />
                <DealField name="mezz_spread" />
                <UseTranchesButton />
              </>
            )}
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
          <RailGroup title={t("groupCredit")}>
            <BusinessRiskSelect />
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
  const { inputs, run, current } = useDeal();
  const t = useTranslations("deal");
  const x = useTranslations("export");
  const fiscalLabels = useFiscalLabels();
  const trancheName = useEngineLabel("tranche");
  const std = useStandardLabel(inputs.accounting_standard);
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
      <Kpi title={t("kpiDebtAtClose")} value={fmtMoney(atClose)} sub={t("debtAtCloseSub", { share: fmtPct(debtShareOfEv(inputs)), money: mu })} lead />
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
            { label: std("rowNetIncome", t("rowNetIncome")), values: cf.net_income ?? [] },
            { label: t("rowDa"), values: cf.da ?? [] },
            { label: t("rowCapex"), values: cf.capex ?? [], kind: "outflow" },
            { label: t("rowChangeInNwc"), values: cf.delta_nwc ?? [], kind: "outflow" },
            { label: t("rowLeveredFcf"), values: cf.levered_fcf ?? [], total: true },
            { label: t("rowCumulativeFcf"), values: cf.cumulative_fcf ?? [] },
          ]}
        />
      </Tile>

      <DealDistressTile d={res.distress} years={years} />

      {(res.capital_structure?.length ?? 0) > 0 && (
        <Tile span={6} title={t("tileCapitalStructure")} unit={mu}>
          <CapitalStructureTable rows={res.capital_structure ?? []} />
        </Tile>
      )}

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
                  run.result?.model,
                  current?.id ?? null,
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
              // Shown only when the facility has them, so a plain loan's table is as it always was
              ...detailRows(rows, [
                ["cash_interest", t("rowCashInterest"), "outflow", (r) => (r.pik_interest ?? 0) !== 0 || (r.commitment_fee ?? 0) !== 0],
                ["pik_interest", t("rowPikAccrued"), "money"],
                ["commitment_fee", t("rowCommitmentFee"), "outflow"],
                ["redrawn", t("rowRedrawn"), "money"],
              ]),
            ]}
          />
        </Tile>
      ))}
    </Tiles>
  );
}
