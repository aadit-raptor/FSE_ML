"use client";

import { useTranslations } from "next-intl";

import { DataTable, type Row } from "@/components/charts/DataTable";
import { StackedBars } from "@/components/charts/StackedBars";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { quote } from "@/components/ui/Explain";
import { Term } from "@/components/ui/Term";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { useMoney } from "@/components/ui/MoneyScope";
import { debtShareOfEv } from "@/lib/deal/capital";
import { dealFiscal, type DealRun } from "@/lib/deal/fields";
import { downloadWorkbook, sheet } from "@/lib/export";
import { fmtMoney, fmtMultiple, fmtPct, fmtRate, isNum } from "@/lib/format";
import { statementTerm, trancheTerm, unitTerm } from "@/lib/glossary";
import { useEngineLabel } from "@/lib/i18n/useEngineText";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";
import { useStandardLabel } from "@/lib/i18n/useStandardLabel";

import { useDeal } from "../DealProvider";
import { BusinessRiskSelect, DealDistressTile } from "../Distress";
import { DealField, DealScreen, LoadingTiles, RailGroup, WspToggle } from "../DealScreen";
import { debtSeries, debtYears, facilityTerm, totals } from "./shared";
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
    .map(([key, label, kind]) => ({ label, values: rows.map((r) => r[key]), kind, term: DETAIL_TERMS[key] }));
}

/** Each detail row's glossary entry (PLAN.md 6.4) */
const DETAIL_TERMS: Record<DetailKey, string> = {
  cash_interest: "cashInterest",
  pik_interest: "pikAccrued",
  commitment_fee: "commitmentFee",
  redrawn: "redrawn",
};

/** One row per facility: what it is, how big, how priced, when it is due, what it sweeps and accrues. */
function CapitalStructureTable({ rows }: { rows: NonNullable<DealRun["capital_structure"]> }) {
  const t = useTranslations("deal");
  const kinds = useTranslations("trancheKinds");
  const rates = useTranslations("referenceRates");
  const trancheName = useEngineLabel("tranche");
  const head: [string, string][] = [
    [t("colFacility"), "facility"],
    [t("colAmount"), "facilityAmount"],
    [t("colTimesEbitda"), "timesEbitda"],
    [t("colRate"), "facilityRate"],
    [t("colMaturity"), "maturity"],
    [t("colSweep"), "sweepShare"],
    [t("colPik"), "pikShare"],
  ];
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse font-mono text-[11px]">
        <caption className="sr-only">{t("tileCapitalStructure")}</caption>
        <thead>
          <tr>
            {head.map(([h, term], i) => (
              <th key={h} scope="col" className={`border-b border-grid px-2 py-1 font-normal text-muted ${i === 0 ? "text-start" : "text-end"}`}>
                <Term id={term}>{h}</Term>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name} className="text-ink" data-facility={r.name}>
              <th scope="row" className="border-b border-grid px-2 py-1 text-start font-normal">
                <span className="type-input-label block text-soft">
                  <Term id={trancheTerm(r.name, r.kind)}>{trancheName(r.name)}</Term>
                </span>
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
  const x2 = useTranslations("explain");
  // Every figure once, as the tiles show it, for the KPIs and the explanation alike (PLAN.md 6.4)
  const shown = {
    atClose: fmtMoney(atClose),
    share: fmtPct(debtShareOfEv(inputs)),
    atExit: fmtMoney(lastEnd),
    repaid: fmtPct(repaidPct),
    interest: fmtMoney(interest[0]),
    cumFcf: fmtMoney(cumFcf),
    years: String(end.length),
  };
  // A sentence is said only when every figure it quotes exists (an unlevered deal repays nothing)
  const debtExplained = [
    ...(atClose > 0 ? [x2.rich("debt_raise", { ...shown, f: quote })] : []),
    ...([lastEnd, repaidPct, cumFcf].every(isNum) ? [x2.rich("debt_repay", { ...shown, f: quote })] : []),
    ...(isNum(interest[0]) ? [x2.rich("debt_interest", { ...shown, f: quote })] : []),
  ];

  return (
    <Tiles>
      <Kpi term="debtAtClose" title={t("kpiDebtAtClose")} value={shown.atClose} sub={t("debtAtCloseSub", { share: shown.share, money: mu })} lead />
      <Kpi term="debtAtExit" title={t("kpiDebtAtExit")} value={shown.atExit} sub={t("repaidSub", { share: shown.repaid })} />
      <Kpi term="netDebtAtExit" title={t("kpiNetDebtAtExit")} value={fmtMoney(res.returns.net_debt_at_exit)} sub={t("afterCash", { money: mu })} />
      <Kpi term="interestYear1" title={t("kpiInterestYear1")} value={shown.interest} sub={mu} />
      <Kpi term="cumulativeFcf" title={t("kpiCumulativeFcf")} value={shown.cumFcf} sub={t("leveredYears", { years: end.length })} />
      <Kpi
        term="interestLoop"
        title={t("kpiInterestLoop")}
        value={res.interest_converged ? t("converged") : t("notConverged")}
        sub={t("interestGapSub", { gap: fmtMoney(interestGap), money: mu })}
        tone={res.interest_converged ? "gain" : "attention"}
      />

      <Tile span={6} term="debtBalance" title={t("tileDebtBalance")} unit={t("byTranche", { money: mu })} explain={debtExplained}>
        <StackedBars {...debtSeries(res, fiscalLabels.deal(debtYears(res), fiscal), { close: t("close"), chart: t("debtBalanceByTranche"), trancheName })} />
      </Tile>
      <Tile span={6} term="cashFlow" title={t("tileCashFlow")} unit={mu} unitTerm={unitTerm(money.unit)}>
        <DataTable
          caption={t("leveredFcfByYear")}
          columns={years}
          rows={[
            { label: std("rowNetIncome", t("rowNetIncome")), values: cf.net_income ?? [], term: statementTerm("netIncome", inputs.accounting_standard) },
            { label: t("rowDa"), values: cf.da ?? [], term: "da" },
            { label: t("rowCapex"), values: cf.capex ?? [], kind: "outflow", term: "capex" },
            { label: t("rowChangeInNwc"), values: cf.delta_nwc ?? [], kind: "outflow", term: "changeInNwc" },
            { label: t("rowLeveredFcf"), values: cf.levered_fcf ?? [], total: true, term: "leveredFcf" },
            { label: t("rowCumulativeFcf"), values: cf.cumulative_fcf ?? [], term: "cumulativeFcf" },
          ]}
        />
      </Tile>

      <DealDistressTile d={res.distress} years={years} />

      {(res.capital_structure?.length ?? 0) > 0 && (
        <Tile span={6} term="capitalStructure" title={t("tileCapitalStructure")} unit={mu} unitTerm={unitTerm(money.unit)}>
          <CapitalStructureTable rows={res.capital_structure ?? []} />
        </Tile>
      )}

      {Object.entries(res.tranches).map(([name, rows]) => (
        <Tile
          key={name}
          span={6}
          term={facilityTerm(res, name)}
          data={{ "data-facility-tile": name }}
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
              { label: t("rowOpening"), values: rows.map((r) => r.beginning_balance), term: "opening" },
              { label: t("rowMandatory"), values: rows.map((r) => r.mandatory_repayment), kind: "outflow", term: "mandatory" },
              { label: t("rowCashSweep"), values: rows.map((r) => r.cash_sweep), kind: "outflow", term: "cashSweep" },
              { label: t("rowClosing"), values: rows.map((r) => r.ending_balance), total: true, term: "closing" },
              { label: t("rowInterest"), values: rows.map((r) => r.interest_expense), kind: "outflow", term: "interest" },
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
