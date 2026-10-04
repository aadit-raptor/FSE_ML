"use client";

import { useTranslations } from "next-intl";
import { useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

import { DataTable, type Row } from "@/components/charts/DataTable";
import { LineChart } from "@/components/charts/LineChart";
import { Waterfall } from "@/components/charts/Waterfall";
import { CellInput } from "@/components/ui/CellInput";
import { DownloadButton } from "@/components/ui/DownloadButton";
import { EmptyState, LoadingTiles, Notice, RailGroup, Screen, SecondaryButton } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { MoneyScope, useMoney } from "@/components/ui/MoneyScope";
import { FiscalSelects } from "@/components/ui/FiscalSelects";
import { MoneySelects, SELECT_CLASS } from "@/components/ui/MoneySelects";
import { downloadWorkbook, sheet, tableSheet } from "@/lib/export";
import { ASSUMPTION_GROUPS, HISTORY_GROUPS } from "@/lib/forecast";
import { fmtCount, fmtInput, fmtMoney, fmtNumber, fmtRate, isNum } from "@/lib/format";
import { useStandardLabel } from "@/lib/i18n/useStandardLabel";
import { withMoney } from "@/lib/money";

import { useDeal } from "@/components/deal/DealProvider";
import type { Schemas } from "@/lib/api/client";

import { type ForecastRun, type Standard, useForecast } from "./ForecastProvider";

function Rail() {
  const { source, fetchEdgar, edgar, resetSample, metrics, status, result, money, setMoney, fiscal, setFiscal, histLabels, standard, setStandard } =
    useForecast();
  const t = useTranslations("forecast");
  const dealText = useTranslations("deal");
  const fiscalText = useTranslations("fiscal");
  const [ticker, setTicker] = useState("");
  const latest = metrics?.at(-1);
  return (
    <>
      <div className="flex items-center justify-between gap-2 border-b border-line px-3.5 py-2">
        <span className="type-control">{source.kind === "edgar" ? source.ticker : t("sampleCompany")}</span>
        <span role="status" className={`font-mono text-[10px] ${status === "error" ? "text-loss" : status === "running" ? "text-attention" : "text-dim"}`}>
          {status === "running" ? t("stateUpdating") : status === "error" ? t("stateError") : result ? t("stateUpToDate") : t("stateLoading")}
        </span>
      </div>
      <RailGroup title={t("groupReportingCurrency")}>
        <MoneySelects money={money} onChange={setMoney} of={t("of")} />
      </RailGroup>
      <RailGroup title={t("groupStandard")}>
        <label className="grid grid-cols-[minmax(0,1fr)_128px] items-center gap-1.5 py-px">
          <span className="type-input-label">{t("groupStandard")}</span>
          <select value={standard} onChange={(e) => setStandard(e.target.value as Standard)} className={SELECT_CLASS}>
            {STANDARDS.map((s) => (
              <option key={s} value={s}>
                {dealText(`standard_${s}`)}
              </option>
            ))}
          </select>
        </label>
      </RailGroup>
      <RailGroup title={t("groupEdgar")}>
        <form
          className="grid grid-cols-[1fr_auto] gap-2"
          onSubmit={(ev) => {
            ev.preventDefault();
            void fetchEdgar(ticker);
          }}
        >
          <label className="sr-only" htmlFor="edgar-ticker">
            {t("ticker")}
          </label>
          <input
            id="edgar-ticker"
            value={ticker}
            onChange={(ev) => setTicker(ev.target.value)}
            placeholder={t("tickerPlaceholder")}
            autoComplete="off"
            className="border border-line bg-field px-2 py-1 font-mono text-[11.5px] text-ink uppercase outline-none placeholder:text-dim placeholder:normal-case focus:border-accent"
          />
          <button
            type="submit"
            disabled={edgar.status === "loading" || !ticker.trim()}
            className="type-action-secondary px-2.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)] disabled:opacity-50"
          >
            {edgar.status === "loading" ? t("fetching") : t("fetch")}
          </button>
        </form>
        {edgar.status === "error" && <p className="font-mono text-[10px] text-loss">{edgar.error}</p>}
        {source.kind === "edgar" && (
          <div className="grid gap-1 pt-1">
            <p className="type-input-label">{source.company}</p>
            <p className="font-mono text-[10px] text-muted">{t("fiscalYears", { years: histLabels.join(", ") })}</p>
            {source.warnings.map((w) => (
              <p key={w} className="font-mono text-[10px] text-attention">
                {w}
              </p>
            ))}
            {source.dealInputs && <UseInDeal inputs={source.dealInputs} />}
          </div>
        )}
        <div className="pt-1.5">
          <SecondaryButton onClick={resetSample}>{t("resetToSample")}</SecondaryButton>
        </div>
      </RailGroup>
      <RailGroup title={t("groupFiscal")}>
        <FiscalSelects fiscal={fiscal} onChange={setFiscal} of={t("of")} yearLabel={fiscalText("latestFiscalYear")} />
      </RailGroup>
      {latest && (
        <RailGroup title={t("groupRatios")}>
          <dl className="grid grid-cols-[1fr_auto] gap-x-2 gap-y-1">
            {(
              [
                [t("ratioRevenue"), fmtMoney(latest.revenue)],
                [t("ratioRevenueGrowth"), fmtRate(latest.revenue_growth)],
                [t("ratioGrossMargin"), fmtRate(latest.gross_margin)],
                [t("ratioRd"), fmtRate(latest.rd_pct)],
                [t("ratioSga"), fmtRate(latest.sga_pct)],
                [t("ratioEbitdaMargin"), fmtRate(latest.ebitda_margin)],
                [t("ratioAdjEbitdaMargin"), fmtRate(latest.adj_ebitda_margin)],
              ] as const
            ).map(([k, v]) => (
              <div key={k} className="contents">
                <dt className="type-input-label">{k}</dt>
                <dd className="text-end font-mono text-[11.5px] text-ink">{v}</dd>
              </div>
            ))}
          </dl>
        </RailGroup>
      )}
    </>
  );
}

/** i18n-keys: deal.standard_, deal.standard_ifrs, deal.standard_us_gaap */
const STANDARDS: Standard[] = ["", "ifrs", "us_gaap"];

/**
 * A filing's latest year as the open deal's inputs (PLAN.md 2.6): its EBITDA as the standard
 * reports it, its currency and unit, its standard and its leases. The deal's other money is
 * converted to the filing's unit first, so nothing else changes size.
 */
function UseInDeal({ inputs }: { inputs: Schemas["EdgarDealInputs"] }) {
  const { setMoney, setFields } = useDeal();
  const router = useRouter();
  const t = useTranslations("forecast");
  return (
    <div className="grid gap-1 pt-1.5">
      <SecondaryButton
        onClick={() => {
          setMoney({ currency: inputs.currency, unit: inputs.unit });
          setFields({
            ebitda: Number(inputs.ebitda.toFixed(6)),
            accounting_standard: inputs.accounting_standard,
            lease_cost: Number(inputs.lease_cost.toFixed(6)),
            lease_liability: Number(inputs.lease_liability.toFixed(6)),
          });
          router.push("/deal/inputs");
        }}
      >
        {t("useInDeal")}
      </SecondaryButton>
      <p className="font-mono text-[10px] text-muted">{t("useInDealNote", { ebitda: fmtMoney(inputs.ebitda) })}</p>
    </div>
  );
}

function ForecastScreen({ children, needsResult = true }: { children: ReactNode; needsResult?: boolean }) {
  const { activate, ready, result, status, error, money } = useForecast();
  const t = useTranslations("forecast");
  useEffect(() => activate(), [activate]);
  const bar =
    status === "error" ? (
      <Notice tone="loss" title={t("didntRun")} role="alert">
        {error}
      </Notice>
    ) : undefined;
  const body = !ready ? (
    status === "error" ? (
      <EmptyState title={t("noForecast")}>{t("noForecastBody")}</EmptyState>
    ) : (
      <LoadingTiles />
    )
  ) : needsResult && !result ? (
    <LoadingTiles />
  ) : (
    children
  );
  // The company's reporting currency, not the open deal's
  return (
    <MoneyScope money={money}>
      <Screen rail={<Rail />} bar={bar}>
        {body}
      </Screen>
    </MoneyScope>
  );
}

export function HistoricalsStep() {
  return (
    <ForecastScreen needsResult={false}>
      <Historicals />
    </ForecastScreen>
  );
}

function Historicals() {
  const { label: mu } = useMoney();
  const { history, setHistory, histLabels: cols } = useForecast();
  const t = useTranslations("forecast");
  return (
    <Tiles>
      {HISTORY_GROUPS.map((g) => (
        <Tile key={g.titleKey} span={6} title={withMoney(t(g.titleKey), mu)}>
          <EditableGrid rows={g.rows} columns={cols} values={history} onCommit={setHistory} />
        </Tile>
      ))}
    </Tiles>
  );
}

/**
 * A row's label and each cell's screen-reader label.
 *
 * i18n-keys: forecast.h_*, forecast.rev_g, forecast.gm, forecast.rd
 * i18n-keys: forecast.sga, forecast.tax, forecast.da, forecast.sbc
 * i18n-keys: forecast.capex, forecast.ar_d, forecast.inv_d, forecast.ap_d
 * i18n-keys: forecast.ocl_pct, forecast.def_pct, forecast.nca_pct
 * i18n-keys: forecast.other_inc, forecast.divs, forecast.buybacks
 * i18n-keys: forecast.ltd_chg, forecast.r_cash, forecast.r_debt, forecast.min_cash
 */
function EditableGrid({
  rows,
  columns,
  values,
  onCommit,
  extra,
}: {
  rows: string[];
  columns: string[];
  values: Record<string, number[]>;
  onCommit: (key: string, col: number, value: number) => void;
  extra?: (key: string) => ReactNode;
}) {
  const { label: mu } = useMoney();
  const t = useTranslations("forecast");
  const rowLabel = (key: string) => withMoney(t(key), mu);
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse">
        <thead>
          <tr>
            <th />
            {columns.map((c) => (
              <th key={c} scope="col" className="px-1 py-1 text-end font-mono text-[10.5px] font-normal text-muted">
                {c}
              </th>
            ))}
            {extra && <th />}
          </tr>
        </thead>
        <tbody>
          {rows.map((key) => (
            <tr key={key}>
              <th scope="row" className="type-input-label py-0.5 pe-2 text-start text-[9px] font-normal whitespace-nowrap">
                {rowLabel(key)}
              </th>
              {columns.map((c, i) => (
                <td key={c} className="px-1 py-0.5">
                  <CellInput label={t("cellLabel", { row: rowLabel(key), column: c })} value={values[key]?.[i] ?? NaN} onCommit={(v) => onCommit(key, i, v)} />
                </td>
              ))}
              {extra && <td className="py-0.5 ps-2 whitespace-nowrap">{extra(key)}</td>}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function AssumptionsStep() {
  return (
    <ForecastScreen needsResult={false}>
      <Assumptions />
    </ForecastScreen>
  );
}

function Assumptions() {
  const { label: mu } = useMoney();
  const { assumptions, setAssumption, fillAssumption, seeded, reseed, fwdLabels: cols } = useForecast();
  const t = useTranslations("forecast");
  return (
    <Tiles>
      <Notice
        tone="info"
        title={t("suggestedTitle")}
        className="col-span-12"
        actions={
          <SecondaryButton onClick={reseed} disabled={!seeded}>
            {t("useAllSuggestions")}
          </SecondaryButton>
        }
      >
        {t("suggestedBody")}
      </Notice>
      {ASSUMPTION_GROUPS.map((g) => (
        <Tile key={g.titleKey} span={6} title={withMoney(t(g.titleKey), mu)}>
          <EditableGrid
            rows={g.rows}
            columns={cols}
            values={assumptions}
            onCommit={setAssumption}
            extra={(key) =>
              seeded && isNum(seeded[key]) ? (
                <button
                  type="button"
                  onClick={() => fillAssumption(key, seeded[key])}
                  title={t("useForEveryYear")}
                  className="font-mono text-[10px] text-accent hover:underline"
                >
                  {fmtInput(seeded[key], 1)}
                </button>
              ) : null
            }
          />
        </Tile>
      ))}
    </Tiles>
  );
}

function statementRows(res: ForecastRun, keys: [string, string, Row["kind"]?, boolean?][]): Row[] {
  const ltm = res.ltm as Record<string, unknown>;
  return keys.map(([key, label, kind, total]) => ({
    label,
    kind,
    total,
    values: [typeof ltm[key] === "number" ? (ltm[key] as number) : null, ...res.years.map((y) => (y as Record<string, unknown>)[key] as number | null)],
  }));
}

type Schedules = {
  ppe: Row[];
  retained: Row[];
  workingCapital: Row[];
  cycleDays: number[];
  interest: Row[];
  revolver: Row[];
};

/** Rows for a forecast-years-only schedule. */
const fwdRow = (label: string, values: number[], kind?: Row["kind"], total?: boolean): Row => ({ label, values, kind, total });

function useStatementTables(): (res: ForecastRun) => { income: Row[]; balance: Row[]; cash: Row[] } {
  const t = useTranslations("forecast");
  const std = useStandardLabel(useForecast().standard);
  return (res) => ({
    income: statementRows(res, [
      ["revenue", t("rowRevenue")],
      ["gross_profit", t("rowGrossProfit")],
      ["rd", t("rowRd")],
      ["sga", t("rowSga")],
      ["ebit", t("rowEbit"), undefined, true],
      ["interest_inc", std("rowInterestIncome", t("rowInterestIncome"))],
      ["interest_exp", std("rowInterestExpense", t("rowInterestExpense"))],
      ["pretax", std("rowPretax", t("rowPretax"))],
      ["taxes", std("rowTaxes", t("rowTaxes"))],
      ["net_income", std("rowNetIncome", t("rowNetIncome")), undefined, true],
      ["ebitda", t("rowEbitda")],
      ["ebitda_margin", t("rowEbitdaMargin"), "rate"],
    ]),
    balance: statementRows(res, [
      ["cash", t("rowCash")],
      ["ar", std("rowAr", t("rowAr"))],
      ["inventory", t("rowInventory")],
      ["ppe_net", t("rowPpeNet")],
      ["total_assets", t("rowTotalAssets"), undefined, true],
      ["ap", std("rowAp", t("rowAp"))],
      ["revolver", t("rowRevolver")],
      ["ltd", t("rowLtd")],
      ["total_liab", t("rowTotalLiab")],
      ["total_equity", t("rowTotalEquity")],
      ["balance_check", t("rowBalanceCheck")],
    ]),
    cash: statementRows(res, [
      ["cfo", t("rowCfo")],
      ["cfi", t("rowCfi")],
      ["cff", t("rowCff")],
      ["net_cash_chg", t("rowNetCashChange"), undefined, true],
      ["delta_nwc", t("rowDeltaNwc")],
      ["revolver_draw", t("rowRevolverDraw")],
    ]),
  });
}

/** Supporting schedules, derived exactly from the three statements. */
function useScheduleTables(): (res: ForecastRun, assumptions: Record<string, number[]>) => Schedules {
  const t = useTranslations("forecast");
  const std = useStandardLabel(useForecast().standard);
  return (res, assumptions) => {
  const y = res.years;
  const ltm = res.ltm;
  const at = (key: string, i: number) => assumptions[key]?.[i] ?? NaN;
  const prevCash = y.map((_, i) => (i === 0 ? ltm.cash : y[i - 1].cash) ?? 0);
  const prevDebt = y.map((_, i) => (i === 0 ? ltm.ltd : y[i - 1].ltd) ?? 0);
  return {
    ppe: [
      fwdRow(t("rowOpeningPpe"), y.map((r) => r.ppe_beg ?? 0)),
      // Capex isn't reported separately: the roll-forward implies it exactly
      fwdRow(t("rowCapex"), y.map((r) => (r.ppe_end ?? 0) - (r.ppe_beg ?? 0) + (r.da ?? 0))),
      fwdRow(t("rowDepreciation"), y.map((r) => r.da ?? 0), "outflow"),
      fwdRow(t("rowClosingPpe"), y.map((r) => r.ppe_end ?? 0), undefined, true),
    ],
    retained: [
      fwdRow(t("rowOpeningRe"), y.map((r) => r.re_beg ?? 0)),
      fwdRow(std("rowNetIncome", t("rowNetIncome")), y.map((r) => r.net_income ?? 0)),
      fwdRow(t("rowDividendsBuybacks"), y.map((r) => (r.re_beg ?? 0) + (r.net_income ?? 0) - (r.re_end ?? 0)), "outflow"),
      fwdRow(t("rowClosingRe"), y.map((r) => r.re_end ?? 0), undefined, true),
    ],
    workingCapital: [
      fwdRow(t("rowReceivables"), y.map((r) => r.ar ?? 0)),
      fwdRow(t("rowInventory"), y.map((r) => r.inventory ?? 0)),
      fwdRow(t("rowPayables"), y.map((r) => r.ap ?? 0), "outflow"),
      fwdRow(t("rowNwc"), y.map((r) => r.nwc ?? 0), undefined, true),
      fwdRow(t("rowDeltaNwc"), y.map((r) => r.delta_nwc ?? 0)),
    ],
    cycleDays: y.map((_, i) => at("ar_d", i) + at("inv_d", i) - at("ap_d", i)),
    interest: [
      fwdRow(t("rowOpeningCash"), prevCash),
      fwdRow(t("rowRateOnCash"), y.map((_, i) => at("r_cash", i) / 100), "rate"),
      fwdRow(std("rowInterestIncome", t("rowInterestIncome")), y.map((r) => r.interest_inc ?? 0)),
      fwdRow(t("rowOpeningDebt"), prevDebt),
      fwdRow(t("rowRateOnDebt"), y.map((_, i) => at("r_debt", i) / 100), "rate"),
      fwdRow(std("rowInterestExpense", t("rowInterestExpense")), y.map((r) => r.interest_exp ?? 0), "outflow", true),
    ],
    revolver: [
      fwdRow(t("rowDrawRepay"), y.map((r) => r.revolver_draw ?? 0)),
      fwdRow(t("rowClosingRevolver"), y.map((r) => r.revolver ?? 0), undefined, true),
      fwdRow(t("rowClosingCash"), y.map((r) => r.cash ?? 0)),
    ],
  };
  };
}

export function StatementsStep() {
  return (
    <ForecastScreen>
      <Statements />
    </ForecastScreen>
  );
}

function Statements() {
  const { label: mu, money } = useMoney();
  const { result: res, assumptions, source, histLabels, fwdLabels: fwd, standard } = useForecast();
  const t = useTranslations("forecast");
  const x = useTranslations("export");
  const std = useStandardLabel(standard);
  const fiscalText = useTranslations("fiscal");
  const statementTables = useStatementTables();
  const scheduleTables = useScheduleTables();
  if (!res) return null;
  const tables = statementTables(res);
  const schedules = scheduleTables(res, assumptions);
  const cols = [histLabels.at(-1) ?? fiscalText("ltm"), ...fwd];
  const last = res.years.at(-1);
  const maxGap = Math.max(0, ...res.forecast_balance_gaps.map(Math.abs));
  const company = source.kind === "edgar" ? source.ticker : t("sampleCompany");
  const everything = () => [
    tableSheet(x("sheetIncomeStatement"), cols, tables.income),
    tableSheet(x("sheetBalanceSheet"), cols, tables.balance),
    tableSheet(x("sheetCashFlow"), cols, tables.cash),
    tableSheet(x("sheetPpe"), fwd, schedules.ppe),
    tableSheet(x("sheetRetainedEarnings"), fwd, schedules.retained),
    tableSheet(x("sheetWorkingCapital"), fwd, schedules.workingCapital),
    tableSheet(x("sheetInterest"), fwd, schedules.interest),
    tableSheet(x("sheetRevolver"), fwd, schedules.revolver),
  ];
  const lastCol = cols.at(-1) ?? "";
  return (
    <Tiles>
      <Kpi title={t("kpiRevenue", { year: lastCol })} value={fmtMoney(last?.revenue)} sub={t("cagrSub", { cagr: fmtRate(res.revenue_cagr) })} lead />
      <Kpi title={t("kpiEbitda", { year: lastCol })} value={fmtMoney(last?.ebitda)} sub={t("marginSub", { margin: fmtRate(last?.ebitda_margin) })} />
      <Kpi title={std("kpiNetIncome", t("kpiNetIncome", { year: lastCol }), { year: lastCol })} value={fmtMoney(last?.net_income)} sub={t("marginSub", { margin: fmtRate(last?.net_margin) })} />
      <Kpi title={t("kpiCash", { year: lastCol })} value={fmtMoney(last?.cash)} sub={mu} />
      <Kpi
        title={t("kpiBalanceSheet")}
        value={res.balanced ? t("balances") : t("doesntBalance")}
        sub={t("largestGap", { gap: fmtMoney(maxGap), money: mu })}
        tone={res.balanced ? "gain" : "loss"}
      />
      <Kpi
        title={t("kpiOpeningGap")}
        value={fmtMoney(res.opening_balance_gap)}
        sub={t("openingGapSub", { money: mu })}
        tone={Math.abs(res.opening_balance_gap) > 0.5 ? "attention" : undefined}
      />
      <div className="col-span-12 flex items-center justify-between gap-4 bg-canvas px-3 py-2">
        <p className="type-body">{t("modelNote")}</p>
        <DownloadButton label={t("modelDownload")} onDownload={() => downloadWorkbook(x("fileThreeStatement", { company }), everything(), money, res?.model)} />
      </div>
      <Tile
        span={12}
        title={t("tileIncomeStatement")}
        unit={mu}
        action={
          <DownloadButton onDownload={() => downloadWorkbook(x("fileIncomeStatement"), [tableSheet(x("sheetIncomeStatement"), cols, tables.income)], money, res?.model)} />
        }
      >
        <DataTable caption={t("chartIncomeStatement")} columns={cols} rows={tables.income} />
      </Tile>
      <Tile
        span={6}
        title={t("tileBalanceSheet")}
        unit={mu}
        action={<DownloadButton onDownload={() => downloadWorkbook(x("fileBalanceSheet"), [tableSheet(x("sheetBalanceSheet"), cols, tables.balance)], money, res?.model)} />}
      >
        <DataTable caption={t("chartBalanceSheet")} columns={cols} rows={tables.balance} />
      </Tile>
      <Tile
        span={6}
        title={t("tileCashFlow")}
        unit={mu}
        action={<DownloadButton onDownload={() => downloadWorkbook(x("fileForecastCashFlow"), [tableSheet(x("sheetCashFlow"), cols, tables.cash)], money, res?.model)} />}
      >
        <DataTable caption={t("chartCashFlow")} columns={cols} rows={tables.cash} />
      </Tile>
    </Tiles>
  );
}

export function SchedulesStep() {
  return (
    <ForecastScreen>
      <Schedules />
    </ForecastScreen>
  );
}

function Schedules() {
  const { label: mu, money } = useMoney();
  const { result: res, assumptions, fwdLabels: fwd, standard } = useForecast();
  const t = useTranslations("forecast");
  const x = useTranslations("export");
  const std = useStandardLabel(standard);
  const scheduleTables = useScheduleTables();
  if (!res) return null;
  const s = scheduleTables(res, assumptions);
  const y = res.years.at(-1);
  const bridge = y
    ? [
        { label: t("bridgeEbitda"), value: y.ebitda ?? 0, isTotal: true },
        { label: t("bridgeDa"), value: (y.ebit ?? 0) - (y.ebitda ?? 0), isTotal: false },
        { label: t("bridgeIntIncome"), value: y.interest_inc ?? 0, isTotal: false },
        { label: t("bridgeIntExpense"), value: y.interest_exp ?? 0, isTotal: false },
        { label: t("bridgeOther"), value: (y.pretax ?? 0) - (y.ebit ?? 0) - (y.interest_inc ?? 0) - (y.interest_exp ?? 0), isTotal: false },
        { label: t("bridgeTaxes"), value: (y.net_income ?? 0) - (y.pretax ?? 0), isTotal: false },
        { label: std("bridgeNetIncome", t("bridgeNetIncome")), value: y.net_income ?? 0, isTotal: true },
      ]
    : [];
  const all = () => [
    tableSheet(x("sheetPpe"), fwd, s.ppe),
    tableSheet(x("sheetRetainedEarnings"), fwd, s.retained),
    tableSheet(x("sheetWorkingCapital"), fwd, s.workingCapital),
    tableSheet(x("sheetInterest"), fwd, s.interest),
    tableSheet(x("sheetRevolver"), fwd, s.revolver),
  ];
  const lastFwd = fwd.at(-1) ?? "";
  return (
    <Tiles>
      <div className="col-span-12 flex items-center justify-between gap-4 bg-canvas px-3 py-2">
        <p className="type-body">{t("schedulesNote")}</p>
        <DownloadButton label={t("allSchedules")} onDownload={() => downloadWorkbook(x("fileSchedules"), all(), money, res?.model)} />
      </div>
      <Tile span={6} title={t("tilePpe")} unit={mu}>
        <DataTable caption={t("chartPpe")} columns={fwd} rows={s.ppe} />
      </Tile>
      <Tile span={6} title={t("tileRetained")} unit={mu}>
        <DataTable caption={t("chartRetained")} columns={fwd} rows={s.retained} />
      </Tile>
      <Tile span={6} title={t("tileWorkingCapital")} unit={mu}>
        <DataTable caption={t("chartWorkingCapital")} columns={fwd} rows={s.workingCapital} />
        <p className="font-mono text-[10.5px] text-muted">
          {t("cycleDays", { days: s.cycleDays.map((d) => (Number.isFinite(d) ? `${fmtNumber(d, 0)}d` : fmtNumber(null))).join(" · ") })}
        </p>
      </Tile>
      <Tile span={6} title={t("tileInterest")} unit={mu}>
        <DataTable caption={t("chartInterest")} columns={fwd} rows={s.interest} />
      </Tile>
      <Tile span={6} title={t("tileRevolver")} unit={t("modelPlug", { money: mu })}>
        <DataTable caption={t("chartRevolver")} columns={fwd} rows={s.revolver} />
        <p className="type-body text-[9px]">{t("revolverNote")}</p>
      </Tile>
      <Tile span={6} title={t("tileBridge", { year: lastFwd })} unit={mu}>
        <Waterfall label={t("chartBridge", { year: lastFwd })} steps={bridge} />
      </Tile>
    </Tiles>
  );
}

export function SimulationStep() {
  return (
    <ForecastScreen>
      <Simulation />
    </ForecastScreen>
  );
}

function Simulation() {
  const { label: mu, money } = useMoney();
  const { result: res, simPaths, fwdLabels: cols } = useForecast();
  const t = useTranslations("forecast");
  const x = useTranslations("export");
  const sim = res?.simulation;
  if (!res || !sim) return <EmptyState title={t("noSimulation")}>{t("noSimulationBody")}</EmptyState>;
  const fan = (b: typeof sim.revenue_bands, det: number[], name: string) => (
    <LineChart
      label={t("chartFan", { name })}
      xLabels={cols}
      yFormat={(v) => fmtNumber(v, 0)}
      bands={[
        { lower: b.p5, upper: b.p95, color: "var(--color-accent)", opacity: 0.14 },
        { lower: b.p25, upper: b.p75, color: "var(--color-accent)", opacity: 0.28 },
      ]}
      lines={[
        { name: t("seriesP50"), values: b.p50, color: "var(--color-accent)" },
        { name: t("seriesPlan"), values: det, color: "var(--color-muted)", dashed: true, width: 1.5 },
      ]}
    />
  );
  const lastCol = cols.at(-1) ?? "";
  // i18n-keys: forecast.case*
  const caseKey = (scenario: string) => (scenario === "Bull" ? "caseBull" : scenario === "Bear" ? "caseBear" : "caseBase");
  return (
    <Tiles>
      <Kpi
        title={t("kpiRevenue", { year: lastCol })}
        value={fmtMoney(sim.revenue_final.median)}
        sub={t("planSub", { value: fmtMoney(sim.revenue_final.deterministic), money: mu })}
        lead
      />
      <Kpi title={t("kpiRevenueRange")} value={fmtMoney(sim.revenue_final.p5)} sub={t("toValue", { value: fmtMoney(sim.revenue_final.p95), money: mu })} />
      <Kpi
        title={t("kpiEbitda", { year: lastCol })}
        value={fmtMoney(sim.ebitda_final.median)}
        sub={t("planSub", { value: fmtMoney(sim.ebitda_final.deterministic), money: mu })}
      />
      <Kpi title={t("kpiEbitdaRange")} value={fmtMoney(sim.ebitda_final.p5)} sub={t("toValue", { value: fmtMoney(sim.ebitda_final.p95), money: mu })} />
      <Kpi title={t("kpiSimulatedGrowth")} value={fmtRate(sim.growth_final_mean)} sub={t("meanFinalYear")} />
      <Kpi title={t("kpiPaths")} value={fmtCount(sim.n)} sub={t("requestedSub", { count: fmtCount(simPaths) })} />
      <div className="col-span-12 flex items-center justify-between gap-4 bg-canvas px-3 py-2">
        <p className="type-body">{t("simulationNote")}</p>
        <DownloadButton
          onDownload={() =>
            downloadWorkbook(
              x("fileSimulation"),
              [
                sheet(
                  x("sheetRevenueBands"),
                  [x("colPercentile"), ...cols],
                  (["p5", "p25", "p50", "p75", "p95"] as const).map((q) => [q.toUpperCase(), ...sim.revenue_bands[q]]),
                  { columns: ["text", ...cols.map(() => "money" as const)] },
                ),
                sheet(
                  x("sheetEbitdaBands"),
                  [x("colPercentile"), ...cols],
                  (["p5", "p25", "p50", "p75", "p95"] as const).map((q) => [q.toUpperCase(), ...sim.ebitda_bands[q]]),
                  { columns: ["text", ...cols.map(() => "money" as const)] },
                ),
                sheet(
                  x("sheetTargets"),
                  [x("colEbitdaTarget", { money: mu }), x("colProbability"), x("colCase")],
                  sim.target_probabilities.map((p) => [p.target, p.probability, t(caseKey(p.scenario))]),
                  { columns: ["money", "percent", "text"] },
                ),
              ],
              money,
              res?.model,
            )
          }
        />
      </div>
      <Tile span={6} title={t("tileRevenueFan")} unit={t("fanUnit", { money: mu })}>
        {fan(sim.revenue_bands, res.years.map((y) => y.revenue ?? NaN), t("seriesRevenue"))}
      </Tile>
      <Tile span={6} title={t("tileEbitdaFan")} unit={t("fanUnit", { money: mu })}>
        {fan(sim.ebitda_bands, res.years.map((y) => y.ebitda ?? NaN), t("seriesEbitda"))}
      </Tile>
      <Tile span={12} title={t("tileTargets", { year: lastCol })} unit={t("targetsUnit")}>
        <table className="w-full border-collapse font-mono text-[11.5px]">
          <tbody>
            {sim.target_probabilities.map((p) => (
              <tr key={p.target}>
                <th scope="row" className="w-48 border-b border-grid py-1.5 text-start font-normal text-ink">
                  {fmtMoney(p.target)} {mu}{" "}
                  <span className={`chip ms-1 ${p.scenario === "Bull" ? "text-gain" : p.scenario === "Bear" ? "text-loss" : "text-dim"}`}>
                    {t(caseKey(p.scenario))}
                  </span>
                </th>
                <td className="border-b border-grid py-1.5">
                  <div className="h-3 bg-accent" style={{ width: `${Math.max(0.5, p.probability * 100)}%`, opacity: 0.35 + 0.65 * p.probability }} />
                </td>
                <td className="w-20 border-b border-grid py-1.5 text-end text-bright">{fmtRate(p.probability)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="type-body text-[9px]">{t("targetsNote")}</p>
      </Tile>
    </Tiles>
  );
}
