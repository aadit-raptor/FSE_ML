"use client";

import Link from "next/link";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { MoneySelects } from "@/components/ui/MoneySelects";
import { EmptyGuide, LoadingTiles, Notice, RailGroup, Screen, SecondaryButton } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { TipLabel, TipText, useTip } from "@/components/ui/Tip";
import { fmtMoney, fmtNumber, fmtRate } from "@/lib/format";
import { formatDateTime } from "@/lib/locale";
import { type Money, moneyLabel, type MoneyUnit } from "@/lib/money";
import { BOARD_COLUMNS, getPortfolio, type Portfolio, type PortfolioHolding, problemKey } from "@/lib/portfolio";

/**
 * Portfolio -> Dashboard (PLAN.md 7.4): every owned deal the person can read, its figures in its own currency and
 * converted to a reporting currency at one day's ECB reference rates, stated beside the totals, and the alerts its
 * latest quarterly results raise.
 *
 * i18n-keys: portfolio.col_*
 * i18n-keys: portfolio.alert_*
 * i18n-keys: portfolio.problem_*
 */
const field = "border border-line bg-field px-1.5 py-0.5 font-mono text-[11px] text-ink outline-none focus:border-accent";

export function DashboardStep() {
  const t = useTranslations("portfolio");
  const onTip = useTip("portfolioRateDay");
  const [money, setMoney] = useState<Money | null>(null);
  const [on, setOn] = useState<string>("");
  const [board, setBoard] = useState<Portfolio | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void getPortfolio(money?.currency ?? null, (money?.unit ?? "millions") as MoneyUnit, on || null).then((answer) => {
      if (cancelled) return;
      if (answer.portfolio) {
        setBoard(answer.portfolio);
        if (!money) setMoney(answer.portfolio.money);
      }
      setProblem(answer.problem ?? null);
    });
    return () => {
      cancelled = true;
    };
  }, [money, on]);

  const reporting = board?.money ?? money;
  return (
    <Screen
      rail={
        <RailGroup title={t("reportingGroup")}>
          {reporting && <MoneySelects money={reporting} onChange={setMoney} of={t("reportingOf")} />}
          <label className="grid grid-cols-[1fr_128px] items-center gap-1.5 py-px">
            <span className="type-input-label"><TipLabel tip={onTip}>{t("rateDay")}</TipLabel></span>
            <input type="date" aria-describedby={onTip.id} className={field} value={on} onChange={(e) => setOn(e.target.value)} />
          </label>
          <TipText tip={onTip} />
          {board && (
            <p className="type-body mt-1.5 text-[10px] text-muted" data-rates-day={board.published_on ?? ""}>
              {board.published_on ? t("ratesOf", { day: formatDateTime(new Date(`${board.published_on}T00:00:00Z`), { dateStyle: "medium", timeZone: "UTC" }) }) : t("noRates")}
            </p>
          )}
        </RailGroup>
      }
    >
      {problem && <Notice tone="loss" title={t("problemTitle")} role="alert">{t(problemKey(problem))}</Notice>}
      {!board && !problem && <LoadingTiles />}
      {board && board.holdings.length === 0 && (
        <EmptyGuide id="portfolio" title={t("emptyTitle")} steps={[t("emptyStep1"), t("emptyStep2"), t("emptyStep3")]}
          actions={<Link href="/portfolio/quarters" className="type-action text-accent">{t("emptyAction")}</Link>}>
          {t("emptyBody")}
        </EmptyGuide>
      )}
      {board && board.holdings.length > 0 && <Board board={board} />}
    </Screen>
  );
}

function Board({ board }: { board: Portfolio }) {
  const t = useTranslations("portfolio");
  const label = moneyLabel(board.money);
  const total = (key: string) => board.totals[key];
  return (
    <>
      {board.missing_rates.length > 0 && (
        <Notice tone="attention" title={t("missingRatesTitle")} role="note">
          {t("missingRates", { currencies: board.missing_rates.join(", ") })}
        </Notice>
      )}
      <Tiles>
        <Kpi span={3} title={t("totalEntryEquity")} lead value={fmtMoney(total("entry_equity")?.value)}
          sub={t("inMoney", { money: label, count: total("entry_equity")?.holdings ?? 0 })} />
        <Kpi span={3} title={t("totalPlanExit")} value={fmtMoney(total("plan_exit_equity")?.value)}
          sub={t("inMoney", { money: label, count: total("plan_exit_equity")?.holdings ?? 0 })} />
        <Kpi span={3} title={t("totalLtmEbitda")} value={fmtMoney(total("ltm_ebitda")?.value)}
          sub={t("inMoney", { money: label, count: total("ltm_ebitda")?.holdings ?? 0 })} />
        <Kpi span={3} title={t("alertsKpi")} value={String(board.alerts)}
          tone={board.alerts > 0 ? "loss" : "gain"} sub={board.alerts > 0 ? t("alertsSome") : t("alertsNone")} />
        <Tile span={12} title={t("holdingsTitle")} unit={label}>
          <div className="relative overflow-x-auto" tabIndex={0}>
            <table className="w-full border-collapse text-[11px]" aria-label={t("holdingsTitle")}>
              <thead>
                <tr>
                  <th className="type-chart-category border-b border-grid px-1 py-1 text-start">{t("colDeal")}</th>
                  {BOARD_COLUMNS.map((c) => (
                    <th key={c} className="type-chart-category border-b border-grid px-1 py-1 text-end">{t(`col_${c}`)}</th>
                  ))}
                  <th className="type-chart-category border-b border-grid px-1 py-1 text-end">{t("colPlanIrr")}</th>
                  <th className="type-chart-category border-b border-grid px-1 py-1 text-end">{t("colReforecastIrr")}</th>
                  <th className="type-chart-category border-b border-grid px-1 py-1 text-start">{t("colAlerts")}</th>
                </tr>
              </thead>
              <tbody>
                {board.holdings.map((h) => <Row key={h.deal_id} holding={h} />)}
                <tr data-total>
                  <th scope="row" className="type-result-title px-1 py-1 text-start">{t("total")}</th>
                  {BOARD_COLUMNS.map((c) => (
                    <td key={c} className="px-1 py-1 text-end font-mono text-bright" data-total-col={c}>{fmtMoney(total(c)?.value)}</td>
                  ))}
                  <td colSpan={3} />
                </tr>
              </tbody>
            </table>
          </div>
        </Tile>
        <Tile span={12} title={t("ratesTitle")}>
          {Object.keys(board.rates).length === 0 ? (
            <p className="type-body text-[10.5px]">{t("ratesNone", { money: board.money.currency })}</p>
          ) : (
            <ul className="grid gap-0.5">
              {Object.entries(board.rates).map(([currency, rate]) => (
                <li key={currency} className="type-body font-mono text-[10.5px]" data-rate={currency}>
                  {t("rateLine", { currency, rate: fmtNumber(rate, 4), reporting: board.money.currency })}
                </li>
              ))}
            </ul>
          )}
          <p className="type-body mt-1.5 text-[10px] text-muted">
            {t.rich("ratesSource", { link: (chunks) => <a href={board.rates_url} className="text-accent underline" target="_blank" rel="noreferrer">{chunks}</a> })}
          </p>
        </Tile>
      </Tiles>
    </>
  );
}

function Row({ holding }: { holding: PortfolioHolding }) {
  const t = useTranslations("portfolio");
  const { openDeal } = useDeal();
  const local = moneyLabel(holding.money);
  return (
    <>
      <tr data-holding={holding.name}>
        <th scope="row" className="border-b border-grid/40 px-1 pt-1 text-start font-normal">
          <SecondaryButton onClick={() => void openDeal(holding.deal_id)}>{holding.name}</SecondaryButton>
        </th>
        {BOARD_COLUMNS.map((c) => (
          <td key={c} className="border-b border-grid/40 px-1 pt-1 text-end font-mono text-ink" data-col={c}>
            {holding.converted ? fmtMoney(holding.converted[c]) : fmtMoney(null)}
          </td>
        ))}
        <td className="border-b border-grid/40 px-1 pt-1 text-end font-mono">{fmtRate(holding.plan_irr)}</td>
        <td className="border-b border-grid/40 px-1 pt-1 text-end font-mono">
          {holding.reforecast_irr == null ? fmtRate(null) : fmtRate(holding.reforecast_irr)}
          {holding.reforecast_irr != null && !holding.reforecast_current && <span className="text-attention"> *</span>}
        </td>
        <td rowSpan={2} className="border-b border-grid px-1 pt-1 align-top">
          {holding.alerts.length === 0 ? <span className="type-body text-[10px] text-muted">{t("noAlertsShort")}</span> : (
            <ul className="grid gap-0.5">
              {holding.alerts.map((a) => (
                <li key={a.id} className="type-body text-[10px] text-loss" data-alert={a.id}>
                  {t(`alert_${a.id}`, { actual: fmtMoney(a.actual), plan: fmtMoney(a.plan),
                    gap: a.gap_pct == null ? "" : fmtRate(a.gap_pct / 100) })}
                </li>
              ))}
            </ul>
          )}
        </td>
      </tr>
      <tr data-holding-local={holding.name}>
        <td className="border-b border-grid px-1 pb-1 text-start text-[10px] text-muted">
          {holding.as_of ? t("localAsOf", { money: local, year: String(holding.as_of.year), quarter: String(holding.as_of.quarter) })
            : t("localNoQuarters", { money: local })}
        </td>
        {BOARD_COLUMNS.map((c) => (
          <td key={c} className="border-b border-grid px-1 pb-1 text-end font-mono text-[10px] text-muted" data-local-col={c}>
            {fmtMoney(holding.local[c])}
          </td>
        ))}
        <td colSpan={2} className="border-b border-grid" />
      </tr>
    </>
  );
}
