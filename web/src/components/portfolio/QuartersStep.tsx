"use client";

import { useTranslations } from "next-intl";
import { useEffect, useRef, useState } from "react";

import { useDeal } from "@/components/deal/DealProvider";
import { CellInput } from "@/components/ui/CellInput";
import { MoneyScope, useMoney } from "@/components/ui/MoneyScope";
import { Notice, PrimaryButton, RailGroup, Screen, SecondaryButton, Switch } from "@/components/ui/Screen";
import { Kpi, Tile, Tiles } from "@/components/ui/Tile";
import { TipLabel, TipText, useTip, useTips } from "@/components/ui/Tip";
import { dealFiscal } from "@/lib/deal/fields";
import { fmtMoney, fmtNumber, fmtRate, isNum } from "@/lib/format";
import { useFiscalLabels } from "@/lib/i18n/useFiscalLabels";
import { formatDateTime, parseNumber } from "@/lib/locale";
import {
  type Drift, draftsOf, getHolding, type Holding, nextQuarter, problemKey, type QuarterDraft, QUARTER_LINES,
  type QuarterLine, reforecast, removeHolding, rowsOf, saveQuarters, setHolding,
} from "@/lib/portfolio";

/**
 * Portfolio -> Quarterly results (PLAN.md 7.4): the open deal marked as owned, its quarterly results against the
 * plan, the alerts they raise and a reforecast from the last four quarters, run as a background job.
 *
 * i18n-keys: portfolio.problem_*
 * i18n-keys: portfolio.alert_*
 * i18n-keys: tips.quarter_*
 */
const field = "border border-line bg-field px-1.5 py-0.5 font-mono text-[11px] text-ink outline-none focus:border-accent";

export function QuartersStep() {
  const t = useTranslations("portfolio");
  const { current, inputs } = useDeal();
  const dealId = current?.id ?? null;
  const [holding, setHoldingState] = useState<Holding | null>(null);
  const [drafts, setDrafts] = useState<QuarterDraft[]>([]);
  const [problem, setProblem] = useState<string | null>(null);
  const [stage, setStage] = useState<string | null>(null);
  const inflight = useRef<AbortController | null>(null);

  const take = (answer: { holding?: Holding; problem?: string }) => {
    if (answer.holding) {
      setHoldingState(answer.holding);
      setDrafts(draftsOf(answer.holding));
    }
    setProblem(answer.problem ?? null);
  };

  useEffect(() => {
    if (!dealId) return;
    let cancelled = false;
    void getHolding(dealId).then((answer) => {
      if (cancelled) return;
      take(answer);
    });
    return () => {
      cancelled = true;
      inflight.current?.abort();
    };
  }, [dealId]);

  if (!current || !dealId) {
    return (
      <Screen rail={null}>
        <Notice tone="attention" title={t("noDealTitle")} role="note">{t("noDeal")}</Notice>
      </Screen>
    );
  }

  const money = { currency: inputs.currency ?? "USD", unit: inputs.unit ?? "millions" } as const;
  const readOnly = holding?.access === "viewer";
  const owned = holding?.owned ?? false;

  const run = async () => {
    inflight.current?.abort();
    const ctrl = new AbortController();
    inflight.current = ctrl;
    setStage(t("reforecasting"));
    setProblem(null);
    try {
      take(await reforecast(dealId, ctrl.signal, (s) => setStage(s ?? t("reforecasting"))));
    } catch {
      if (!ctrl.signal.aborted) setProblem("failed");
    } finally {
      if (inflight.current === ctrl) setStage(null);
    }
  };

  return (
    <MoneyScope money={money}>
      <Screen
        rail={
          <>
            <RailGroup title={t("holdingGroup")}>
              {holding === null ? <p className="type-step">{t("loading")}</p> : (
                <HoldingControls holding={holding} readOnly={readOnly}
                  onChange={async (next) => take(next === null ? await removeHolding(dealId)
                    : await setHolding(dealId, next.since, next.tolerance))} />
              )}
            </RailGroup>
            {owned && (
              <RailGroup title={t("reforecastGroup")}>
                <p className="type-body mb-1.5 text-[10.5px]">{t("reforecastHint")}</p>
                <PrimaryButton onClick={() => void run()} disabled={readOnly || stage !== null}>{t("reforecast")}</PrimaryButton>
                {stage && <p className="type-body mt-1 text-[10px]" role="status">{stage}</p>}
              </RailGroup>
            )}
          </>
        }
      >
        <Notice tone="info" title={t("aboutTitle")} role="note">{t("about")}</Notice>
        {problem && <Notice tone="loss" title={t("problemTitle")} role="alert">{t(problemKey(problem))}</Notice>}
        {holding?.quarters_problem && (
          <Notice tone="attention" title={t("problemTitle")} role="note">{t(problemKey(holding.quarters_problem))}</Notice>
        )}
        {holding && owned && (
          <Tiles>
            {holding.drift && <DriftTiles drift={holding.drift} />}
            <QuarterEditor key={`${dealId}:${holding.quarters_updated_at ?? ""}`} drafts={drafts} hold={inputs.hold}
              fiscal={dealFiscal(inputs)} readOnly={readOnly} onSave={async (next) => {
                setDrafts(next);
                take(await saveQuarters(dealId, money, rowsOf(next)));
              }} />
            {holding.reforecast && <ReforecastTile holding={holding} />}
          </Tiles>
        )}
      </Screen>
    </MoneyScope>
  );
}

function HoldingControls({ holding, readOnly, onChange }: {
  holding: Holding; readOnly: boolean;
  onChange: (next: { since: string | null; tolerance: number } | null) => Promise<void>;
}) {
  const t = useTranslations("portfolio");
  const sinceTip = useTip("portfolioSince");
  const toleranceTip = useTip("portfolioTolerance");
  const [tolerance, setTolerance] = useState<string | null>(null);
  const since = holding.since ?? null;
  const shownTolerance = tolerance ?? fmtNumber(holding.tolerance ?? 5, 1);
  const commitTolerance = () => {
    const n = parseNumber(shownTolerance);
    setTolerance(null);
    if (Number.isFinite(n) && n >= 0 && n <= 100 && n !== holding.tolerance) void onChange({ since, tolerance: n });
  };
  return (
    <div className="grid gap-1.5">
      {readOnly ? (
        <p className="type-body text-[10.5px]">{holding.owned ? t("ownedViewOnly") : t("notOwnedViewOnly")}</p>
      ) : (
        <Switch checked={holding.owned} label={t("owned")} tip="portfolioOwned"
          onChange={(v) => void onChange(v ? { since, tolerance: holding.tolerance ?? 5 } : null)} />
      )}
      {holding.owned && (
        <>
          <label className="grid grid-cols-[1fr_128px] items-center gap-1.5">
            <span className="type-input-label"><TipLabel tip={sinceTip}>{t("since")}</TipLabel></span>
            <input type="date" aria-describedby={sinceTip.id} className={field} value={since ?? ""} disabled={readOnly}
              onChange={(e) => void onChange({ since: e.target.value || null, tolerance: holding.tolerance ?? 5 })} />
          </label>
          <TipText tip={sinceTip} />
          <label className="grid grid-cols-[1fr_128px] items-center gap-1.5">
            <span className="type-input-label"><TipLabel tip={toleranceTip}>{t("tolerance")}</TipLabel></span>
            <input inputMode="decimal" aria-describedby={toleranceTip.id} className={`${field} text-end`} disabled={readOnly}
              value={shownTolerance} onChange={(e) => setTolerance(e.target.value)} onBlur={commitTolerance}
              onKeyDown={(e) => e.key === "Enter" && commitTolerance()} />
          </label>
          <TipText tip={toleranceTip} />
        </>
      )}
    </div>
  );
}

function QuarterEditor({ drafts, hold, fiscal, readOnly, onSave }: {
  drafts: QuarterDraft[]; hold: number; fiscal: ReturnType<typeof dealFiscal>; readOnly: boolean;
  onSave: (next: QuarterDraft[]) => Promise<void>;
}) {
  const t = useTranslations("portfolio");
  const { label } = useMoney();
  const years = useFiscalLabels().deal(hold, fiscal);
  const removeTips = useTips(["quarterRemove"]);
  const [rows, setRows] = useState<QuarterDraft[]>(drafts);
  const next = nextQuarter(rows, hold);
  const dirty = JSON.stringify(rows) !== JSON.stringify(drafts);
  const quarterName = (d: { year: number; quarter: number }) =>
    t("quarterName", { year: years[d.year - 1] ?? String(d.year), quarter: String(d.quarter) });
  const set = (i: number, line: QuarterLine, v: number | null) =>
    setRows(rows.map((r, j) => (j === i ? { ...r, values: { ...r.values, [line]: v } } : r)));
  return (
    <Tile span={12} title={t("quartersTitle")} unit={label}
      action={readOnly ? undefined : (
        <span className="flex gap-1.5">
          <SecondaryButton disabled={next === null} onClick={() => next && setRows([...rows, {
            ...next, values: { revenue: null, ebitda: null, net_income: null, fcf: null, total_debt: null } }])}>
            {t("addQuarter")}
          </SecondaryButton>
          <PrimaryButton disabled={!dirty} onClick={() => void onSave(rows)}>{t("save")}</PrimaryButton>
        </span>
      )}>
      {rows.length === 0 ? <p className="type-body text-[10.5px]">{t("noQuarters")}</p> : (
        <div className="overflow-x-auto" tabIndex={0}>
          <table className="w-full border-collapse text-[11px]" aria-label={t("quartersTitle")}>
            <thead>
              <tr>
                <th className="type-chart-category border-b border-grid px-1 py-1 text-start">{t("quarter")}</th>
                {QUARTER_LINES.map((l) => (
                  <th key={l.key} className="type-chart-category border-b border-grid px-1 py-1 text-end">{t(l.labelKey)}</th>
                ))}
                <td className="border-b border-grid" aria-hidden />
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={`${r.year}-${r.quarter}`} data-quarter={`${r.year}-${r.quarter}`}>
                  <th scope="row" className="type-body border-b border-grid px-1 py-0.5 text-start font-normal">{quarterName(r)}</th>
                  {QUARTER_LINES.map((l) => (
                    <td key={l.key} className="border-b border-grid px-1 py-0.5">
                      {readOnly ? <span className="block text-end font-mono">{fmtMoney(r.values[l.key])}</span> : (
                        <CellInput label={t("cellLabel", { line: t(l.labelKey), quarter: quarterName(r) })}
                          tip={`quarter_${l.key}`} value={r.values[l.key] ?? NaN}
                          onCommit={(v) => set(i, l.key, v)} onClear={() => set(i, l.key, null)} />
                      )}
                    </td>
                  ))}
                  <td className="border-b border-grid px-1 py-0.5 text-end">
                    {!readOnly && i === rows.length - 1 && (
                      <SecondaryButton describedBy={removeTips.quarterRemove.id} onClick={() => setRows(rows.slice(0, -1))}>
                        {t("removeQuarter")}
                      </SecondaryButton>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <TipText tip={removeTips.quarterRemove} />
        </div>
      )}
    </Tile>
  );
}

function gapText(gap: number | null | undefined): string | undefined {
  return isNum(gap) ? fmtRate(gap / 100, 1) : undefined;
}

function DriftTiles({ drift }: { drift: Drift }) {
  const t = useTranslations("portfolio");
  const ebitda = drift.ytd.ebitda ?? null;
  const revenue = drift.ytd.revenue ?? null;
  const tone = (g: number | null | undefined, worse: "below" | "above") =>
    !isNum(g) ? undefined : (worse === "below" ? g < -drift.tolerance : g > drift.tolerance) ? "loss" as const : "gain" as const;
  return (
    <>
      <Kpi span={4} title={t("ytdEbitda")} value={fmtMoney(ebitda?.actual)} lead
        sub={ebitda ? t("againstPlan", { plan: fmtMoney(ebitda.plan), gap: gapText(ebitda.gap_pct) ?? "" }) : t("ytdMissing")}
        tone={tone(ebitda?.gap_pct, "below")} />
      <Kpi span={4} title={t("ytdRevenue")} value={fmtMoney(revenue?.actual)}
        sub={revenue ? t("againstPlan", { plan: fmtMoney(revenue.plan), gap: gapText(revenue.gap_pct) ?? "" }) : t("ytdMissing")}
        tone={tone(revenue?.gap_pct, "below")} />
      <Kpi span={4} title={t("latestDebt")} value={fmtMoney(drift.debt?.actual)}
        sub={drift.debt ? t("againstPlan", { plan: fmtMoney(drift.debt.plan), gap: gapText(drift.debt.gap_pct) ?? "" }) : t("ytdMissing")}
        tone={tone(drift.debt?.gap_pct, "above")} />
      <Tile span={12} title={t("alertsTitle")}>
        {drift.alerts.length === 0 ? (
          <p className="type-body text-[10.5px]" data-alerts="0">{t("noAlerts", { tolerance: fmtNumber(drift.tolerance, 1) })}</p>
        ) : (
          <ul className="grid gap-1" data-alerts={String(drift.alerts.length)}>
            {drift.alerts.map((a) => (
              <li key={a.id} className="type-body text-[10.5px] text-loss" data-alert={a.id}>
                {t(`alert_${a.id}`, { actual: fmtMoney(a.actual), plan: fmtMoney(a.plan), gap: gapText(a.gap_pct) ?? "" })}
              </li>
            ))}
          </ul>
        )}
        <p className="type-body mt-1.5 text-[10px] text-muted">{t("seasonality")}</p>
      </Tile>
    </>
  );
}

function ReforecastTile({ holding }: { holding: Holding }) {
  const t = useTranslations("portfolio");
  const ref = holding.reforecast!;
  return (
    <>
      <Kpi span={3} title={t("planIrr")} value={fmtRate(ref.plan.irr)} />
      <Kpi span={3} title={t("reforecastIrr")} value={fmtRate(ref.reforecast.irr)} lead
        sub={t("growthFromTo", { from: fmtNumber(ref.plan_growth, 1), to: fmtNumber(ref.reforecast_growth, 1) })} />
      <Kpi span={3} title={t("reforecastRange")} value={`${fmtRate(ref.simulated.p5)} – ${fmtRate(ref.simulated.p95)}`}
        sub={t("medianIrr", { irr: fmtRate(ref.simulated.p50) })} />
      <Kpi span={3} title={t("belowPlan")} value={fmtRate(ref.simulated.below_plan)} />
      <Tile span={12} title={t("reforecastTitle")}>
        {!holding.reforecast_current && <p className="type-alert mb-1 text-[10.5px] text-attention" role="note">{t("reforecastStale")}</p>}
        <p className="type-body text-[10px] text-muted">
          {t("reforecastBasis", { year: String(ref.based_on.year), quarter: String(ref.based_on.quarter),
            when: formatDateTime(new Date(ref.run_at)) })}
        </p>
      </Tile>
    </>
  );
}
