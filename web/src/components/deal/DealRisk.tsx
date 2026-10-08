"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { Tile } from "@/components/ui/Tile";
import { api } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";
import { multiplesFromPct } from "@/lib/deal/capital";
import { apiInputs } from "@/lib/deal/fields";
import { fmtMultiple, fmtNumber, fmtPct } from "@/lib/format";
import { monthYear } from "@/lib/locale";

import { useDeal } from "./DealProvider";

type Risk = components["schemas"]["DealRiskResponse"];
type Comparison = components["schemas"]["PeerComparison"];

/** Colour by how far the figure sits on the risky side of its industry's. */
function riskTone(riskZ: number | null | undefined): string {
  if (riskZ == null) return "text-muted";
  return riskZ >= 2 ? "text-loss" : riskZ >= 1 ? "text-attention" : "text-ink";
}

/**
 * The deal against companies and deals like it (PLAN.md 5.2): its leverage,
 * price and margin beside its industry's listed companies in its own region,
 * the score where its model card says it beats leverage alone, and the
 * reference transactions like it while the library is on.
 *
 * i18n-keys: dealRisk.metric_*, dealRisk.position_*, dealRisk.verdict_*, starting.area_*, library.region_*,
 * i18n-keys: library.sector_*, library.bucket_*, library.outcome_*, library.event_*
 */
export function DealRisk() {
  const { inputs } = useDeal();
  const t = useTranslations("dealRisk");
  const [risk, setRisk] = useState<Risk | null>(null);
  const [error, setError] = useState<string>();
  const { seniorX, mezzX } = multiplesFromPct(inputs.entry_mult, inputs.debt_pct, inputs.senior_pct);
  const unavailable = t("unavailable");

  useEffect(() => {
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      api
        .POST("/api/ml/deal-risk", {
          body: { inputs: apiInputs(inputs), senior_x: Number(seniorX.toFixed(6)), mezz_x: Number(mezzX.toFixed(6)) },
          signal: ctrl.signal,
        })
        .then(({ data, error: err }) => {
          if (data) {
            setRisk(data);
            setError(undefined);
          } else setError(String((err as { detail?: unknown })?.detail ?? unavailable));
        })
        .catch(() => {});
    }, 400);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [inputs, seniorX, mezzX, unavailable]);

  return (
    <Tile span={12} title={t("tile")} unit={t("unit")}>
      {error && <p className="font-mono text-[10.5px] text-loss">{error}</p>}
      {!risk && !error && <p className="type-body">{t("comparing")}</p>}
      {risk && <RiskBody risk={risk} />}
    </Tile>
  );
}

function RiskBody({ risk }: { risk: Risk }) {
  const t = useTranslations("dealRisk");
  const area = useTranslations("starting");
  const industry = risk.industry_name ?? risk.industry;

  if (risk.status === "not_enough_data") {
    const thin = [...new Set(risk.comparisons.flatMap((c) => c.skipped.map((s) => area(`area_${s.area}`))))];
    return (
      <p className="type-body" data-risk-status="not_enough_data">
        <span className="type-alert text-[9px]">{t("notEnoughData")}</span>{" "}
        {risk.reason === "no_country" ? t("noCountry") : t("noPeers", { min: risk.min_firms, industry, areas: thin.join(", ") })}
      </p>
    );
  }

  const sample = risk.sample;
  const source = risk.comparisons.find((c) => c.status === "ok");
  return (
    <div className="grid gap-3" data-risk-status="ok">
      <div className="grid grid-cols-[200px_minmax(0,1fr)] gap-4">
        <Score risk={risk} />
        <table className="w-full font-mono text-[10.5px]" aria-label={t("tableLabel")}>
          <thead>
            <tr className="text-dim">
              <th className="text-start font-normal">{t("colFigure")}</th>
              <th className="text-end font-normal">{t("colDeal")}</th>
              <th className="text-end font-normal">{t("colIndustry")}</th>
              <th className="ps-3 text-start font-normal">{t("colPlace")}</th>
            </tr>
          </thead>
          <tbody>
            {risk.comparisons.map((c) => (
              <ComparisonRow key={c.metric} c={c} />
            ))}
          </tbody>
        </table>
      </div>
      {sample && (
        <p className="type-body text-[9px]" data-provenance="risk">
          <span className="type-alert text-[9px]">
            {t("basedOn", { firms: sample.firms, group: area(`area_${sample.group}`), industry })}
          </span>{" "}
          {t("sourceNote", { publisher: risk.source.publisher, date: source?.published ? monthYear(source.published.slice(0, 7)) : "" })}{" "}
          <a href={risk.source.url} target="_blank" rel="noopener noreferrer" className="text-accent underline">
            {risk.source.title}
          </a>
        </p>
      )}
      {risk.deals.enabled && <Deals risk={risk} />}
    </div>
  );
}

function Score({ risk }: { risk: Risk }) {
  const t = useTranslations("dealRisk");
  const lib = useTranslations("library");
  const s = risk.score;
  const region = s.region ? lib(`region_${s.region}`) : "";
  const tone = s.value == null ? "text-muted" : s.value < 1 ? "text-gain" : s.value < 2 ? "text-attention" : "text-loss";
  return (
    <div className="grid content-start gap-1">
      <p className="font-mono text-[10px] text-dim">{t("score")}</p>
      {s.shown ? (
        <p className={`type-figure text-[22px] ${tone}`} data-kpi={t("score")}>
          {fmtNumber(s.value, 1)}
        </p>
      ) : (
        <p className="type-alert text-[10px] text-attention" data-kpi={t("score")}>
          {t(`verdict_${s.verdict}`)}
        </p>
      )}
      <p className="font-mono text-[10px] text-muted">
        {s.shown
          ? t("scoreTested", { cases: s.cases, region, model: fmtNumber(s.model, 2), baseline: fmtNumber(s.baseline, 2) })
          : s.verdict === "does_not_beat_baseline"
            ? t("scoreNotBetter", { region })
            : t("scoreTooFew", { cases: s.cases, region })}
      </p>
      <p className={`font-mono text-[10px] ${risk.unusual ? "text-loss" : "text-gain"}`}>{risk.unusual ? t("unusual") : t("inLine")}</p>
    </div>
  );
}

function figure(c: Comparison, v: number | null | undefined): string {
  return c.metric === "ebitda_margin" ? fmtPct(v, 1) : fmtMultiple(v, 1);
}

function ComparisonRow({ c }: { c: Comparison }) {
  const t = useTranslations("dealRisk");
  return (
    <tr data-metric={c.metric}>
      <td className="text-ink">{t(`metric_${c.metric}`)}</td>
      <td className="text-end text-ink">{figure(c, c.deal)}</td>
      <td className="text-end text-muted">{c.status === "ok" ? figure(c, c.peer) : ""}</td>
      <td className={`ps-3 ${riskTone(c.risk_z)}`}>
        {c.status === "ok" && c.position ? t(`position_${c.position}`) : t("rowNotEnough")}
      </td>
    </tr>
  );
}

function Deals({ risk }: { risk: Risk }) {
  const t = useTranslations("dealRisk");
  const lib = useTranslations("library");
  const d = risk.deals;
  const region = d.region ? lib(`region_${d.region}`) : "";
  if (!d.sector) return <p className="type-body text-[9px]">{t("dealsNoSector")}</p>;
  const sector = lib(`sector_${d.sector}`);
  return (
    <div className="grid gap-1" data-risk-deals>
      <p className="font-mono text-[10px] text-dim">{t("dealsTitle", { count: d.deals.length, region, sector })}</p>
      {d.deals.length === 0 && <p className="type-body text-[9px]">{t("dealsNone")}</p>}
      <ul className="grid gap-1" aria-label={t("dealsLabel")}>
        {d.deals.map((x) => (
          <li key={x.key} className="font-mono text-[10.5px] text-ink">
            <span className={x.outcome === "distress" ? "text-loss" : x.outcome === "success" ? "text-gain" : "text-muted"}>
              {lib(`event_${x.event}`)}
            </span>{" "}
            {x.target} · {String(x.year)} ·{" "}
            {t("dealFigures", { entry: fmtMultiple(x.entry_multiple, 1), leverage: fmtMultiple(x.leverage, 1) })}
            {x.size && <> · {lib(`bucket_${x.size}`)}</>}
            {x.same_size && <span className="ms-1 text-accent">{t("sameSize")}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
