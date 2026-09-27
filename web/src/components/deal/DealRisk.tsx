"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";

import { Tile } from "@/components/ui/Tile";
import { api } from "@/lib/api/client";
import { fmtMultiple, fmtNumber, fmtPct } from "@/lib/format";
import { useCapabilities } from "@/lib/capabilities";
import { multiplesFromPct } from "@/lib/deal/capital";
import { apiInputs } from "@/lib/deal/fields";
import { useProvenance } from "@/lib/i18n/useProvenance";
import { type HistoricalSample } from "@/lib/provenance";

import { useDeal } from "./DealProvider";

type Risk = {
  risk_score: number;
  is_anomalous: boolean;
  warnings: string[];
  nearest_deals: { name: string; entry_mult: number; leverage: number; growth: number; success: boolean }[];
  inputs: { leverage: number; ebitda_margin: number };
  historical_sample: HistoricalSample;
};

/** Anomaly-detector score for the deal against historical LBOs. Hidden when the server has no ML layer. */
export function DealRisk() {
  const caps = useCapabilities();
  const { inputs } = useDeal();
  const t = useTranslations("deal");
  const provenance = useProvenance();
  const [risk, setRisk] = useState<Risk | null>(null);
  const [error, setError] = useState<string>();
  const { seniorX, mezzX } = multiplesFromPct(inputs.entry_mult, inputs.debt_pct, inputs.senior_pct);
  const unavailable = t("riskUnavailable");

  useEffect(() => {
    if (!caps?.anomaly_detector) return;
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      api
        .POST("/api/ml/deal-risk", {
          body: { inputs: apiInputs(inputs), senior_x: Number(seniorX.toFixed(6)), mezz_x: Number(mezzX.toFixed(6)) },
          signal: ctrl.signal,
        })
        .then(({ data, error: err }) => {
          if (data) {
            setRisk(data as unknown as Risk);
            setError(undefined);
          } else setError(String((err as { detail?: unknown })?.detail ?? unavailable));
        })
        .catch(() => {});
    }, 400);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [caps?.anomaly_detector, inputs, seniorX, mezzX, unavailable]);

  if (!caps?.anomaly_detector) return null;
  const tone = !risk ? "text-muted" : risk.risk_score < 4 ? "text-gain" : risk.risk_score < 7 ? "text-attention" : "text-loss";
  return (
    <Tile span={12} title={t("riskTile")} unit={t("riskUnit")}>
      {error && <p className="font-mono text-[10.5px] text-loss">{error}</p>}
      {risk && (
        <div className="grid grid-cols-[180px_1fr_1fr] gap-4">
          <div>
            <p className={`type-figure text-[22px] ${tone}`} data-kpi={t("riskScore")}>
              {t("riskScoreOutOf", { score: fmtNumber(risk.risk_score, 1) })}
            </p>
            <p className="font-mono text-[10px] text-muted">
              {risk.is_anomalous ? t("riskUnusual") : t("riskInLine")} · {t("riskLeverage", { leverage: fmtMultiple(risk.inputs.leverage, 1) })}
            </p>
          </div>
          <ul className="grid content-start gap-1" aria-label={t("riskFlags")}>
            {risk.warnings.length ? (
              risk.warnings.map((w) => (
                <li key={w} className="type-body text-[9.5px] shadow-[inset_2px_0_0_var(--color-loss)] ps-2">
                  {w}
                </li>
              ))
            ) : (
              <li className="type-body text-[9.5px]">{t("riskNoFlags")}</li>
            )}
          </ul>
          <ul className="grid content-start gap-1" aria-label={t("riskSimilar")}>
            {risk.nearest_deals.map((d) => (
              <li key={d.name} className="font-mono text-[10.5px] text-ink">
                <span className={d.success ? "text-gain" : "text-loss"}>{d.success ? t("riskSuccess") : t("riskDistress")}</span> {d.name} ·{" "}
                {t("riskNearest", {
                  entry: fmtMultiple(d.entry_mult, 1),
                  leverage: fmtMultiple(d.leverage, 1),
                  growth: `${d.growth >= 0 ? "+" : ""}${fmtPct(d.growth, 1)}`,
                })}
              </li>
            ))}
          </ul>
        </div>
      )}
      {risk && (
        <p className="type-body text-[9px]" data-provenance="risk">
          <span className="type-alert text-[9px]">{provenance.risk(risk.historical_sample)}</span>
          {provenance.riskDetail}
        </p>
      )}
      {!risk && !error && <p className="type-body">{t("riskScoring")}</p>}
    </Tile>
  );
}
