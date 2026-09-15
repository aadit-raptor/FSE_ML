"use client";

import { useEffect, useState } from "react";

import { Tile } from "@/components/ui/Tile";
import { api } from "@/lib/api/client";
import { useCapabilities } from "@/lib/capabilities";
import { multiplesFromPct } from "@/lib/deal/capital";

import { useDeal } from "./DealProvider";

type Risk = {
  risk_score: number;
  is_anomalous: boolean;
  warnings: string[];
  nearest_deals: { name: string; entry_mult: number; leverage: number; growth: number; success: boolean }[];
  inputs: { leverage: number; ebitda_margin: number };
};

/** Anomaly-detector score for the deal against historical LBOs. Hidden when the server has no ML layer. */
export function DealRisk() {
  const caps = useCapabilities();
  const { inputs } = useDeal();
  const [risk, setRisk] = useState<Risk | null>(null);
  const [error, setError] = useState<string>();
  const { seniorX, mezzX } = multiplesFromPct(inputs.entry_mult, inputs.debt_pct, inputs.senior_pct);

  useEffect(() => {
    if (!caps?.anomaly_detector) return;
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      api
        .POST("/api/ml/deal-risk", { body: { inputs, senior_x: Number(seniorX.toFixed(6)), mezz_x: Number(mezzX.toFixed(6)) }, signal: ctrl.signal })
        .then(({ data, error: err }) => {
          if (data) {
            setRisk(data as unknown as Risk);
            setError(undefined);
          } else setError(String((err as { detail?: unknown })?.detail ?? "Risk score unavailable"));
        })
        .catch(() => {});
    }, 400);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [caps?.anomaly_detector, inputs, seniorX, mezzX]);

  if (!caps?.anomaly_detector) return null;
  const tone = !risk ? "text-muted" : risk.risk_score < 4 ? "text-gain" : risk.risk_score < 7 ? "text-attention" : "text-loss";
  return (
    <Tile span={12} title="Deal risk" unit="ML anomaly detector vs historical LBOs">
      {error && <p className="font-mono text-[10.5px] text-loss">{error}</p>}
      {risk && (
        <div className="grid grid-cols-[180px_1fr_1fr] gap-4">
          <div>
            <p className={`type-figure text-[22px] ${tone}`} data-kpi="Risk score">
              {risk.risk_score.toFixed(1)} / 10
            </p>
            <p className="font-mono text-[10px] text-muted">
              {risk.is_anomalous ? "unusual versus history" : "in line with history"} · {risk.inputs.leverage.toFixed(1)}x leverage
            </p>
          </div>
          <ul className="grid content-start gap-1" aria-label="Risk flags">
            {risk.warnings.length ? (
              risk.warnings.map((w) => (
                <li key={w} className="type-body text-[9.5px] shadow-[inset_2px_0_0_var(--color-loss)] pl-2">
                  {w}
                </li>
              ))
            ) : (
              <li className="type-body text-[9.5px]">No risk flags.</li>
            )}
          </ul>
          <ul className="grid content-start gap-1" aria-label="Most similar historical deals">
            {risk.nearest_deals.map((d) => (
              <li key={d.name} className="font-mono text-[10.5px] text-ink">
                <span className={d.success ? "text-gain" : "text-loss"}>{d.success ? "success" : "distress"}</span> {d.name} · {d.entry_mult.toFixed(1)}x entry ·{" "}
                {d.leverage.toFixed(1)}x lev · {d.growth >= 0 ? "+" : ""}
                {d.growth.toFixed(1)}% growth
              </li>
            ))}
          </ul>
        </div>
      )}
      {!risk && !error && <p className="type-body">Scoring the deal…</p>}
    </Tile>
  );
}
