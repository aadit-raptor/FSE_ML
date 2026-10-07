"use client";

import { useEffect, useState } from "react";

import { api, type Schemas } from "@/lib/api/client";
import type { DealInputs } from "@/lib/deal/fields";

export type Starting = Schemas["StartingAssumptionsResponse"];
export type StartingFigure = Schemas["StartingFigure"];
export type StartingField = StartingFigure["field"];
export type Industries = Schemas["BenchmarkIndustriesResponse"];

/**
 * The industries the stored averages cover (PLAN.md 4.3), read once per page load: they change
 * once a year. A failed read is asked again next time rather than remembered as empty.
 */
let industriesPending: Promise<Industries | null> | null = null;

function loadIndustries(): Promise<Industries | null> {
  industriesPending ??= api
    .GET("/api/benchmarks/industries")
    .then(({ data }) => data ?? null)
    .catch(() => null)
    .then((found) => {
      if (!found) industriesPending = null;
      return found;
    });
  return industriesPending;
}

export function useIndustries(): Industries | null {
  const [found, setFound] = useState<Industries | null>(null);
  useEffect(() => {
    let live = true;
    loadIndustries().then((f) => {
      if (live) setFound(f);
    });
    return () => {
      live = false;
    };
  }, []);
  return found;
}

/** A deal's starting figures for its country, industry and currency, each with its source. */
export async function fetchStarting(country: string, industry: string, currency: string): Promise<Starting | null> {
  try {
    const { data } = await api.GET("/api/benchmarks/starting", { params: { query: { country, industry, currency } } });
    return data ?? null;
  } catch {
    return null;
  }
}

/** The starting figures as deal inputs, with the labels saying where they were sourced for. */
export function startingPatch(s: Starting): Partial<DealInputs> {
  return { ...(s.inputs as Partial<DealInputs>), country: s.country, industry: s.industry };
}

/** How a starting figure reads: a per cent, a multiple or days. */
export function figureKind(field: StartingField): "pct" | "multiple" | "days" {
  if (field === "entry_mult" || field === "exit_mult") return "multiple";
  if (field === "ar_days" || field === "inv_days" || field === "ap_days") return "days";
  return "pct";
}

export type Risk = Schemas["RiskAssumptionsResponse"];
export type RiskFigure = Risk["figures"][number];
export type RiskSetting = RiskFigure["field"];
export type RiskScenario = Risk["scenarios"][keyof Risk["scenarios"]];

/**
 * The Monte Carlo means, spreads, correlations and scenario presets sourced for a country, industry
 * and currency (PLAN.md 4.4), each with its source and years. One read per combination and page
 * load: the rail and the Scenarios tile share it. A failed read is asked again next time.
 */
const riskPending = new Map<string, Promise<Risk | null>>();

export function fetchRisk(country: string, industry: string, currency: string): Promise<Risk | null> {
  const key = `${country}|${industry}|${currency}`;
  let found = riskPending.get(key);
  if (!found) {
    found = api
      .GET("/api/benchmarks/risk", { params: { query: { country, industry, currency } } })
      .then(({ data }) => data ?? null)
      .catch(() => null)
      .then((r) => {
        if (!r) riskPending.delete(key);
        return r;
      });
    riskPending.set(key, found);
  }
  return found;
}

/** How a sourced Setting reads: a per cent, a multiple, a multiplier, percentage points or a correlation. */
export function riskKind(setting: RiskSetting): "pct" | "multiple" | "multiplier" | "points" | "correlation" {
  if (setting.startsWith("corr_")) return "correlation";
  if (setting === "mc_exit_mean" || setting === "mc_exit_std") return "multiple";
  if (setting.endsWith("_mult")) return "multiplier";
  if (setting.endsWith("_adj") || setting.endsWith("_floor")) return "points";
  return "pct";
}
