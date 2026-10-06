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
