"use client";

import { useEffect, useState } from "react";

import { api, type Schemas } from "@/lib/api/client";
import type { FullTranche, ReferenceRate } from "@/lib/deal/capital";

export type ReferenceRates = Schemas["ReferenceRatesResponse"];
export type EconomicFigure = Schemas["EconomicFigure"];

/**
 * The current level of each floating-rate benchmark (PLAN.md 4.2), read once per page load:
 * the API answers what the nightly refresh stored, which changes once a day. A failed read
 * is the same as no data -- a facility then keeps whatever rate the user typed.
 */
let pending: Promise<ReferenceRates | null> | null = null;
let loadedAt = 0;
// A tab left open overnight asks again, so "today's level" is today's
const KEEP_MS = 6 * 60 * 60 * 1000;

function load(): Promise<ReferenceRates | null> {
  if (pending && Date.now() - loadedAt > KEEP_MS) pending = null;
  if (!pending) loadedAt = Date.now();
  pending ??= api
    .GET("/api/economy/reference-rates")
    .then(({ data }) => data ?? null)
    .catch(() => null)
    .then((rates) => {
      if (!rates) pending = null; // asked again next time, rather than remembered as missing
      return rates;
    });
  return pending;
}

export function useReferenceRates(): ReferenceRates | null {
  const [rates, setRates] = useState<ReferenceRates | null>(null);
  useEffect(() => {
    let live = true;
    load().then((r) => {
      if (live) setRates(r);
    });
    return () => {
      live = false;
    };
  }, []);
  return rates;
}

/** The benchmark's current level, when one is stored and current. */
export function currentRate(rates: ReferenceRates | null, code: string): EconomicFigure | undefined {
  return rates?.rates[code];
}

/**
 * A facility priced on ``code`` at its current level: a flat reference, so every year of the
 * hold starts from today's rate (core/debt.py repeats a path's last year). Without a current
 * level only the label changes and the rate the user typed stays.
 */
export function onBenchmark(rates: ReferenceRates | null, code: ReferenceRate): Partial<FullTranche> {
  const now = currentRate(rates, code);
  return now ? { reference_rate: code, reference_level: now.value, reference_path: [] } : { reference_rate: code };
}

/**
 * A new floating facility starts on the deal currency's usual benchmark at its current level
 * (SONIA for sterling, SOFR for dollars ...), when the API has one; otherwise as before.
 */
export function startOnBenchmark(rates: ReferenceRates | null, currency: string, tranche: FullTranche): FullTranche {
  const code = rates?.currency_benchmarks[currency];
  if (!tranche.floating || !code || !currentRate(rates, code)) return tranche;
  return { ...tranche, ...onBenchmark(rates, code) };
}

/**
 * The benchmark chosen in the editor: today's level for a facility on a flat rate, the label
 * only for one with a typed year-by-year path (switching to look must not erase it; "Use
 * today's level" is one click away).
 */
export function chooseBenchmark(rates: ReferenceRates | null, tranche: FullTranche, code: ReferenceRate): Partial<FullTranche> {
  return new Set(tranche.reference_path).size > 1 ? { reference_rate: code } : onBenchmark(rates, code);
}

/** Whether a facility's reference is exactly today's level (flat, no path of its own). */
export function onCurrentRate(rates: ReferenceRates | null, tranche: FullTranche): boolean {
  const now = currentRate(rates, tranche.reference_rate);
  return !!now && tranche.reference_path.length === 0 && tranche.reference_level === now.value;
}
