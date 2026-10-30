/**
 * The guided first deal and the sample deals (PLAN.md 7.1).
 *
 * A guided deal asks four things -- the company's country, its industry, its size (EBITDA) and the
 * currency -- and then starts from the sourced figures for them (PLAN.md 4.3: Damodaran's industry
 * averages for the region, the IMF's growth, the country's tax rate, the currency's benchmark rate).
 * A sample deal is the same four answers given in advance, so its figures are as sourced as a guided
 * deal's and only its size is made up. The answers and the countries' currencies live in `data.json`,
 * which tests/test_onboarding.py checks against the economic data's own list.
 */
import type { DealInputs } from "@/lib/deal/fields";
import { startingPatch, type Starting } from "@/lib/benchmarks";
import type { MoneyUnit } from "@/lib/money";

import data from "./data.json";

export type GuidedAnswers = { country: string; industry: string; ebitda: number; currency: string; unit: MoneyUnit };

/** i18n-keys: onboarding.sample_* */
export type Sample = GuidedAnswers & { id: string };

export const SAMPLES: Sample[] = data.samples.map((s) => ({ ...s, unit: s.unit as MoneyUnit }));

/** The guide's steps, in order. */
export const GUIDE_STEPS = ["region", "sector", "size", "currency", "figures"] as const;
export type GuideStep = (typeof GUIDE_STEPS)[number];

const CURRENCIES: Record<string, string> = data.currencies;

/** The currency a company in `country` most likely reports in, where the economic data covers it. */
export function currencyOf(country: string): string | null {
  return CURRENCIES[country] ?? null;
}

/** The inputs a guided or sample deal is created with: its answers, then every sourced figure. */
export function guidedInputs(answers: GuidedAnswers, starting: Starting | null): Partial<DealInputs> {
  return {
    currency: answers.currency,
    unit: answers.unit,
    ebitda: answers.ebitda,
    country: answers.country,
    industry: answers.industry,
    ...(starting ? startingPatch(starting) : {}),
  };
}
