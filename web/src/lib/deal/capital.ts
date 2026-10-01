import type { Schemas } from "@/lib/api/client";

/**
 * Debt sized two ways: as % of EV (what the model takes) or as multiples of
 * EBITDA (how the wizard's first step asks for it). Mirrors
 * core/deal.py::capital_structure_from_multiples so both views agree.
 */

export function multiplesFromPct(entryMult: number, debtPct: number, seniorPct: number) {
  const debtX = (entryMult * debtPct) / 100;
  const seniorX = (debtX * seniorPct) / 100;
  return { seniorX, mezzX: debtX - seniorX };
}

export function pctFromMultiples(entryMult: number, seniorX: number, mezzX: number) {
  const totalX = seniorX + mezzX;
  return {
    debtPct: entryMult > 0 ? Math.min((totalX / entryMult) * 100, 99) : 0,
    seniorPct: totalX > 0 ? (seniorX / totalX) * 100 : 70,
  };
}

// ---------------------------------------------------------------------------
// Debt facility by facility (PLAN.md 2.4)
// ---------------------------------------------------------------------------

/** One facility in the deal's debt, every field present (the editor shows them all). */
export type FullTranche = Required<Omit<Schemas["TrancheIn"], "currency">>;
export type TrancheKind = FullTranche["kind"];
export type ReferenceRate = FullTranche["reference_rate"];

/** i18n-keys: trancheKinds.* */
export const TRANCHE_KINDS: TrancheKind[] = [
  "amortising_term_loan",
  "institutional_term_loan",
  "unitranche",
  "second_lien",
  "senior_notes",
  "pik_notes",
  "vendor_loan",
  "revolver",
  "shareholder_loan",
];

/** i18n-keys: referenceRates.* */
export const REFERENCE_RATES: ReferenceRate[] = ["SOFR", "SONIA", "ESTR", "EURIBOR", "TONA", "SARON", "BBSY", "MIBOR", "custom"];

/** TrancheSpec's own defaults (core/debt.py): what a facility is before its kind says more. */
const SPEC_DEFAULTS: Omit<FullTranche, "name" | "kind"> = {
  amount: 0,
  drawn_pct: 100,
  floating: false,
  fixed_rate: 0,
  reference_rate: "custom",
  reference_level: 0,
  reference_path: [],
  margin: 0,
  floor: 0,
  maturity_years: 7,
  amort_pct: 0,
  amort_schedule: [],
  upfront_fee_pct: 0,
  commitment_fee_pct: 0,
  sweep: false,
  sweep_share: 100,
  pik_share: 0,
  sweep_priority: 0,
  allow_redraw: false,
};

/**
 * What each kind is when the user picks it: core/debt.py `PRESETS`, field for
 * field (tests/test_debt_structures.py reads this block and compares). No
 * rates and no sizes -- those are the deal's, and a made-up one would be a
 * default without a source (PLAN.md principle 4).
 */
// presets-begin
export const KIND_PRESETS: Record<TrancheKind, Partial<FullTranche>> = {
  amortising_term_loan: { floating: true, amort_pct: 5.0, sweep: true, maturity_years: 6 },
  institutional_term_loan: { floating: true, amort_pct: 1.0, sweep: true, maturity_years: 7 },
  unitranche: { floating: true, sweep: true, maturity_years: 7 },
  second_lien: { floating: true, maturity_years: 8 },
  senior_notes: { floating: false, maturity_years: 8 },
  pik_notes: { floating: false, pik_share: 100.0, maturity_years: 8 },
  vendor_loan: { floating: false, pik_share: 100.0, maturity_years: 8 },
  revolver: { floating: true, drawn_pct: 0.0, commitment_fee_pct: 0.5, sweep: true, allow_redraw: true, maturity_years: 6 },
  shareholder_loan: { floating: false, pik_share: 100.0, maturity_years: 10 },
};
// presets-end

/** A facility of `kind`, shaped the way that kind usually is, with nothing drawn yet. */
export function newTranche(kind: TrancheKind, name: string): FullTranche {
  return { ...SPEC_DEFAULTS, ...KIND_PRESETS[kind], name, kind };
}

/** Every field some kind's preset sets: what a change of kind resets. */
const SHAPE_FIELDS = new Set(Object.values(KIND_PRESETS).flatMap((p) => Object.keys(p)));

/**
 * The same facility as another kind: the new kind's shape (the preset's
 * fields, or TrancheSpec's defaults where the preset says nothing), and
 * everything else -- name, size, pricing, fees, schedule, sweep share -- kept.
 */
export function withKind(t: FullTranche, kind: TrancheKind): FullTranche {
  const kept = Object.fromEntries(Object.entries(t).filter(([k]) => !SHAPE_FIELDS.has(k)));
  return { ...newTranche(kind, t.name), ...kept, kind };
}

/** A facility as the API sent it back (defaults left out), with every field present. */
export function fullTranche(t: Schemas["TrancheIn"]): FullTranche {
  return { ...SPEC_DEFAULTS, ...t } as FullTranche;
}

/**
 * The deal's senior + mezzanine sizing, written out as two facilities:
 * core/debt.py `equivalent_tranches`, line for line. Running it gives exactly
 * the answer the percentages give; the deal's own IRR does not move.
 *
 * Both are floating, the senior at the base rate plus nothing and the
 * mezzanine at the base rate plus its spread, because that is how the Monte
 * Carlo simulation has always moved them. Their names are the engine's, so
 * the schedule tiles keep their titles.
 */
type LeaseFields = Pick<Schemas["DealInputsIn"], "accounting_standard" | "lease_view" | "lease_cost" | "lease_liability">;

/**
 * The EBITDA a deal is valued on (PLAN.md 2.6), a line-for-line mirror of core/accounting.py
 * lease_terms: after lease costs for an IFRS figure, with them added back when priced
 * post-IFRS 16. A deal without leases gets its own EBITDA.
 */
export function valuationEbitda(inputs: { ebitda: number } & LeaseFields): number {
  const cost = inputs.lease_cost ?? 0;
  const ifrs = inputs.accounting_standard === "ifrs";
  const view = inputs.lease_view || (ifrs ? "post_ifrs16" : "pre_ifrs16");
  const operating = ifrs ? inputs.ebitda - cost : inputs.ebitda;
  return view === "post_ifrs16" ? operating + cost : operating;
}

export function equivalentTranches(
  inputs: { ebitda: number; entry_mult: number; debt_pct: number; senior_pct: number; base_rate: number; mezz_spread: number; hold: number } & LeaseFields,
  seniorAmortPct: number,
): FullTranche[] {
  const debt = (valuationEbitda(inputs) * inputs.entry_mult * inputs.debt_pct) / 100;
  const round2 = (x: number) => Math.round(x * 100) / 100;
  const senior = round2((debt * inputs.senior_pct) / 100);
  const mezz = round2(debt * (1 - inputs.senior_pct / 100));
  return [
    {
      ...SPEC_DEFAULTS,
      name: "Senior Term Loan", // core/debt.py's name, so useEngineLabel translates it: text-ok
      kind: "amortising_term_loan",
      amount: senior,
      floating: true,
      reference_rate: "custom",
      reference_level: inputs.base_rate,
      amort_pct: seniorAmortPct,
      sweep: true,
      maturity_years: inputs.hold,
    },
    {
      ...SPEC_DEFAULTS,
      name: "Mezzanine", // as above: text-ok
      kind: "second_lien",
      amount: mezz,
      floating: true,
      reference_rate: "custom",
      reference_level: inputs.base_rate,
      margin: inputs.mezz_spread,
      sweep: true,
      maturity_years: inputs.hold,
    },
  ];
}

/** Debt drawn at close (money): a revolver's undrawn commitment is nobody's money yet. */
export function drawnDebt(tranches: Schemas["TrancheIn"][]): number {
  return tranches.reduce((sum, t) => sum + ((t.amount ?? 0) * (t.drawn_pct ?? 100)) / 100, 0);
}

/** Total debt as a share of EV (%), whichever way the deal sizes it. */
export function debtShareOfEv(
  inputs: { ebitda: number; entry_mult: number; debt_pct: number; tranches: Schemas["TrancheIn"][] } & LeaseFields,
): number {
  if (!inputs.tranches.length) return inputs.debt_pct;
  const ev = valuationEbitda(inputs) * inputs.entry_mult;
  return ev > 0 ? (drawnDebt(inputs.tranches) / ev) * 100 : 0;
}

/** Whether the rate draw of the Monte Carlo simulation moves any of the deal's facilities. */
export function floatingCount(tranches: Schemas["TrancheIn"][]): number {
  return tranches.filter((t) => t.floating).length;
}
