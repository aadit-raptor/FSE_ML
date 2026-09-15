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
