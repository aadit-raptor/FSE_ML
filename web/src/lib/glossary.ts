/**
 * Which glossary entry (PLAN.md 6.4, `web/messages/<language>.json` `glossary`) a label looked up at run time
 * belongs to: the engine's own labels, a facility named by the user, a money unit. The ids are plain strings;
 * `components/ui/Term.tsx` draws the definition, and draws nothing for an id the glossary lacks.
 *
 * tests/test_explain.py requires an entry for every engine label, tranche kind and money unit these can give.
 */
import { camel } from "@/lib/i18n/useEngineText";

/** A facility: the engine's name when it is one of the engine's own (Senior term loan), else its kind's. */
export function trancheTerm(name: string, kind?: string | null): string | null {
  if (kind) return `kind_${kind}`;
  return `tranche_${camel(name)}`;
}

/** A Monte Carlo driver, as the simulation names it ("Exit Multiple"). */
export const driverTerm = (name: string) => `driver_${camel(name)}`;

/** An equity bridge step, by the API's short axis label ("EBITDA growth"). */
export const bridgeTerm = (key: string) => `bridgeAxis_${camel(key)}`;

/** A money unit (lib/money.ts): thousands, lakhs, millions, crores, billions. */
export const unitTerm = (unit: string) => `unit_${unit}`;

/** A research company's unit as the analyst names it ("Rs lakh", "₹ crore"). */
export function unitNameTerm(name: string | null | undefined): string | null {
  const found = /(thousand|lakh|million|crore|billion)/i.exec(name ?? "");
  return found ? `unit_${found[1].toLowerCase()}s` : null;
}

/** A statement line whose name follows the accounting standard (lib/i18n/useStandardLabel.ts). */
export function statementTerm(line: "interest" | "netIncome", standard: string | null | undefined): string {
  if (standard === "ifrs") return line === "interest" ? "financeCosts" : "profitForYear";
  return line;
}
