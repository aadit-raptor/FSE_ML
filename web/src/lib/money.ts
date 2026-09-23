/**
 * Currency and money units (PLAN.md 2.2). Every money figure belongs to a
 * currency (any ISO 4217 code) and is counted in thousands, millions or
 * billions. Labels come from the browser's own CLDR data, so no currency is
 * written into the code: `moneyLabel({ currency: "EUR", unit: "thousands" })`
 * is "€k".
 */
import type { Schemas } from "@/lib/api/client";

export type Money = Schemas["Money"];
export type MoneyUnit = Money["unit"];

/** What an answer's money is in when it doesn't say: the API's own default. */
export const DEFAULT_MONEY: Money = { currency: "USD", unit: "millions" };

export const MONEY_UNITS: { value: MoneyUnit; label: string; short: string; scale: number }[] = [
  { value: "thousands", label: "Thousands", short: "k", scale: 1e3 },
  { value: "millions", label: "Millions", short: "M", scale: 1e6 },
  { value: "billions", label: "Billions", short: "bn", scale: 1e9 },
];

/** A FieldSpec unit meaning "the money label of whatever is on screen". */
export const MONEY = "money";

const FALLBACK_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CNY", "INR", "BRL", "ZAR", "AUD", "CAD", "CHF"];

/** Every currency code this browser knows (ISO 4217). */
export function currencyCodes(): string[] {
  const intl = Intl as typeof Intl & { supportedValuesOf?: (k: string) => string[] };
  try {
    const values = intl.supportedValuesOf?.("currency");
    return values && values.length ? values : FALLBACK_CURRENCIES;
  } catch {
    return FALLBACK_CURRENCIES;
  }
}

const symbols = new Map<string, string>();

/**
 * The currency's symbol, disambiguated (USD and CAD get different ones), with "€" and "₹" as they are,
 * or its code when there is no symbol ("CHF"). English formatting until the
 * locale work (PLAN.md 2.3).
 */
export function currencySymbol(currency: string): string {
  const cached = symbols.get(currency);
  if (cached) return cached;
  let symbol = currency;
  try {
    const parts = new Intl.NumberFormat("en-US", { style: "currency", currency, currencyDisplay: "symbol" }).formatToParts(0);
    symbol = parts.find((p) => p.type === "currency")?.value ?? currency;
  } catch {
    symbol = currency;
  }
  symbols.set(currency, symbol);
  return symbol;
}

/** Name of the currency for a list: "EUR · Euro". */
export function currencyName(currency: string): string {
  try {
    const name = new Intl.DisplayNames(["en"], { type: "currency" }).of(currency);
    return name && name !== currency ? `${currency} · ${name}` : currency;
  } catch {
    return currency;
  }
}

export function unitShort(unit: MoneyUnit): string {
  return MONEY_UNITS.find((u) => u.value === unit)?.short ?? unit;
}

/** "€k" for euro thousands, "£bn" for sterling billions; a symbol that ends in a letter gets a space: "CHF M". */
export function moneyLabel(money: Money): string {
  const symbol = currencySymbol(money.currency);
  const short = unitShort(money.unit);
  return /\p{L}$/u.test(symbol) ? `${symbol} ${short}` : `${symbol}${short}`;
}

/** A field's unit on screen: the money label for money fields, else the unit as written. */
export function fieldUnit(unit: string, money: Money): string {
  return unit === MONEY ? moneyLabel(money) : unit;
}

/** Factor that turns an amount in `from` into the same amount in `to` (millions to thousands: 1,000). */
export function unitFactor(from: MoneyUnit, to: MoneyUnit): number {
  const scale = (u: MoneyUnit) => MONEY_UNITS.find((x) => x.value === u)?.scale ?? 1e6;
  return scale(from) / scale(to);
}

/** Fill a "{money}" placeholder in a label: "Dividends, {money}" -> "Dividends, €k". */
export function withMoney(text: string, label: string): string {
  return text.replace("{money}", label);
}
