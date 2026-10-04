/**
 * A deal's actual results for plan vs actual (PLAN.md 2.7): the editor's own shape, the conversion to
 * the API's `DealActuals`, and the CSV a user uploads or downloads.
 *
 * The CSV is laid out the way the year-by-year table is, years across:
 *
 *     line,FY2022,FY2023,FY2024
 *     revenue,1200,1310,1405
 *     ebitda,240,268,301
 *     net_income,61,77,90
 *     fcf,95,104,122
 *     total_debt,1100,1010,905
 *     exit_ev,3400
 *     net_debt_at_exit,850
 *     sponsor_equity_entry,1000
 *
 * The header's first cell is ignored and the rest count the years. Exit lines take one value. A blank
 * cell is a figure not known. Comma, semicolon or tab separated; with a semicolon or a tab a decimal
 * comma is read too ("1.234,5"), and a number with no comma keeps its decimal point. Unknown lines are skipped and reported.
 */
import type { Schemas } from "@/lib/api/client";
import type { Money } from "@/lib/money";

export type DealActuals = Schemas["DealActuals"];
export type ActualYear = Schemas["ActualYearIn"];

/** One year's figures, in the order the screens show them. i18n-keys: backtest.line* */
export const ACTUAL_LINES = [
  { key: "revenue", labelKey: "lineRevenue" },
  { key: "ebitda", labelKey: "lineEbitda" },
  { key: "net_income", labelKey: "lineNetIncome" },
  { key: "fcf", labelKey: "lineFcf" },
  { key: "total_debt", labelKey: "lineTotalDebt" },
] as const;
export type LineKey = (typeof ACTUAL_LINES)[number]["key"];

/** The exit: three figures needed, two optional (computed when left out). i18n-keys: backtest.exit* */
export const EXIT_FIELDS = [
  { key: "exit_ev", labelKey: "exitEv", money: true, required: true },
  { key: "net_debt_at_exit", labelKey: "exitNetDebt", money: true, required: true },
  { key: "sponsor_equity_entry", labelKey: "exitEquityEntry", money: true, required: true },
  { key: "moic", labelKey: "exitMoic", money: false, required: false },
  { key: "irr", labelKey: "exitIrr", money: false, required: false },
] as const;
export type ExitKey = (typeof EXIT_FIELDS)[number]["key"];

export type Figure = number | null;
export type YearDraft = Record<LineKey, Figure>;
export type ExitDraft = Record<ExitKey, Figure>;

/** What the editor holds: blanks allowed anywhere, the exit possibly half filled. */
export type ActualsDraft = { money: Money; years: YearDraft[]; exit: ExitDraft | null };

export const MAX_YEARS = 15;

export const blankYear = (): YearDraft => ({ revenue: null, ebitda: null, net_income: null, fcf: null, total_debt: null });
export const blankExit = (): ExitDraft => ({ exit_ev: null, net_debt_at_exit: null, sponsor_equity_entry: null, moic: null, irr: null });

/** A blank draft: every year of the plan, nothing known yet. */
export function blankDraft(money: Money, years: number): ActualsDraft {
  return { money, years: Array.from({ length: years }, blankYear), exit: null };
}

const fig = (v: unknown): Figure => (typeof v === "number" && Number.isFinite(v) ? v : null);

/** The draft for stored (or example) actuals. */
export function draftFrom(a: DealActuals): ActualsDraft {
  return {
    money: { currency: a.currency, unit: a.unit },
    years: a.years.map((y) => Object.fromEntries(ACTUAL_LINES.map(({ key }) => [key, fig(y[key])])) as YearDraft),
    exit: a.exit ? (Object.fromEntries(EXIT_FIELDS.map(({ key }) => [key, fig(a.exit?.[key])])) as ExitDraft) : null,
  };
}

/** The draft with every money figure times k (a deal whose unit changed since the actuals were saved). */
export function scaleDraft(d: ActualsDraft, k: number, money: Money): ActualsDraft {
  const s = (v: Figure) => (v === null ? null : v * k);
  return {
    money,
    years: d.years.map((y) => Object.fromEntries(ACTUAL_LINES.map(({ key }) => [key, s(y[key])])) as YearDraft),
    exit: d.exit && {
      ...d.exit,
      ...Object.fromEntries(EXIT_FIELDS.filter((f) => f.money).map(({ key }) => [key, s(d.exit![key])])),
    },
  };
}

/** The draft with exactly n years: later ones dropped, new ones blank. */
export function resizeYears(d: ActualsDraft, n: number): ActualsDraft {
  const years = d.years.slice(0, n);
  while (years.length < n) years.push(blankYear());
  return { ...d, years };
}

/** True once the draft holds any figure at all. */
export const hasFigures = (d: ActualsDraft): boolean => d.years.some((y) => ACTUAL_LINES.some(({ key }) => y[key] !== null));

/** The exit only when its three required figures are in. */
export function completeExit(e: ExitDraft | null): Schemas["ActualExitIn"] | undefined {
  if (!e || EXIT_FIELDS.some((f) => f.required && e[f.key] === null)) return undefined;
  const out: Record<string, number> = {};
  for (const { key } of EXIT_FIELDS) if (e[key] !== null) out[key] = e[key]!;
  return out as Schemas["ActualExitIn"];
}

/** The API's shape: unknown figures left out, a half-filled exit not sent. */
export function apiActuals(d: ActualsDraft): DealActuals {
  const exit = completeExit(d.exit);
  return {
    currency: d.money.currency,
    unit: d.money.unit,
    years: d.years.map((y) => Object.fromEntries(ACTUAL_LINES.filter(({ key }) => y[key] !== null).map(({ key }) => [key, y[key]]))),
    ...(exit ? { exit } : {}),
  };
}

// ---------------------------------------------------------------------------
// CSV
// ---------------------------------------------------------------------------

/** Names a spreadsheet might use for each line, normalised (lower case, words joined by _). */
const ALIASES: Record<string, LineKey | ExitKey> = {
  revenue: "revenue", revenues: "revenue", sales: "revenue", turnover: "revenue",
  ebitda: "ebitda",
  net_income: "net_income", net_profit: "net_income", profit_after_tax: "net_income",
  fcf: "fcf", free_cash_flow: "fcf",
  total_debt: "total_debt", debt: "total_debt",
  exit_ev: "exit_ev", exit_enterprise_value: "exit_ev", enterprise_value_at_exit: "exit_ev",
  net_debt_at_exit: "net_debt_at_exit", exit_net_debt: "net_debt_at_exit",
  sponsor_equity_entry: "sponsor_equity_entry", equity_at_entry: "sponsor_equity_entry", entry_equity: "sponsor_equity_entry",
  moic: "moic",
  irr: "irr", irr_pct: "irr",
};
const LINE_KEYS = new Set<string>(ACTUAL_LINES.map((l) => l.key));

export type CsvProblem = "empty" | "noYears" | "tooManyYears" | "badNumber" | "exitIncomplete";
export type CsvResult =
  | { ok: true; years: YearDraft[]; exit: ExitDraft | null; ignored: string[] }
  | { ok: false; problem: CsvProblem; where?: string };

const normalise = (s: string) => s.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");

function splitRow(line: string, delimiter: string): string[] {
  const cells: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === delimiter) {
      cells.push(cell);
      cell = "";
    } else cell += ch;
  }
  cells.push(cell);
  return cells.map((c) => c.trim());
}

/** A CSV figure: blank or a dash is unknown; (12) is -12; NaN when it isn't a number. */
export function csvNumber(text: string, decimalComma: boolean): Figure | typeof NaN {
  let s = text.replace(/[\s  ']/g, "");
  if (s === "" || s === "-" || s === "–") return null;
  const negative = /^\(.*\)$/.test(s);
  if (negative) s = s.slice(1, -1);
  // A decimal comma only where a comma is written: "1234.5" in a semicolon file keeps its dot
  if (decimalComma && s.includes(",")) s = s.replace(/\./g, "").replace(",", ".");
  else s = s.replace(/,/g, "");
  if (!/^[+-]?(\d+\.?\d*|\.\d+)(e[+-]?\d+)?$/i.test(s)) return NaN;
  const n = Number(s);
  return negative ? -n : n;
}

export function parseActualsCsv(text: string): CsvResult {
  const lines = text.replace(/^﻿/, "").split(/\r?\n/).filter((l) => l.trim() !== "");
  if (!lines.length) return { ok: false, problem: "empty" };
  const delimiter = lines[0].includes(";") ? ";" : lines[0].includes("\t") ? "\t" : ",";
  const decimalComma = delimiter !== ",";
  const rows = lines.map((l) => splitRow(l, delimiter));
  const header = rows[0];
  // Years: the header's labelled columns, or, with no header labels, the longest line
  let n = header.slice(1).filter((c) => c !== "").length;
  const body = ALIASES[normalise(header[0])] ? rows : rows.slice(1);
  if (body === rows || n === 0) n = Math.max(0, ...body.filter((r) => LINE_KEYS.has(ALIASES[normalise(r[0])] ?? "")).map((r) => r.length - 1));
  if (n === 0) return { ok: false, problem: "noYears" };
  if (n > MAX_YEARS) return { ok: false, problem: "tooManyYears" };

  const years = Array.from({ length: n }, blankYear);
  const exit = blankExit();
  let exitSeen = false;
  const ignored: string[] = [];
  for (const row of body) {
    const key = ALIASES[normalise(row[0])];
    if (!key) {
      if (row[0]) ignored.push(row[0]);
      continue;
    }
    if (LINE_KEYS.has(key)) {
      for (let i = 0; i < n; i++) {
        const v = csvNumber(row[i + 1] ?? "", decimalComma);
        if (Number.isNaN(v)) return { ok: false, problem: "badNumber", where: row[0] };
        years[i][key as LineKey] = v;
      }
    } else {
      const v = csvNumber(row[1] ?? "", decimalComma);
      if (Number.isNaN(v)) return { ok: false, problem: "badNumber", where: row[0] };
      exit[key as ExitKey] = v;
      exitSeen = exitSeen || v !== null;
    }
  }
  const found = exitSeen ? exit : null;
  if (found && EXIT_FIELDS.some((f) => f.required && found[f.key] === null)) return { ok: false, problem: "exitIncomplete" };
  return { ok: true, years, exit: found, ignored };
}

/** The draft as a CSV in the layout above, years labelled as on screen. */
export function actualsCsv(d: ActualsDraft, yearLabels: string[]): string {
  const cell = (v: Figure) => (v === null ? "" : String(v));
  const quote = (s: string) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  const rows = [["line", ...d.years.map((_, i) => yearLabels[i] ?? String(i + 1))].map(quote).join(",")];
  for (const { key } of ACTUAL_LINES) rows.push([key, ...d.years.map((y) => cell(y[key]))].join(","));
  if (d.exit) for (const { key } of EXIT_FIELDS) rows.push(`${key},${cell(d.exit[key])}`);
  return rows.join("\n") + "\n";
}
