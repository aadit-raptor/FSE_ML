/**
 * Writing a checked quarter into the open workbook (PLAN.md R3).
 *
 * The cells come from the API (`GET /api/research/periods/{id}/cells`), from
 * the same description the download is written from (research/coverage.py
 * `layout`), so the add-in and the download hold the same cells:
 * subtotals as formulas of their parts, figures as numbers, every figure's
 * page in a note. Words go in as words even when they start with "=" (a
 * cell formatted as text first), so a company's name can never become a
 * formula. Each sheet is new: a name the workbook already has gets a number.
 */
import { api, type Schemas } from "@/lib/api/client";

import type { ExcelContext, ExcelWorksheet } from "./office";

export type PeriodCells = Schemas["PeriodCells"];
export type SheetOut = Schemas["SheetOut"];

export async function periodCells(periodId: string): Promise<PeriodCells | null> {
  const { data } = await api.GET("/api/research/periods/{period_id}/cells", { params: { path: { period_id: periodId } } });
  return data ?? null;
}

/** Excel's own limit on a sheet's name, and the characters it refuses in one. */
const MAX_NAME = 31;
const REFUSED = /[[\]:*?/\\]/g;

/** A name for `wanted` that no sheet in `taken` has (Excel compares names without case). */
export function freeName(wanted: string, taken: string[]): string {
  const base = wanted.replace(REFUSED, " ").slice(0, MAX_NAME) || "Sheet";
  const used = new Set(taken.map((n) => n.toLowerCase()));
  if (!used.has(base.toLowerCase())) return base;
  for (let n = 2; ; n++) {
    const name = `${base.slice(0, MAX_NAME - String(n).length - 1)} ${n}`;
    if (!used.has(name.toLowerCase())) return name;
  }
}

export function columnLetter(column: number): string {
  let n = column;
  let out = "";
  while (n > 0) {
    const r = (n - 1) % 26;
    out = String.fromCharCode(65 + r) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

/** Characters to points: Excel's column width counts characters of the default font, about 7 points each. */
const POINTS_PER_CHARACTER = 7;

/** The cell above-left of `freeze` ("C6" freezes A1:B5). */
function frozenRange(freeze: string): string | null {
  const match = /^([A-Z]+)(\d+)$/.exec(freeze);
  if (!match) return null;
  const columns = match[1].split("").reduce((n, ch) => n * 26 + ch.charCodeAt(0) - 64, 0) - 1;
  const rows = Number(match[2]) - 1;
  if (columns < 1 || rows < 1) return null;
  return `A1:${columnLetter(columns)}${rows}`;
}

// The sheet's name is passed in: a sheet just added has no name loaded to read back
type NoteWriter = (ws: ExcelWorksheet, name: string, address: string, text: string) => void;

function noteWriter(context: ExcelContext, notes: boolean, comments: boolean): NoteWriter | null {
  // Notes (ExcelApi 1.18) are what the download holds; comments (1.10) where notes are not yet supported
  const at = (name: string, address: string) => `'${name.replace(/'/g, "''")}'!${address}`;
  if (notes) return (ws, name, address, text) => ws.notes?.add(at(name, address), text);
  if (comments) return (_ws, name, address, text) => context.workbook.comments?.add(at(name, address), text);
  return null;
}

export type Written = { sheets: string[]; notes: "notes" | "comments" | "none" };

/**
 * Writes every sheet of `cells` as new sheets of the open workbook, in the
 * download's order, and opens the first statement.
 */
export async function writeCells(
  run: (batch: (context: ExcelContext) => Promise<Written>) => Promise<Written>,
  cells: PeriodCells,
  support: { notes: boolean; comments: boolean },
): Promise<Written> {
  return run(async (context) => {
    const sheets = context.workbook.worksheets;
    sheets.load("items/name");
    await context.sync();
    const taken = sheets.items.map((ws) => ws.name);
    const writeNote = noteWriter(context, support.notes, support.comments);
    const names: string[] = [];
    const added: ExcelWorksheet[] = [];
    for (const sheet of cells.sheets) {
      const name = freeName(sheet.title, taken);
      taken.push(name);
      names.push(name);
      const ws = sheets.add(name);
      added.push(ws);
      for (const cell of sheet.cells) {
        const address = `${columnLetter(cell.column)}${cell.row}`;
        const range = ws.getRange(address);
        if (cell.text !== null && cell.text !== undefined) {
          range.numberFormat = [["@"]];
          range.values = [[cell.text]];
        } else {
          if (cell.number_format) range.numberFormat = [[cell.number_format]];
          if (cell.formula) range.formulas = [[cell.formula]];
          else range.values = [[cell.number]];
        }
        if (cell.bold) range.format.font.bold = true;
        if (cell.note && writeNote) writeNote(ws, name, address, cell.note);
      }
      for (const [letter, width] of Object.entries(sheet.widths)) {
        ws.getRange(`${letter}:${letter}`).format.columnWidth = width * POINTS_PER_CHARACTER;
      }
      const frozen = sheet.freeze ? frozenRange(sheet.freeze) : null;
      if (frozen) ws.freezePanes.freezeAt(frozen);
      await context.sync();
    }
    // the first statement, not the About sheet
    (added[1] ?? added[0])?.activate();
    await context.sync();
    return { sheets: names, notes: support.notes ? "notes" : support.comments ? "comments" : "none" };
  });
}
