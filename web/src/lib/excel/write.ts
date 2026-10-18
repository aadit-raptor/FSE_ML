/**
 * Writing a checked quarter into the open workbook (PLAN.md R3).
 *
 * The cells come from the API (`GET /api/research/periods/{id}/cells`), from
 * the same description the download is written from (research/coverage.py
 * `layout`), so the add-in and the download hold the same cells:
 * subtotals as formulas of their parts, figures as numbers, every figure's
 * page in a note. Words go in as words even when they start with "=" (a
 * cell formatted as text, and Excel's leading apostrophe), so a company's
 * name can never become a formula. Each sheet is new: a name the workbook already has gets a number.
 *
 * A sector model's sheets (PLAN.md R4, "Load model") come the same way from `/api/research/models/cells`, with
 * fills, font colours and defined names besides: the yellow assumption cells are named on their sheet.
 */
import { api, type Schemas } from "@/lib/api/client";

import type { ExcelContext, ExcelWorksheet } from "./office";

export type PeriodCells = Schemas["PeriodCells"];
export type SheetOut = Schemas["SheetOut"];

export async function periodCells(periodId: string): Promise<PeriodCells | null> {
  const { data } = await api.GET("/api/research/periods/{period_id}/cells", { params: { path: { period_id: periodId } } });
  return data ?? null;
}

const ALIGN = { left: "Left", right: "Right", center: "Center" } as const;

/** Excel's own limit on a sheet's name, the characters it refuses in one, and the name it keeps for itself. */
const MAX_NAME = 31;
const REFUSED = /[[\]:*?/\\]/g;
const RESERVED = "history";

/** A name for `wanted` that no sheet in `taken` has (Excel compares names without case). */
export function freeName(wanted: string, taken: string[]): string {
  // nor may a name start or end with an apostrophe
  const base = wanted.replace(REFUSED, " ").slice(0, MAX_NAME).replace(/^'+|'+$/g, "").trim() || "Sheet";
  const used = new Set([...taken.map((n) => n.toLowerCase()), RESERVED]);
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

/**
 * A width in characters of the default font (what the download holds) as Office.js's points: Excel draws a
 * column of n characters n x 7 + 5 pixels wide, and a pixel is three quarters of a point.
 */
export function widthInPoints(characters: number): number {
  return Math.round((characters * 7 + 5) * 0.75 * 100) / 100;
}

/** Words Excel would read as a formula when written: written with its text marker, a leading apostrophe. */
function asWords(text: string): string {
  return /^[=+\-@]/.test(text) ? `'${text}` : text;
}

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
  cells: { sheets: SheetOut[] },
  support: { notes: boolean; comments: boolean },
  open = 1,
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
          range.values = [[asWords(cell.text)]];
        } else {
          if (cell.number_format) range.numberFormat = [[cell.number_format]];
          if (cell.formula) range.formulas = [[cell.formula]];
          else if (cell.number !== null && cell.number !== undefined) range.values = [[cell.number]];
        }
        if (cell.bold) range.format.font.bold = true;
        if (cell.color) range.format.font.color = `#${cell.color}`;
        if (cell.size) range.format.font.size = cell.size;
        if (cell.fill) range.format.fill.color = `#${cell.fill}`;
        if (cell.align) range.format.horizontalAlignment = ALIGN[cell.align];
        if (cell.wrap) range.format.wrapText = true;
        if (cell.note && writeNote) writeNote(ws, name, address, cell.note);
      }
      // a sector model's assumption cells and constants, named on their own sheet (ExcelApi 1.4) so a second
      // model loaded into the same workbook keeps its own
      for (const [defined, address] of Object.entries(sheet.names ?? {})) {
        ws.names?.add(defined, ws.getRange(address.replace(/\$/g, "")));
      }
      for (const [row, height] of Object.entries(sheet.heights ?? {})) {
        ws.getRange(`${row}:${row}`).format.rowHeight = height;
      }
      for (const [letter, width] of Object.entries(sheet.widths)) {
        ws.getRange(`${letter}:${letter}`).format.columnWidth = widthInPoints(width);
      }
      const frozen = sheet.freeze ? frozenRange(sheet.freeze) : null;
      if (frozen) ws.freezePanes.freezeAt(frozen);
      await context.sync();
    }
    // the checked results open on their first statement, not the About sheet; a model on its own sheet
    (added[open] ?? added[0])?.activate();
    await context.sync();
    return { sheets: names, notes: support.notes ? "notes" : support.comments ? "comments" : "none" };
  });
}
