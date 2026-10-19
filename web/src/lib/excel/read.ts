/**
 * Reading the analyst's sheet back for "Save to Variater" (PLAN.md R5): the active sheet's values from A1 and its
 * sheet-scoped names with their addresses, which is all the API needs to read the named cells back and to refuse a
 * sheet whose structure was changed (research/sheetback.py). Nothing is written to the workbook here.
 */
import type { ExcelContext } from "./office";

export type SheetRead = { title: string; grid: (number | string | boolean | null)[][]; names: Record<string, string> };

/** The first cell of an address such as "Model!B3:H120" as (row, column), from 1. */
function start(address: string): [number, number] {
  const match = /!?\$?([A-Z]+)\$?(\d+)(?::|$)/.exec(address.split("!").pop() ?? "");
  if (!match) return [1, 1];
  const column = match[1].split("").reduce((n, ch) => n * 26 + ch.charCodeAt(0) - 64, 0);
  return [Number(match[2]), column];
}

function plain(value: unknown): number | string | boolean | null {
  if (typeof value === "number" || typeof value === "string" || typeof value === "boolean") return value;
  return null;
}

type Batch = (context: ExcelContext) => Promise<SheetRead>;
type Runner = (batch: Batch) => Promise<SheetRead>;

export async function readActiveSheet(run: Runner): Promise<SheetRead> {
  return run(async (context) => {
    const ws = context.workbook.worksheets.getActiveWorksheet();
    ws.load?.("name");
    const used = ws.getUsedRange();
    used.load("values,address");
    ws.names?.load("items/name,items/formula");
    await context.sync();
    const [row, column] = start(used.address);
    // the grid always starts at A1, so a row and a column are where Excel says they are
    const grid: (number | string | boolean | null)[][] = [];
    for (let r = 1; r < row; r++) grid.push([]);
    for (const values of used.values) grid.push([...Array(column - 1).fill(""), ...values.map(plain)]);
    const names: Record<string, string> = {};
    for (const item of ws.names?.items ?? []) names[item.name] = item.formula.replace(/^=/, "");
    return { title: ws.name, grid, names };
  });
}
