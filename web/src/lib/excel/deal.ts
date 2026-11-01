/**
 * A saved deal in the Excel add-in (PLAN.md 7.3b): the person's deals, and one deal as the live-formula workbook's
 * cells (`GET /api/deals/{id}/model/cells`), the same description the Summary step's "Live workbook" download is
 * written from (api/deal_workbook.py), run on the deal as last saved. The model sheet says which deal it is beside
 * its title (B1, named `variater_deal`), so "Refresh" can find the deal again and write the sheet over.
 */
import { api, type Schemas } from "@/lib/api/client";

import type { ExcelContext } from "./office";
import { namedValue, type SheetRead } from "./read";

export type DealCells = Schemas["DealModelCells"];
export type DealListed = Schemas["DealSummary"];

export const SAVED_TEXT = "Variater deal: "; // api/deal_workbook.py SAVED_TEXT: text-ok
const UUID = /^[0-9a-fA-F-]{36}$/;

export async function listDeals(): Promise<DealListed[] | null> {
  const { data } = await api.GET("/api/deals", {});
  return data?.deals ?? null;
}

export async function dealCells(dealId: string): Promise<DealCells | null> {
  const { data } = await api.GET("/api/deals/{deal_id}/model/cells", { params: { path: { deal_id: dealId } } });
  return data ?? null;
}

/** The saved deal a sheet read back says it is, or null when it is no deal's sheet. */
export function dealOfSheet(sheet: SheetRead): string | null {
  const text = namedValue(sheet, "variater_deal", "B1");
  if (typeof text !== "string" || !text.startsWith(SAVED_TEXT)) return null;
  const id = text.slice(SAVED_TEXT.length).trim();
  return UUID.test(id) ? id : null;
}

type Runner = (batch: (context: ExcelContext) => Promise<boolean>) => Promise<boolean>;

/**
 * Removes the sheet named `name` so a refreshed one can take its name; false (and nothing removed) when it is the
 * workbook's only sheet, which Excel will not delete: the refreshed sheet then sits beside it.
 */
export async function removeSheet(run: Runner, name: string): Promise<boolean> {
  return run(async (context) => {
    const sheets = context.workbook.worksheets;
    sheets.load("items/name");
    await context.sync();
    if (sheets.items.length < 2) return false;
    const found = sheets.items.find((ws) => ws.name === name);
    if (!found?.delete) return false;
    found.delete();
    await context.sync();
    return true;
  });
}
