import type { Metadata } from "next";

import { ExcelPane } from "@/components/excel/ExcelPane";
import { EXCEL_PANE_TITLE } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: EXCEL_PANE_TITLE };

export default function Page() {
  return <ExcelPane />;
}
