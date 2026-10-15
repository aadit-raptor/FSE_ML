import type { Metadata } from "next";

import { ConnectDialog } from "@/components/excel/ConnectDialog";
import { EXCEL_CONNECT_TITLE } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: EXCEL_CONNECT_TITLE };

export default function Page() {
  return <ConnectDialog />;
}
