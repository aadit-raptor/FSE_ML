import type { Metadata } from "next";

import { MemoStep } from "@/components/deal/steps/MemoStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeDeal", "dealMemo") };

export default function Page() {
  return <MemoStep />;
}
