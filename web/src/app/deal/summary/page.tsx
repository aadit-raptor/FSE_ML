import type { Metadata } from "next";

import { SummaryStep } from "@/components/deal/steps/SummaryStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeDeal", "dealSummary") };

export default function Page() {
  return <SummaryStep />;
}
