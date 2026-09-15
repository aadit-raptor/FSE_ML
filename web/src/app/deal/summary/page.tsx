import type { Metadata } from "next";

import { SummaryStep } from "@/components/deal/steps/SummaryStep";

export const metadata: Metadata = { title: "Deal · Summary" };

export default function Page() {
  return <SummaryStep />;
}
