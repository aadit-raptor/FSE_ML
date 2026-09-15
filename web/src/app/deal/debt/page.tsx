import type { Metadata } from "next";

import { DebtStep } from "@/components/deal/steps/DebtStep";

export const metadata: Metadata = { title: "Deal · Debt & cash flow" };

export default function Page() {
  return <DebtStep />;
}
