import type { Metadata } from "next";

import { DebtStep } from "@/components/deal/steps/DebtStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeDeal", "dealDebt") };

export default function Page() {
  return <DebtStep />;
}
