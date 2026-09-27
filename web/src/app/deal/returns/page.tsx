import type { Metadata } from "next";

import { ReturnsStep } from "@/components/deal/steps/ReturnsStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeDeal", "dealReturns") };

export default function Page() {
  return <ReturnsStep />;
}
