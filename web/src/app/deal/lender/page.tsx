import type { Metadata } from "next";

import { LenderStep } from "@/components/deal/steps/LenderStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeDeal", "dealLender") };

export default function Page() {
  return <LenderStep />;
}
