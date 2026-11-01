import type { Metadata } from "next";

import { QuartersStep } from "@/components/portfolio/QuartersStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modePortfolio", "portfolioQuarters") };

export default function Page() {
  return <QuartersStep />;
}
