import type { Metadata } from "next";

import { YearsStep } from "@/components/backtest/BacktestSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeBacktest", "backtestYears") };

export default function Page() {
  return <YearsStep />;
}
