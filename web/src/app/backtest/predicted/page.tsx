import type { Metadata } from "next";

import { PredictedStep } from "@/components/backtest/BacktestSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeBacktest", "backtestPredicted") };

export default function Page() {
  return <PredictedStep />;
}
