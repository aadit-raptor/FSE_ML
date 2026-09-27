import type { Metadata } from "next";

import { AttributionStep } from "@/components/backtest/BacktestSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeBacktest", "backtestAttribution") };

export default function Page() {
  return <AttributionStep />;
}
