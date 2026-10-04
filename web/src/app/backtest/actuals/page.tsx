import type { Metadata } from "next";

import { ActualsStep } from "@/components/backtest/BacktestSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeBacktest", "backtestActuals") };

export default function Page() {
  return <ActualsStep />;
}
