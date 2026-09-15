import type { Metadata } from "next";

import { AttributionStep } from "@/components/backtest/BacktestSteps";

export const metadata: Metadata = { title: "Backtest · Error attribution" };

export default function Page() {
  return <AttributionStep />;
}
