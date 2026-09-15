import type { Metadata } from "next";

import { PredictedStep } from "@/components/backtest/BacktestSteps";

export const metadata: Metadata = { title: "Backtest · Predicted vs actual" };

export default function Page() {
  return <PredictedStep />;
}
