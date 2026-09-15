import type { Metadata } from "next";

import { YearsStep } from "@/components/backtest/BacktestSteps";

export const metadata: Metadata = { title: "Backtest · Year by year" };

export default function Page() {
  return <YearsStep />;
}
