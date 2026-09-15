import type { Metadata } from "next";

import { HeatmapStep } from "@/components/montecarlo/MonteCarloSteps";

export const metadata: Metadata = { title: "Monte Carlo · Heatmap" };

export default function Page() {
  return <HeatmapStep />;
}
