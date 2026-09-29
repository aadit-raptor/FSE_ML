import type { Metadata } from "next";

import { HeatmapStep } from "@/components/montecarlo/MonteCarloSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeMonteCarlo", "mcHeatmap") };

export default function Page() {
  return <HeatmapStep />;
}
