import type { Metadata } from "next";

import { DistributionStep } from "@/components/montecarlo/MonteCarloSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeMonteCarlo", "mcDistribution") };

export default function Page() {
  return <DistributionStep />;
}
