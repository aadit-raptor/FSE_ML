import type { Metadata } from "next";

import { DistributionStep } from "@/components/montecarlo/MonteCarloSteps";

export const metadata: Metadata = { title: "Monte Carlo · Distribution" };

export default function Page() {
  return <DistributionStep />;
}
