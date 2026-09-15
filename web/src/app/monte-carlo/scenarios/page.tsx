import type { Metadata } from "next";

import { ScenariosStep } from "@/components/montecarlo/MonteCarloSteps";

export const metadata: Metadata = { title: "Monte Carlo · Scenarios" };

export default function Page() {
  return <ScenariosStep />;
}
