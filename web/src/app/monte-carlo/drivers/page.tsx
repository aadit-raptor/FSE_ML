import type { Metadata } from "next";

import { DriversStep } from "@/components/montecarlo/MonteCarloSteps";

export const metadata: Metadata = { title: "Monte Carlo · Drivers" };

export default function Page() {
  return <DriversStep />;
}
