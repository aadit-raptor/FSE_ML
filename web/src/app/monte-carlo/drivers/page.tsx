import type { Metadata } from "next";

import { DriversStep } from "@/components/montecarlo/MonteCarloSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeMonteCarlo", "mcDrivers") };

export default function Page() {
  return <DriversStep />;
}
