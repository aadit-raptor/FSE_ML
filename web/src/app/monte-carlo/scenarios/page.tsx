import type { Metadata } from "next";

import { ScenariosStep } from "@/components/montecarlo/MonteCarloSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeMonteCarlo", "mcScenarios") };

export default function Page() {
  return <ScenariosStep />;
}
