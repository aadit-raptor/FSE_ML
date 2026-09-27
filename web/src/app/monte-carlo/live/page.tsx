import type { Metadata } from "next";

import { LiveStep } from "@/components/montecarlo/MonteCarloML";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeMonteCarlo", "mcLive") };

export default function Page() {
  return <LiveStep />;
}
