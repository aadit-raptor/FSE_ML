import type { Metadata } from "next";

import { LiveStep } from "@/components/montecarlo/MonteCarloML";

export const metadata: Metadata = { title: "Monte Carlo · Live" };

export default function Page() {
  return <LiveStep />;
}
