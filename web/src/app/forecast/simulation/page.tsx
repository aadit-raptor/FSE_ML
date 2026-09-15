import type { Metadata } from "next";

import { SimulationStep } from "@/components/forecast/ForecastSteps";

export const metadata: Metadata = { title: "Forecast · Simulation" };

export default function Page() {
  return <SimulationStep />;
}
