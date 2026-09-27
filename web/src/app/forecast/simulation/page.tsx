import type { Metadata } from "next";

import { SimulationStep } from "@/components/forecast/ForecastSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeForecast", "forecastSimulation") };

export default function Page() {
  return <SimulationStep />;
}
