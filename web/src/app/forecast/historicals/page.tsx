import type { Metadata } from "next";

import { HistoricalsStep } from "@/components/forecast/ForecastSteps";

export const metadata: Metadata = { title: "Forecast · Historicals" };

export default function Page() {
  return <HistoricalsStep />;
}
