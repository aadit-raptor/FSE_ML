import type { Metadata } from "next";

import { HistoricalsStep } from "@/components/forecast/ForecastSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeForecast", "forecastHistoricals") };

export default function Page() {
  return <HistoricalsStep />;
}
