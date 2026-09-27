import type { Metadata } from "next";

import { StatementsStep } from "@/components/forecast/ForecastSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeForecast", "forecastStatements") };

export default function Page() {
  return <StatementsStep />;
}
