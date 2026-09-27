import type { Metadata } from "next";

import { AssumptionsStep } from "@/components/forecast/ForecastSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeForecast", "forecastAssumptions") };

export default function Page() {
  return <AssumptionsStep />;
}
