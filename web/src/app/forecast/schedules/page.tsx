import type { Metadata } from "next";

import { SchedulesStep } from "@/components/forecast/ForecastSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeForecast", "forecastSchedules") };

export default function Page() {
  return <SchedulesStep />;
}
