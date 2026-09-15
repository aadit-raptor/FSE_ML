import type { Metadata } from "next";

import { SchedulesStep } from "@/components/forecast/ForecastSteps";

export const metadata: Metadata = { title: "Forecast · Schedules" };

export default function Page() {
  return <SchedulesStep />;
}
