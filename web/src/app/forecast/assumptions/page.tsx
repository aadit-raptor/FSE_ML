import type { Metadata } from "next";

import { AssumptionsStep } from "@/components/forecast/ForecastSteps";

export const metadata: Metadata = { title: "Forecast · Assumptions" };

export default function Page() {
  return <AssumptionsStep />;
}
