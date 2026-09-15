import type { Metadata } from "next";

import { StatementsStep } from "@/components/forecast/ForecastSteps";

export const metadata: Metadata = { title: "Forecast · Statements" };

export default function Page() {
  return <StatementsStep />;
}
