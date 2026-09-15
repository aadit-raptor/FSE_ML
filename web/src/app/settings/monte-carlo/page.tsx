import type { Metadata } from "next";

import { MonteCarloDefaultsStep } from "@/components/settings/SettingsSteps";

export const metadata: Metadata = { title: "Settings · Monte Carlo" };

export default function Page() {
  return <MonteCarloDefaultsStep />;
}
