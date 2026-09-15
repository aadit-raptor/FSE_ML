import type { Metadata } from "next";

import { CorrelationsStep } from "@/components/settings/SettingsSteps";

export const metadata: Metadata = { title: "Settings · Correlations" };

export default function Page() {
  return <CorrelationsStep />;
}
