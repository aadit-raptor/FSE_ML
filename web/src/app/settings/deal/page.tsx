import type { Metadata } from "next";

import { DealDefaultsStep } from "@/components/settings/SettingsSteps";

export const metadata: Metadata = { title: "Settings · Deal defaults" };

export default function Page() {
  return <DealDefaultsStep />;
}
