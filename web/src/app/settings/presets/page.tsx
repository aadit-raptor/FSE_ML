import type { Metadata } from "next";

import { PresetsStep } from "@/components/settings/SettingsSteps";

export const metadata: Metadata = { title: "Settings · Scenario presets" };

export default function Page() {
  return <PresetsStep />;
}
