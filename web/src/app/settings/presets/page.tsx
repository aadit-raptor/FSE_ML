import type { Metadata } from "next";

import { PresetsStep } from "@/components/settings/SettingsSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeSettings", "settingsPresets") };

export default function Page() {
  return <PresetsStep />;
}
