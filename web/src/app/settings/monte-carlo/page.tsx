import type { Metadata } from "next";

import { MonteCarloDefaultsStep } from "@/components/settings/SettingsSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeSettings", "settingsMonteCarlo") };

export default function Page() {
  return <MonteCarloDefaultsStep />;
}
