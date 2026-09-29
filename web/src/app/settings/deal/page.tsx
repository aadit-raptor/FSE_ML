import type { Metadata } from "next";

import { DealDefaultsStep } from "@/components/settings/SettingsSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeSettings", "settingsDeal") };

export default function Page() {
  return <DealDefaultsStep />;
}
