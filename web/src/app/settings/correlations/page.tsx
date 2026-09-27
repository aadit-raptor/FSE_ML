import type { Metadata } from "next";

import { CorrelationsStep } from "@/components/settings/SettingsSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeSettings", "settingsCorrelations") };

export default function Page() {
  return <CorrelationsStep />;
}
