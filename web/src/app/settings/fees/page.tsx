import type { Metadata } from "next";

import { FeesStep } from "@/components/settings/SettingsSteps";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeSettings", "settingsFees") };

export default function Page() {
  return <FeesStep />;
}
