import type { Metadata } from "next";

import { FeesStep } from "@/components/settings/SettingsSteps";

export const metadata: Metadata = { title: "Settings · Fees" };

export default function Page() {
  return <FeesStep />;
}
