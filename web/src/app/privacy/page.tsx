import type { Metadata } from "next";

import { PrivacyScreen } from "@/components/legal/PrivacyScreen";
import { PRIVACY_TITLE } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: PRIVACY_TITLE };

export default function Page() {
  return <PrivacyScreen />;
}
