import type { Metadata } from "next";

import { SupportScreen } from "@/components/legal/SupportScreen";
import { SUPPORT_TITLE } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: SUPPORT_TITLE };

export default function Page() {
  return <SupportScreen />;
}
