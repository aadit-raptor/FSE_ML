import type { Metadata } from "next";

import { DeveloperScreen } from "@/components/developer/DeveloperScreen";
import { DEVELOPER_TITLE } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: DEVELOPER_TITLE };

export default function Page() {
  return <DeveloperScreen />;
}
