import type { Metadata } from "next";

import { HelpIndex } from "@/components/help/Help";
import { HELP_TITLE } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: HELP_TITLE };

export default function Page() {
  return <HelpIndex />;
}
