import type { Metadata } from "next";

import { WhatsNew } from "@/components/help/Help";
import { WHATS_NEW_TITLE } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: WHATS_NEW_TITLE };

export default function Page() {
  return <WhatsNew />;
}
