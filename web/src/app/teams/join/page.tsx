import type { Metadata } from "next";

import { JoinScreen } from "@/components/teams/JoinScreen";
import { JOIN_TITLE } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: JOIN_TITLE };

export default function Page() {
  return <JoinScreen />;
}
