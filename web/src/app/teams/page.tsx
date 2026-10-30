import type { Metadata } from "next";

import { TeamsScreen } from "@/components/teams/TeamsScreen";
import { TEAMS_TITLE } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: TEAMS_TITLE };

export default function Page() {
  return <TeamsScreen />;
}
