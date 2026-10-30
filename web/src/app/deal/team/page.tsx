import type { Metadata } from "next";

import { TeamStep } from "@/components/deal/steps/TeamStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeDeal", "dealTeam") };

export default function Page() {
  return <TeamStep />;
}
