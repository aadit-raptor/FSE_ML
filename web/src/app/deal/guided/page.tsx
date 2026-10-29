import type { Metadata } from "next";

import { GuidedDeal } from "@/components/onboarding/GuidedDeal";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeDeal", "dealGuided") };

export default function Page() {
  return <GuidedDeal />;
}
