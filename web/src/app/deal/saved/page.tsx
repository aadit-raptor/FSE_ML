import type { Metadata } from "next";

import { SavedStep } from "@/components/deal/steps/SavedStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeDeal", "dealSaved") };

export default function Page() {
  return <SavedStep />;
}
