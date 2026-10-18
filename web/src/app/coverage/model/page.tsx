import type { Metadata } from "next";

import { ModelStep } from "@/components/coverage/ModelStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeCoverage", "coverageModel") };

export default function Page() {
  return <ModelStep />;
}
