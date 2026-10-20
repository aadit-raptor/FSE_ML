import type { Metadata } from "next";

import { ReportStep } from "@/components/coverage/ReportStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeCoverage", "coverageReport") };

export default function Page() {
  return <ReportStep />;
}
