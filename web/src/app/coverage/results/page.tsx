import type { Metadata } from "next";

import { ResultsStep } from "@/components/coverage/ResultsStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeCoverage", "coverageResults") };

export default function Page() {
  return <ResultsStep />;
}
