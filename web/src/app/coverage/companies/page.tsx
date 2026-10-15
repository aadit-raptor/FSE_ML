import type { Metadata } from "next";

import { CompaniesStep } from "@/components/coverage/CompaniesStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeCoverage", "coverageCompanies") };

export default function Page() {
  return <CompaniesStep />;
}
