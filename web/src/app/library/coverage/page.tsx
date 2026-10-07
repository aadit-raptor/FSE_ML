import type { Metadata } from "next";

import { CoverageStep } from "@/components/library/Coverage";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeLibrary", "libraryCoverage") };

export default function Page() {
  return <CoverageStep />;
}
