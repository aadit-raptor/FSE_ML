import type { Metadata } from "next";

import { ExamplesStep } from "@/components/library/Examples";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeLibrary", "libraryExamples") };

export default function Page() {
  return <ExamplesStep />;
}
