import type { Metadata } from "next";

import { ReferencesStep } from "@/components/library/References";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeLibrary", "libraryReferences") };

export default function Page() {
  return <ReferencesStep />;
}
