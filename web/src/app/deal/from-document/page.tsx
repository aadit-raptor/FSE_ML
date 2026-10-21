import type { Metadata } from "next";

import { FromDocumentStep } from "@/components/deal/steps/FromDocumentStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeDeal", "dealFromDocument") };

export default function Page() {
  return <FromDocumentStep />;
}
