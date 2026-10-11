import type { Metadata } from "next";

import { DocumentsStep } from "@/components/deal/steps/DocumentsStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeDeal", "dealDocuments") };

export default function Page() {
  return <DocumentsStep />;
}
