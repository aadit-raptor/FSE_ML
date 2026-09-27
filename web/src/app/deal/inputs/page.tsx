import type { Metadata } from "next";

import { InputsStep } from "@/components/deal/steps/InputsStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeDeal", "dealInputs") };

export default function Page() {
  return <InputsStep />;
}
