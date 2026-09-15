import type { Metadata } from "next";

import { InputsStep } from "@/components/deal/steps/InputsStep";

export const metadata: Metadata = { title: "Deal · Deal inputs" };

export default function Page() {
  return <InputsStep />;
}
