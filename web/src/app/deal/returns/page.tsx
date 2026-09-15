import type { Metadata } from "next";

import { ReturnsStep } from "@/components/deal/steps/ReturnsStep";

export const metadata: Metadata = { title: "Deal · Returns" };

export default function Page() {
  return <ReturnsStep />;
}
