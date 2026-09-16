import type { Metadata } from "next";

import { SavedStep } from "@/components/deal/steps/SavedStep";

export const metadata: Metadata = { title: "Deal · Saved deals" };

export default function Page() {
  return <SavedStep />;
}
