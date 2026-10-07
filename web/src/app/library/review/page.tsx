import type { Metadata } from "next";

import { ReviewStep } from "@/components/library/Review";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeLibrary", "libraryReview") };

export default function Page() {
  return <ReviewStep />;
}
