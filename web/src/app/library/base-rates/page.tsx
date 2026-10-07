import type { Metadata } from "next";

import { BaseRatesStep } from "@/components/library/BaseRates";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeLibrary", "libraryBaseRates") };

export default function Page() {
  return <BaseRatesStep />;
}
