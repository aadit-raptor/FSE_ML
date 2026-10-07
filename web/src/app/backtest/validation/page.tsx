import type { Metadata } from "next";

import { ValidationStep } from "@/components/backtest/Validation";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modeBacktest", "backtestValidation") };

export default function Page() {
  return <ValidationStep />;
}
