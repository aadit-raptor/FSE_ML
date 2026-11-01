import type { Metadata } from "next";

import { DashboardStep } from "@/components/portfolio/DashboardStep";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("modePortfolio", "portfolioDashboard") };

export default function Page() {
  return <DashboardStep />;
}
