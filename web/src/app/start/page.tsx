import type { Metadata } from "next";

import { Launcher } from "@/components/shell/Launcher";
import { pageTitle } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: pageTitle("launcher") };

export default function Page() {
  return <Launcher />;
}
