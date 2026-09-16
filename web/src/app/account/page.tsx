import type { Metadata } from "next";

import { AccountScreen } from "@/components/auth/AccountScreen";

export const metadata: Metadata = { title: "Account" };

export default function Page() {
  return <AccountScreen />;
}
