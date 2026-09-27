import type { Metadata } from "next";

import { AccountScreen } from "@/components/auth/AccountScreen";
import { ACCOUNT_TITLE } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: ACCOUNT_TITLE };

export default function Page() {
  return <AccountScreen />;
}
