import type { Metadata } from "next";

import { SignInScreen } from "@/components/auth/AuthScreens";
import { SIGN_IN_TITLE } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: SIGN_IN_TITLE };

export default function Page() {
  return <SignInScreen />;
}
