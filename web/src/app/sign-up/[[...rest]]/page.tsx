import type { Metadata } from "next";

import { SignUpScreen } from "@/components/auth/AuthScreens";
import { SIGN_UP_TITLE } from "@/lib/i18n/titles";

export const metadata: Metadata = { title: SIGN_UP_TITLE };

export default function Page() {
  return <SignUpScreen />;
}
