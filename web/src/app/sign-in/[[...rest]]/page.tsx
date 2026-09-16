import type { Metadata } from "next";

import { SignInScreen } from "@/components/auth/AuthScreens";

export const metadata: Metadata = { title: "Sign in" };

export default function Page() {
  return <SignInScreen />;
}
