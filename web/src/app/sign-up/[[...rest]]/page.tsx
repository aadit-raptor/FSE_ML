import type { Metadata } from "next";

import { SignUpScreen } from "@/components/auth/AuthScreens";

export const metadata: Metadata = { title: "Sign up" };

export default function Page() {
  return <SignUpScreen />;
}
