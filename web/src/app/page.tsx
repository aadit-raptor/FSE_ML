"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { lastScreen } from "@/lib/lastScreen";
import { LAUNCHER_HREF } from "@/lib/nav";

/**
 * The site's root, opened by a visitor who is already signed in (a sign-in
 * lands on the launcher itself): carry on at the screen this browser last
 * showed, or offer the launcher when there is none.
 */
export default function Home() {
  const router = useRouter();
  useEffect(() => {
    router.replace(lastScreen() ?? LAUNCHER_HREF);
  }, [router]);
  return null;
}
