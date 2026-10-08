"use client";

import { useTranslations } from "next-intl";
import { usePathname } from "next/navigation";
import { useEffect } from "react";

import { LAUNCHER_HREF, parsePath } from "@/lib/nav";

/**
 * The browser tab's title in the account's language (PLAN.md 2.3b).
 *
 * Each page's `metadata` is English, because it is produced on the server
 * before anyone is known to be signed in (lib/i18n/titles.ts). This runs after
 * the route settles and rewrites the title from the same keys the step row
 * uses, so the tab matches the screen.
 *
 * i18n-keys: nav.*
 */
export function DocumentTitle() {
  const pathname = usePathname();
  const nav = useTranslations("nav");
  const app = useTranslations("app");
  const account = useTranslations("account");

  useEffect(() => {
    const { mode, step } = parsePath(pathname);
    const brand = app("brand");
    const parts =
      mode && step
        ? [nav(mode.labelKey), nav(step.labelKey)]
        : pathname === "/account"
          ? [account("title")]
          : pathname === LAUNCHER_HREF
            ? [nav("launcher")]
            : [];
    document.title = [...parts, brand].join(" · ");
  }, [pathname, nav, app, account]);

  return null;
}
