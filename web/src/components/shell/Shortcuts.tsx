"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";

import { MODES, parsePath, stepHref } from "@/lib/nav";

import { useWorkspace } from "./workspace";

function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  return !!el && (el.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName));
}

/** Ctrl/Cmd K search, Alt 1-5 modes, [ and ] previous / next step. */
export function Shortcuts() {
  const router = useRouter();
  const pathname = usePathname();
  const { setSearchOpen } = useWorkspace();

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setSearchOpen(true);
        return;
      }
      if (isTyping(e.target)) return;

      if (e.altKey && /^[1-9]$/.test(e.key)) {
        const mode = MODES[Number(e.key) - 1];
        if (mode) {
          e.preventDefault();
          router.push(stepHref(mode.slug, mode.steps[0].slug));
        }
        return;
      }
      if ((e.key === "[" || e.key === "]") && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const { mode, step } = parsePath(pathname);
        if (!mode || !step) return;
        const i = mode.steps.indexOf(step) + (e.key === "]" ? 1 : -1);
        const next = mode.steps[i];
        if (next) router.push(stepHref(mode.slug, next.slug));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [router, pathname, setSearchOpen]);

  return null;
}
