"use client";

import { MagnifyingGlassIcon, UserIcon } from "@phosphor-icons/react";
import { motion } from "motion/react";
import { useTranslations } from "next-intl";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { useSession } from "@/components/auth/AuthProvider";
import { BrandLockup } from "@/components/brand/BrandMark";
import { useMonteCarlo } from "@/components/montecarlo/MonteCarloProvider";
import { LAUNCHER_HREF, parsePath, stepHref, WORKSPACES, workspaceModes } from "@/lib/nav";

import { useCurrentWorkspace, useModes, useWorkspaceModes } from "./useModes";
import { useWorkspace } from "./workspace";

export function TopBar() {
  const pathname = usePathname();
  const { mode: current } = parsePath(pathname);
  const { setSearchOpen } = useWorkspace();
  const { label } = useSession();
  const modes = useWorkspaceModes();
  const shown = useModes();
  const workspace = useCurrentWorkspace();
  const t = useTranslations("shell");
  const nav = useTranslations("nav");
  const app = useTranslations("app");
  const mcText = useTranslations("montecarlo");
  // Modes whose results no longer match their inputs, and runs in progress
  // (a Monte Carlo run carries on while you work elsewhere)
  const mc = useMonteCarlo();
  const staleModes = new Set(mc.stale && mc.run.status !== "running" ? ["monte-carlo"] : []);
  const runningModes = new Set(mc.run.status === "running" ? ["monte-carlo"] : []);

  return (
    <header className="flex min-h-[42px] flex-none items-stretch border-b border-line bg-panel">
      <Link href={LAUNCHER_HREF} title={t("launcherTitle")} className="flex items-center border-e border-line px-4">
        <BrandLockup name={app("brand")} />
      </Link>

      {/* The workspace switcher: each workspace opens on its first mode's first step */}
      <nav aria-label={t("workspaces")} className="flex flex-col justify-center border-e border-line px-3 py-1">
        {WORKSPACES.map((w) => {
          const first = workspaceModes(w, shown)[0];
          if (!first) return null;
          const active = w.slug === workspace?.slug;
          return (
            <Link
              key={w.slug}
              href={stepHref(first.slug, first.steps[0].slug)}
              aria-current={active ? "true" : undefined}
              className={`type-control flex items-center gap-1.5 leading-[1.6] whitespace-nowrap hover:text-ink ${active ? "text-accent" : ""}`}
            >
              {nav(w.labelKey)}
              {/* A run carries on in another workspace: say so here, where its tab isn't */}
              {!active && w.modes.some((m) => staleModes.has(m)) && <span className="chip text-attention">{mcText("chipStale")}</span>}
              {!active && w.modes.some((m) => runningModes.has(m)) && <span className="chip text-accent">{mcText("chipRunning")}</span>}
            </Link>
          );
        })}
      </nav>

      <nav aria-label={t("modes")} className="flex">
        {modes.map((mode, i) => {
          const active = mode.slug === current?.slug;
          return (
            <Link
              key={mode.slug}
              href={stepHref(mode.slug, mode.steps[0].slug)}
              aria-current={active ? "page" : undefined}
              title={t("modeShortcut", { number: i + 1 })}
              className={`type-tab relative flex items-center gap-2 border-e border-line px-[15px] whitespace-nowrap hover:text-ink ${active ? "bg-raised" : ""}`}
            >
              {nav(mode.labelKey)}
              {staleModes.has(mode.slug) && <span className="chip text-attention">{mcText("chipStale")}</span>}
              {runningModes.has(mode.slug) && <span className="chip text-accent">{mcText("chipRunning")}</span>}
              {active && (
                <motion.span
                  layoutId="mode-underline"
                  className="absolute inset-x-0 bottom-0 h-0.5 bg-accent"
                  transition={{ type: "spring", stiffness: 500, damping: 40 }}
                />
              )}
            </Link>
          );
        })}
      </nav>

      <div className="flex-1" />

      <button
        type="button"
        onClick={() => setSearchOpen(true)}
        className="my-[7px] mx-2.5 flex min-w-[300px] items-center gap-2 border border-[#2a343a] bg-field px-2.5 text-start hover:border-line-strong"
      >
        <MagnifyingGlassIcon size={13} weight="bold" className="text-dim" aria-hidden />
        <span className="type-step flex-1">{t("searchPlaceholderShort")}</span>
        <kbd>Ctrl K</kbd>
      </button>

      <Link
        href="/account"
        aria-current={pathname === "/account" ? "page" : undefined}
        title={t("accountTitle")}
        className={`type-tab flex items-center gap-2 border-s border-line px-3.5 whitespace-nowrap hover:text-ink ${pathname === "/account" ? "bg-raised" : ""}`}
      >
        <UserIcon size={13} weight="bold" aria-hidden />
        <span className="max-w-[180px] truncate normal-case">{label ?? t("account")}</span>
      </Link>
    </header>
  );
}
