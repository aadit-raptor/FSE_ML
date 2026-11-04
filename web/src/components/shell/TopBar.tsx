"use client";

import { MagnifyingGlassIcon, QuestionIcon, UserIcon, UsersThreeIcon } from "@phosphor-icons/react";
import { motion } from "motion/react";
import { useTranslations } from "next-intl";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { useSession } from "@/components/auth/AuthProvider";
import { BrandLockup } from "@/components/brand/BrandMark";
import { useMonteCarlo } from "@/components/montecarlo/MonteCarloProvider";
import { useUnseenNews } from "@/lib/help/whatsNew";
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
  const help = useTranslations("help");
  const unseen = useUnseenNews();
  const inHelp = pathname === "/help" || pathname.startsWith("/help/");
  // Modes whose results no longer match their inputs, and runs in progress
  // (a Monte Carlo run carries on while you work elsewhere)
  const mc = useMonteCarlo();
  const staleModes = new Set(mc.stale && mc.run.status !== "running" ? ["monte-carlo"] : []);
  const runningModes = new Set(mc.run.status === "running" ? ["monte-carlo"] : []);

  return (
    // Phones and narrow tablets (PLAN.md 7.6): two rows. The brand and the icon links stay on the first; the
    // workspaces and modes move to a second row of their own that scrolls sideways
    <header className="flex min-h-[42px] flex-none flex-wrap items-stretch border-b border-line bg-panel lg:flex-nowrap">
      <Link href={LAUNCHER_HREF} title={t("launcherTitle")} className="flex items-center border-e border-line px-4 max-md:px-3">
        <BrandLockup name={app("brand")} />
      </Link>

      <div className="order-last flex w-full min-w-0 overflow-x-auto border-t border-line lg:order-none lg:w-auto lg:overflow-visible lg:border-t-0">
      {/* The workspace switcher: each workspace opens on its first mode's first step */}
      <nav aria-label={t("workspaces")} className="flex flex-none flex-col justify-center border-e border-line px-3 py-1">
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

      <nav aria-label={t("modes")} className="flex flex-none">
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
      </div>

      <div className="flex-1" />

      {/* On a phone the search is its icon; its words stay its name */}
      <button
        type="button"
        onClick={() => setSearchOpen(true)}
        className="my-[7px] mx-2.5 flex min-w-[300px] items-center gap-2 border border-[#2a343a] bg-field px-2.5 text-start hover:border-line-strong max-xl:min-w-[200px] max-md:mx-1.5 max-md:min-w-0"
      >
        <MagnifyingGlassIcon size={13} weight="bold" className="text-dim" aria-hidden />
        <span className="type-step flex-1 max-md:sr-only">{t("searchPlaceholderShort")}</span>
        <kbd className="max-md:hidden">Ctrl K</kbd>
      </button>

      {/* Help (PLAN.md 7.1): the methodology as articles, and what's new, marked while something is unseen */}
      <Link
        href="/help"
        aria-current={inHelp ? "page" : undefined}
        title={unseen ? help("unseen") : undefined}
        className={`type-tab relative flex items-center gap-2 border-s border-line px-3.5 whitespace-nowrap max-md:px-2.5 hover:text-ink ${inHelp ? "bg-raised" : ""}`}
      >
        <QuestionIcon size={13} weight="bold" aria-hidden />
        <span className="max-md:sr-only">{help("topBarLink")}</span>
        {unseen && <span data-unseen aria-hidden className="absolute top-2 end-2 size-1.5 bg-accent" />}
      </Link>

      {/* Teams (PLAN.md 7.2): the caller's teams, their people and invitation links */}
      <Link
        href="/teams"
        aria-current={pathname.startsWith("/teams") ? "page" : undefined}
        className={`type-tab flex items-center gap-2 border-s border-line px-3.5 whitespace-nowrap max-md:px-2.5 hover:text-ink ${pathname.startsWith("/teams") ? "bg-raised" : ""}`}
      >
        <UsersThreeIcon size={13} weight="bold" aria-hidden />
        <span className="max-md:sr-only">{t("teamsLink")}</span>
      </Link>

      <Link
        href="/account"
        aria-current={pathname === "/account" ? "page" : undefined}
        title={t("accountTitle")}
        className={`type-tab flex items-center gap-2 border-s border-line px-3.5 whitespace-nowrap max-md:px-2.5 hover:text-ink ${pathname === "/account" ? "bg-raised" : ""}`}
      >
        <UserIcon size={13} weight="bold" aria-hidden />
        <span className="max-w-[180px] truncate normal-case max-md:sr-only">{label ?? t("account")}</span>
      </Link>
    </header>
  );
}
