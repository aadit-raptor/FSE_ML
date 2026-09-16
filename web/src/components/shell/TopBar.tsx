"use client";

import { MagnifyingGlassIcon, UserIcon } from "@phosphor-icons/react";
import { motion } from "motion/react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { useSession } from "@/components/auth/AuthProvider";
import { useMonteCarlo } from "@/components/montecarlo/MonteCarloProvider";
import { MODES, parsePath, stepHref } from "@/lib/nav";

import { useWorkspace } from "./workspace";

export function TopBar() {
  const pathname = usePathname();
  const { mode: current } = parsePath(pathname);
  const { setSearchOpen } = useWorkspace();
  const { label } = useSession();
  // Modes whose results no longer match their inputs
  const staleModes = new Set(useMonteCarlo().stale ? ["monte-carlo"] : []);

  return (
    <header className="flex min-h-[42px] flex-none items-stretch border-b border-line bg-panel">
      <Link href="/" className="flex items-center border-r border-line px-4">
        <span className="type-brand">FSE/ML</span>
      </Link>

      <nav aria-label="Modes" className="flex">
        {MODES.map((mode, i) => {
          const active = mode.slug === current?.slug;
          return (
            <Link
              key={mode.slug}
              href={stepHref(mode.slug, mode.steps[0].slug)}
              aria-current={active ? "page" : undefined}
              title={`Alt ${i + 1}`}
              className={`type-tab relative flex items-center gap-2 border-r border-line px-[15px] whitespace-nowrap hover:text-ink ${active ? "bg-raised" : ""}`}
            >
              {mode.label}
              {staleModes.has(mode.slug) && <span className="chip text-attention">stale</span>}
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
        className="my-[7px] mx-2.5 flex min-w-[300px] items-center gap-2 border border-[#2a343a] bg-field px-2.5 text-left hover:border-line-strong"
      >
        <MagnifyingGlassIcon size={13} weight="bold" className="text-dim" aria-hidden />
        <span className="type-step flex-1">Search screens and actions</span>
        <kbd>Ctrl K</kbd>
      </button>

      <Link
        href="/account"
        aria-current={pathname === "/account" ? "page" : undefined}
        title="Your account: country, currency, format and time zone"
        className={`type-tab flex items-center gap-2 border-l border-line px-3.5 whitespace-nowrap hover:text-ink ${pathname === "/account" ? "bg-raised" : ""}`}
      >
        <UserIcon size={13} weight="bold" aria-hidden />
        <span className="max-w-[180px] truncate normal-case">{label ?? "Account"}</span>
      </Link>
    </header>
  );
}
