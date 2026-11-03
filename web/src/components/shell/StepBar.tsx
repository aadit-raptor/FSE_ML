"use client";

import { useTranslations } from "next-intl";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef } from "react";

import { parsePath, stepHref } from "@/lib/nav";

export function StepBar() {
  const { mode, step: current } = parsePath(usePathname());
  const t = useTranslations("shell");
  const nav = useTranslations("nav");
  const bar = useRef<HTMLElement>(null);
  // On a phone the row scrolls sideways (PLAN.md 7.6): keep the current step in sight
  // (scrolled by hand, not with scrollIntoView, which would also move where the Tab key starts from)
  useEffect(() => {
    const row = bar.current;
    const step = row?.querySelector<HTMLElement>('[aria-current="page"]');
    if (!row || !step || row.scrollWidth <= row.clientWidth) return;
    const start = step.getBoundingClientRect().left - row.getBoundingClientRect().left + row.scrollLeft;
    row.scrollLeft = Math.max(0, start - (row.clientWidth - step.offsetWidth) / 2);
  }, [current?.slug]);
  if (!mode) return <div className="min-h-[34px] flex-none border-b border-line bg-[#0f1417]" />;

  return (
    <nav
      ref={bar}
      aria-label={t("stepsOf", { mode: nav(mode.labelKey) })}
      className="flex min-h-[34px] flex-none items-stretch overflow-x-auto border-b border-line bg-[#0f1417]"
    >
      {mode.steps.map((step) => (
        <Link
          key={step.slug}
          href={stepHref(mode.slug, step.slug)}
          aria-current={step.slug === current?.slug ? "page" : undefined}
          className="type-step flex items-center px-3 whitespace-nowrap hover:text-ink"
        >
          {nav(step.labelKey)}
        </Link>
      ))}
      <div className="flex-1" />
      <span className="type-control flex items-center gap-2 pe-3 max-md:hidden" aria-hidden>
        <kbd>[</kbd>
        <kbd>]</kbd>
        {t("steps")}
      </span>
    </nav>
  );
}
