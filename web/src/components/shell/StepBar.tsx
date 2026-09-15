"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { parsePath, stepHref } from "@/lib/nav";

export function StepBar() {
  const { mode, step: current } = parsePath(usePathname());
  if (!mode) return <div className="min-h-[34px] flex-none border-b border-line bg-[#0f1417]" />;

  return (
    <nav
      aria-label={`${mode.label} steps`}
      className="flex min-h-[34px] flex-none items-stretch border-b border-line bg-[#0f1417]"
    >
      {mode.steps.map((step) => (
        <Link
          key={step.slug}
          href={stepHref(mode.slug, step.slug)}
          aria-current={step.slug === current?.slug ? "page" : undefined}
          className="type-step flex items-center px-3 whitespace-nowrap hover:text-ink"
        >
          {step.label}
        </Link>
      ))}
      <div className="flex-1" />
      <span className="type-control flex items-center gap-2 pr-3" aria-hidden>
        <kbd>[</kbd>
        <kbd>]</kbd>
        steps
      </span>
    </nav>
  );
}
