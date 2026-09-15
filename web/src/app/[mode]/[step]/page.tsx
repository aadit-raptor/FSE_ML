import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { StepPending } from "@/components/StepPending";
import { findStep, MODES } from "@/lib/nav";

export const dynamicParams = false;

export function generateStaticParams() {
  return MODES.flatMap((m) => m.steps.map((s) => ({ mode: m.slug, step: s.slug })));
}

export async function generateMetadata(props: PageProps<"/[mode]/[step]">): Promise<Metadata> {
  const { mode, step } = await props.params;
  const found = findStep(mode, step);
  return { title: found ? `${found.mode.label} · ${found.step.label}` : "Not found" };
}

export default async function StepPage(props: PageProps<"/[mode]/[step]">) {
  const { mode, step } = await props.params;
  const found = findStep(mode, step);
  if (!found) notFound();
  return <StepPending mode={found.mode} step={found.step} />;
}
