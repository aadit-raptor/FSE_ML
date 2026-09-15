import { notFound, redirect } from "next/navigation";

import { findMode, PENDING_MODES, stepHref } from "@/lib/nav";

export const dynamicParams = false;

export function generateStaticParams() {
  return PENDING_MODES.map((m) => ({ mode: m.slug }));
}

/** /deal goes to the mode's first step. */
export default async function ModePage(props: PageProps<"/[mode]">) {
  const { mode: slug } = await props.params;
  const mode = findMode(slug);
  if (!mode) notFound();
  redirect(stepHref(mode.slug, mode.steps[0].slug));
}
