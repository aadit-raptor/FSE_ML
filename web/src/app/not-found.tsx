import Link from "next/link";

import { DEFAULT_HREF } from "@/lib/nav";

export default function NotFound() {
  return (
    <div className="grid content-start gap-3 p-6">
      <h1 className="type-result-title text-[18px]">Screen not found</h1>
      <p className="type-body">That address isn&apos;t a screen in FSE/ML. Use the menu above or search with Ctrl K.</p>
      <Link href={DEFAULT_HREF} className="type-action-secondary justify-self-start px-2.5 py-1.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)]">
        Go to deal inputs
      </Link>
    </div>
  );
}
