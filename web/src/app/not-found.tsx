"use client";

import { useTranslations } from "next-intl";
import Link from "next/link";

import { DEFAULT_HREF } from "@/lib/nav";

export default function NotFound() {
  const t = useTranslations("app");
  return (
    <div className="grid content-start gap-3 p-6">
      <h1 className="type-result-title text-[18px]">{t("notFoundTitle")}</h1>
      <p className="type-body">{t("notFoundBody")}</p>
      <Link href={DEFAULT_HREF} className="type-action-secondary justify-self-start px-2.5 py-1.5 text-accent shadow-[inset_0_0_0_1px_var(--color-accent)]">
        {t("notFoundAction")}
      </Link>
    </div>
  );
}
