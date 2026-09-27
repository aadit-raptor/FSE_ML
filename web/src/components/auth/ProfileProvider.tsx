"use client";

import { useTranslations } from "next-intl";
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api/client";
import type { DigitGrouping } from "@/lib/locale";

/** How this account wants money, dates and numbers shown (PLAN.md 1.4). */
export type Profile = {
  country: string;
  preferred_currency: string;
  locale: string;
  time_zone: string;
  /** How long numbers are grouped (PLAN.md 2.3a) */
  digit_grouping: DigitGrouping;
};

type ProfileContext = {
  /** The account id from the identity provider. */
  subject: string | null;
  /** Null while loading, and null again when the account hasn't answered yet. */
  profile: Profile | null;
  /** True once the API has answered, whatever the answer was. */
  loaded: boolean;
  /** The account exists but has no profile: sign-up isn't finished. */
  needsProfile: boolean;
  save: (profile: Profile) => Promise<{ ok: boolean; error?: string }>;
  error?: string;
};

const Ctx = createContext<ProfileContext | null>(null);

export function useProfile(): ProfileContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useProfile must be used inside ProfileProvider");
  return ctx;
}

/** Wording for the refusals this provider reports, in the account's language (PLAN.md 2.3b). */
type ProfileMessages = { rejected: (field: string, reason: string) => string; profileField: string; rejectedFallback: string; saveFailed: string };

function describe(error: unknown, m: ProfileMessages): string {
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    const first = detail[0] as { loc?: (string | number)[]; msg?: string };
    return m.rejected(String(first.loc?.at(-1) ?? m.profileField), first.msg ?? m.rejectedFallback);
  }
  return m.saveFailed;
}

export function ProfileProvider({ children }: { children: React.ReactNode }) {
  const t = useTranslations("account");
  const messages = useMemo<ProfileMessages>(
    () => ({
      rejected: (field, reason) => t("rejected", { field, reason }),
      profileField: t("profileField"),
      rejectedFallback: t("rejectedFallback"),
      saveFailed: t("saveFailed"),
    }),
    [t],
  );
  const unreachable = t("unreachable");
  const [subject, setSubject] = useState<string | null>(null);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState<string>();

  useEffect(() => {
    let cancelled = false;
    api
      .GET("/api/account")
      .then(({ data, error: err }) => {
        if (cancelled) return;
        if (data) {
          setSubject(data.subject);
          setProfile((data.profile as Profile | null) ?? null);
          setError(undefined);
        } else if (err) {
          setError(describe(err, messages));
        }
      })
      .catch(() => !cancelled && setError(unreachable))
      .finally(() => !cancelled && setLoaded(true));
    return () => {
      cancelled = true;
    };
  }, [messages, unreachable]);

  const save = useCallback(async (next: Profile) => {
    const { data, error: err } = await api.POST("/api/account", { body: next });
    if (data) {
      setSubject(data.subject);
      setProfile((data.profile as Profile | null) ?? null);
      setError(undefined);
      return { ok: true };
    }
    const message = describe(err, messages);
    setError(message);
    return { ok: false, error: message };
  }, [messages]);

  const value = useMemo<ProfileContext>(
    () => ({ subject, profile, loaded, needsProfile: loaded && !profile && !error, save, error }),
    [subject, profile, loaded, error, save],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}
