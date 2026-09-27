"use client";

import { NextIntlClientProvider } from "next-intl";
import { useMemo } from "react";

import { useProfile } from "@/components/auth/ProfileProvider";
import { setDownloadFailedMessage } from "@/lib/export";
import { setMissingText } from "@/lib/format";
import { DEFAULT_LANGUAGE, directionFor, languageFor } from "@/lib/i18n/config";
import { messagesFor } from "@/lib/i18n/messages";

/**
 * Which language the words are in, and which way the layout runs (PLAN.md
 * 2.3b). Both come from the account's locale, the same answer that chooses
 * number and date formats (lib/i18n/config.ts); there is no locale in the
 * URL, so `src/proxy.ts` and its content security policy are untouched.
 *
 * It does not hold the screens back: before the account answers, the shell
 * chrome reads the browser's own language, and `LocaleScope` (AppShell) is
 * still what waits for the profile before any figure is drawn. `lang` and
 * `dir` are set on the document from here rather than in the root layout,
 * which is rendered before anyone is known to be signed in.
 */
export function I18nScope({
  locale,
  setsDocument = true,
  children,
}: {
  locale?: string;
  /** False for an outer scope that a nested one will override (see `Shell`). */
  setsDocument?: boolean;
  children: React.ReactNode;
}) {
  const language = languageFor(locale);
  const direction = directionFor(locale);
  const messages = useMemo(() => messagesFor(language), [language]);
  // The stand-in for a figure the model didn't produce, for the plain
  // formatters in lib/format.ts (same pattern as the number style)
  setMissingText(messages.format.missing);
  setDownloadFailedMessage((status) => messages.errors.downloadFailed.replace("{status}", String(status)));

  return (
    // next-intl formats no dates here -- they go through lib/locale.ts with the
    // account's own time zone -- so a fixed zone only silences its warning
    // about server and browser disagreeing.
    <NextIntlClientProvider locale={locale || language} messages={messages} timeZone="UTC">
      {setsDocument && <DocumentLanguage locale={locale || language} direction={direction} />}
      {children}
    </NextIntlClientProvider>
  );
}

/** The account's locale for the shell: its own locale, or the browser's until it answers. */
export function ProfileI18nScope({ children }: { children: React.ReactNode }) {
  const { profile } = useProfile();
  return <I18nScope locale={profile?.locale ?? browserLocale()}>{children}</I18nScope>;
}

/** What the browser says, used only until the account's own answer arrives. */
export function browserLocale(): string {
  return typeof navigator === "undefined" ? DEFAULT_LANGUAGE : navigator.language || DEFAULT_LANGUAGE;
}

/**
 * `<html lang>` and `<html dir>`, set during render so the very first paint is
 * already the right way round. Writing to `documentElement` here rather than
 * in an effect keeps a right-to-left layout from flashing left-to-right.
 */
function DocumentLanguage({ locale, direction }: { locale: string; direction: "ltr" | "rtl" }) {
  if (typeof document !== "undefined") {
    const html = document.documentElement;
    if (html.lang !== locale) html.lang = locale;
    if (html.dir !== direction) html.dir = direction;
  }
  return null;
}
