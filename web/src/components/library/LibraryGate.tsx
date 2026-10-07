"use client";

import { useTranslations } from "next-intl";
import { useState, type ReactNode } from "react";

import { EmptyState, Notice, Switch } from "@/components/ui/Screen";
import { formatDateTime } from "@/lib/locale";

import { useLibrary } from "./LibraryProvider";

/**
 * The library's screens render only while it is on. Off, a visitor who follows a link here is told
 * so (every other screen works as usual); an administrator also gets the switch back on.
 */
export function LibraryGate({ children, switchElsewhere = false }: { children: ReactNode; switchElsewhere?: boolean }) {
  const { state, loaded } = useLibrary();
  const t = useTranslations("library");
  if (!loaded) {
    return (
      <p className="type-step p-6" role="status">
        {t("loading")}
      </p>
    );
  }
  if (!state) return <EmptyState title={t("unreachableTitle")}>{t("unreachable")}</EmptyState>;
  if (state.enabled) return <>{children}</>;
  return (
    <EmptyState title={t("offTitle")} action={state.can_switch && !switchElsewhere ? <LibrarySwitch /> : undefined}>
      {state.locked_off ? t("offLocked") : state.can_switch ? t("offAdmin") : t("offEveryone")}
    </EmptyState>
  );
}

/** The administrators' switch: shows or hides the library for every account in this environment. */
export function LibrarySwitch() {
  const { state, setEnabled } = useLibrary();
  const t = useTranslations("library");
  const [busy, setBusy] = useState(false);
  const [failed, setFailed] = useState(false);
  if (!state?.can_switch) return null;
  const change = async (on: boolean) => {
    setBusy(true);
    setFailed(false);
    const ok = await setEnabled(on);
    setBusy(false);
    setFailed(!ok);
  };
  return (
    <div className="grid gap-1.5" aria-busy={busy}>
      <Switch checked={state.enabled} onChange={(on) => void change(on)} label={t("switchLabel")} />
      <p className="type-body text-[9px]" data-testid="library-switch-state">
        {state.enabled ? t("shownToEveryone") : t("hiddenFromEveryone")}
        {state.updated_at ? ` ${t("switchedAt", { at: formatDateTime(new Date(state.updated_at)) })}` : ""}
      </p>
      {failed && (
        <Notice tone="loss" title={t("switchFailedTitle")} role="alert">
          {t("switchFailed")}
        </Notice>
      )}
    </div>
  );
}
