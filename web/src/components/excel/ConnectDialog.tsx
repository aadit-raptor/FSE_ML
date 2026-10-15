"use client";

import { useTranslations } from "next-intl";
import { useEffect, useRef, useState } from "react";

import { useSession } from "@/components/auth/AuthProvider";
import { DialogSignIn } from "@/components/auth/AuthScreens";
import { Notice } from "@/components/ui/Screen";
import { loadOffice, messageParent } from "@/lib/excel/office";
import { newPass, passMessage } from "@/lib/excel/session";

/**
 * Office's sign-in dialog for the Excel add-in (PLAN.md R3, /excel/connect).
 *
 * A window of its own on variater.com, where the site's sign-in works as in any tab. Signed in, it asks the
 * API for an add-in pass (`POST /api/addin/sessions`) and hands it to the pane with Office's `messageParent`,
 * which only this site's own pages receive. Email sign-in never leaves this page; Google's leaves and comes
 * back here, which Office allows, but Google may refuse its sign-in inside Excel's own window on Windows and
 * Mac, so the page says to use email then.
 *
 * i18n-keys: excel.connectError_*
 */
const KNOWN = ["no_account", "no_database"];

export function ConnectDialog() {
  const t = useTranslations("excel");
  const { ready, signedIn, label } = useSession();
  const [office, setOffice] = useState<Awaited<ReturnType<typeof loadOffice>> | undefined>(undefined);
  const [state, setState] = useState<"idle" | "connecting" | "connected">("idle");
  const [error, setError] = useState<string | null>(null);
  const sent = useRef(false);

  useEffect(() => {
    let cancelled = false;
    void loadOffice({ insideOffice: false }).then((found) => {
      if (!cancelled) setOffice(found);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!office || !ready || !signedIn || sent.current) return;
    sent.current = true;
    setState("connecting");
    void newPass().then((made) => {
      if (!made.pass) {
        setError(made.error ?? "failed");
        setState("idle");
        sent.current = false;
        return;
      }
      messageParent(office, passMessage({ ...made.pass, label }));
      setState("connected");
    });
  }, [office, ready, signedIn, label]);

  return (
    <div className="grid h-full place-items-start justify-center p-5" data-connect-state={state}>
      <div className="grid w-full max-w-[420px] gap-4">
        <div className="grid gap-1.5">
          <h1 className="type-result-title text-[14px]">{t("connectTitle")}</h1>
          <p className="type-body">{t("connectHint")}</p>
        </div>
        {office === null && <Notice tone="attention" title={t("connectOutsideOffice")} />}
        {error && <Notice tone="loss" role="alert" title={t(`connectError_${KNOWN.includes(error) ? error : "failed"}`)} />}
        {state === "connecting" && <p className="type-step" role="status">{t("connecting")}</p>}
        {state === "connected" && <p className="type-step" role="status">{t("connected")}</p>}
        {ready && !signedIn && (
          <>
            <p className="type-body">{t("googleNote")}</p>
            <p className="type-body">{t("noAccountHint")}</p>
            <DialogSignIn redirectTo="/excel/connect" />
          </>
        )}
      </div>
    </div>
  );
}
