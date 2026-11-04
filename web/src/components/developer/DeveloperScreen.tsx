"use client";

import Link from "next/link";
import { useTranslations } from "next-intl";
import { useCallback, useEffect, useState } from "react";

import { Notice, PrimaryButton, SecondaryButton } from "@/components/ui/Screen";
import { TipLabel, TipText, useTip } from "@/components/ui/Tip";
import {
  addWebhook, type ApiKeyList, createKey, DOCS_URL, listKeys, listWebhooks, pingWebhook, problemKey, removeWebhook,
  revokeKey, type WebhookList,
} from "@/lib/developer";
import { formatDateTime } from "@/lib/locale";

const FIELD = "border border-line bg-field px-2 py-1.5 font-mono text-[11px] text-ink outline-none focus:border-accent";
const API_ORIGIN = process.env.NEXT_PUBLIC_FSE_API_ORIGIN ?? "";

/**
 * The public API (PLAN.md 7.7): the caller's API keys (made, listed by name and last four characters, revoked)
 * and webhooks (added, sent a test message, removed). A new key and a new webhook's secret are shown once.
 */
export function DeveloperScreen() {
  const t = useTranslations("developer");
  const nameTip = useTip("apiKeyName");
  const urlTip = useTip("webhookUrl");
  const [keys, setKeys] = useState<ApiKeyList | null>(null);
  const [hooks, setHooks] = useState<WebhookList | null>(null);
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");
  const [shown, setShown] = useState<{ kind: "key" | "secret"; value: string } | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  const show = useCallback(([k, h]: [Awaited<ReturnType<typeof listKeys>>, Awaited<ReturnType<typeof listWebhooks>>]) => {
    if (!k.ok) return setProblem(k.code);
    if (!h.ok) return setProblem(h.code);
    setKeys(k.data);
    setHooks(h.data);
  }, []);
  const fetchBoth = useCallback(() => Promise.all([listKeys(), listWebhooks()]), []);
  const reload = useCallback(async () => show(await fetchBoth()), [fetchBoth, show]);
  useEffect(() => {
    let cancelled = false;
    void fetchBoth().then((found) => {
      if (!cancelled) show(found);
    });
    return () => {
      cancelled = true;
    };
  }, [fetchBoth, show]);

  const act = async <T,>(call: Promise<{ ok: true; data: T } | { ok: false; code: string }>, after?: (data: T) => void) => {
    const done = await call;
    if (!done.ok) return setProblem(done.code);
    setProblem(null);
    after?.(done.data);
    await reload();
  };

  return (
    <div className="grid max-w-[880px] content-start gap-4 p-6">
      <div className="grid gap-1.5">
        <h1 className="type-result-title text-[14px]">{t("title")}</h1>
        <p className="type-body">{t("intro")}</p>
        <p className="type-body text-[10.5px]">
          {t("baseLabel")} <code className="font-mono text-ink" data-developer="base">{(API_ORIGIN || t("baseLocal")) + "/api/v1"}</code>
          {" · "}
          <a href={DOCS_URL} target="_blank" rel="noreferrer" className="text-accent hover:text-ink">{t("docsLink")}</a>
          {" · "}
          <a href={`${API_ORIGIN}/api/docs`} target="_blank" rel="noreferrer" className="text-accent hover:text-ink">{t("schemaLink")}</a>
        </p>
      </div>
      {problem && <Notice tone="loss" title={t("title")} role="alert">{t(problemKey(problem))}</Notice>}
      {shown && (
        <Notice tone="attention" title={shown.kind === "key" ? t("keyMadeTitle") : t("secretMadeTitle")} role="status"
          actions={<SecondaryButton onClick={() => setShown(null)}>{t("hideOnce")}</SecondaryButton>}>
          {shown.kind === "key" ? t("keyMade") : t("secretMade")}{" "}
          <code className="font-mono break-all text-bright select-all" data-developer={shown.kind}>{shown.value}</code>
        </Notice>
      )}

      <section className="grid gap-2" aria-label={t("keys")}>
        <h2 className="type-input-group">{t("keys")}</h2>
        {keys === null ? (
          <p className="type-step">{t("loading")}</p>
        ) : keys.keys.length === 0 ? (
          <p className="type-body text-[10.5px]" data-guide="api-keys">{t("noKeys")}</p>
        ) : (
          <ul className="grid gap-1">
            {keys.keys.map((key) => (
              <li key={key.id} className="flex flex-wrap items-center gap-3 border-b border-line py-1.5" data-key={key.name}>
                <span className="type-body text-ink">{key.name}</span>
                <span className="font-mono text-[10.5px] text-muted">{t("hint", { hint: key.hint })}</span>
                <span className="type-body text-[10.5px]">
                  {key.last_used_at ? t("lastUsed", { when: formatDateTime(new Date(key.last_used_at)) }) : t("neverUsed")}
                </span>
                <span className="ms-auto">
                  <SecondaryButton onClick={() => void act(revokeKey(key.id))}>{t("revoke")}</SecondaryButton>
                </span>
              </li>
            ))}
          </ul>
        )}
        <div className="flex flex-wrap items-end gap-2">
          <label className="grid gap-1">
            <span className="type-input-label"><TipLabel tip={nameTip}>{t("newKeyLabel")}</TipLabel></span>
            <input aria-describedby={nameTip.id} className={FIELD} value={name} maxLength={60}
              onChange={(e) => setName(e.target.value)} />
          </label>
          <TipText tip={nameTip} />
          <PrimaryButton disabled={!name.trim() || (keys !== null && keys.keys.length >= keys.max)}
            onClick={() => void act(createKey(name.trim()), (made) => {
              setName("");
              setShown({ kind: "key", value: made.token });
            })}>
            {t("createKey")}
          </PrimaryButton>
        </div>
        <p className="type-body text-[10.5px]">{t("keysNote", { max: String(keys?.max ?? "") })}</p>
      </section>

      <section className="grid gap-2" aria-label={t("webhooks")}>
        <h2 className="type-input-group">{t("webhooks")}</h2>
        <p className="type-body text-[10.5px]">{t("webhooksIntro")}</p>
        {hooks === null ? (
          <p className="type-step">{t("loading")}</p>
        ) : hooks.webhooks.length === 0 ? (
          <p className="type-body text-[10.5px]" data-guide="webhooks">{t("noWebhooks")}</p>
        ) : (
          <ul className="grid gap-1">
            {hooks.webhooks.map((hook) => (
              <li key={hook.id} className="flex flex-wrap items-center gap-3 border-b border-line py-1.5" data-webhook={hook.url}>
                <span className="min-w-0 truncate font-mono text-[10.5px] text-ink">{hook.url}</span>
                <span className="type-body text-[10.5px]" data-webhook-state={hook.last_state ?? "none"}>
                  {hook.last_state
                    ? t(`state_${hook.last_state}`, { status: String(hook.last_status ?? 0) })
                    : t("state_none")}
                </span>
                <span className="ms-auto flex gap-2">
                  <SecondaryButton onClick={() => void act(pingWebhook(hook.id))}>{t("ping")}</SecondaryButton>
                  <SecondaryButton onClick={() => void act(removeWebhook(hook.id))}>{t("remove")}</SecondaryButton>
                </span>
              </li>
            ))}
          </ul>
        )}
        <div className="flex flex-wrap items-end gap-2">
          <label className="grid min-w-0 flex-1 gap-1">
            <span className="type-input-label"><TipLabel tip={urlTip}>{t("newWebhookLabel")}</TipLabel></span>
            <input aria-describedby={urlTip.id} className={FIELD} value={url} maxLength={500} inputMode="url"
              onChange={(e) => setUrl(e.target.value)} />
          </label>
          <TipText tip={urlTip} />
          <PrimaryButton disabled={!url.trim() || (hooks !== null && hooks.webhooks.length >= hooks.max)}
            onClick={() => void act(addWebhook(url.trim()), (made) => {
              setUrl("");
              setShown({ kind: "secret", value: made.secret });
            })}>
            {t("addWebhook")}
          </PrimaryButton>
        </div>
      </section>
      <p className="type-body text-[10.5px]"><Link href="/account" className="text-accent hover:text-ink">{t("backToAccount")}</Link></p>
    </div>
  );
}
