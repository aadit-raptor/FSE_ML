"use client";

import { useTranslations } from "next-intl";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { api } from "@/lib/api/client";
import { formatForViewer } from "@/lib/monitoring";
import { parsePath } from "@/lib/nav";

type Health =
  | { state: "checking" }
  | { state: "ok"; version: string; engine?: string; environment?: string; time?: string }
  | { state: "down" };

const POLL_MS = 30_000;
// Retry sooner while the API is down: a sleeping host can take ~a minute to start
const RETRY_MS = 5_000;
const LOCAL = process.env.NODE_ENV === "development";

export function StatusBar() {
  const { mode, step } = parsePath(usePathname());
  const t = useTranslations("shell");
  const nav = useTranslations("nav");
  const app = useTranslations("app");
  const [health, setHealth] = useState<Health>({ state: "checking" });

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const { data, response } = await api.GET("/api/health");
        if (cancelled) return false;
        const { version, engine_version: engine, environment, time } =
          (data as { version?: string; engine_version?: string; environment?: string; time?: string } | undefined) ?? {};
        const ok = !!(response.ok && version);
        setHealth(ok ? { state: "ok", version: version!, engine, environment, time } : { state: "down" });
        return ok;
      } catch {
        if (!cancelled) setHealth({ state: "down" });
        return false;
      }
    };
    let id: ReturnType<typeof setTimeout>;
    const loop = async () => {
      const ok = await check();
      if (!cancelled) id = setTimeout(loop, ok ? POLL_MS : RETRY_MS);
    };
    void loop();
    return () => {
      cancelled = true;
      clearTimeout(id);
    };
  }, []);

  return (
    <footer className="flex min-h-7 flex-none items-stretch border-t border-line bg-panel font-mono text-[10.5px] text-muted">
      {/* The API reports its clock in UTC; show it in the viewer's time zone */}
      <span
        className="flex items-center gap-1.5 border-e border-line px-3"
        role="status"
        title={health.state === "ok" && health.time ? t("apiCheckedAt", { when: formatForViewer(health.time) }) : undefined}
      >
        {t("apiLabel")}{" "}
        {health.state === "ok" && <b className="font-medium text-ink">{t("apiOk", { version: health.version })}</b>}
        {/* The model's own version (PLAN.md 3.1): what every result on screen was computed with */}
        {health.state === "ok" && health.engine && (
          <span data-engine-version={health.engine}>{t("modelVersion", { version: health.engine })}</span>
        )}
        {/* Name any copy that isn't production, so staging is never mistaken for it */}
        {health.state === "ok" && health.environment && health.environment !== "production" && (
          <b className="font-medium text-attention">· {health.environment}</b>
        )}
        {health.state === "checking" && <b className="font-medium text-dim">{t("apiChecking")}</b>}
        {health.state === "down" && (
          <b className="font-medium text-loss">{t("apiUnreachable", { hint: LOCAL ? t("apiHintLocal") : t("apiHintDeployed") })}</b>
        )}
      </span>
      <span className="ms-auto flex items-center border-s border-line px-3">
        {mode && step ? `${nav(mode.labelKey)} · ${nav(step.labelKey)}` : app("brand")}
      </span>
    </footer>
  );
}
