"use client";

import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { api } from "@/lib/api/client";
import { parsePath } from "@/lib/nav";

type Health = { state: "checking" } | { state: "ok"; version: string; environment?: string } | { state: "down" };

const POLL_MS = 30_000;
// Retry sooner while the API is down: a sleeping host can take ~a minute to start
const RETRY_MS = 5_000;
const DOWN_HINT = process.env.NODE_ENV === "development" ? "start it with uvicorn api.main:app --port 8000" : "it may be starting up, retrying";

export function StatusBar() {
  const { mode, step } = parsePath(usePathname());
  const [health, setHealth] = useState<Health>({ state: "checking" });

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const { data, response } = await api.GET("/api/health");
        if (cancelled) return false;
        const { version, environment } = (data as { version?: string; environment?: string } | undefined) ?? {};
        const ok = !!(response.ok && version);
        setHealth(ok ? { state: "ok", version: version!, environment } : { state: "down" });
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
      <span className="flex items-center gap-1.5 border-r border-line px-3" role="status">
        API{" "}
        {health.state === "ok" && <b className="font-medium text-ink">ok · v{health.version}</b>}
        {/* Name any copy that isn't production, so staging is never mistaken for it */}
        {health.state === "ok" && health.environment && health.environment !== "production" && (
          <b className="font-medium text-attention">· {health.environment}</b>
        )}
        {health.state === "checking" && <b className="font-medium text-dim">checking</b>}
        {health.state === "down" && (
          <b className="font-medium text-loss">
            unreachable · {DOWN_HINT}
          </b>
        )}
      </span>
      <span className="ml-auto flex items-center border-l border-line px-3">
        {mode && step ? `${mode.label} · ${step.label}` : "FSE/ML"}
      </span>
    </footer>
  );
}
