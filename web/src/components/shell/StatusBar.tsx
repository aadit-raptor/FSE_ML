"use client";

import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { api } from "@/lib/api/client";
import { parsePath } from "@/lib/nav";

type Health = { state: "checking" } | { state: "ok"; version: string } | { state: "down" };

const POLL_MS = 30_000;

export function StatusBar() {
  const { mode, step } = parsePath(usePathname());
  const [health, setHealth] = useState<Health>({ state: "checking" });

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const { data, response } = await api.GET("/api/health");
        if (cancelled) return;
        const version = (data as { version?: string } | undefined)?.version;
        setHealth(response.ok && version ? { state: "ok", version } : { state: "down" });
      } catch {
        if (!cancelled) setHealth({ state: "down" });
      }
    };
    check();
    const id = setInterval(check, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  return (
    <footer className="flex min-h-7 flex-none items-stretch border-t border-line bg-panel font-mono text-[10.5px] text-muted">
      <span className="flex items-center gap-1.5 border-r border-line px-3" role="status">
        API{" "}
        {health.state === "ok" && <b className="font-medium text-ink">ok · v{health.version}</b>}
        {health.state === "checking" && <b className="font-medium text-dim">checking</b>}
        {health.state === "down" && (
          <b className="font-medium text-loss">
            unreachable · start it with uvicorn api.main:app --port 8000
          </b>
        )}
      </span>
      <span className="ml-auto flex items-center border-l border-line px-3">
        {mode && step ? `${mode.label} · ${step.label}` : "FSE/ML"}
      </span>
    </footer>
  );
}
