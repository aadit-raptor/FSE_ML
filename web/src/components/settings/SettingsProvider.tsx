"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api/client";

export type SettingValue = number | boolean;
export type Settings = Record<string, SettingValue>;

const STORAGE_KEY = "fse.settings.v1";

type SettingsContext = {
  /** Server defaults (core/config.py DEFAULTS); null until loaded */
  defaults: Settings | null;
  /** Only the keys the user changed. Sent as `settings` on every model run. */
  overrides: Settings;
  /** defaults with overrides applied */
  effective: Settings;
  set: (key: string, value: SettingValue) => void;
  reset: (key?: string) => void;
  /** Result of /api/settings/validate for the current overrides */
  correlation: { valid: boolean; matrix: number[][] } | null;
  error?: string;
};

const Ctx = createContext<SettingsContext | null>(null);

function readStored(): Settings {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? (parsed as Settings) : {};
  } catch {
    return {};
  }
}

export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [defaults, setDefaults] = useState<Settings | null>(null);
  const [overrides, setOverrides] = useState<Settings>({});
  const [correlation, setCorrelation] = useState<SettingsContext["correlation"]>(null);
  const [error, setError] = useState<string>();

  // Defaults from the API, then any saved overrides that still exist there
  useEffect(() => {
    let cancelled = false;
    api
      .GET("/api/settings/defaults")
      .then(({ data }) => {
        if (cancelled || !data) return;
        setDefaults(data.defaults);
        const stored = readStored();
        setOverrides(Object.fromEntries(Object.entries(stored).filter(([k, v]) => k in data.defaults && data.defaults[k] !== v)));
        setCorrelation({ valid: data.correlation_valid, matrix: data.correlation_matrix });
      })
      .catch(() => !cancelled && setError("Can't load settings from the API."));
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!defaults) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(overrides));
    } catch {
      // Private mode or blocked storage: settings still apply for this session
    }
    const ctrl = new AbortController();
    const id = setTimeout(() => {
      api
        .POST("/api/settings/validate", { body: { settings: overrides }, signal: ctrl.signal })
        .then(({ data, error: err }) => {
          if (data) {
            setCorrelation({ valid: data.correlation_valid, matrix: data.correlation_matrix });
            setError(undefined);
          } else {
            const d = (err as { detail?: unknown } | undefined)?.detail;
            setError(typeof d === "string" ? d : "Settings rejected by the API.");
          }
        })
        .catch(() => {});
    }, 250);
    return () => {
      clearTimeout(id);
      ctrl.abort();
    };
  }, [overrides, defaults]);

  const set = useCallback(
    (key: string, value: SettingValue) => {
      setOverrides((prev) => {
        const next = { ...prev };
        if (defaults && defaults[key] === value) delete next[key];
        else next[key] = value;
        return next;
      });
    },
    [defaults],
  );

  const reset = useCallback((key?: string) => {
    setOverrides((prev) => {
      if (!key) return {};
      const next = { ...prev };
      delete next[key];
      return next;
    });
  }, []);

  const effective = useMemo(() => ({ ...(defaults ?? {}), ...overrides }), [defaults, overrides]);
  const value = useMemo(
    () => ({ defaults, overrides, effective, set, reset, correlation, error }),
    [defaults, overrides, effective, set, reset, correlation, error],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useSettings(): SettingsContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useSettings must be used inside SettingsProvider");
  return ctx;
}

/** Numeric setting with a fallback while defaults load. */
export function num(settings: Settings, key: string, fallback: number): number {
  const v = settings[key];
  return typeof v === "number" ? v : fallback;
}
