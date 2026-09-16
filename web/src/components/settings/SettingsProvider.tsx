"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api/client";

export type SettingValue = number | boolean;
export type Settings = Record<string, SettingValue>;

/** Where overrides lived before they moved to the account (PLAN.md 1.5); read once, then cleared. */
const LEGACY_STORAGE_KEY = "fse.settings.v1";
/** Debounce between the last edit and saving to the account. */
const SAVE_DELAY_MS = 300;

export type SaveState = "loading" | "saved" | "saving" | "error";

type SettingsContext = {
  /** Server defaults (core/config.py DEFAULTS); null until loaded */
  defaults: Settings | null;
  /** Only the keys the user changed. Sent as `settings` on every model run. */
  overrides: Settings;
  /** defaults with overrides applied */
  effective: Settings;
  set: (key: string, value: SettingValue) => void;
  reset: (key?: string) => void;
  /** Replace every override at once (opening a saved deal brings its settings). */
  replace: (overrides: Settings) => void;
  /** True once the defaults and the account's overrides have both loaded. */
  loaded: boolean;
  /** Whether the overrides are saved to the account. */
  saveState: SaveState;
  /** Result of /api/settings/validate for the current overrides */
  correlation: { valid: boolean; matrix: number[][] } | null;
  error?: string;
};

const Ctx = createContext<SettingsContext | null>(null);

function takeLegacy(): Settings {
  try {
    const raw = window.localStorage.getItem(LEGACY_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : {};
    return parsed && typeof parsed === "object" ? (parsed as Settings) : {};
  } catch {
    return {};
  }
}

function clearLegacy() {
  try {
    window.localStorage.removeItem(LEGACY_STORAGE_KEY);
  } catch {
    // Blocked storage: nothing to clear
  }
}

function knownChanges(settings: Settings, defaults: Settings): Settings {
  return Object.fromEntries(Object.entries(settings).filter(([k, v]) => k in defaults && defaults[k] !== v));
}

/**
 * Settings belong to the account (PLAN.md 1.5): the same on every device and
 * browser. Overrides load from /api/account/settings and every change is
 * saved back shortly after it's made.
 */
export function SettingsProvider({ children }: { children: React.ReactNode }) {
  const [defaults, setDefaults] = useState<Settings | null>(null);
  const [overrides, setOverrides] = useState<Settings>({});
  const [loaded, setLoaded] = useState(false);
  const [saveState, setSaveState] = useState<SaveState>("loading");
  const [correlation, setCorrelation] = useState<SettingsContext["correlation"]>(null);
  const [error, setError] = useState<string>();
  // JSON of the overrides the account holds, so loading doesn't save them straight back
  const savedKey = useRef<string | null>(null);

  // Defaults, then the account's overrides that still exist in them
  useEffect(() => {
    let cancelled = false;
    Promise.all([api.GET("/api/settings/defaults"), api.GET("/api/account/settings")])
      .then(([{ data }, account]) => {
        if (cancelled || !data) return;
        setDefaults(data.defaults);
        setCorrelation({ valid: data.correlation_valid, matrix: data.correlation_matrix });
        const stored = knownChanges(account.data?.settings ?? {}, data.defaults);
        savedKey.current = account.data ? JSON.stringify(stored) : null;
        // Overrides left in this browser from before settings followed the
        // account: bring them over once, if the account has none yet
        const legacy = knownChanges(takeLegacy(), data.defaults);
        const start = account.data && !Object.keys(stored).length && Object.keys(legacy).length ? legacy : stored;
        if (account.data) clearLegacy();
        setOverrides(start);
        setSaveState(account.data ? "saved" : "error");
        if (!account.data) setError("Can't load your saved settings; changes won't be kept.");
        setLoaded(true);
      })
      .catch(() => !cancelled && setError("Can't load settings from the API."));
    return () => {
      cancelled = true;
    };
  }, []);

  // Save to the account shortly after each change
  useEffect(() => {
    if (!loaded || savedKey.current === null) return;
    const key = JSON.stringify(overrides);
    if (key === savedKey.current) {
      setSaveState("saved");
      return;
    }
    setSaveState("saving");
    const id = setTimeout(() => {
      api
        .PUT("/api/account/settings", { body: { settings: overrides } })
        .then(({ data }) => {
          if (data) {
            savedKey.current = key;
            setSaveState((s) => (JSON.stringify(overrides) === key ? "saved" : s));
          } else setSaveState("error");
        })
        .catch(() => setSaveState("error"));
    }, SAVE_DELAY_MS);
    return () => clearTimeout(id);
  }, [overrides, loaded]);

  useEffect(() => {
    if (!defaults) return;
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

  const replace = useCallback(
    (next: Settings) => {
      const clean = defaults ? knownChanges(next, defaults) : next;
      setOverrides((prev) => (JSON.stringify(prev) === JSON.stringify(clean) ? prev : clean));
    },
    [defaults],
  );

  const effective = useMemo(() => ({ ...(defaults ?? {}), ...overrides }), [defaults, overrides]);
  const value = useMemo(
    () => ({ defaults, overrides, effective, set, reset, replace, loaded, saveState, correlation, error }),
    [defaults, overrides, effective, set, reset, replace, loaded, saveState, correlation, error],
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
