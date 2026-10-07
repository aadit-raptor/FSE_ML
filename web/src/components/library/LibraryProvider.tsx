"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { fetchLibraryState, type LibraryState, libraryVisible, putLibrarySwitch } from "@/lib/library";

type LibraryContext = {
  /** Null until the API answers (or when it can't be reached: the library then stays out of the way). */
  state: LibraryState | null;
  loaded: boolean;
  /** The Library tab is listed: the library is on, or this caller may switch it back on. */
  visible: boolean;
  /** An administrator's switch; resolves false when the API refused it. */
  setEnabled: (enabled: boolean) => Promise<boolean>;
};

const Ctx = createContext<LibraryContext | null>(null);

/**
 * Whether the optional reference library is shown (PLAN.md 4.5), read once per session. When it is
 * off, the Library tab disappears for everyone but administrators and nothing else changes.
 */
export function LibraryProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<LibraryState | null>(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let live = true;
    void fetchLibraryState().then((s) => {
      if (!live) return;
      setState(s);
      setLoaded(true);
    });
    return () => {
      live = false;
    };
  }, []);

  const setEnabled = useCallback(async (enabled: boolean) => {
    try {
      const next = await putLibrarySwitch(enabled);
      if (next) setState(next);
      return !!next;
    } catch {
      return false;
    }
  }, []);

  const value = useMemo(() => ({ state, loaded, visible: libraryVisible(state), setEnabled }), [state, loaded, setEnabled]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useLibrary(): LibraryContext {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useLibrary must be used inside LibraryProvider");
  return ctx;
}
