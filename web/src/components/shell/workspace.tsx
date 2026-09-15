"use client";

import { MotionConfig } from "motion/react";
import { createContext, useCallback, useContext, useMemo, useState } from "react";

/**
 * Shell state shared with screens.
 * - stale: modes whose results no longer match their inputs (e.g. Monte Carlo
 *   after a deal input changed). The top menu marks them.
 * - search: whether the command search is open.
 */
type Workspace = {
  staleModes: ReadonlySet<string>;
  setStale: (mode: string, stale: boolean) => void;
  searchOpen: boolean;
  setSearchOpen: (open: boolean) => void;
};

const WorkspaceContext = createContext<Workspace | null>(null);

export function WorkspaceProvider({ children }: { children: React.ReactNode }) {
  const [staleModes, setStaleModes] = useState<ReadonlySet<string>>(new Set());
  const [searchOpen, setSearchOpen] = useState(false);

  const setStale = useCallback((mode: string, stale: boolean) => {
    setStaleModes((prev) => {
      if (prev.has(mode) === stale) return prev;
      const next = new Set(prev);
      if (stale) next.add(mode);
      else next.delete(mode);
      return next;
    });
  }, []);

  const value = useMemo(
    () => ({ staleModes, setStale, searchOpen, setSearchOpen }),
    [staleModes, setStale, searchOpen],
  );
  return (
    <WorkspaceContext.Provider value={value}>
      <MotionConfig reducedMotion="user">{children}</MotionConfig>
    </WorkspaceContext.Provider>
  );
}

export function useWorkspace(): Workspace {
  const ctx = useContext(WorkspaceContext);
  if (!ctx) throw new Error("useWorkspace must be used inside WorkspaceProvider");
  return ctx;
}
