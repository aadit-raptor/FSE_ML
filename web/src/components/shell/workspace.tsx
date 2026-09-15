"use client";

import { MotionConfig } from "motion/react";
import { createContext, useContext, useMemo, useState } from "react";

/** Shell state shared with screens: whether the command search is open. */
type Workspace = {
  searchOpen: boolean;
  setSearchOpen: (open: boolean) => void;
};

const WorkspaceContext = createContext<Workspace | null>(null);

export function WorkspaceProvider({ children }: { children: React.ReactNode }) {
  const [searchOpen, setSearchOpen] = useState(false);
  const value = useMemo(() => ({ searchOpen, setSearchOpen }), [searchOpen]);
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
